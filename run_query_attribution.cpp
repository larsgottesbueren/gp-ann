#include <iostream>
#include <filesystem>
#include <map>

#include "points_io.h"
#include "metis_io.h"
#include "recall.h"
#include "kmeans_tree_router.h"
#include "hnsw_router.h"

void PinThread(int cpu_id) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu_id, &mask);
    const int err = sched_setaffinity(0, sizeof(mask), &mask);
    if (err) {
        std::cerr << "Thread pinning failed" << std::endl;
        std::abort();
    }
}

void PrintAffinityMask() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    sched_getaffinity(0, sizeof(mask), &mask);

    // -1 = before first range
    // 0 = not in range
    // 1 = range just started
    // 2 = long range
    int range_status = -1;

    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &mask)) {
            if (range_status <= 0) {
                if (range_status >= 0) {
                    std::cout << ",";
                }
                std::cout << cpu;
                range_status = 1;
            } else if (range_status == 1) {
                ++range_status;
            }
        } else if (range_status == 1) {
            range_status = 0;
        } else if (range_status == 2) {
            std::cout << "-" << cpu - 1;
            range_status = 0;
        }
    }
    std::cout << std::endl;
}

void PinThread(cpu_set_t old_affinity) {
    sched_setaffinity(0, sizeof(old_affinity), &old_affinity);
}

void UnpinThread() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        CPU_SET(cpu, &mask);
    }
    PinThread(mask);
    PrintAffinityMask();
}

struct RoutingConfig {
    std::string routing_algorithm = "";
    size_t hnsw_num_voting_neighbors = 0;
    double routing_time = 0.0;
    bool try_increasing_num_shards = false;
    KMeansTreeRouterOptions routing_index_options;
    std::vector<std::vector<int>> buckets_to_probe;
};

std::vector<RoutingConfig> IterateRoutingConfigs(PointSet& points, PointSet& queries, std::vector<int>& partition, int num_shards, KMeansTreeRouterOptions routing_index_options) {
    KMeansTreeRouter router;
    std::vector<RoutingConfig> routes;
    Timer routing_timer; routing_timer.Start();
    router.Train(points, partition, routing_index_options);
    std::cout << "Training the router took " << routing_timer.Stop() << std::endl;

    {   // Standard tree-search routing
        std::vector<std::vector<int>> buckets_to_probe_by_query(queries.n);
        routing_timer.Start();
        for (size_t i = 0; i < queries.n; ++i) {
            buckets_to_probe_by_query[i] = router.Query(queries.GetPoint(i), routing_index_options.search_budget);
        }
        double time_routing = routing_timer.Stop();
        std::cout << "Routing took " << time_routing << " s overall, and " << time_routing / queries.n << " s per query" << std::endl;
        auto& new_route = routes.emplace_back();
        new_route.routing_algorithm = "KMeansTree";
        new_route.hnsw_num_voting_neighbors = 0;
        new_route.routing_time = time_routing;
        new_route.routing_index_options = routing_index_options;
        new_route.try_increasing_num_shards = true;
        new_route.buckets_to_probe = std::move(buckets_to_probe_by_query);
    }
    routing_timer.Start();
    auto [routing_points, partition_offsets] = router.ExtractPoints();
    std::cout << "Extraction finished" << std::endl;
    HNSWRouter hnsw_router(std::move(routing_points), std::move(partition_offsets),
                           HNSWParameters {
                                   .M = 32,
                                   .ef_construction = 200,
                                   .ef_search = 250 }
    );
    std::cout << "Training HNSW router took " << routing_timer.Stop() << " s" << std::endl;

    for (size_t num_voting_neighbors : {20, 40, 80, 120, 200, 400, 500}) {
        std::vector<std::vector<int>> buckets_to_probe_by_query_hnsw(queries.n);
        routing_timer.Start();
        for (size_t i = 0; i < queries.n; ++i) {
            buckets_to_probe_by_query_hnsw[i] = hnsw_router.Query(queries.GetPoint(i), num_voting_neighbors);
            if (buckets_to_probe_by_query_hnsw[i].size() != num_shards) {
                std::cerr << "What's going on" << std::endl;
            }
        }
        double time_routing = routing_timer.Stop();
        std::cout << "HNSW routing took " << time_routing << " s" << std::endl;
        auto& new_route = routes.emplace_back();
        new_route.routing_algorithm = "HNSW";
        new_route.hnsw_num_voting_neighbors = num_voting_neighbors;
        new_route.routing_time = time_routing;
        new_route.routing_index_options = routing_index_options;
        new_route.try_increasing_num_shards = true;
        new_route.buckets_to_probe = std::move(buckets_to_probe_by_query_hnsw);
    }

    // Pyramid routing where you visit the shards touched during the search
    for (size_t num_voting_neighbors : {20, 40, 80, 120, 200, 400, 500}) {
        std::vector<std::vector<int>> buckets_to_probe_by_query_hnsw(queries.n);
        routing_timer.Start();
        for (size_t i = 0; i < queries.n; ++i) {
            buckets_to_probe_by_query_hnsw[i] = hnsw_router.PyramidRoutingQuery(queries.GetPoint(i), num_voting_neighbors);
        }
        double time_routing = routing_timer.Stop();
        std::cout << "Pyramid routing took " << time_routing << " s" << std::endl;
        auto& new_route = routes.emplace_back();
        new_route.routing_algorithm = "Pyramid";
        new_route.hnsw_num_voting_neighbors = num_voting_neighbors;
        new_route.routing_time = time_routing;
        new_route.routing_index_options = routing_index_options;
        new_route.try_increasing_num_shards = false;
        new_route.buckets_to_probe = std::move(buckets_to_probe_by_query_hnsw);
    }

    // SPANN routing where you prune next shards based on how much further they are than the closest shard
    // --> i.e., dist(q, shard_i) > (1+eps) dist(q, shard_1) then cut off before i. eps in [0.6, 7] in the paper
    for (size_t num_voting_neighbors : {200, 400, 500}) {
        std::vector<std::vector<int>> buckets_to_probe_by_query_hnsw(queries.n);
        routing_timer.Start();
        for (size_t i = 0; i < queries.n; ++i) {
            buckets_to_probe_by_query_hnsw[i] = hnsw_router.SPANNRoutingQuery(queries.GetPoint(i), num_voting_neighbors, 0.6);
        }
        double time_routing = routing_timer.Stop();
        std::cout << "SPANN routing took " << time_routing << " s" << std::endl;
        auto& new_route = routes.emplace_back();
        new_route.routing_algorithm = "SPANN";
        new_route.hnsw_num_voting_neighbors = num_voting_neighbors;
        new_route.routing_time = time_routing;
        new_route.routing_index_options = routing_index_options;
        new_route.try_increasing_num_shards = false;
        new_route.buckets_to_probe = std::move(buckets_to_probe_by_query_hnsw);
    }

    return routes;
}

struct ShardSearch {
    ShardSearch() {};
    void Init(size_t ef_search, int num_shards, size_t num_queries) {
        this->ef_search = ef_search;
        query_hits_in_shard.assign(num_shards, std::vector<int>(num_queries, 0));
        time_query_in_shard.assign(num_shards, std::vector<double>(num_queries, 0.0));
    }
    size_t ef_search = 0;
    std::vector<std::vector<int>> query_hits_in_shard;
    std::vector<std::vector<double>> time_query_in_shard;
};

void AttributeRecallAndQueryTimeIncreasingNumProbes(const RoutingConfig& route, const ShardSearch& search, size_t num_queries, size_t num_shards, int num_neighbors, std::function<void(double,double)>& emit) {
    size_t total_hits = 0;
    std::vector<int> hits_per_query(num_queries, 0);
    std::vector<double> local_work(num_shards, 0.0);
    for (int n_probes = 1; n_probes <= num_shards; ++n_probes) {
        for (size_t q = 0; q < num_queries; ++q) {
            int b = route.buckets_to_probe[q][n_probes - 1];
            int diff = std::min(search.query_hits_in_shard[b][q], num_neighbors - hits_per_query[q]);
            hits_per_query[q] += diff;
            total_hits += diff;
            local_work[b] += search.time_query_in_shard[b][q];
        }

        double recall = static_cast<double>(total_hits) / (num_neighbors * num_queries);
        double max_latency = *std::max_element(local_work.begin(), local_work.end());
        double total_time = max_latency + (route.routing_time / num_shards);
        double QPS = num_queries / total_time;
        emit(recall, QPS);
        std::cout   << "NProbes = " << n_probes << " recall@k = " << recall << " total time " << total_time << " QPS = " << num_queries / total_time << std::endl;
        std::cout << "local work\t";
        for (double t : local_work) std::cout << t << " ";
        std::cout << std::endl;
    }
}

void AttributeRecallAndQueryTimeVariableNumProbes(const RoutingConfig& route, const ShardSearch& search, size_t num_queries, size_t num_shards, int num_neighbors, std::function<void(double,double)>& emit) {
    std::vector<double> local_work(num_shards, 0.0);
    size_t total_hits = 0;
    for (size_t q = 0; q < num_queries; ++q) {
        int hits = 0;
        for (int b : route.buckets_to_probe[q]) {
            hits += search.query_hits_in_shard[b][q];
            local_work[b] += search.time_query_in_shard[b][q];
        }
        hits = std::min(hits, num_neighbors);
        total_hits += hits;
    }

    double recall = static_cast<double>(total_hits) / (num_queries * num_neighbors);
    double max_latency = *std::max_element(local_work.begin(), local_work.end());
    double total_time = max_latency + (route.routing_time / num_shards);
    double QPS = num_queries / total_time;
    emit(recall, QPS);

    std::cout  << " recall@k = " << recall << " total time " << total_time << " QPS = " << num_queries / total_time << std::endl;
    std::cout << "local work\t";
    for (double t : local_work) std::cout << t << " ";
    std::cout << std::endl;
}

std::vector<ShardSearch> RunInShardSearches(
        PointSet& points, PointSet& queries, HNSWParameters hnsw_parameters, int num_neighbors,
        const std::vector<std::vector<uint32_t>>& clusters, int num_shards,
        const std::vector<float>& distance_to_kth_neighbor) {
    std::vector<size_t> ef_search_param_values = { 50, 80, 100, 150, 200, 250, 300, 400, 500 };

    Timer init_timer; init_timer.Start();
    std::vector<ShardSearch> shard_searches(ef_search_param_values.size());
    for (size_t i = 0; i < ef_search_param_values.size(); ++i) {
        shard_searches[i].Init(ef_search_param_values[i], num_shards, queries.n);
    }
    std::cout << "Init search output took " << init_timer.Stop() << std::endl;

    for (int b = 0; b < num_shards; ++b) {
        const auto& cluster = clusters[b];

        std::cout << "Start building HNSW for shard " << b << " of size " << cluster.size() << std::endl;

        #ifdef MIPS_DISTANCE
        using SpaceType = hnswlib::InnerProductSpace;
        #else
        using SpaceType = hnswlib::L2Space;
        #endif


        // PinThread(0);

        SpaceType space(points.d);

        Timer build_timer; build_timer.Start();
        hnswlib::HierarchicalNSW<float> hnsw(&space, cluster.size(), hnsw_parameters.M, hnsw_parameters.ef_construction, 555 + b);

        // UnpinThread();

        parlay::parallel_for(0, cluster.size(), [&](size_t i) {
            float* p = points.GetPoint(cluster[i]);
            hnsw.addPoint(p, i);
        });

        std::cout << "HNSW build took " << build_timer.Stop() << std::endl;

        // PinThread(0);

        size_t ef_search_param_id = 0;
        for (size_t ef_search : ef_search_param_values) {
            hnsw.setEf(ef_search);

            Timer timer;
            for (size_t q = 0; q < queries.n; ++q) {
                float* Q = queries.GetPoint(q);
                timer.Start();
                auto result = hnsw.searchKnn(Q, num_neighbors);
                shard_searches[ef_search_param_id].time_query_in_shard[b][q] = timer.Stop();
                while (!result.empty()) {
                    auto top = result.top();
                    result.pop();
                    if (top.first <= distance_to_kth_neighbor[q]) {
                        shard_searches[ef_search_param_id].query_hits_in_shard[b][q]++;
                    }
                }
            }

            std::cout << "Shard search with ef-search = " << ef_search << " took " << timer.total_duration.count() << std::endl;

            ef_search_param_id++;
        }

        std::cout << "Finished searches in bucket " << b << std::endl;

        // UnpinThread();
    }

    return shard_searches;
}


int main(int argc, const char* argv[]) {
    if (argc != 7) {
        std::cerr << "Usage ./QueryAttribution input-points queries ground-truth-file k partition output-file" << std::endl;
        std::abort();
    }

    cpu_set_t old_affinity;
    CPU_ZERO(&old_affinity);
    sched_getaffinity(0, sizeof(old_affinity), &old_affinity);

    std::string point_file = argv[1];
    std::string query_file = argv[2];
    std::string ground_truth_file = argv[3];
    std::string k_string = argv[4];
    int num_neighbors = std::stoi(k_string);
    std::string partition_file = argv[5];
    std::string output_file = argv[6];

    PointSet points = ReadPoints(point_file);
    PointSet queries = ReadPoints(query_file);

    #ifdef MIPS_DISTANCE
    Normalize(points);
    Normalize(queries);
    #endif

    std::vector<NNVec> ground_truth;
    if (std::filesystem::exists(ground_truth_file)) {
        ground_truth = ReadGroundTruth(ground_truth_file);
        std::cout << "Read ground truth file" << std::endl;
    } else {
        std::cout << "start computing ground truth" << std::endl;
        ground_truth = ComputeGroundTruth(points, queries, num_neighbors);
        std::cout << "computed ground truth" << std::endl;
    }
    std::vector<float> distance_to_kth_neighbor = ConvertGroundTruthToDistanceToKthNeighbor(ground_truth, num_neighbors, points, queries);
    std::cout << "Finished computing distance to kth neighbor" << std::endl;

    std::vector<int> partition = ReadMetisPartition(partition_file);
    int num_shards = *std::max_element(partition.begin(), partition.end()) + 1;

    KMeansTreeRouterOptions router_options;
    router_options.budget = points.n / num_shards;
    std::vector<RoutingConfig> routes = IterateRoutingConfigs(points, queries, partition, num_shards, router_options);
    std::cout << "Finished routing configs" << std::endl;


    Timer timer;
    timer.Start();
    std::vector<std::vector<uint32_t>> clusters(num_shards);
    for (uint32_t i = 0; i < partition.size(); ++i) {
        clusters[partition[i]].push_back(i);
    }
    std::cout << "Convert partition to clusters took " << timer.Stop() << std::endl;

    std::cout << "Start shard searches" << std::endl;
    std::vector<ShardSearch> shard_searches = RunInShardSearches(points, queries, HNSWParameters(), num_neighbors, clusters, num_shards, distance_to_kth_neighbor);
    std::cout << "Finished shard searches" << std::endl;

    std::ofstream out(output_file);
    // header
    out << "partitioning,shard query,routing query,routing index,ef-search-shard,num voting points,recall,QPS";

    struct Desc {
        std::string format_string;
        double recall;
        double QPS;
    };

    std::map<std::string, std::vector<Desc>> outputs;

    for (const auto& route : routes) {
        for (const auto& search : shard_searches) {
            std::function<void(double, double)> format_output = [&](double recall, double QPS) -> void {
                std::stringstream str;
                str << "GP,HNSW," << route.routing_algorithm << ",KMeansTree,"
                    << search.ef_search << "," << route.hnsw_num_voting_neighbors
                    << "," << recall << "," << QPS;
                out << str.str() << std::endl;
                std::cout << str.str() << std::endl;
                outputs[route.routing_algorithm].push_back(Desc{ .format_string = str.str(), .recall = recall, .QPS = QPS });
            };
            if (route.try_increasing_num_shards) {
                AttributeRecallAndQueryTimeIncreasingNumProbes(route, search, queries.n, num_shards, num_neighbors, format_output);
            } else {
                AttributeRecallAndQueryTimeVariableNumProbes(route, search, queries.n, num_shards, num_neighbors, format_output);
            }
        }
    }

    std::cout << "Only Pareto" << std::endl;

    std::ofstream pareto_out(output_file + ".pareto");
    for (auto& [routing_algo, configs] : outputs) {
        if (configs.empty()) continue;

        auto dominates = [](const Desc& l, const Desc& r) -> bool { return l.recall < r.recall && l.QPS < r.QPS; };

        std::vector<Desc> pareto;
        for (const auto& c : configs) {
            bool insert_new = true;
            for (int64_t i = 0; i < pareto.size(); ++i) {
                if (dominates(pareto[i], c)) {  // remove pareto[i]
                    pareto[i] = std::move(pareto.back());
                    pareto.pop_back();
                    --i;
                } else if (dominates(c, pareto[i])) {
                    insert_new = false;
                    break;
                }
            }
            if (insert_new) {
                pareto.push_back(c);
            }
        }

        for (const auto& c : pareto) {
            pareto_out << c.format_string << std::endl;
            std::cout << c.format_string << std::endl;
        }
    }


}
