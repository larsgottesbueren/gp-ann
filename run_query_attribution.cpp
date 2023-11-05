#include <iostream>
#include <filesystem>
#include <map>

#include "points_io.h"
#include "metis_io.h"
#include "recall.h"
#include "kmeans_tree_router.h"
#include "hnsw_router.h"

void PrintAffinityMask();

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
}

struct RoutingConfig {
    std::string routing_algorithm = "None";
    std::string index_trainer = "KMeansTree";
    size_t hnsw_num_voting_neighbors = 0;
    size_t hnsw_ef_search = 250;
    double routing_time = 0.0;
    size_t routing_distance_calcs = 0;
    bool try_increasing_num_shards = false;
    KMeansTreeRouterOptions routing_index_options;
    std::vector<std::vector<int>> buckets_to_probe;

    std::string Serialize() const {
        std::stringstream sb;
        sb  << routing_algorithm << " " << index_trainer << " " << hnsw_num_voting_neighbors << " " << hnsw_ef_search << " "
            << routing_time << " " << std::boolalpha << try_increasing_num_shards << std::noboolalpha << " " << buckets_to_probe.size() << "\n";
        for (const auto& visit_order : buckets_to_probe) {
            for (const int b : visit_order) {
                sb << b << " ";
            }
            sb << "\n";
        }
        return sb.str();
    }

    static RoutingConfig Deserialize(std::ifstream& in) {
        RoutingConfig r;
        int num_queries = 0;
        std::string line;
        std::getline(in, line);
        std::istringstream iss(line);
        iss >> r.routing_algorithm >> r.index_trainer >> r.hnsw_num_voting_neighbors >> r.hnsw_ef_search >> r.routing_time >> std::boolalpha >> r.try_increasing_num_shards >> std::noboolalpha >> num_queries;
        std::cout << r.routing_algorithm << " " << r.index_trainer << " " << r.hnsw_num_voting_neighbors << " " << r.hnsw_ef_search << " " << r.routing_time << " " << std::boolalpha << " " << r.try_increasing_num_shards << " " << std::noboolalpha << num_queries << std::endl;
        for (int i = 0; i < num_queries; ++i) {
            std::getline(in, line);
            std::istringstream line_stream(line);
            int b = 0;
            auto& visit_order = r.buckets_to_probe.emplace_back();
            while (line_stream >> b) {
                visit_order.push_back(b);
            }
        }
        return r;
    }
};


void IterateHNSWRouterConfigs(HNSWRouter& hnsw_router, PointSet& queries, std::vector<RoutingConfig>& routes, const RoutingConfig& blueprint) {
    Timer routing_timer;
    for (size_t num_voting_neighbors : {20, 40, 80, 120, 200, 400, 500}) {
        std::cout << "num voting neighbors " << num_voting_neighbors << " num queries " << queries.n << std::endl;
        std::vector<std::vector<int>> buckets_to_probe_by_query_hnsw(queries.n);
        routing_timer.Start();
        for (size_t i = 0; i < queries.n; ++i) {
            buckets_to_probe_by_query_hnsw[i] = hnsw_router.Query(queries.GetPoint(i), num_voting_neighbors);
        }
        double time_routing = routing_timer.Stop();
        std::cout << "HNSW routing took " << time_routing << " s" << std::endl;
        routes.push_back(blueprint);
        auto& new_route = routes.back();
        new_route.routing_algorithm = "HNSW";
        new_route.hnsw_num_voting_neighbors = num_voting_neighbors;
        new_route.routing_time = time_routing;

        new_route.try_increasing_num_shards = true;
        new_route.buckets_to_probe = std::move(buckets_to_probe_by_query_hnsw);

        new_route.routing_distance_calcs = hnsw_router.hnsw->metric_distance_computations / queries.n;
        hnsw_router.hnsw->metric_distance_computations = 0;
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
        routes.push_back(blueprint);
        auto& new_route = routes.back();
        new_route.routing_algorithm = "Pyramid";
        new_route.hnsw_num_voting_neighbors = num_voting_neighbors;
        new_route.routing_time = time_routing;
        new_route.try_increasing_num_shards = false;
        new_route.buckets_to_probe = std::move(buckets_to_probe_by_query_hnsw);

        new_route.routing_distance_calcs = hnsw_router.hnsw->metric_distance_computations / queries.n;
        hnsw_router.hnsw->metric_distance_computations = 0;
    }

    // SPANN routing where you prune next shards based on how much further they are than the closest shard
    // --> i.e., dist(q, shard_i) > (1+eps) dist(q, shard_1) then cut off before i. eps in [0.6, 7] in the paper
    for (size_t num_voting_neighbors : {20, 40, 80, 120, 200, 400, 500}) {
        std::vector<std::vector<int>> buckets_to_probe_by_query_hnsw(queries.n);
        routing_timer.Start();
        for (size_t i = 0; i < queries.n; ++i) {
            buckets_to_probe_by_query_hnsw[i] = hnsw_router.SPANNRoutingQuery(queries.GetPoint(i), num_voting_neighbors, 0.6);
        }
        double time_routing = routing_timer.Stop();
        std::cout << "SPANN routing took " << time_routing << " s" << std::endl;
        routes.push_back(blueprint);
        auto& new_route = routes.back();
        new_route.routing_algorithm = "SPANN";
        new_route.hnsw_num_voting_neighbors = num_voting_neighbors;
        new_route.routing_time = time_routing;
        new_route.try_increasing_num_shards = false;
        new_route.buckets_to_probe = std::move(buckets_to_probe_by_query_hnsw);

        new_route.routing_distance_calcs = hnsw_router.hnsw->metric_distance_computations / queries.n;
        hnsw_router.hnsw->metric_distance_computations = 0;
    }
}

std::vector<RoutingConfig> IterateRoutingConfigs(PointSet& points, PointSet& queries, const std::vector<int>& partition, int num_shards,
                                                 KMeansTreeRouterOptions routing_index_options, const std::string& routing_index_file,
                                                 const std::string& pyramid_index_file, const std::string& our_pyramid_index_file,
                                                 bool our_pyramid_is_hnsw_partition = false) {
    std::vector<RoutingConfig> routes;

    {
        KMeansTreeRouter router;
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
            new_route.routing_distance_calcs = routing_index_options.search_budget;
            new_route.try_increasing_num_shards = true;
            new_route.buckets_to_probe = std::move(buckets_to_probe_by_query);
        }
        routing_timer.Start();
        auto [routing_points, partition_offsets] = router.ExtractPoints();
        std::cout << "Extraction finished" << std::endl;

        std::vector<int> routing_index_partition;
        for (size_t i = 1; i < partition_offsets.size(); ++i) {
            for (int j = partition_offsets[i-1]; j < partition_offsets[i]; ++j) {
                routing_index_partition.push_back(i-1);
            }
        }

        HNSWRouter hnsw_router(routing_points, num_shards, routing_index_partition,
                               HNSWParameters {
                                       .M = 32,
                                       .ef_construction = 200,
                                       .ef_search = 200 }
        );
        std::cout << "Training HNSW router took " << routing_timer.Restart() << " s" << std::endl;
        hnsw_router.Serialize(routing_index_file);
        std::cout << "Serializing HNSW router took " << routing_timer.Stop() << " s" << std::endl;

        RoutingConfig blueprint;
        blueprint.index_trainer = "HierKMeans";
        IterateHNSWRouterConfigs(hnsw_router, queries, routes, blueprint);
    }

    if (!pyramid_index_file.empty()) {
        std::cout << "Run Pyramid routing" << std::endl;
        std::vector<int> routing_index_partition = ReadMetisPartition(pyramid_index_file + ".routing_index_partition");
        HNSWRouter hnsw_router(pyramid_index_file, points.d, routing_index_partition);
        RoutingConfig blueprint;
        blueprint.index_trainer = "Pyramid";
        IterateHNSWRouterConfigs(hnsw_router, queries, routes, blueprint);
    }

    if (!our_pyramid_index_file.empty()) {
        std::cout << "Run OurPyramid++ routing" << std::endl;
        std::vector<int> routing_index_partition = ReadMetisPartition(
                pyramid_index_file + (our_pyramid_is_hnsw_partition ? ".hnsw" : ".knn") + ".routing_index_partition");
        HNSWRouter hnsw_router(pyramid_index_file, points.d, routing_index_partition);
        RoutingConfig blueprint;
        blueprint.index_trainer = std::string("OurPyramid+") + std::string(our_pyramid_is_hnsw_partition ? "HNSW" : "KNN");
        IterateHNSWRouterConfigs(hnsw_router, queries, routes, blueprint);
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
    std::string Serialize() const {
        std::stringstream out;
        out << ef_search << " " << query_hits_in_shard.size() << " " << query_hits_in_shard[0].size() << "\n";
        for (const auto& qh : query_hits_in_shard) {
            for (int x : qh) { out << x << " "; }
            out << "\n";
        }
        for (const auto& tq : time_query_in_shard) {
            for (double x : tq) { out << x << " "; }
            out << "\n";
        }
        return out.str();
    }

    static ShardSearch Deserialize(std::ifstream& in) {
        ShardSearch s;
        int num_shards, num_queries;
        std::string line;
        std::getline(in, line);
        std::istringstream iss(line);
        iss >> s.ef_search >> num_shards >> num_queries;
        for (int i = 0; i < num_shards; ++i) {
            std::getline(in, line);
            std::istringstream line_stream(line);
            s.query_hits_in_shard.emplace_back();
            int hits;
            while (line_stream >> hits) {
                s.query_hits_in_shard.back().push_back(hits);
            }
            assert(s.query_hits_in_shard.back().size() == num_queries);
        }

        for (int i = 0; i < num_shards; ++i) {
            std::getline(in, line);
            std::istringstream line_stream(line);
            s.time_query_in_shard.emplace_back();
            double time;
            while (line_stream >> time) {
                s.time_query_in_shard.back().push_back(time);
            }
        }
        return s;
    }
};

struct EmitResult {
    std::vector<double> local_work;
    size_t total_hits;
    double n_probes;
};

void AttributeRecallAndQueryTimeIncreasingNumProbes(const RoutingConfig& route, const ShardSearch& search, size_t num_queries, size_t num_shards, int num_neighbors, std::function<void(EmitResult)>& emit) {
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
        emit(EmitResult{
            .local_work = local_work,
            .total_hits = total_hits,
            .n_probes = static_cast<double>(n_probes),
        });
    }
}

void AttributeRecallAndQueryTimeVariableNumProbes(const RoutingConfig& route, const ShardSearch& search, size_t num_queries, size_t num_shards, int num_neighbors, std::function<void(EmitResult)>& emit) {
    std::vector<double> local_work(num_shards, 0.0);
    size_t total_hits = 0;
    size_t total_num_probes = 0;
    for (size_t q = 0; q < num_queries; ++q) {
        total_num_probes += route.buckets_to_probe[q].size();
        int hits = 0;
        for (int b : route.buckets_to_probe[q]) {
            hits += search.query_hits_in_shard[b][q];
            local_work[b] += search.time_query_in_shard[b][q];
        }
        hits = std::min(hits, num_neighbors);
        total_hits += hits;
    }
    emit(EmitResult{
            .local_work = local_work,
            .total_hits = total_hits,
            .n_probes = double(total_num_probes) / num_queries,
    });
}

std::vector<ShardSearch> RunInShardSearches(
        PointSet& points, PointSet& queries, HNSWParameters hnsw_parameters, int num_neighbors,
        std::vector<std::vector<uint32_t>>& clusters, int num_shards,
        const std::vector<float>& distance_to_kth_neighbor) {
    std::vector<size_t> ef_search_param_values = { 50, 80, 100, 150, 200, 250, 300, 400, 500 };

    Timer init_timer; init_timer.Start();
    std::vector<ShardSearch> shard_searches(ef_search_param_values.size());
    for (size_t i = 0; i < ef_search_param_values.size(); ++i) {
        shard_searches[i].Init(ef_search_param_values[i], num_shards, queries.n);
    }
    std::cout << "Init search output took " << init_timer.Stop() << std::endl;

    for (int b = 0; b < num_shards; ++b) {
        auto& cluster = clusters[b];

        std::cout << "Start building HNSW for shard " << b << " of size " << cluster.size() << std::endl;

        #ifdef MIPS_DISTANCE
        using SpaceType = hnswlib::InnerProductSpace;
        #else
        using SpaceType = hnswlib::L2Space;
        #endif


        PinThread(0);

        SpaceType space(points.d);

        Timer build_timer; build_timer.Start();
        hnswlib::HierarchicalNSW<float> hnsw(&space, cluster.size(), hnsw_parameters.M, hnsw_parameters.ef_construction, 555 + b);

        UnpinThread();

        std::mt19937 prng(555 + b);
        std::shuffle(cluster.begin(), cluster.end(), prng);

        // do some insertion sequentially
        size_t seq_insertion = std::min(1UL << 11, cluster.size());
        for (size_t i = 0; i < seq_insertion; ++i) {
            hnsw.addPoint(points.GetPoint(cluster[i]), i);
        }
        parlay::parallel_for(seq_insertion, cluster.size(), [&](size_t i) {
            hnsw.addPoint(points.GetPoint(cluster[i]), i);
        }, 512);

        std::cout << "HNSW build took " << build_timer.Stop() << std::endl;

        PinThread(0);

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

        UnpinThread();
    }

    return shard_searches;
}

void Serialize(const std::vector<RoutingConfig>& routes, const std::vector<ShardSearch>& shard_searches, const std::string& output_file) {
    std::ofstream out(output_file);
    out << routes.size() << " " << shard_searches.size() << std::endl;
    for (const RoutingConfig& r : routes) {
        out << "R" << std::endl;
        out << r.Serialize();
    }
    for (const ShardSearch& search : shard_searches) {
        out << "S" << std::endl;
        out << search.Serialize();
    }
}

void Deserialize(std::vector<RoutingConfig>& routes, std::vector<ShardSearch>& shard_searches, const std::string& input_file) {
    std::ifstream in(input_file);
    size_t num_routes, num_searches;
    std::string header;
    std::getline(in, header);
    std::istringstream iss(header);
    iss >> num_routes >> num_searches;
    std::cout << "nr=" << num_routes << " ns=" << num_searches << std::endl;
    for (size_t i = 0; i < num_routes; ++i) {
        std::getline(in, header);
        std::cout << "i = " << i << " for routes " << std::endl;
        if (header != "R") std::cout << "routing config doesn't start with marker R. Instead: " << header << std::endl;
        RoutingConfig r = RoutingConfig::Deserialize(in);
        routes.push_back(std::move(r));
    }
    for (size_t i = 0; i < num_searches; ++i) {
        std::getline(in, header);
        if (header != "S") std::cout << "search config doesn't start with marker S. Instead: " << header << std::endl;
        ShardSearch s = ShardSearch::Deserialize(in);
        shard_searches.push_back(std::move(s));
    }
}


int main(int argc, const char* argv[]) {
    if (argc != 9) {
        std::cerr << "Usage ./QueryAttribution input-points queries ground-truth-file num_neighbors partition-file output-file partition_method requested-num-shards" << std::endl;
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
    std::string part_method = argv[7];
    std::string requested_num_shards_str = argv[8];
    int requested_num_shards = std::stoi(requested_num_shards_str);

    PointSet points = ReadPoints(point_file);
    PointSet queries = ReadPoints(query_file);

    #ifdef MIPS_DISTANCE
    Normalize(points);
    Normalize(queries);
    #endif

    std::vector<int> partition = ReadMetisPartition(partition_file);
    int num_shards = NumPartsInPartition(partition);

    KMeansTreeRouterOptions router_options;
    router_options.budget = points.n / num_shards;
    std::string pyramid_index_file, our_pyramid_index_file;
    bool our_pyramid_is_hnsw_partition = false;
    if (part_method == "Pyramid") {
        pyramid_index_file = partition_file + ".pyramid_routing_index";
    }
    if (part_method == "OurPyramid") {
        our_pyramid_index_file = partition_file + ".our_pyramid_routing_index";
        if (partition_file.ends_with(".hnsw_graph_part")) {
            our_pyramid_is_hnsw_partition = true;
        }
    }
    std::vector<RoutingConfig> routes = IterateRoutingConfigs(points, queries, partition, num_shards, router_options,
                                                              partition_file + ".routing_index", pyramid_index_file, our_pyramid_index_file,
                                                              our_pyramid_is_hnsw_partition);
    std::cout << "Finished routing configs" << std::endl;

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

    Serialize(routes, shard_searches, output_file + ".routes_and_searches.txt");

    std::ofstream out(output_file);
    // header
    std::string header = "partitioning,shard query,routing query,routing index,ef-search-shard,num voting points,routing time,num probes,recall,QPS,QPS per host,"
                         "QPS without routing, QPS without routing per host,num hosts,num shards,requested num shards\n";
    out << header;

    struct Desc {
        std::string format_string;
        double recall;
        double QPS_per_host;
    };

    std::map<std::string, std::vector<Desc>> outputs;

    for (const auto& route : routes) {
        for (const auto& search : shard_searches) {
            std::function<void(EmitResult)> format_output = [&](const EmitResult& r) -> void {
                double recall = static_cast<double>(r.total_hits) / static_cast<double>(num_neighbors * queries.n);

                auto lwr = r.local_work;
                std::vector<size_t> assigned_hosts(num_shards, 1);
                const size_t num_queries = queries.n;
                size_t num_hosts = num_shards;

                for (size_t extra_hosts = 0; extra_hosts < 21; ++extra_hosts, ++num_hosts) {
                    const size_t max_shard = std::distance(lwr.begin(), std::max_element(lwr.begin(), lwr.end()));
                    const double max_latency = lwr[max_shard];

                    {   // output and formatting bits
                        double QPS_without_routing = num_queries / max_latency;
                        double QPS_without_routing_per_host = QPS_without_routing / num_hosts;

                        double total_time = max_latency + (route.routing_time / num_hosts);
                        double QPS = num_queries / total_time;
                        double QPS_per_host = QPS / num_hosts;

                        std::stringstream str;
                        str << part_method << ",HNSW," << route.routing_algorithm << "," << route.index_trainer << ","
                            << search.ef_search << "," << route.hnsw_num_voting_neighbors
                            << "," << route.routing_time / queries.n
                            << "," << r.n_probes << "," << recall << "," << QPS << "," << QPS_per_host
                            << "," << QPS_without_routing << "," << QPS_without_routing_per_host
                            << "," << num_hosts << "," << num_shards << "," << requested_num_shards << "\n";
                        out << str.str() << std::flush;
                        std::cout << str.str() << std::flush;
                        outputs[route.routing_algorithm].push_back(Desc{ .format_string = str.str(), .recall = recall, .QPS_per_host = QPS_per_host });
                    }

                    // assign one more replica to the slowest shard
                    assigned_hosts[max_shard]++;
                    lwr[max_shard] = r.local_work[max_shard] / assigned_hosts[max_shard];
                }
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
    pareto_out << header;
    for (auto& [routing_algo, configs] : outputs) {
        if (configs.empty()) continue;

        auto dominates = [](const Desc& l, const Desc& r) -> bool { return l.recall < r.recall && l.QPS_per_host < r.QPS_per_host; };

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
