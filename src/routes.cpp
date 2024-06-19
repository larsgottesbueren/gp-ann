#include "routes.h"

#include <filesystem>
#include <fstream>
#include <sstream>

#include "hnsw_router.h"
#include "kmeans_tree_router.h"
#include "metis_io.h"

double MaxFirstShardRoutingRecall(const std::vector<std::vector<int>>& buckets_to_probe, const std::vector<NNVec>& ground_truth, int num_neighbors,
                                  const Cover& cover) {
    if (ground_truth.empty()) {
        std::cerr << "Ground truth empty. Max recall calculation during routing not possible (not necessarily an issue).";
        return 555.0;
    }
    size_t hits = 0;
    const size_t num_queries = buckets_to_probe.size();
    for (size_t q = 0; q < num_queries; ++q) {
        if (buckets_to_probe[q].empty())
            continue;
        int probe = buckets_to_probe[q][0];
        for (int i = 0; i < num_neighbors; ++i) {
            const uint32_t neigh = ground_truth[q][i].second;
            if (std::find(cover[neigh].begin(), cover[neigh].end(), probe) != cover[neigh].end()) {
                hits++;
            }
        }
    }
    return static_cast<double>(hits) / num_queries / num_neighbors;
}

void IterateHNSWRouterConfigsInScheduler(HNSWRouter& hnsw_router, PointSet& queries, std::vector<RoutingConfig>& routes, const RoutingConfig& blueprint,
                                         const std::vector<NNVec>& ground_truth, int num_neighbors, const Cover& cover) {
    Timer routing_timer;
    for (size_t num_voting_neighbors : { 20, 40, 80, 120, 200, 250, 300, 400, 500 }) {
        std::cout << "num voting neighbors " << num_voting_neighbors << " num queries " << queries.n << std::endl;
        hnsw_router.hnsw->setEf(num_voting_neighbors);
        std::vector<HNSWRouter::ShardPriorities> routing_objects(queries.n);

        routing_timer.Start();
        parlay::parallel_for(0, queries.n, [&](size_t i) { routing_objects[i] = hnsw_router.Query(queries.GetPoint(i), num_voting_neighbors); });
        double time_routing = routing_timer.Stop();

        std::cout << "HNSW routing took " << time_routing << " s" << std::endl;

        // HNSW routing
        {
            std::vector<std::vector<int>> buckets_to_probe_by_query_hnsw(queries.n);
            parlay::parallel_for(0, queries.n, [&](size_t i) { buckets_to_probe_by_query_hnsw[i] = routing_objects[i].RoutingQuery(); });
            double first_shard_recall = MaxFirstShardRoutingRecall(buckets_to_probe_by_query_hnsw, ground_truth, num_neighbors, cover);
            std::cout << "HNSW routing first shard recall = " << first_shard_recall << std::endl;

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
        // Pyramid routing
        {
            std::vector<std::vector<int>> buckets_to_probe_by_query_hnsw(queries.n);
            parlay::parallel_for(0, queries.n, [&](size_t i) { buckets_to_probe_by_query_hnsw[i] = routing_objects[i].PyramidRoutingQuery(); });

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
        // SPANN routing
        {
            // where you prune next shards based on how much further they are than the closest shard
            // --> i.e., dist(q, shard_i) > (1+eps) dist(q, shard_1) then cut off before i. eps in [0.6, 7] in the paper
            std::vector<std::vector<int>> buckets_to_probe_by_query_hnsw(queries.n);
            parlay::parallel_for(0, queries.n, [&](size_t i) { buckets_to_probe_by_query_hnsw[i] = routing_objects[i].SPANNRoutingQuery(/*eps=*/0.6); });
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

        // frequency routing
        {
            std::vector<std::vector<int>> buckets_to_probe_by_query_hnsw(queries.n);
            parlay::parallel_for(0, queries.n, [&](size_t i) { buckets_to_probe_by_query_hnsw[i] = routing_objects[i].FrequencyQuery(); });
            double first_shard_recall = MaxFirstShardRoutingRecall(buckets_to_probe_by_query_hnsw, ground_truth, num_neighbors, cover);
            std::cout << "HNSW frequency routing first shard recall = " << first_shard_recall << std::endl;
            routes.push_back(blueprint);
            auto& new_route = routes.back();
            new_route.routing_algorithm = "HNSW-Frequency";
            new_route.hnsw_num_voting_neighbors = num_voting_neighbors;
            new_route.routing_time = time_routing;
            new_route.try_increasing_num_shards = true;
            new_route.buckets_to_probe = std::move(buckets_to_probe_by_query_hnsw);
        }
    }
}

void IterateHNSWRouterConfigs(HNSWRouter& hnsw_router, PointSet& queries, std::vector<RoutingConfig>& routes, const RoutingConfig& blueprint,
                              const std::vector<NNVec>& ground_truth, int num_neighbors, const Cover& cover) {
    parlay::execute_with_scheduler(std::min<size_t>(32, parlay::num_workers()),
                                   [&] { IterateHNSWRouterConfigsInScheduler(hnsw_router, queries, routes, blueprint, ground_truth, num_neighbors, cover); });
}

std::vector<KMeansTreeRouterOptions> GenerateRouterConfigs(KMeansTreeRouterOptions routing_index_options_blueprint) {
    std::vector<KMeansTreeRouterOptions> routing_index_option_vals;

#if false
    for (double factor : {0.2, 0.4, 0.8, 1.0 }) {
        KMeansTreeRouterOptions ro = routing_index_options_blueprint;
        ro.budget *= factor;
        routing_index_option_vals.push_back(ro);
    }
#else
    for (int64_t budget : { 20'000, 100'000, 200'000, 500'000, 1'000'000, 2'000'000, 5'000'000, 10'000'000 }) {
        KMeansTreeRouterOptions ro = routing_index_options_blueprint;
        ro.budget = budget;
        routing_index_option_vals.push_back(ro);
    }
#endif

    auto copy = routing_index_option_vals;
    routing_index_option_vals.clear();
    for (auto ro : copy) {
        for (int min_cluster_size : { 350 }) {
            ro.min_cluster_size = min_cluster_size;
            routing_index_option_vals.push_back(ro);
        }
    }

    copy = routing_index_option_vals;
    routing_index_option_vals.clear();
    for (auto ro : copy) {
        for (int num_centroids : { 64 }) {
            ro.num_centroids = num_centroids;
            routing_index_option_vals.push_back(ro);
        }

        if (ro.budget >= 5'000'000) {
            ro.num_centroids = 128;
            routing_index_option_vals.push_back(ro);
        }
    }

    return routing_index_option_vals;
}


std::vector<RoutingConfig> IterateRoutingConfigs(PointSet& points, PointSet& queries, const Clusters& clusters, int num_shards,
                                                 KMeansTreeRouterOptions routing_index_options_blueprint, const std::vector<NNVec>& ground_truth,
                                                 int num_neighbors, const std::string& routing_index_file, const std::string& pyramid_index_file,
                                                 const std::string& our_pyramid_index_file) {
    std::vector<RoutingConfig> routes;

    std::vector<KMeansTreeRouterOptions> routing_index_option_vals = GenerateRouterConfigs(routing_index_options_blueprint);


    const Cover cover = ConvertClustersToCover(clusters);

    for (const KMeansTreeRouterOptions& routing_index_options : routing_index_option_vals) {
        std::cout << "Train router on " << routing_index_options.num_centroids << " centroids " << routing_index_options.min_cluster_size
                  << " min cluster size " << routing_index_options.budget << " size budget " << std::endl;

        PointSet routing_points;
        std::vector<int> routing_index_partition;
        Timer routing_timer;
        routing_timer.Start();
        // compute routing points and run tree-search routing
        {
            KMeansTreeRouter router;

            router.Train(points, clusters, routing_index_options);
            std::cout << "Training the router took " << routing_timer.Stop() << std::endl;

            // Standard tree-search routing
            {
                std::vector<std::vector<int>> buckets_to_probe_by_query(queries.n);
                double time_routing;
                parlay::execute_with_scheduler(std::min<size_t>(32, parlay::num_workers()), [&] {
                    routing_timer.Start();
                    parlay::parallel_for(0, queries.n, [&](size_t i) {
                        buckets_to_probe_by_query[i] = router.Query(queries.GetPoint(i), routing_index_options.search_budget);
                    });
                    time_routing = routing_timer.Stop();
                });

                double first_shard_recall = MaxFirstShardRoutingRecall(buckets_to_probe_by_query, ground_truth, num_neighbors, cover);
                std::cout << "KMTR Routing took " << time_routing << " s overall, and " << time_routing / queries.n
                          << " s per query. Max first shard recall = " << first_shard_recall << std::endl;
                auto& new_route = routes.emplace_back();
                new_route.routing_algorithm = "KMeansTree";
                new_route.hnsw_num_voting_neighbors = 0;
                new_route.routing_time = time_routing;
                new_route.routing_index_options = routing_index_options;
                new_route.routing_distance_calcs = routing_index_options.search_budget;
                new_route.try_increasing_num_shards = true;
                new_route.buckets_to_probe = std::move(buckets_to_probe_by_query);
            }


            // frequency tree-search routing
            {
                std::vector<KMeansTreeRouter::FrequencyQueryData> routing_data(queries.n);
                double time_routing;
                parlay::execute_with_scheduler(std::min<size_t>(32, parlay::num_workers()), [&] {
                    routing_timer.Start();
                    parlay::parallel_for(0, queries.n, [&](size_t i) {
                        routing_data[i] = router.FrequencyQuery(queries.GetPoint(i), routing_index_options.search_budget, 500);
                    });
                    time_routing = routing_timer.Stop();
                });
                std::cout << "KMTR Frequency Routing took " << time_routing << " s overall, and " << time_routing / queries.n << " s per query" << std::endl;

                for (size_t num_voting_neighbors : { 20, 40, 80, 120, 200, 250, 300, 400, 500 }) {
                    std::vector<std::vector<int>> buckets_to_probe_by_query(queries.n);
                    parlay::parallel_for(0, queries.n,
                                         [&](size_t i) { buckets_to_probe_by_query[i] = routing_data[i].Query(num_shards, num_voting_neighbors); });
                    double first_shard_recall = MaxFirstShardRoutingRecall(buckets_to_probe_by_query, ground_truth, num_neighbors, cover);
                    std::cout << "Num voting points " << num_voting_neighbors << " Max first shard recall = " << first_shard_recall << std::endl;
                    auto& new_route = routes.emplace_back();
                    new_route.routing_algorithm = "KMeansTree-Frequency";
                    new_route.hnsw_num_voting_neighbors = num_voting_neighbors;
                    new_route.routing_time = time_routing;
                    new_route.routing_index_options = routing_index_options;
                    new_route.routing_distance_calcs = routing_index_options.search_budget;
                    new_route.try_increasing_num_shards = true;
                    new_route.buckets_to_probe = std::move(buckets_to_probe_by_query);
                }
            }


            // extract
            std::tie(routing_points, routing_index_partition) = router.ExtractPoints();
            std::cout << "Extraction finished. Index size = " << routing_points.n << std::endl;
        }
        // HNSW routing index
        {
            routing_timer.Start();
            HNSWRouter hnsw_router(routing_points, num_shards, routing_index_partition, HNSWParameters{ .M = 32, .ef_construction = 200, .ef_search = 200 });
            hnsw_router.Train(routing_points);
            std::cout << "Training HNSW router took " << routing_timer.Restart() << " s" << std::endl;
            // hnsw_router.Serialize(routing_index_file);
            // std::cout << "Serializing HNSW router took " << routing_timer.Stop() << " s" << std::endl;

            RoutingConfig blueprint;
            blueprint.index_trainer = "HierKMeans";
            blueprint.routing_index_options = routing_index_options;

            IterateHNSWRouterConfigs(hnsw_router, queries, routes, blueprint, ground_truth, num_neighbors, cover);
        }
    }


    { // Random routing -- saves runtime when all shards are probed anyways
        RoutingConfig rc;
        rc.index_trainer = "None";
        rc.routing_algorithm = "Random";
        rc.routing_time = 0;
        rc.try_increasing_num_shards = true;
        std::vector<std::vector<int>> buckets_to_probe(queries.n);
        std::mt19937 prng(555);
        for (size_t q = 0; q < queries.n; ++q) {
            std::vector<int> probes(num_shards);
            std::iota(probes.begin(), probes.end(), 0);
            std::shuffle(probes.begin(), probes.end(), prng);
            buckets_to_probe[q] = std::move(probes);
        }
        rc.buckets_to_probe = std::move(buckets_to_probe);
        routes.push_back(rc);
    }

    if (!pyramid_index_file.empty()) {
        if (!std::filesystem::exists(pyramid_index_file)) {
            std::cerr << "Tried to open " << pyramid_index_file << " but it doesnt exist-..." << std::endl;
            std::abort();
        }
        std::cout << "Run Pyramid routing" << std::endl;
        std::vector<int> routing_index_partition = ReadMetisPartition(pyramid_index_file + ".routing_index_partition");
        HNSWRouter hnsw_router(pyramid_index_file, points.d, routing_index_partition);
        RoutingConfig blueprint;
        blueprint.index_trainer = "Pyramid";
        IterateHNSWRouterConfigs(hnsw_router, queries, routes, blueprint, ground_truth, num_neighbors, cover);
    }

    if (!our_pyramid_index_file.empty()) {
        if (!std::filesystem::exists(our_pyramid_index_file)) {
            std::cerr << "Tried to open " << our_pyramid_index_file << " but it doesnt exist-..." << std::endl;
            std::abort();
        }
        std::cout << "Run OurPyramid++ routing" << std::endl;
        std::vector<int> routing_index_partition = ReadMetisPartition(our_pyramid_index_file + ".knn.routing_index_partition");
        HNSWRouter hnsw_router(our_pyramid_index_file, points.d, routing_index_partition);
        RoutingConfig blueprint;
        blueprint.index_trainer = "OurPyramid+KNN";
        IterateHNSWRouterConfigs(hnsw_router, queries, routes, blueprint, ground_truth, num_neighbors, cover);
    }

    return routes;
}


std::string RoutingConfig::Serialize() const {
    std::stringstream sb;
    sb << routing_algorithm << " " << index_trainer << " " << hnsw_num_voting_neighbors << " " << hnsw_ef_search << " " << routing_time << " " << std::boolalpha
       << try_increasing_num_shards << std::noboolalpha << " " << buckets_to_probe.size() << " " << routing_index_options.budget << " "
       << routing_index_options.num_centroids << " " << routing_index_options.min_cluster_size << "\n";
    for (const auto& visit_order : buckets_to_probe) {
        for (const int b : visit_order) {
            sb << b << " ";
        }
        sb << "\n";
    }
    return sb.str();
}

RoutingConfig RoutingConfig::Deserialize(std::ifstream& in) {
    RoutingConfig r;
    std::string line;
    std::getline(in, line);
    std::istringstream iss(line);
    int num_queries;
    iss >> r.routing_algorithm >> r.index_trainer >> r.hnsw_num_voting_neighbors >> r.hnsw_ef_search >> r.routing_time >> std::boolalpha >>
            r.try_increasing_num_shards >> std::noboolalpha >> num_queries >> r.routing_index_options.budget >> r.routing_index_options.num_centroids >>
            r.routing_index_options.min_cluster_size;
    // std::cout << r.routing_algorithm << " " << r.index_trainer << " " << r.hnsw_num_voting_neighbors << " " << r.hnsw_ef_search << " " << r.routing_time << "
    // " << std::boolalpha << r.try_increasing_num_shards << std::noboolalpha << " " << num_queries << std::endl;
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


void SerializeRoutes(const std::vector<RoutingConfig>& routes, const std::string& output_file) {
    std::ofstream out(output_file);
    out << routes.size() << std::endl;
    for (const RoutingConfig& r : routes) {
        out << "R" << std::endl;
        out << r.Serialize();
    }
}

std::vector<RoutingConfig> DeserializeRoutes(const std::string& input_file) {
    std::ifstream in(input_file);
    size_t num_routes;
    std::string header;
    std::getline(in, header);
    std::istringstream iss(header);
    iss >> num_routes;
    std::vector<RoutingConfig> routes;
    for (size_t i = 0; i < num_routes; ++i) {
        std::getline(in, header);
        // std::cout << "i = " << i << " for routes " << std::endl;
        if (header != "R")
            std::cout << "routing config doesn't start with marker R. Instead: " << header << std::endl;
        RoutingConfig r = RoutingConfig::Deserialize(in);
        routes.push_back(std::move(r));
    }
    return routes;
}
