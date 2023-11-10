#pragma once

#include "metis_io.h"
#include "kmeans_tree_router.h"
#include "hnsw_router.h"


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
            << routing_time << " " << std::boolalpha << try_increasing_num_shards << std::noboolalpha << " " << buckets_to_probe.size()
            << routing_index_options.budget << " " << routing_index_options.num_centroids << " " << routing_index_options.min_cluster_size << "\n";
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
        iss >> r.routing_algorithm >> r.index_trainer >> r.hnsw_num_voting_neighbors >> r.hnsw_ef_search >> r.routing_time >> std::boolalpha >> r.try_increasing_num_shards >> std::noboolalpha >> num_queries
            >> r.routing_index_options.budget >> r.routing_index_options.num_centroids >> r.routing_index_options.min_cluster_size;
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
        std::cout << "i = " << i << " for routes " << std::endl;
        if (header != "R") std::cout << "routing config doesn't start with marker R. Instead: " << header << std::endl;
        RoutingConfig r = RoutingConfig::Deserialize(in);
        routes.push_back(std::move(r));
    }
    return routes;
}

std::vector<RoutingConfig> IterateRoutingConfigs(PointSet& points, PointSet& queries, const std::vector<int>& partition, int num_shards,
                                                 KMeansTreeRouterOptions routing_index_options_blueprint, const std::string& routing_index_file,
                                                 const std::string& pyramid_index_file, const std::string& our_pyramid_index_file,
                                                 bool our_pyramid_is_hnsw_partition = false) {
    std::vector<RoutingConfig> routes;

    std::vector<KMeansTreeRouterOptions> routing_index_option_vals;
    {
        for (double factor : {0.2, 0.4, 0.8, 1.0}) {
            KMeansTreeRouterOptions ro = routing_index_options_blueprint;
            ro.budget *= factor;
            routing_index_option_vals.push_back(ro);
        }

        auto copy = routing_index_option_vals;
        routing_index_option_vals.clear();
        for (auto ro : copy) {
            for (int min_cluster_size : {250, 300, 350, 400}) {
                ro.min_cluster_size = min_cluster_size;
                routing_index_option_vals.push_back(ro);
            }
        }

        copy = routing_index_option_vals;
        routing_index_option_vals.clear();
        for (auto ro : copy) {
            for (int num_centroids : {32,64,128,256}) {
                ro.num_centroids = num_centroids;
                routing_index_option_vals.push_back(ro);
            }
        }
    }


    for (const KMeansTreeRouterOptions& routing_index_options : routing_index_option_vals)
    {
        PointSet routing_points;
        std::vector<int> routing_index_partition;
        Timer routing_timer; routing_timer.Start();
        {
            KMeansTreeRouter router;

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
            std::tie(routing_points, routing_index_partition) = router.ExtractPoints();
            std::cout << "Extraction finished" << std::endl;
        }

        {
            routing_timer.Start();
            HNSWRouter hnsw_router(routing_points, num_shards, routing_index_partition,
                                   HNSWParameters {
                                           .M = 32,
                                           .ef_construction = 200,
                                           .ef_search = 200 }
            );
            std::cout << "Training HNSW router took " << routing_timer.Restart() << " s" << std::endl;
            // hnsw_router.Serialize(routing_index_file);
            // std::cout << "Serializing HNSW router took " << routing_timer.Stop() << " s" << std::endl;

            RoutingConfig blueprint;
            blueprint.index_trainer = "HierKMeans";
            IterateHNSWRouterConfigs(hnsw_router, queries, routes, blueprint);
        }
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
