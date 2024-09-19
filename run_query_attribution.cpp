#include <filesystem>
#include <iostream>

#include "metis_io.h"
#include "points_io.h"
#include "recall.h"
#include "route_search_combination.h"

void SetAffinity() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (int cpu = 0; cpu < 64; ++cpu) {
        CPU_SET(cpu, &mask);
    }
    int err = sched_setaffinity(0, sizeof(mask), &mask);
    if (err) {
        std::cerr << "thread pinning failed" << std::endl;
        std::abort();
    }
}

int main(int argc, const char* argv[]) {
    if (argc != 9) {
        std::cerr << "Usage ./QueryAttribution input-points queries ground-truth-file num_neighbors partition-file output-file partition_method "
                     "requested-num-shards"
                  << std::endl;
        std::abort();
    }

    // SetAffinity();

    std::string point_file = argv[1];
    std::string query_file = argv[2];
    std::string ground_truth_file = argv[3];
    std::string k_string = argv[4];
    int max_num_neighbors = std::stoi(k_string);
    std::string partition_file = argv[5];
    std::string output_file = argv[6];
    std::string part_method = argv[7];
    std::string requested_num_shards_str = argv[8];
    int requested_num_shards = std::stoi(requested_num_shards_str);

    PointSet points = ReadPoints(point_file);
    PointSet queries = ReadPoints(query_file);

    std::vector<NNVec> ground_truth;
    if (std::filesystem::exists(ground_truth_file)) {
        ground_truth = ReadGroundTruth(ground_truth_file);
        std::cout << "Read ground truth file" << std::endl;
    } else {
        std::cout << "start computing ground truth" << std::endl;
        ground_truth = ComputeGroundTruth(points, queries, max_num_neighbors);
        std::cout << "computed ground truth" << std::endl;
        WriteGroundTruth(ground_truth_file, ground_truth);
    }

    std::vector<int> num_neighbors_values = { 100, 10, 1 };
    // CleanGroundTruth(ground_truth, points, queries);
    // std::cout << "Finished reordering ground truth" << std::endl;

#if false
    std::vector<int> partition = ReadMetisPartition(partition_file);
    int num_shards = NumPartsInPartition(partition);
    Clusters clusters = ConvertPartitionToClusters(partition);
#else
    Clusters clusters = ReadClusters(partition_file);
    int num_shards = static_cast<int>(clusters.size());
#endif

    KMeansTreeRouterOptions router_options;
    router_options.budget = points.n / requested_num_shards;
    std::string pyramid_index_file, our_pyramid_index_file;
    if (part_method == "Pyramid") {
        pyramid_index_file = partition_file + ".pyramid_routing_index";
    }
    if (part_method == "OurPyramid") {
        our_pyramid_index_file = partition_file + ".our_pyramid_routing_index";
    }

    std::vector<RoutingConfig> routes;
    if (false) {
        routes = IterateRoutingConfigs(points, queries, clusters, num_shards, router_options, ground_truth, max_num_neighbors,
                                                              partition_file + ".routing_index", pyramid_index_file, our_pyramid_index_file);
        std::cout << "Finished routing configs" << std::endl;
        SerializeRoutes(routes, output_file + ".routes");
    } else {
        std::cout << "Load routes from file" << std::endl;
        routes = DeserializeRoutes(output_file + ".routes");
        std::cout << "Loading routes finished" << std::endl;
    }
    
    std::cout << "Start shard searches" << std::endl;
    std::vector<std::vector<ShardSearch>> shard_searches =
            RunInShardSearches(points, queries, HNSWParameters(), num_neighbors_values, clusters, num_shards, ground_truth);
    std::cout << "Finished shard searches" << std::endl;
    for (size_t i = 0; i < num_neighbors_values.size(); ++i) {
        int num_neighbors = num_neighbors_values[i];
        SerializeShardSearches(shard_searches[i], output_file + ".nn=" + std::to_string(num_neighbors) + ".searches");

        PrintCombinationsOfRoutesAndSearches(routes, shard_searches[i], output_file + ".nn=" + std::to_string(num_neighbors), ground_truth, num_neighbors, queries.n,
                                             num_shards, requested_num_shards, part_method);
    }
}
