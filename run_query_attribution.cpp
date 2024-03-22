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

    SetAffinity();

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

    std::vector<NNVec> ground_truth;
    if (std::filesystem::exists(ground_truth_file)) {
        ground_truth = ReadGroundTruth(ground_truth_file);
        std::cout << "Read ground truth file" << std::endl;
    } else {
        std::cout << "start computing ground truth" << std::endl;
        ground_truth = ComputeGroundTruth(points, queries, num_neighbors);
        std::cout << "computed ground truth" << std::endl;
        WriteGroundTruth(ground_truth_file, ground_truth);
    }
    std::vector<float> distance_to_kth_neighbor = ConvertGroundTruthToDistanceToKthNeighbor(ground_truth, num_neighbors, points, queries);
    std::cout << "Finished computing distance to kth neighbor" << std::endl;

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

    std::vector<RoutingConfig> routes = IterateRoutingConfigs(points, queries, clusters, num_shards, router_options, ground_truth, num_neighbors,
                                                              partition_file + ".routing_index", pyramid_index_file, our_pyramid_index_file);
    std::cout << "Finished routing configs" << std::endl;
    SerializeRoutes(routes, output_file + ".routes");

    std::cout << "Start shard searches" << std::endl;
    std::vector<ShardSearch> shard_searches =
            RunInShardSearches(points, queries, HNSWParameters(), num_neighbors, clusters, num_shards, distance_to_kth_neighbor);
    std::cout << "Finished shard searches" << std::endl;
    SerializeShardSearches(shard_searches, output_file + ".searches");

    PrintCombinationsOfRoutesAndSearches(routes, shard_searches, output_file, num_neighbors, queries.n, num_shards, requested_num_shards, part_method);
}
