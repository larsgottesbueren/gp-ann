#pragma once

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

    std::string Serialize() const;

    static RoutingConfig Deserialize(std::ifstream& in);
};

double MaxFirstShardRoutingRecall(const std::vector<std::vector<int>>& buckets_to_probe, const std::vector<NNVec>& ground_truth, int num_neighbors,
                                  const Cover& cover);

void IterateHNSWRouterConfigsInScheduler(HNSWRouter& hnsw_router, PointSet& queries, std::vector<RoutingConfig>& routes, const RoutingConfig& blueprint,
                                         const std::vector<NNVec>& ground_truth, int num_neighbors, const Cover& cover);

void IterateHNSWRouterConfigs(HNSWRouter& hnsw_router, PointSet& queries, std::vector<RoutingConfig>& routes, const RoutingConfig& blueprint,
                              const std::vector<NNVec>& ground_truth, int num_neighbors, const Cover& cover);

void SerializeRoutes(const std::vector<RoutingConfig>& routes, const std::string& output_file);

std::vector<RoutingConfig> DeserializeRoutes(const std::string& input_file);

std::vector<RoutingConfig> IterateRoutingConfigs(PointSet& points, PointSet& queries, const Clusters& clusters, int num_shards,
                                                 KMeansTreeRouterOptions routing_index_options_blueprint, const std::vector<NNVec>& ground_truth,
                                                 int num_neighbors, const std::string& routing_index_file, const std::string& pyramid_index_file,
                                                 const std::string& our_pyramid_index_file);
