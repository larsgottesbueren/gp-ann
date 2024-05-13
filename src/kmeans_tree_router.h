#pragma once

#include "defs.h"


struct KMeansTreeRouterOptions {
    size_t num_centroids = 64;
    size_t min_cluster_size = 250;
    int64_t budget = 50000;
    int64_t search_budget = 50000;
};

class KMeansTreeRouter {
public:
    void Train(PointSet& points, const Clusters& clusters, KMeansTreeRouterOptions options);

    void TrainWithQueries(PointSet& points, PointSet& queries, const std::vector<NNVec>& ground_truth, const Clusters& clusters, int search_budget);

    std::vector<int> Query(float* Q, int budget);

    std::pair<PointSet, std::vector<int>> ExtractPoints();

private:
    struct TreeNode {
        std::vector<TreeNode> children;
        PointSet centroids;
    };

    struct PQEntry {
        float dist = 0.f;
        int shard_id = -1;
        TreeNode* node = nullptr;
        bool operator>(const PQEntry& other) const { return dist > other.dist; }
    };

    std::vector<PQEntry> QueryWithEntriesReturned(float* Q, int budget);

    void TrainRecursive(PointSet& points, KMeansTreeRouterOptions options, TreeNode& tree_node, int seed);

    PointSet ReorderCentroids(PointSet& centroids, std::vector<std::pair<size_t, size_t>>& permutation);

    bool centroids_in_roots = false;
    uint32_t dim = 0;
    std::vector<TreeNode> roots;
    int num_shards = 0;
};
