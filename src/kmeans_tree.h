#pragma once

#include "kmeans.h"
#include "inverted_index.h"

struct TreeNode {
	std::vector<TreeNode> children;
	PointSet centroids;
};

struct Options {
    int num_centroids = 64;
    int min_cluster_size = 250;
};

struct KMeansTreeRouter {
    std::vector<TreeNode> roots;
    int num_shards;

    void Train(PointSet& points, const std::vector<int>& partition, const Options& options) {
        auto buckets = ConvertPartitionToBuckets(partition);
        num_shards = buckets.size();
        roots.resize(num_shards);
        for (int b = 0; b < num_shards; ++b) {
            PointSet ps = ExtractPointsInBucket(buckets[b], points);
            TrainRecursive(ps, options, roots[b], 555 * b);
        }
    }

    void TrainRecursive(PointSet& points, const Options& options, TreeNode& tree_node, int seed) {
        PointSet centroids = RandomSample(points, options.num_centroids, seed);
        auto partition = KMeans(points, centroids);
        tree_node.centroids = std::move(centroids);
        auto buckets = ConvertPartitionToBuckets(partition);

        // check bucket size. stop recursion if small enough
    }

	PointSet ExtractPointsInBucket(const std::vector<uint32_t>& bucket, PointSet& points) {
        PointSet ps;
        ps.n = bucket.size();
        ps.d = points.d;
        ps.coordinates.reserve(ps.n * ps.d);
        for (auto u : bucket) {
            float* p = points.GetPoint(u);
            for (size_t j = 0; j < points.d; ++j) {
                ps.coordinates.push_back(p[j]);
            }
        }
        return ps;
    }

    std::vector<std::vector<uint32_t>> ConvertPartitionToBuckets(const std::vector<int>& partition) {
        int num_buckets = *std::max_element(partition.begin(), partition.end()) + 1;
        std::vector<std::vector<uint32_t>> buckets(num_buckets);
        for (uint32_t u = 0; u < partition.size(); ++u) {
            buckets[partition[u]].push_back(u);
        }
        return buckets;
    }


    std::vector<int> Query(float* Q, int num_shards) {
        std::vector<int> probes(num_shards);
        for (int i = 0; i < num_shards; ++i) { probes[i] = i; }
        return probes;
    }

    std::vector<size_t> SplitBudgetEqually(const std::vector<std::vector<uint32_t>>& buckets, size_t budget) {
        size_t total_size = 0; for (const auto& b : buckets) total_size += b.size();
        double split = static_cast<double>(total_size) / static_cast<double>(budget);
        std::vector<size_t> result; for (const auto& b : buckets) result.push_back(b.size() * split);
        return result;
    }

};
