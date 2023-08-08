#pragma once

#include "kmeans.h"
#include "inverted_index.h"

struct TreeNode {
	std::vector<TreeNode> children;
	PointSet centroids;
};

struct Options {
    size_t num_centroids = 64;
    size_t min_cluster_size = 250;
    int64_t budget = 50000;
};

struct KMeansTreeRouter {
    std::vector<TreeNode> roots;
    int num_shards;

    void Train(PointSet& points, const std::vector<int>& partition, Options options) {
        auto buckets = ConvertPartitionToBuckets(partition);
        num_shards = buckets.size();
        roots.resize(num_shards);

        double split = static_cast<double> (options.budget) / static_cast<double> (partition.size());

        parlay::parallel_for(0, num_shards, [&](size_t b) {
            PointSet ps = ExtractPointsInBucket(buckets[b], points);
            Options recursive_options = options;
            recursive_options.budget = split * options.budget;
            TrainRecursive(ps, recursive_options, roots[b], 555 * b);
        }, 1);
    }

    void TrainRecursive(PointSet& points, Options options, TreeNode& tree_node, int seed) {
        PointSet centroids = RandomSample(points, options.num_centroids, seed);
        auto partition = KMeans(points, centroids);
        auto buckets = ConvertPartitionToBuckets(partition);
        // std::cout << "num buckets " << buckets.size() << " num centroids " << centroids.n << " num points " << points.n << " options.budget " << options.budget << std::endl;

        // check bucket size. stop recursion if small enough. --> partition buckets into those who get a sub-tree and those who don't
        // this is needed for 1-to-1 mapping between centroid and sub-tree in the query phase
        std::vector<std::pair<size_t, size_t>> bucket_size_and_ids(buckets.size());
        for (size_t i = 0; i < buckets.size(); ++i) bucket_size_and_ids[i] = std::make_pair(buckets[i].size(), i);
        size_t num_buckets_in_recursion = std::distance(
                bucket_size_and_ids.begin(),
                std::partition(bucket_size_and_ids.begin(), bucket_size_and_ids.end(), [&](const auto& pair) { return pair.first > options.min_cluster_size; })
                );

        tree_node.centroids = ReorderCentroids(centroids, bucket_size_and_ids);

        options.budget -= tree_node.centroids.n;
        if (options.budget <= 0) return;

        tree_node.children.resize(num_buckets_in_recursion);

        // split budget equally
        size_t total_size = 0; for (size_t i = 0; i < num_buckets_in_recursion; ++i) total_size += bucket_size_and_ids[i].first;
        double split = static_cast<double> (options.budget) / static_cast<double> (total_size);

        parlay::parallel_for(0, num_buckets_in_recursion, [&](size_t i) {
            PointSet ps = ExtractPointsInBucket(buckets[bucket_size_and_ids[i].second], points);
            Options recursive_options = options;
            recursive_options.budget = split * bucket_size_and_ids[i].first;
            TrainRecursive(ps, recursive_options, tree_node.children[i], seed + i);
        }, 1);
    }

    PointSet ReorderCentroids(PointSet& centroids, std::vector<std::pair<size_t, size_t>>& permutation) {
        PointSet re;
        re.d = centroids.d;
        re.n = centroids.n;
        for (const auto& [_, centroid_id] : permutation) {
            float* c = centroids.GetPoint(centroid_id);
            for (size_t j = 0; j < centroids.d; ++j) {
                re.coordinates.push_back(c[j]);
            }
        }
        return re;
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


    struct PQEntry {
        float dist = 0.f;
        int shard_id = -1;
        TreeNode* node = nullptr;
    };


    std::vector<int> Query(float* Q, int num_shards) {
        std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<>> pq;
        std::vector<int> probes(num_shards);
        for (int i = 0; i < num_shards; ++i) { probes[i] = i; }
        return probes;


    }

};
