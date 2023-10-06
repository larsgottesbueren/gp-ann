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
    int64_t search_budget = 10000;
};

struct KMeansTreeRouter {
    std::vector<TreeNode> roots;
    int num_shards;

    void Train(PointSet& points, const std::vector<int>& partition, Options options) {
        auto buckets = ConvertPartitionToBuckets(partition);
        num_shards = buckets.size();
        roots.resize(num_shards);

        dim = points.d;
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

        if (tree_node.centroids.n == 1) {
            std::cout << "LOADS of DUPES" << std::endl;
            // Don't split it further. There's no use.
            return;
        }

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
        bool operator>(const PQEntry& other) const {
            return dist > other.dist;
        }
    };


    bool centroids_in_roots = false;
    uint32_t dim = 0;

    #define PRINT false

    std::vector<int> Query(float* Q, int budget) {
        // TODO optimize
        // a) avoid re-allocs of PQ and probes vector
        //
        std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<>> pq;
        std::vector<float> min_dist(num_shards, std::numeric_limits<float>::max());

        for (size_t u = 0; u < roots.size(); ++u) {
            float dist = std::numeric_limits<float>::lowest();
            if (centroids_in_roots) {
                dist = distance(roots[u].centroids.GetPoint(0), Q, dim);
                budget--;
            }
            pq.push(PQEntry{dist, u, &roots[u]});
        }

        size_t iter = 0;

        while (!pq.empty() && budget > 0) {
            PQEntry top = pq.top(); pq.pop();
            budget -= top.node->centroids.n;
            #if PRINT
            std::cout << "iter " << iter++ << " dist " << top.dist << " shard " << top.shard_id << " budget " << budget << " num centroids " << top.node->centroids.n << std::endl;
            #endif
            for (size_t i = 0; i < top.node->centroids.n; ++i) {
                float dist = distance(top.node->centroids.GetPoint(i), Q, dim);
                min_dist[top.shard_id] = std::min(min_dist[top.shard_id], dist);
                if (i < top.node->children.size()) {
                    pq.push(PQEntry{dist, top.shard_id, &top.node->children[i]});
                }
            }
        }

        std::vector<int> probes(num_shards);
        std::iota(probes.begin(), probes.end(), 0);
        std::sort(probes.begin(), probes.end(), [&](int l, int r) {
            return min_dist[l] < min_dist[r];
        });

        #if PRINT
        std::cout << "done" << std::endl;
        std::cout << "min dists ";
        for (int i = 0; i < num_shards; ++i) {
            std::cout << min_dist[i] << " ";
        }
        std::cout << std::endl;
        #endif
        return probes;
    }

    std::pair<PointSet, std::vector<int>> ExtractPoints() {
        PointSet points;
        points.d = roots[0].centroids.d;
        std::vector<int> offsets(1, 0);

        for (TreeNode& root : roots) {
            std::queue<TreeNode*> queue;
            queue.push(&root);
            while (!queue.empty()) {
                TreeNode* u = queue.front(); queue.pop();
                for (TreeNode& c : u->children) {
                    queue.push(&c);
                }
                points.coordinates.insert(points.coordinates.end(), u->centroids.coordinates.begin(), u->centroids.coordinates.end());
                points.n += u->centroids.n;
            }
            offsets.push_back(points.n);
        }

        return std::make_pair(std::move(points), std::move(offsets));
    }
};
