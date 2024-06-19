#include "kmeans_tree_router.h"

#include <iostream>
#include <numeric>
#include <parlay/parallel.h>
#include "dist.h"
#include "kmeans.h"

void KMeansTreeRouter::Train(PointSet& points, const Clusters& clusters, KMeansTreeRouterOptions options) {
    num_shards = clusters.size();
    roots.resize(num_shards);
    dim = points.d;

    std::cout << "Train. num-shards = " << num_shards << " dim = " << dim << " budget = " << options.budget << std::endl;

    int num_shards_processed_in_parallel = 8;
#ifdef MIPS_DISTANCE
    // not for MIPS_DISTANCE (just for text-to-image...)
    num_shards_processed_in_parallel = 4;
#endif

    parlay::parallel_for(
            0, num_shards,
            [&](int b) {
                // for (int b = 0; b < num_shards; ++b) {      // go sequential for the big datasets on not the biggest memory machines
                PointSet ps = ExtractPointsInBucket(clusters[b], points);
                KMeansTreeRouterOptions recursive_options = options;
                recursive_options.budget = double(clusters[b].size() * options.budget) / double(points.n);
                TrainRecursive(ps, recursive_options, roots[b], 555 * b);
                // }
            },
            num_shards / num_shards_processed_in_parallel);
}

void KMeansTreeRouter::TrainRecursive(PointSet& points, KMeansTreeRouterOptions options, TreeNode& tree_node, int seed) {
    PointSet centroids = RandomSample(points, std::max(2, std::min<int>(options.num_centroids, options.budget)), seed);
    auto partition = KMeans(points, centroids);
    auto buckets = ConvertPartitionToClusters(partition);
    // std::cout << "num buckets " << buckets.size() << " num centroids " << centroids.n << " num points " << points.n << " options.budget " << options.budget
    // << std::endl;

    // check bucket size. stop recursion if small enough. --> partition buckets into those who get a sub-tree and those who don't
    // this is needed for 1-to-1 mapping between centroid and sub-tree in the query phase
    std::vector<std::pair<size_t, size_t>> bucket_size_and_ids(buckets.size());
    for (size_t i = 0; i < buckets.size(); ++i)
        bucket_size_and_ids[i] = std::make_pair(buckets[i].size(), i);
    size_t num_buckets_in_recursion =
            std::distance(bucket_size_and_ids.begin(), std::partition(bucket_size_and_ids.begin(), bucket_size_and_ids.end(),
                                                                      [&](const auto& pair) { return pair.first > options.min_cluster_size; }));

    tree_node.centroids = ReorderCentroids(centroids, bucket_size_and_ids);

    options.budget -= tree_node.centroids.n;
    if (options.budget <= 0)
        return;

    if (tree_node.centroids.n == 1) {
        // Don't split it further. There's no use.
        return;
    }

    tree_node.children.resize(num_buckets_in_recursion);

    // split budget equally
    size_t total_size = 0;
    for (size_t i = 0; i < num_buckets_in_recursion; ++i)
        total_size += bucket_size_and_ids[i].first;

    parlay::parallel_for(
            0, num_buckets_in_recursion,
            [&](int i) {
                PointSet ps = ExtractPointsInBucket(buckets[bucket_size_and_ids[i].second], points);
                KMeansTreeRouterOptions recursive_options = options;
                recursive_options.budget = double(bucket_size_and_ids[i].first * options.budget) / double(total_size);
                TrainRecursive(ps, recursive_options, tree_node.children[i], seed + i);
            },
            1);
}

PointSet KMeansTreeRouter::ReorderCentroids(PointSet& centroids, std::vector<std::pair<size_t, size_t>>& permutation) {
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

std::vector<int> KMeansTreeRouter::Query(float* Q, int budget) {
    struct PQEntry {
        float dist = 0.f;
        int shard_id = -1;
        TreeNode* node = nullptr;
        bool operator>(const PQEntry& other) const { return dist > other.dist; }
    };
    std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<>> pq;
    std::vector<float> min_dist(num_shards, std::numeric_limits<float>::max());

    for (int u = 0; u < int(roots.size()); ++u) {
        float dist = std::numeric_limits<float>::lowest();
        if (centroids_in_roots) {
            dist = distance(roots[u].centroids.GetPoint(0), Q, dim);
            budget--;
        }
        pq.push(PQEntry{ dist, u, &roots[u] });
    }

    while (!pq.empty() && budget > 0) {
        PQEntry top = pq.top();
        pq.pop();
        budget -= top.node->centroids.n;
        for (size_t i = 0; i < top.node->centroids.n; ++i) {
            float dist = distance(top.node->centroids.GetPoint(i), Q, dim);
            min_dist[top.shard_id] = std::min(min_dist[top.shard_id], dist);
            if (i < top.node->children.size()) {
                pq.push(PQEntry{ dist, top.shard_id, &top.node->children[i] });
            }
        }
    }

    std::vector<int> probes(num_shards);
    std::iota(probes.begin(), probes.end(), 0);
    std::sort(probes.begin(), probes.end(), [&](int l, int r) { return min_dist[l] < min_dist[r]; });

    return probes;
}

KMeansTreeRouter::FrequencyQueryData KMeansTreeRouter::FrequencyQuery(float* Q, int budget, int num_voting_neighbors) {
    struct PQEntry {
        float dist = 0.f;
        int shard_id = -1;
        TreeNode* node = nullptr;
        bool operator>(const PQEntry& other) const { return dist > other.dist; }
    };
    std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<>> pq;
    TopN top_neighbors(num_voting_neighbors);
    std::vector<float> min_dist(num_shards, std::numeric_limits<float>::max());

    for (int u = 0; u < int(roots.size()); ++u) {
        float dist = std::numeric_limits<float>::lowest();
        if (centroids_in_roots) {
            dist = distance(roots[u].centroids.GetPoint(0), Q, dim);
            budget--;
        }
        pq.push(PQEntry{ dist, u, &roots[u] });
    }

    while (!pq.empty() && budget > 0) {
        PQEntry top = pq.top();
        pq.pop();
        budget -= top.node->centroids.n;
        for (size_t i = 0; i < top.node->centroids.n; ++i) {
            float dist = distance(top.node->centroids.GetPoint(i), Q, dim);
            min_dist[top.shard_id] = std::min(min_dist[top.shard_id], dist);
            top_neighbors.Add(std::make_pair(dist, top.shard_id));
            if (i < top.node->children.size()) {
                pq.push(PQEntry{ dist, top.shard_id, &top.node->children[i] });
            }
        }
    }


    FrequencyQueryData ret;
    ret.min_dist = std::move(min_dist);
    ret.near_neighbors = top_neighbors.Take();
    ret.Init();
    return ret;
}

std::pair<PointSet, std::vector<int>> KMeansTreeRouter::ExtractPoints() {
    PointSet points;
    points.d = dim;
    std::vector<int> partition;

    int b = 0;
    for (TreeNode& root : roots) {
        std::queue<TreeNode*> queue;
        queue.push(&root);
        while (!queue.empty()) {
            TreeNode* u = queue.front();
            queue.pop();
            for (TreeNode& c : u->children) {
                queue.push(&c);
            }
            if (u->centroids.coordinates.size() != u->centroids.n * u->centroids.d) {
                std::cout << u->centroids.coordinates.size() << " " << u->centroids.n << " " << u->centroids.d << std::endl;
                throw std::runtime_error("C");
            }
            points.coordinates.insert(points.coordinates.end(), u->centroids.coordinates.begin(), u->centroids.coordinates.end());
            points.n += u->centroids.n;
            partition.insert(partition.end(), u->centroids.n, b);
        }
        b++;
    }

    return std::make_pair(std::move(points), std::move(partition));
}
