#include "kmeans_tree_router.h"

#include <iostream>
#include <numeric>
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>
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

std::vector<KMeansTreeRouter::VisitEntry> KMeansTreeRouter::QueryWithEntriesReturned(float* Q, int budget) {
    std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<>> pq;
    std::vector<float> min_dist(num_shards, std::numeric_limits<float>::max());
    std::vector<VisitEntry> visited;

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
        for (int i = 0; i < int(top.node->centroids.n); ++i) {
            float dist = distance(top.node->centroids.GetPoint(i), Q, dim);
            min_dist[top.shard_id] = std::min(min_dist[top.shard_id], dist);
            if (i < top.node->children.size()) {
                pq.push(PQEntry{ dist, top.shard_id, &top.node->children[i] });
                visited.push_back(VisitEntry{ .dist = dist, .shard_id = top.shard_id, .node = top.node, .centroid_id = i });
            }
        }
    }

    return visited;
}

void KMeansTreeRouter::TrainWithQueries(PointSet& points, PointSet& queries, const std::vector<NNVec>& ground_truth, const Clusters& clusters,
                                        int search_budget) {
    auto cover = ConvertClustersToCover(clusters);
    struct ShardFrequency {
        int shard_id = -1;
        int num_neighbors = 0;
    };
    auto top_gt_shards = parlay::map(parlay::iota(queries.n), [&](size_t q) -> ShardFrequency {
        std::vector<int> frequency(clusters.size(), 0);
        for (const auto& [neigh, _] : ground_truth[q]) {
            for (int c : cover[neigh]) {
                frequency[c]++;
            }
        }
        auto it = std::max_element(frequency.begin(), frequency.end());
        return ShardFrequency{ .shard_id = std::distance(frequency.begin(), it), .num_neighbors = *it };
    });

    struct Move {
        TreeNode* node = nullptr;
        int centroid_id = -1;
        std::vector<float> direction;
    };
    
    int num_neighbors_retrieved = 0;
    int num_ranked_correctly = 0;
    auto moves_nested = parlay::map(parlay::iota(queries.n), [&](size_t q) -> parlay::sequence<Move> {
        auto visits = QueryWithEntriesReturned(queries.GetPoint(q), search_budget);
        int top_routed_shard = -1;
        float min_dist = std::numeric_limits<float>::max();
        for (const auto& el : visits) {
            if (el.dist < min_dist) {
                min_dist = el.dist;
                top_routed_shard = el.shard_id;
            }
        }

        if (top_routed_shard == top_gt_shards[q].shard_id) {
            __atomic_fetch_add(&num_ranked_correctly, 1, __ATOMIC_RELAXED);
            __atomic_fetch_add(&num_neighbors_retrieved, top_gt_shards[q].num_neighbors, __ATOMIC_RELAXED);
            return {}; // continue
        }

        constexpr float alpha = 0.1;
        int num_vectors_to_move = 5;

        visits.erase(std::remove_if(visits.begin(), visits.end(), [&](const VisitEntry& visit) { return visit.shard_id != top_gt_shards[q].shard_id; }),
                     visits.end());
        std::sort(visits.begin(), visits.end());
        for (int i = 0; i < num_vectors_to_move; ++i) {

        }

    });

    std::cout << "num neighbors retrieved " << num_neighbors_retrieved << " num ranked correctly " << num_ranked_correctly << std::endl;

    auto moves = parlay::flatten(moves_nested);
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
