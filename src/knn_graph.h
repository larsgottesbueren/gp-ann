#pragma once

#include "defs.h"
#include "dist.h"
#include "topn.h"
#include "spinlock.h"
#include <sstream>
#include <iostream>
#include <parlay/parallel.h>
#include <random>


std::vector<int> TopKNeighbors(PointSet& P, uint32_t my_id, int k) {
	TopN top_k(k);
	float* Q = P.GetPoint(my_id);
	for (uint32_t i = 0; i < P.n; ++i) {
	    if (i == my_id) continue;
	    float new_dist = distance(P.GetPoint(i), Q, P.d);
		top_k.Add(std::make_pair(new_dist, i));
	}
	auto x = top_k.Take();
	std::vector<int> y;
	for (const auto& a : x) y.push_back(a.second);
	return y;
}

AdjGraph BuildKNNGraph(PointSet& P, int k) {
	AdjGraph graph(P.n);
	parlay::parallel_for(0, P.n, [&](size_t i) { graph[i] = TopKNeighbors(P, i, k); });
	return graph;
}


struct ApproximateKNNGraphBuilder {
    using Bucket = std::vector<uint32_t>;

    static TopN AsymmetricTopNeighbors(PointSet& points, PointSet& queries, uint32_t my_id, int k) {
        TopN top_k(k);
        float* Q = queries.GetPoint(my_id);
        for (uint32_t j = 0; j < points.n; ++j) {
            float* P = points.GetPoint(j);
            float dist = distance(P, Q, points.d);
            top_k.Add(std::make_pair(dist, j));
        }
        return top_k;
    }

    static TopN ClosestLeaders(PointSet& points, const Bucket& leaders, uint32_t my_id, int k) {
        TopN top_k(k);
        float* Q = points.GetPoint(my_id);
        for (uint32_t j = 0; j < leaders.size(); ++j) {
            float* P = points.GetPoint(leaders[j]);
            float dist = distance(P, Q, points.d);
            top_k.Add(std::make_pair(dist, j));
        }
        return top_k;
    }

    std::vector<Bucket> RecursivelySketch(PointSet& points, const Bucket& ids, int depth, int fanout=1) {
        if (ids.size() <= MAX_CLUSTER_SIZE) {
            return { ids };
        }

        // sample leaders
        size_t num_leaders = depth == 0 ? TOP_LEVEL_NUM_LEADERS : ids.size() * FRACTION_LEADERS;
        num_leaders = std::min<size_t>(num_leaders, 5000);
        Bucket leaders(num_leaders);
        std::mt19937 prng(seed);
        std::sample(ids.begin(), ids.end(), leaders.begin(), leaders.size(), prng);

        // find closest leaders and build clusters around leaders
        std::vector<Bucket> clusters(leaders.size());
        std::vector<SpinLock> cluster_locks(leaders.size());
        parlay::parallel_for(0, ids.size(), [&](size_t i) {
            auto point_id = ids[i];
            auto closest_leaders = ClosestLeaders(points, leaders, point_id, fanout).Take();
            while (!closest_leaders.empty()) {
                for (size_t j = 0; j < closest_leaders.size(); ++j) {
                    const auto leader = closest_leaders[j].second;
                    if (cluster_locks[leader].tryLock()) {
                        clusters[leader].push_back(point_id);
                        cluster_locks[leader].unlock();

                        closest_leaders[j] = closest_leaders.back();
                        closest_leaders.pop_back();
                        --j;
                    }
                }
            }
        });
        cluster_locks.clear(); cluster_locks.shrink_to_fit();

        // recurse on clusters
        std::vector<Bucket> buckets;
        SpinLock bucket_lock;
        parlay::parallel_for(0, clusters.size(), [&](size_t cluster_id) {
            auto recursive_buckets = RecursivelySketch(points, clusters[cluster_id], depth + 1, /*fanout=*/1);
            bucket_lock.lock();
            buckets.insert(buckets.end(), recursive_buckets.begin(), recursive_buckets.end());
            bucket_lock.unlock();
        }, 1);

        return buckets;
    }

    AdjGraph BuildApproximateNearestNeighborGraph(PointSet& points, int num_neighbors) {
        Bucket all_ids(points.n);
        std::iota(all_ids.begin(), all_ids.end(), 0);
        std::vector<Bucket> buckets;
        for (int rep = 0; rep < REPETITIONS; ++rep) {
            std::vector<Bucket> new_buckets = RecursivelySketch(points, all_ids, 0, FANOUT);
            buckets.insert(buckets.end(), new_buckets.begin(), new_buckets.end());
        }
        return BruteForceBuckets(points, buckets, num_neighbors);
    }


    std::vector<std::vector<TopN::value_type>> CrunchBucket(PointSet& points, const Bucket& bucket, int num_neighbors) {
        std::vector<TopN> neighbors(bucket.size(), TopN(num_neighbors));
        for (size_t i = 0; i < bucket.size(); ++i) {
            float* P = points.GetPoint(bucket[i]);
            for (size_t j = i + 1; j < bucket.size(); ++j) {
                float* Q = points.GetPoint(bucket[j]);
                float dist = distance(P, Q, points.d);
                neighbors[i].Add(std::make_pair(dist, bucket[j]));
                neighbors[j].Add(std::make_pair(dist, bucket[i]));
            }
        }

        std::vector<std::vector<TopN::value_type>> result(bucket.size());
        for (size_t i = 0; i < bucket.size(); ++i) {
            result[i] = neighbors[i].Take();
        }
        return result;
    }

    AdjGraph BruteForceBuckets(PointSet& points, std::vector<Bucket>& buckets, int num_neighbors) {
        std::vector<SpinLock> locks(points.n);
        std::vector<std::vector<TopN::value_type>> top_neighbors(points.n);
        parlay::parallel_for(0, buckets.size(), [&](size_t bucket_id) {
            auto& bucket = buckets[bucket_id];
            auto bucket_neighbors = CrunchBucket(points, bucket, num_neighbors);
            while (!bucket.empty()) {
                for (size_t j = 0; j < bucket.size(); ++j) {
                    auto point_id = bucket[j];
                    if (locks[point_id].tryLock()) {
                        // insert new neighbors. due to possible duplicate neighbors, we can't insert directly into the top-k data structure, and instead have to do this
                        auto& n = top_neighbors[point_id];
                        n.insert(n.end(), bucket_neighbors[j].begin(), bucket_neighbors[j].end());
                        std::sort(n.begin(), n.end(), [](const auto& l, const auto& r) {return std::tie(l.second, l.first) < std::tie(r.second, r.first);});
                        n.erase(std::unique(n.begin(), n.end(), [&](const auto& l, const auto& r) { return l.second == r.second; }), n.end());
                        std::sort(n.begin(), n.end());
                        n.resize(std::min<size_t>(n.size(), num_neighbors));

                        locks[point_id].unlock();

                        bucket[j] = bucket.back();
                        bucket.pop_back();
                        bucket_neighbors[j] = std::move(bucket_neighbors.back());
                        bucket_neighbors.pop_back();
                        --j;
                    }
                }
            }
        }, 1);

        AdjGraph graph(points.n);
        parlay::parallel_for(0, points.n, [&](size_t i) {
            for (const auto& [dist, point_id] : top_neighbors[i]) {
                graph[i].push_back(point_id);
            }
        });
        return graph;
    }


    int seed = 555;
    static constexpr double FRACTION_LEADERS = 0.01;
    static constexpr size_t TOP_LEVEL_NUM_LEADERS = 950;
    static constexpr size_t MAX_CLUSTER_SIZE = 3500;
    static constexpr int REPETITIONS = 3;
    static constexpr int FANOUT = 3;
};

void Symmetrize(AdjGraph& graph) {
    std::vector<size_t> degrees; degrees.reserve(graph.size());
    for (const auto& n : graph) degrees.push_back(n.size());
    for (size_t u = 0; u < graph.size(); ++u) {
        for (size_t j = 0; j < degrees[u]; ++j) {
            auto v = graph[u][j];
            graph[v].push_back(u);
        }
    }
}
