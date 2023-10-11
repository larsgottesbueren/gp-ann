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

    TopN ClosestLeaders(PointSet& points, PointSet& leader_points, uint32_t my_id, int k) {
        TopN top_k(k);
        float* Q = points.GetPoint(my_id);
        for (uint32_t j = 0; j < leader_points.n; ++j) {
            float* P = leader_points.GetPoint(j);
            float dist = distance(P, Q, points.d);
            top_k.Add(std::make_pair(dist, j));
        }
        return top_k;
    }

    PointSet ExtractPoints(PointSet& points, const Bucket& ids) {
        PointSet bucket_points;
        bucket_points.d = points.d;
        bucket_points.n = ids.size();
        for (auto id : ids) {
            float* P = points.GetPoint(id);
            for (size_t d = 0; d < points.d; ++d) {
                bucket_points.coordinates.push_back(P[d]);
            }
        }
        return bucket_points;
    }

    std::vector<Bucket> RecursivelySketch(PointSet& points, const Bucket& ids, int depth, int fanout) {
        if (ids.size() <= MAX_CLUSTER_SIZE) {
            return { ids };
        }

        if (depth == 0) { timer.Start(); }

        // sample leaders
        size_t num_leaders = depth == 0 ? TOP_LEVEL_NUM_LEADERS : ids.size() * FRACTION_LEADERS;
        num_leaders = std::min<size_t>(num_leaders, MAX_NUM_LEADERS);
        num_leaders = std::max<size_t>(num_leaders, 3);
        Bucket leaders(num_leaders);
        std::mt19937 prng(seed);
        std::sample(ids.begin(), ids.end(), leaders.begin(), leaders.size(), prng);

        PointSet leader_points = ExtractPoints(points, leaders);

        // find closest leaders and build clusters around leaders
        std::vector<Bucket> clusters(leaders.size());
        std::vector<SpinLock> cluster_locks(leaders.size());
        parlay::parallel_for(0, ids.size(), [&](size_t i) {
            auto point_id = ids[i];
            auto closest_leaders = ClosestLeaders(points, leader_points, point_id, fanout).Take();

            for (const auto& [_, leader] : closest_leaders) {
                cluster_locks[leader].lock();
                clusters[leader].push_back(point_id);
                cluster_locks[leader].unlock();
            }
        });
        cluster_locks.clear(); cluster_locks.shrink_to_fit();
        leaders.clear(); leaders.shrink_to_fit();
        leader_points.Drop();

        if (depth == 0) {
            std::cout << "Closest leaders on top level took " << timer.Stop() << std::endl;
        }

        // recurse on clusters
        std::vector<Bucket> buckets;
        SpinLock bucket_lock;
        parlay::parallel_for(0, clusters.size(), [&](size_t cluster_id) {
            std::vector<Bucket> recursive_buckets;
            if (depth > MAX_DEPTH || (depth > CONCERNING_DEPTH && clusters[cluster_id].size() > TOO_SMALL_SHRINKAGE_FRACTION * ids.size())) {
                // Base case for duplicates and near-duplicates. Split the buckets randomly
                auto ids_copy = ids;
                std::mt19937 prng(seed + depth + ids.size());
                std::shuffle(ids_copy.begin(), ids_copy.end(), prng);
                for (size_t i = 0; i < ids_copy.size(); i += MAX_CLUSTER_SIZE) {
                    auto& new_bucket = recursive_buckets.emplace_back();
                    for (size_t j = 0; j < MAX_CLUSTER_SIZE; ++j) {
                        new_bucket.push_back(ids_copy[j]);
                    }
                }
            } else {
                // The normal case
                recursive_buckets = RecursivelySketch(points, clusters[cluster_id], depth + 1, /*fanout=*/1);
            }

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
            std::cout << "Sketching rep " << rep << std::endl;
            Timer timer2;
            timer2.Start();
            std::vector<Bucket> new_buckets = RecursivelySketch(points, all_ids, 0, FANOUT);
            std::cout << "Finished sketching rep. It took " << timer2.Stop() << " seconds."  << std::endl;
            buckets.insert(buckets.end(), new_buckets.begin(), new_buckets.end());
        }
        std::cout << "Start bucket brute force" << std::endl;
        return BruteForceBuckets(points, buckets, num_neighbors);
    }


    std::vector<NNVec> CrunchBucket(PointSet& points, const Bucket& bucket, int num_neighbors) {
        PointSet bucket_points = ExtractPoints(points, bucket);

        std::vector<TopN> neighbors(bucket.size(), TopN(num_neighbors));

        for (size_t i = 0; i < bucket.size(); ++i) {
            float* P = bucket_points.GetPoint(i);
            for (size_t j = i + 1; j < bucket.size(); ++j) {
                float* Q = bucket_points.GetPoint(j);
                float dist = distance(P, Q, points.d);
                neighbors[i].Add(std::make_pair(dist, bucket[j]));
                neighbors[j].Add(std::make_pair(dist, bucket[i]));
            }
        }

        std::vector<NNVec> result(bucket.size());
        for (size_t i = 0; i < bucket.size(); ++i) {
            result[i] = neighbors[i].Take();
        }
        return result;
    }

    AdjGraph BruteForceBuckets(PointSet& points, std::vector<Bucket>& buckets, int num_neighbors) {
        std::vector<SpinLock> locks(points.n);
        std::vector<NNVec> top_neighbors(points.n);


        {
            std::cout << "Number of buckets to crunch " << buckets.size() << std::endl;
            std::vector<size_t> bucket_sizes(buckets.size());
            for (size_t i = 0; i < buckets.size(); ++i) bucket_sizes[i] = buckets[i].size();
            double avg_size = 0.0; for (const auto& b : bucket_sizes) avg_size += b; avg_size /= buckets.size();
            std::cout << "avg bucket size : " << avg_size << std::endl;
            std::sort(bucket_sizes.begin(), bucket_sizes.end());
            std::vector<double> quantiles = { 0.0, 0.01, 0.05, 0.1, 0.15, 0.5, 0.85, 0.9, 0.95, 0.99, 1.0 };
            for (double quantile : quantiles) {
                size_t pos = quantile * buckets.size();
                pos = std::clamp<size_t>(pos, 0, buckets.size() - 1);
                std::cout << "quant " << quantile << " size :  " << bucket_sizes[pos] << std::endl;
            }
        }


        timer.Start();

        parlay::parallel_for(0, buckets.size(), [&](size_t bucket_id) {
            auto& bucket = buckets[bucket_id];
            auto bucket_neighbors = CrunchBucket(points, bucket, num_neighbors);
            for (size_t j = 0; j < bucket.size(); ++j) {
                auto point_id = bucket[j];

                locks[point_id].lock();
                // insert new neighbors. due to possible duplicate neighbors, we can't insert directly into the top-k data structure, and instead have to do this
                auto& n = top_neighbors[point_id];
                n.insert(n.end(), bucket_neighbors[j].begin(), bucket_neighbors[j].end());
                std::sort(n.begin(), n.end(), [](const auto& l, const auto& r) {return std::tie(l.second, l.first) < std::tie(r.second, r.first);});
                n.erase(std::unique(n.begin(), n.end(), [&](const auto& l, const auto& r) { return l.second == r.second; }), n.end());
                std::sort(n.begin(), n.end());
                n.resize(std::min<size_t>(n.size(), num_neighbors));

                locks[point_id].unlock();
            }

        }, 1);

        std::cout << "Brute forcing buckets took " << timer.Stop() << std::endl;

        AdjGraph graph(points.n);
        parlay::parallel_for(0, points.n, [&](size_t i) {
            for (const auto& [dist, point_id] : top_neighbors[i]) {
                graph[i].push_back(point_id);
            }
        });
        return graph;
    }


    int seed = 555;
    static constexpr double FRACTION_LEADERS = 0.005;
    static constexpr size_t TOP_LEVEL_NUM_LEADERS = 950;
    static constexpr size_t MAX_NUM_LEADERS = 2000;
    static constexpr size_t MAX_CLUSTER_SIZE = 3500;
    static constexpr int REPETITIONS = 3;
    static constexpr int FANOUT = 3;
    static constexpr int MAX_DEPTH = 14;
    static constexpr int CONCERNING_DEPTH = 10;
    static constexpr double TOO_SMALL_SHRINKAGE_FRACTION = 0.8;

    Timer timer;
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
