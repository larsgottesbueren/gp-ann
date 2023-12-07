#pragma once

#include <vector>
#include <cstdint>
#include <chrono>
#include <algorithm>

#include "topn.h"
#include "dist.h"

struct PointSet {
  std::vector<float> coordinates;   // potentially empty
  size_t d = 0, n = 0;
  float* GetPoint(size_t i) { return &coordinates[i*d]; }
  void Drop() { coordinates.clear(); coordinates.shrink_to_fit(); }
  void Alloc() { coordinates.resize(n*d, 0.f); }
  bool empty() const { return n == 0; }
};

void Normalize(PointSet& points) {
    for (size_t i = 0; i < points.n; ++i) {
        float* p = points.GetPoint(i);
        if (!L2Normalize(p, points.d)) {
            std::cerr << "Point " << i << " is fully zero --> delete" << std::endl;
        }
    }
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

int NumPartsInPartition(const std::vector<int>& partition) {
    if (partition.empty()) return 0;
    return *std::max_element(partition.begin(), partition.end()) + 1;
}

std::vector<std::vector<uint32_t>> ConvertPartitionToBuckets(const std::vector<int>& partition) {
    int num_buckets = NumPartsInPartition(partition);
    std::vector<std::vector<uint32_t>> buckets(num_buckets);
    for (uint32_t u = 0; u < partition.size(); ++u) {
        buckets[partition[u]].push_back(u);
    }
    return buckets;
}

void RemapPartitionIDs(std::vector<int>& partition) {
    if (partition.empty()) return;
    int num_shards = NumPartsInPartition(partition);
    std::vector<size_t> num_points_in_shard(num_shards, 0);
    for (int x : partition) { num_points_in_shard[x]++; }
    if (std::any_of(num_points_in_shard.begin(), num_points_in_shard.end(), [](const auto& n) { return n == 0; })) {
        std::vector<int> remapped(num_shards, 0);
        int l = 0;
        for (int r = 0; r < num_shards; ++r) {
            if (num_points_in_shard[r] > 0) {
                remapped[r] = l++;
            }
        }
        for (int& x : partition) { x = remapped[x]; }
    }
}

using AdjGraph = std::vector<std::vector<int>>;

using NNVec = std::vector<std::pair<float, uint32_t>>;

NNVec ConvertTopKToNNVec(TopN& top_k) {
    NNVec res = top_k.Take();
    std::reverse(res.begin(), res.end());
    return res;
}

int Top1Neighbor(PointSet& P, float* Q) {
    int best = -1;
    float best_dist = std::numeric_limits<float>::max();
    for (size_t i = 0; i < P.n; ++i) {
        float new_dist = distance(P.GetPoint(i), Q, P.d);
        if (new_dist < best_dist) {
            best_dist = new_dist;
            best = i;
        }
    }
    return best;
}

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

struct HNSWParameters {
    size_t M = 32;
    size_t ef_construction = 200;
    size_t ef_search = 250;
};

using Duration = std::chrono::duration<double>;
using Timepoint = decltype(std::chrono::high_resolution_clock::now());

struct Timer {
    bool running = false;
    Timepoint start;
    Duration total_duration = Duration(0.0);

    Timepoint Start() {
        if (running) throw std::runtime_error("Called Start() but timer is already running");
        start = std::chrono::high_resolution_clock::now();
        running = true;
        return start;
    }

    double Stop() {
        if (!running) throw std::runtime_error("Timer not running but called Stop()");
        auto finish = std::chrono::high_resolution_clock::now();
        Duration duration = finish - start;
        total_duration += duration;
        running = false;
        return duration.count();
    }

    double Restart() {
        if (!running) throw std::runtime_error("Timer not running but called Restart()");
        auto finish = std::chrono::high_resolution_clock::now();
        Duration duration = finish - start;
        total_duration += duration;
        start = finish;
        return duration.count();
    }

    double ElapsedRunning() {
        if (!running) throw std::runtime_error("Timer not running but called Restart()");
        auto finish = std::chrono::high_resolution_clock::now();
        Duration duration = finish - start;
        return duration.count();
    }

};


template <typename T1, typename T2>
inline auto idiv_ceil(T1 a, T2 b) {
    return static_cast<T1>((static_cast<unsigned long long>(a)+b-1) / b);
}

inline std::pair<size_t, size_t> bounds(size_t i, size_t n, size_t chunk_size) {
    return std::make_pair(std::min(n, i * chunk_size), std::min(n, (i+1) * chunk_size));
}
