#pragma once

#include <vector>
#include <cstdint>
#include <chrono>
#include <algorithm>

#include "topn.h"

struct PointSet {
  std::vector<float> coordinates;   // potentially empty
  size_t d = 0, n = 0;
  float* GetPoint(size_t i) { return &coordinates[i*d]; }
  void Drop() { coordinates.clear(); coordinates.shrink_to_fit(); }
  void Alloc() { coordinates.resize(n*d, 0.f); }
  void Resize(size_t _n) {
      n = _n;
      coordinates.resize(_n * d);
  }
  bool empty() const { return n == 0; }
};

PointSet ExtractPointsInBucket(const std::vector<uint32_t>& bucket, PointSet& points);

// maps a point to one cluster
using Partition = std::vector<int>;

// maps a cluster ID to its contained points
using Clusters = std::vector<std::vector<uint32_t>>;

// maps a point to (potentially multiple) clusters
using Cover = std::vector<std::vector<int>>;

int NumPartsInPartition(const Partition& partition);

Clusters ConvertPartitionToClusters(const Partition& partition);

Cover ConvertPartitionToCover(const Partition& partition);

Cover ConvertClustersToCover(const Clusters& clusters);

void RemapPartitionIDs(std::vector<int>& partition);

using AdjGraph = std::vector<std::vector<int>>;

using NNVec = std::vector<std::pair<float, uint32_t>>;

NNVec ConvertTopKToNNVec(TopN& top_k);

TopN ClosestLeaders(PointSet& points, PointSet& leader_points, uint32_t my_id, int k);

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

inline bool DoubleEquals(double x, double y, double eps = 1e-12) { return std::abs(x - y) < eps; }
