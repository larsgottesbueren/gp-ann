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

void PinThread(int cpu_id) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu_id, &mask);
    const int err = sched_setaffinity(0, sizeof(mask), &mask);
    if (err) {
        std::cerr << "Thread pinning failed" << std::endl;
        std::abort();
    }
}

void PrintAffinityMask() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    sched_getaffinity(0, sizeof(mask), &mask);

    // -1 = before first range
    // 0 = not in range
    // 1 = range just started
    // 2 = long range
    int range_status = -1;

    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &mask)) {
            if (range_status <= 0) {
                if (range_status >= 0) {
                    std::cout << ",";
                }
                std::cout << cpu;
                range_status = 1;
            } else if (range_status == 1) {
                ++range_status;
            }
        } else if (range_status == 1) {
            range_status = 0;
        } else if (range_status == 2) {
            std::cout << "-" << cpu - 1;
            range_status = 0;
        }
    }
    std::cout << std::endl;
}

void PinThread(cpu_set_t old_affinity) {
    sched_setaffinity(0, sizeof(old_affinity), &old_affinity);
}

void UnpinThread() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        CPU_SET(cpu, &mask);
    }
    PinThread(mask);
}
