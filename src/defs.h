#pragma once

#include <vector>
#include <cstdint>
#include <chrono>

struct PointSet {
  std::vector<float> coordinates;   // potentially empty
  size_t d = 0, n = 0;
  float* GetPoint(size_t i) { return &coordinates[i*d]; }
  void Drop() { coordinates.clear(); coordinates.shrink_to_fit(); }
};

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

using AdjGraph = std::vector<std::vector<int>>;

using NNVec = std::vector<std::pair<float, uint32_t>>;

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

};
