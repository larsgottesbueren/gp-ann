#pragma once

#include <vector>
#include <cstdint>

struct PointSet {
  std::vector<float> coordinates;   // potentially empty
  size_t d = 0, n = 0;
  float* GetPoint(size_t i) { return &coordinates[i*d]; }
};

using AdjGraph = std::vector<std::vector<int>>;

using NNVec = std::vector<std::pair<float, uint32_t>>;

struct HNSWParameters {
    int M = 16;
    int ef_construction = 200;
};
