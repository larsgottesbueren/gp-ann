#pragma once

#include <vector>

struct PointSet {
  std::vector<float> coordinates;
  size_t d, n;
  float* GetPoint(size_t i) { return coordinates.data() + i*d; }
};

using AdjGraph = std::vector<std::vector<int>>;
