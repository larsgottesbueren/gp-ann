#pragma once

#include "defs.h"

PointSet RandomSample(PointSet& points, size_t num_samples, int seed);
std::vector<int> KMeans(PointSet& P, PointSet& centroids);
double ObjectiveValue(PointSet& points, PointSet& centroids, const std::vector<int>& closest_center);
std::vector<int> BalancedKMeans(PointSet& points, PointSet& centroids, size_t max_cluster_size);
