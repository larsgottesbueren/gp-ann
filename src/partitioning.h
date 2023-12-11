#pragma once

#include "defs.h"

std::vector<int> RecursiveKMeansPartitioning(PointSet& points, size_t max_cluster_size, int depth = 0, int num_clusters = -1);

std::vector<int> KMeansPartitioning(PointSet& points, int num_clusters, double epsilon);


std::vector<int> GraphPartitioning(PointSet& points, int num_clusters, double epsilon, const std::string& graph_output_path = "");

std::vector<int> PyramidPartitioning(PointSet& points, int num_clusters, double epsilon, const std::string& routing_index_path = "");

// want to extract only the leaf-level points here
// and the mapping of top-level points to leaf-level points
std::pair<std::vector<int>, PointSet>
HierarchicalKMeansParlayImpl(PointSet& points, double coarsening_ratio, int depth = 0);

std::pair<std::vector<int>, PointSet>
HierarchicalKMeans(PointSet& points, double coarsening_ratio, int depth = 0);

std::vector<int> OurPyramidPartitioning(PointSet& points, int num_clusters, double epsilon,
                                        const std::string& routing_index_path, double coarsening_rate = 0.002);
