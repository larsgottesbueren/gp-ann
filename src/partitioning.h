#pragma once

#include "defs.h"

Partition RecursiveKMeansPartitioning(PointSet& points, size_t max_cluster_size, int depth = 0, int num_clusters = -1);

Partition RebalancingKMeansPartitioning(PointSet& points, size_t max_cluster_size, int num_clusters = -1);

Partition KMeansPartitioning(PointSet& points, int num_clusters, double epsilon);

Partition PartitionAdjListGraph(const AdjGraph& adj_graph, int num_clusters, double epsilon, int num_threads = 1, bool strong = false, bool quiet = false);

Partition GraphPartitioning(PointSet& points, int num_clusters, double epsilon, bool strong, const std::string& graph_output_path = "");

Partition PyramidPartitioning(PointSet& points, int num_clusters, double epsilon, const std::string& routing_index_path = "");

// want to extract only the leaf-level points here
// and the mapping of top-level points to leaf-level points
std::pair<Partition, PointSet> HierarchicalKMeansParlayImpl(PointSet& points, double coarsening_ratio, int depth = 0);

std::pair<Partition, PointSet> HierarchicalKMeans(PointSet& points, double coarsening_ratio, int depth = 0);

Partition OurPyramidPartitioning(PointSet& points, int num_clusters, double epsilon, const std::string& routing_index_path, double coarsening_rate = 0.002);
