#pragma once

#include "defs.h"


Clusters OverlappingGraphPartitioning(PointSet& points, int requested_num_clusters, double epsilon, double overlap, bool strong);

void MakeOverlappingWithCentroids(PointSet& points, Clusters& clusters, size_t max_cluster_size, size_t num_extra_assignments);

Clusters OverlappingKMeansPartitioningSPANN(PointSet& points, const Partition& partition, int requested_num_clusters, double epsilon, double overlap);
