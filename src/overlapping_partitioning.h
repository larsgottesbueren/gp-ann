#pragma once

#include "defs.h"

Clusters OverlappingGraphPartitioning(PointSet& points, int num_clusters, double epsilon, double overlap);

Clusters OverlappingKMeansPartitioningSPANN(PointSet& points, int num_clusters, double epsilon, double overlap);
