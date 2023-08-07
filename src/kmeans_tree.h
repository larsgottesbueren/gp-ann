#pragma once

#include "kmeans.h"

struct KMeansTreeNode {
	std::vector<KMeansTreeNode> children;
	PointSet centroids;
};

struct KMeansTreeRouter {

	std::vector<int> Query(float* Q, int num_shards) {
	    std::vector<int> probes(num_shards);
	    for (int i = 0; i < num_shards; ++i) { probes[i] = i; }
	    return probes;
	}
};
