#pragma once

#include "kmeans.h"

struct KMeansTreeNode {
	std::vector<KMeansTreeNode> children;
	PointSet centroids;
};

struct KMeansTreeRouter {

	std::vector<int> Query(float* Q, int k) {
	    std::vector<int> probes(k);
	    for (int i = 0; i < k; ++i) { probes[i] = i; }
	    return probes;
	}
};
