#pragma once

#include "kmeans.h"

struct KMeansTreeNode {
	std::vector<KMeansTreeNode> children;
	PointSet centroids;
};

struct KMeansTreeRouter {

	std::vector<int> Query(float* Q);
};
