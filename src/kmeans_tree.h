#pragma once

#include "kmeans.h"

struct KMeansTreeNode {
	std::vector<KMeansTreeNode> children;
};

struct KMeansTree {

	std::vector<int> Query(float* Q);
};
