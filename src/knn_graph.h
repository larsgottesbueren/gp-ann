#pragma once 

#include "defs.h"
#include "dist.h"
#include "topn.h"
#include <parlay/parallel.h>

std::vector<int> TopKNeighbors(const PointSet& P, float* Q, int k) {
	TopN top_k(k);
	for (size_t i = 0; i < P.n; ++i) {
		float new_dist = distance(P.GetPoint(i), Q, P.d);
		top_k.Add(std::make_pair(new_dist, i));
	}
	auto x = top_k.Take();
	std::vector<int> y;
	for (const auto& a : x) y.push_back(x.second);
	return y;
}

AdjGraph BuildKNNGraph(const PointSet& P, int k) {
	AdjGraph graph(P.n);
	parlay::parallel_for(0, P.n, [&](size_t i) {
		graph[i] = TopKNeighbors(P, P.GetPoint(i), k);
	});
	return graph;
}
