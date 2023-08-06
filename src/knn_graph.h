#pragma once

#include "defs.h"
#include "dist.h"
#include "topn.h"
#include <parlay/parallel.h>

std::vector<int> TopKNeighbors(PointSet& P, float* Q, int k) {
	TopN top_k(k);
	for (uint32_t i = 0; i < P.n; ++i) {
	    // TODO optimize with precomputed norms
		float new_dist = distance(P.GetPoint(i), Q, P.d);
		auto x = std::make_pair(new_dist, i);
		top_k.Add(x);
	}
	auto x = top_k.Take();
	std::vector<int> y;
	for (const auto& a : x) y.push_back(a.second);
	return y;
}

AdjGraph BuildKNNGraph(PointSet& P, int k) {
	AdjGraph graph(P.n);
	parlay::parallel_for(0, P.n, [&](size_t i) {
		graph[i] = TopKNeighbors(P, P.GetPoint(i), k);
	});
	return graph;
}
