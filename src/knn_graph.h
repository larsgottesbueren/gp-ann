#pragma once

#include "defs.h"
#include "dist.h"
#include "topn.h"
#include <sstream>
#include <iostream>
#include <parlay/parallel.h>


std::vector<int> TopKNeighbors(PointSet& P, uint32_t my_id, int k) {
	TopN top_k(k);
	float* Q = P.GetPoint(my_id);
	for (uint32_t i = 0; i < P.n; ++i) {
	    if (i == my_id) continue;
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
	parlay::parallel_for(0, P.n, [&](size_t i) { graph[i] = TopKNeighbors(P, i, k); });
	return graph;
}

void Symmetrize(AdjGraph& graph) {
    std::vector<size_t> degrees; degrees.reserve(graph.size());
    for (const auto& n : graph) degrees.push_back(n.size());
    for (size_t u = 0; u < graph.size(); ++u) {
        for (size_t j = 0; j < degrees[u]; ++j) {
            auto v = graph[u][j];
            graph[v].push_back(u);
        }
    }
}
