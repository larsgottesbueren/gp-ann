#pragma once

#include "defs.h"
#include "dist.h"
#include "topn.h"
#include <sstream>
#include <iostream>
#include <parlay/parallel.h>

std::vector<float> ComputeDistanceToKthNeighbor(PointSet& points, PointSet& queries, int k) {
    std::vector<float> d(queries.n);
    parlay::parallel_for(0, queries.n, [&](size_t i) {
        TopN top_k(k);
        float* Q = queries.GetPoint(i);
        for (uint32_t j = 0; j < points.n; ++j) {
            float* P = queries.GetPoint(j);
            float dist = distance(P, Q, P.d);
            top_k.Add(std::make_pair(j, dist));
        }
        d[i] = top_k.Top().first;
    }, 1);
    return d;
}

double Recall(const std::vector<NNVec>& neighbors_per_query, const std::vector<float>& distance_to_kth_neighbor, int k) {
    size_t hits;
    for (size_t i = 0; i < neighbors_per_query.size(); ++i) {
        for (const auto& x : neighbors_per_query[i]) {
            if (x.first <= distance_to_kth_neighbor[i]) {
                hits++;
            }
        }
    }
    double recall = static_cast<double>(hits) / static_cast<double>(neighbors_per_query.size() * k);
    return recall;
}
