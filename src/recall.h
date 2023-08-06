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
            float dist = distance(P, Q, points.d);
            top_k.Add(std::make_pair(j, dist));
        }
        d[i] = top_k.Top().first;
    }, 1);
    return d;
}

std::vector<NNVec> GetGroundTruth(PointSet& points, PointSet& queries, int k) {
    std::vector<NNVec> res(queries.n);
    parlay::parallel_for(0, queries.n, [&](size_t i) {
        TopN top_k(k);
        float* Q = queries.GetPoint(i);
        for (uint32_t j = 0; j < points.n; ++j) {
            float* P = queries.GetPoint(j);
            float dist = distance(P, Q, points.d);
            top_k.Add(std::make_pair(j, dist));
        }
        res[i] = top_k.Take();
    }, 1);
    return res;
}

double OracleRecall(const std::vector<NNVec>& ground_truth, const std::vector<int>& partition) {
    int num_shards = *std::max_element(partition.begin(),  partition.end()) + 1;
    std::vector<int> freq;
    double recall = 0.0;
    for (const auto& neigh : ground_truth) {
        freq.assign(num_shards, 0);
        for (const auto& x : neigh) {
            freq[partition[x.second]]++;
        }


        int hits = *std::max_element(freq.begin(), freq.end());
        recall += static_cast<double>(hits) / neigh.size();
        if (neigh.size() != 10) std::cerr << "neigh.size() != 10..." << neigh.size() << std::endl;
    }

    recall /= ground_truth.size();
    std::cout << "Oracle first shard recall " << recall << std::endl;
    return recall;
}

double Recall(const std::vector<NNVec>& neighbors_per_query, const std::vector<float>& distance_to_kth_neighbor, int k) {
    size_t hits = 0;
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
