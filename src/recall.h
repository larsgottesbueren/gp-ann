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
            float* P = points.GetPoint(j);
            float dist = distance(P, Q, points.d);
            top_k.Add(std::make_pair(dist, j));
        }
        d[i] = top_k.Top().first;
    }, 1);
    return d;
}

std::vector<NNVec> ComputeGroundTruth(PointSet& points, PointSet& queries, int k) {
    std::vector<NNVec> res(queries.n);
    parlay::parallel_for(0, queries.n, [&](size_t i) {
        TopN top_k(k);
        float* Q = queries.GetPoint(i);
        for (uint32_t j = 0; j < points.n; ++j) {
            float* P = points.GetPoint(j);
            float dist = distance(P, Q, points.d);
            top_k.Add(std::make_pair(dist, j));
        }
        res[i] = top_k.Take();
        std::sort(res[i].begin(), res[i].end());    // should be = std::reverse
    }, 1);
    return res;
}

void OracleRecall(const std::vector<NNVec>& ground_truth, const std::vector<int>& partition, int num_neighbors) {
    int num_shards = NumPartsInPartition(partition);
    std::vector<size_t> hits(num_shards, 0);

    for (const auto& neigh : ground_truth) {
        std::vector<std::pair<int, int>> freq(num_shards);
        for (int i = 0; i < num_shards; ++i) {
            freq[i].first = 0;
            freq[i].second = i;
        }
        for (int i = 0; i < num_neighbors; ++i) {
            freq[partition[neigh[i].second]].first++;
        }

        std::sort(freq.begin(), freq.end(), std::greater<>());
        for (int i = 0; i < num_shards; ++i) {
            hits[i] += freq[i].first;
        }
    }

    size_t total = 0;
    for (int i = 0; i < num_shards && hits[i] > 0; ++i) {
        total += hits[i];
        double recall = static_cast<double>(total) / ground_truth.size() / num_neighbors;
        std::cout << "nprobes = " << i + 1 << " oracle recall = " << recall << std::endl;
    }

}


/**
 * This function also checks whether the computed distances and order in the ground truth are correct. If not, it will emit a warning and reorder the candidates.
 */
std::vector<float> ConvertGroundTruthToDistanceToKthNeighbor(std::vector<NNVec>& ground_truth, int k, PointSet& points, PointSet& queries) {
    if (ground_truth.size() != queries.n) {
        std::cout << "Ground truth size and number of queries don't match." << std::endl;
        std::exit(0);
    }
    std::cout << "Convert ground truth. k = " << k << " points.d " << points.d << std::endl;
    std::vector<float> distance_to_kth_neighbor(ground_truth.size());
    size_t distance_mismatches = 0;
    size_t wrong_sorts = 0;
    size_t wrong_sorts_before_before_recalc = 0;

    std::vector<double> epss = { 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10 };
    std::vector<size_t> wrongss(epss.size(), 0);


    parlay::parallel_for(0, queries.n, [&](size_t q) {
        auto& neighs = ground_truth[q];
        auto comp = [](const std::pair<float, uint32_t>& l, const std::pair<float, uint32_t>& r) {
            return l.first < r.first;
        };
        bool is_sorted_before_recalc = std::is_sorted(neighs.begin(), neighs.begin() + k, comp);
        if (!is_sorted_before_recalc) {
            __atomic_fetch_add(&wrong_sorts_before_before_recalc, 1, __ATOMIC_RELAXED);
        }

        float* Q = queries.GetPoint(q);
        size_t local_distance_mismatches = 0;
        for (int j = 0; j < k; ++j) {
            uint32_t point_id = neighs[j].second;
            float dist = neighs[j].first;
            float true_dist = distance(points.GetPoint(point_id), Q, points.d);
            if (std::abs(dist - true_dist) > 1e-8) {
                local_distance_mismatches++;
            }

            for (size_t r = 0; r < epss.size(); ++r) {
                if (std::abs(dist - true_dist) > epss[r]) {
                    __atomic_fetch_add(&wrongss[r], 1, __ATOMIC_RELAXED);
                }
            }
            neighs[j].first = true_dist;
        }

        bool is_sorted = std::is_sorted(neighs.begin(), neighs.begin() + k, comp);
        if (!is_sorted) {
            std::sort(neighs.begin(), neighs.begin() + k, comp);
            __atomic_fetch_add(&wrong_sorts, 1, __ATOMIC_RELAXED);
        }
        distance_to_kth_neighbor[q] = neighs[k-1].first;

        if (local_distance_mismatches > 0) {
            __atomic_fetch_add(&distance_mismatches, local_distance_mismatches, __ATOMIC_RELAXED);
        }
    });

    std::cout << distance_mismatches << " out of " << ground_truth.size() * ground_truth[0].size() << " distances were wrong" << std::endl;
    std::cout << wrong_sorts_before_before_recalc << " out of " << ground_truth.size() << " neighbors lists were ordered incorrectly before recomputing distances. And " << wrong_sorts << " were ordered incorrectly after recomputing" << std::endl;

    for (size_t r = 0; r < epss.size(); ++r) {
        std::cout << "For eps = " << epss[r] << " there were " << wrongss[r] << " many distances wrong, i.e., |d1 - d2| > eps" << std::endl;
    }

    return distance_to_kth_neighbor;
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
