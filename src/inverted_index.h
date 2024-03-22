#pragma once

#include "defs.h"
#include "dist.h"
#include "topn.h"

struct InvertedIndex {
    PointSet clustered_points;
    std::vector<int> offsets;
    std::vector<uint32_t> permutation;

    InvertedIndex(PointSet& points, const Clusters& clusters) {
        size_t num_shards = clusters.size();
        offsets.assign(num_shards + 1, 0);
        for (size_t b = 0; b < clusters.size(); b++) {
            offsets[b + 1] = clusters[b].size();
        }
        for (size_t i = 2; i < num_shards + 1; ++i)
            offsets[i] += offsets[i - 1];
        size_t num_inserts = offsets.back();
        permutation.resize(num_inserts);

        clustered_points.n = num_inserts;
        clustered_points.d = points.d;
        clustered_points.coordinates.resize(num_inserts * points.d);

        parlay::parallel_for(
                0, clusters.size(),
                [&](size_t b) {
                    parlay::parallel_for(0, clusters[b].size(), [&](size_t i_local) {
                        const uint32_t point_id = clusters[b][i_local];
                        permutation[offsets[b] + i_local] = point_id;
                        float* P = clustered_points.GetPoint(offsets[b] + i_local);
                        float* O = points.GetPoint(point_id);
                        for (uint32_t j = 0; j < points.d; ++j) {
                            P[j] = O[j];
                        }
                    });
                },
                1);
    }

    NNVec Query(float* Q, int k, const std::vector<int>& buckets_to_probe, size_t num_buckets_to_probe) {
        TopN top_k(k);
        for (size_t j = 0; j < num_buckets_to_probe; ++j) {
            int b = buckets_to_probe[j];
            for (int i = offsets[b]; i < offsets[b + 1]; ++i) {
                float new_dist = distance(clustered_points.GetPoint(i), Q, clustered_points.d);
                auto x = std::make_pair(new_dist, i);
                top_k.Add(x);
            }
        }

        auto result = top_k.Take();
        // remap the IDs
        for (auto& x : result) {
            x.second = permutation[x.second];
        }
        return result;
    }

    NNVec QueryBucket(float* Q, int k, int bucket) {
        TopN top_k(k);
        for (int i = offsets[bucket]; i < offsets[bucket + 1]; ++i) {
            float new_dist = distance(clustered_points.GetPoint(i), Q, clustered_points.d);
            auto x = std::make_pair(new_dist, i);
            top_k.Add(x);
        }
        auto result = top_k.Take();
        // remap the IDs
        for (auto& x : result) {
            x.second = permutation[x.second];
        }
        return result;
    }
};
