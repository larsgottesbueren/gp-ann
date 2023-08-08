#pragma once

#include "defs.h"
#include "topn.h"

struct InvertedIndex {
    PointSet reordered_P;
    std::vector<int> offsets;
    std::vector<int> permutation;

    InvertedIndex(PointSet& P, const std::vector<int>& partition)
    {
        int num_shards = *std::max_element(partition.begin(), partition.end()) + 1;
        offsets.assign(num_shards+1, 0);
        for (int b : partition) offsets[b+1]++;
        for (int i = 2; i < num_shards+1; ++i) offsets[i] += offsets[i-1];
        auto off = offsets;     // copy we can destroy
        permutation.resize(P.n);
        std::vector<int> inverse_permutation(P.n);
        for (size_t i = 0; i < P.n; ++i) {
            int b = partition[i];
            inverse_permutation[i] = off[b];
            permutation[off[b]++] = i;
        }

        reordered_P.n = P.n; reordered_P.d = P.d;
        reordered_P.coordinates.reserve(P.coordinates.size());
        for (size_t i = 0; i < P.n; ++i) {
            float* p = P.GetPoint(permutation[i]);
            for (size_t j = 0; j < P.d; ++j) {
                reordered_P.coordinates.push_back(p[j]);
            }
        }

        for (size_t i = 0; i < P.n; ++i) {
            float* p = P.GetPoint(i);
            float* q = reordered_P.GetPoint(inverse_permutation[i]);
            for (size_t j = 0; j < P.d; ++j) {
                if (p[j] != q[j]) {
                    std::cerr << "permutation bad? " << p[j] << " " << q[j] << " " << j << " " << i << " " << inverse_permutation[i] << std::endl;
                }
            }

            p = P.GetPoint(permutation[i]);
            q = reordered_P.GetPoint(i);
            for (size_t j = 0; j < P.d; ++j) {
                if (p[j] != q[j]) {
                    std::cerr << "inverse permutation bad? " << p[j] << " " << q[j] << " " << j << " " << i << " " << inverse_permutation[i] << std::endl;
                }
            }
        }

    }

    NNVec Query(float* Q, int k, std::vector<int>& buckets_to_probe, size_t num_buckets_to_probe) {
        TopN top_k(k);
        for (size_t j = 0; j < num_buckets_to_probe; ++j) {
            int b = buckets_to_probe[j];
            for (int i = offsets[b]; i < offsets[b+1]; ++i) {
                // TODO optimize?
                float new_dist = distance(reordered_P.GetPoint(i), Q, reordered_P.d);
                auto x = std::make_pair(new_dist, i);
                top_k.Add(x);
            }
        }

        auto result = top_k.Take();
        // remap the IDs
        for (auto& x : result) { x.second = permutation[x.second]; }
        return result;
    }
};
