#pragma once

#include "defs.h"
#include "topn.h"

struct InvertedIndex {
    PointSet reordered_P;
    std::vector<int> offsets;
    std::vector<int> permutation;
    TopN top_k;

    InvertedIndex(PointSet& P, const std::vector<int>& partition, int k) :
        top_k(k)
    {
        offsets.assign(k+1, 0);
        for (int b : partition) offsets[b+1]++;
        for (int i = 2; i < k+1; ++i) offsets[i] += offsets[i-1];
        auto off = offsets;     // copy we can destroy
        permutation.resize(P.n);
        for (int i = 0; i < P.n; ++i) {
            int b = partition[i];
            permutation[off[b]++] = i;
        }

        reordered_P.n = P.n; reordered_P.d = P.d;
        reordered_P.coordinates.reserve(P.coordinates.size());
        for (int i = 0; i < P.n; ++i) {
            float* p = P.GetPoint(permutation[i]);
            for (int j = 0; j < P.d; ++j) {
                reordered_P.coordinates.push_back(p[j]);
            }
        }
    }

    NNVec Query(float* Q, std::vector<int>& buckets_to_probe, size_t num_buckets_to_probe) {

        for (int b : buckets_to_probe) {
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
