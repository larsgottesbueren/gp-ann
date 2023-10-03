#pragma once

#include "defs.h"
#include "topn.h"

#include "../external/hnswlib/hnswlib/hnswlib.h"

#include <parlay/parallel.h>

struct InvertedIndexHNSW {
    hnswlib::L2Space space;     // TODO also support cosine similarity
    std::vector<hnswlib::HierarchicalNSW<float>* > bucket_hnsws;

    static constexpr int M = 16;
    static constexpr int ef_construction = 200;

    InvertedIndexHNSW(PointSet& points, const std::vector<int>& partition) : space(points.d)
    {
        int num_shards = *std::max_element(partition.begin(), partition.end()) + 1;
        bucket_hnsws.resize(num_shards);
        std::vector<size_t> bucket_size(num_shards, 0);
        for (int x : partition) bucket_size[x]++;

        for (int b = 0; b < num_shards; ++b) {
            bucket_hnsws[b] = new hnswlib::HierarchicalNSW<float>(&space, bucket_size[b], M, ef_construction, /* random_seed = */ 555 + b);
        }

        parlay::parallel_for(0, points.n, [&](size_t i) {
            float* p = points.GetPoint(i);
            bucket_hnsws[partition[i]]->addPoint(p, i);
        });
    }

    ~InvertedIndexHNSW() {
        for (size_t i = 0; i < bucket_hnsws.size(); ++i) {
            delete bucket_hnsws[i];
        }
    }

    NNVec Query(float* Q, int num_neighbors, const std::vector<int>& buckets_to_probe) const {
        TopN top_k(num_neighbors);
        for (int bucket : buckets_to_probe) {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = bucket_hnsws[bucket]->searchKnn(Q, num_neighbors);
            while (!result.empty()) {
                const auto [dist, label] = result.top();
                result.pop();
                top_k.Add(std::make_pair(dist, label));
            }
        }
        return top_k.Take();
    }
};
