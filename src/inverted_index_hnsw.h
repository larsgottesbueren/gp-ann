#pragma once

#include "defs.h"
#include "topn.h"

#include "../external/hnswlib/hnswlib/hnswlib.h"

#include <parlay/parallel.h>

struct InvertedIndexHNSW {
    HNSWParameters hnsw_parameters;
#ifdef MIPS_DISTANCE
    hnswlib::InnerProductSpace space;
#else
    hnswlib::L2Space space;
#endif

    std::vector<hnswlib::HierarchicalNSW<float> *> bucket_hnsws;


    InvertedIndexHNSW(PointSet& points) : space(points.d) {

    }

    void Build(PointSet& points, const Clusters& clusters) {
        size_t num_shards = clusters.size();
        bucket_hnsws.resize(num_shards);
        size_t total_insertions = 0;
        for (int b = 0; b < num_shards; ++b) {
            bucket_hnsws[b] = new hnswlib::HierarchicalNSW<float>(
                &space, clusters[b].size(),
                hnsw_parameters.M, hnsw_parameters.ef_construction,
                /* random_seed = */ 555 + b);

            bucket_hnsws[b]->setEf(hnsw_parameters.ef_search);
            total_insertions += clusters[b].size();
        }

        std::cout << "start HNSW insertions" << std::endl;

        parlay::parallel_for(0, clusters.size(), [&](size_t b) {
            parlay::parallel_for(0, clusters[b].size(), [&](size_t i_local) {
                float* p = points.GetPoint(clusters[b][i_local]);
                bucket_hnsws[b]->addPoint(p, clusters[b][i_local]);
            });
        });
    }

    ~InvertedIndexHNSW() {
        for (size_t i = 0; i < bucket_hnsws.size(); ++i) {
            delete bucket_hnsws[i];
        }
    }

    NNVec Query(float* Q, int num_neighbors, const std::vector<int>& buckets_to_probe, int num_probes) const {
        TopN top_k(num_neighbors);
        for (int i = 0; i < num_probes; ++i) {
            const int bucket = buckets_to_probe[i];
            auto result = bucket_hnsws[bucket]->searchKnn(Q, num_neighbors);
            while (!result.empty()) {
                const auto [dist, label] = result.top();
                result.pop();
                top_k.Add(std::make_pair(dist, label));
            }
        }
        return top_k.Take();
    }

    NNVec QueryBucket(float* Q, int num_neighbors, int bucket) {
        auto result_pq = bucket_hnsws[bucket]->searchKnn(Q, num_neighbors);
        NNVec result;
        while (!result_pq.empty()) {
            result.emplace_back(result_pq.top());
            result_pq.pop();
        }
        return result;
    }
};
