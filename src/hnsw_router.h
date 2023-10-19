#pragma once

#include "defs.h"
#include "dist.h"

#include "../external/hnswlib/hnswlib/hnswlib.h"

struct HNSWRouter {
    PointSet routing_points;
    std::vector<int> partition_offsets;
    std::vector<int> partition;
    int num_shards;

    #ifdef MIPS_DISTANCE
    hnswlib::InnerProductSpace space;
    #else
    hnswlib::L2Space space;
    #endif

    std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
    HNSWParameters hnsw_parameters;

    HNSWRouter(PointSet routing_points_, std::vector<int> partition_offsets_, HNSWParameters parameters) :
        routing_points(std::move(routing_points_)),
        partition_offsets(std::move(partition_offsets_)),
        space(routing_points.d),
        hnsw_parameters(parameters)
    {
        // build partition array
        for (size_t i = 1; i < partition_offsets.size(); ++i) {
            for (int j = partition_offsets[i-1]; j < partition_offsets[i]; ++j) {
                partition.push_back(i-1);
            }
        }

        num_shards = partition_offsets.size() - 1;

        hnsw = std::make_unique<hnswlib::HierarchicalNSW<float>>(&space, routing_points.n, hnsw_parameters.M, hnsw_parameters.ef_construction, /* random seed = */ 500);
        hnsw->setEf(hnsw_parameters.ef_search);

        std::cout << "num routing points " << routing_points.n << std::endl;

        // insert points...
        parlay::parallel_for(0, routing_points.n, [&](size_t i) {
            hnsw->addPoint(routing_points.GetPoint(i), i);
        });
    }

    std::vector<int> Query(float* Q, int num_voting_neighbors) {
        auto near_neighbors = hnsw->searchKnn(Q, num_voting_neighbors);

        std::vector<float> min_dist(num_shards, std::numeric_limits<float>::max());
        while (!near_neighbors.empty()) {
            auto [dist, point_id] = near_neighbors.top();
            near_neighbors.pop();
            min_dist[partition[point_id]] = std::min(min_dist[partition[point_id]], dist);
        }

        std::vector<int> probes(num_shards);
        std::iota(probes.begin(), probes.end(), 0);
        std::sort(probes.begin(), probes.end(), [&](int l, int r) {
            return min_dist[l] < min_dist[r];
        });
        return probes;
    }
};
