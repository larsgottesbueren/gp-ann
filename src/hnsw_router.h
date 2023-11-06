#pragma once

#include <utility>

#include "defs.h"
#include "dist.h"

#include "../external/hnswlib/hnswlib/hnswlib.h"

struct HNSWRouter {
    const std::vector<int>& partition;
    int num_shards;

    #ifdef MIPS_DISTANCE
    hnswlib::InnerProductSpace space;
    #else
    hnswlib::L2Space space;
    #endif

    std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
    HNSWParameters hnsw_parameters;

    HNSWRouter(PointSet& routing_points, int num_shards_, const std::vector<int>& partition_, HNSWParameters parameters) :
        partition(partition_),
        num_shards(num_shards_),
        space(routing_points.d),
        hnsw_parameters(parameters)
    {
        hnsw = std::make_unique<hnswlib::HierarchicalNSW<float>>(&space, routing_points.n, hnsw_parameters.M, hnsw_parameters.ef_construction, /* random seed = */ 500);
        parlay::parallel_for(0, routing_points.n, [&](size_t i) { hnsw->addPoint(routing_points.GetPoint(i), i); });
        hnsw->setEf(hnsw_parameters.ef_search);
    }


    HNSWRouter(const std::string& file, int dim, const std::vector<int>& partition_) :
        partition(partition_),
        num_shards(NumPartsInPartition(partition_)),
        space(dim),
        hnsw(new hnswlib::HierarchicalNSW<float>(&space, file))
    {
        hnsw->setEf(hnsw_parameters.ef_search);
    }

    void Serialize(const std::string& file) {
        hnsw->saveIndex(file);
        WriteMetisPartition(partition, file + ".routing_index_partition");
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

    std::vector<int> PyramidRoutingQuery(float* Q, int num_voting_neighbors) {
        auto near_neighbors = hnsw->searchKnn(Q, num_voting_neighbors);

        std::vector<float> min_dist(num_shards, std::numeric_limits<float>::max());
        while (!near_neighbors.empty()) {
            auto [dist, point_id] = near_neighbors.top();
            near_neighbors.pop();
            min_dist[partition[point_id]] = std::min(min_dist[partition[point_id]], dist);
        }

        std::vector<int> probes;
        for (int b = 0; b < num_shards; ++b) {
            if (min_dist[b] != std::numeric_limits<float>::max()) {
                probes.push_back(b);
            }
        }
        return probes;
    }

    std::vector<int> SPANNRoutingQuery(float* Q, int num_voting_neighbors, double eps) {
        auto near_neighbors = hnsw->searchKnn(Q, num_voting_neighbors);

        std::vector<float> min_dist(num_shards, std::numeric_limits<float>::max());
        while (!near_neighbors.empty()) {
            auto [dist, point_id] = near_neighbors.top();
            near_neighbors.pop();
            min_dist[partition[point_id]] = std::min(min_dist[partition[point_id]], dist);
        }


        double closest_shard_dist = *std::min_element(min_dist.begin(), min_dist.end()) * (1.0 + eps);
        std::vector<int> probes;
        for (int b = 0; b < num_shards; ++b) {
            if (min_dist[b] <= closest_shard_dist) {
                probes.push_back(b);
            }
        }
        return probes;
    }
};
