#pragma once

#include <utility>

#include "defs.h"

#include <parlay/parallel.h>

#include "../external/hnswlib/hnswlib/hnswlib.h"

#include "metis_io.h"

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

    void Train(PointSet& routing_points) {
        parlay::parallel_for(0, routing_points.n, [&](size_t i) { hnsw->addPoint(routing_points.GetPoint(i), i); });
    }

    void Serialize(const std::string& file) {
        hnsw->saveIndex(file);
        WriteMetisPartition(partition, file + ".routing_index_partition");
    }

    struct ShardPriorities {
        std::vector<float> min_dist;
        std::vector<int> frequency;

        std::vector<int> RoutingQuery() const {
            std::vector<int> probes(min_dist.size());
            std::iota(probes.begin(), probes.end(), 0);
            std::sort(probes.begin(), probes.end(), [&](int l, int r) { return min_dist[l] < min_dist[r]; });
            return probes;
        }

        std::vector<int> PyramidRoutingQuery() const {
            std::vector<int> probes;
            for (size_t b = 0; b < min_dist.size(); ++b) {
                if (min_dist[b] != std::numeric_limits<float>::max()) {
                    probes.push_back(b);
                }
            }
            return probes;
        }

        std::vector<int> SPANNRoutingQuery(double eps) const {
            double closest_shard_dist = *std::min_element(min_dist.begin(), min_dist.end()) * (1.0 + eps);
            std::vector<int> probes;
            for (size_t b = 0; b < min_dist.size(); ++b) {
                if (min_dist[b] <= closest_shard_dist) {
                    probes.push_back(b);
                }
            }
            return probes;
        }

        std::vector<int> FrequencyQuery() const {
            std::vector<int> probes;
            size_t highest_freq = 0;
            for (size_t b = 1; b < frequency.size(); ++b) {
                if (frequency[b] > frequency[highest_freq]) {
                    highest_freq = b;
                }
            }
            probes.push_back(highest_freq);
            for (size_t b = 0; b < frequency.size(); ++b) {
                if (b != highest_freq) {
                    probes.push_back(b);
                }
            }
            std::sort(probes.begin() + 1, probes.end(), [&](int l, int r) { return min_dist[l] < min_dist[r]; });
            return probes;
        }
    };

    ShardPriorities Query(float* Q, int num_voting_neighbors) {
        auto near_neighbors = hnsw->searchKnn(Q, num_voting_neighbors);

        ShardPriorities result;
        result.min_dist.assign(num_shards, std::numeric_limits<float>::max());
        result.frequency.assign(num_shards, 0);
        while (!near_neighbors.empty()) {
            auto [dist, point_id] = near_neighbors.top();
            near_neighbors.pop();
            result.min_dist[partition[point_id]] = std::min(result.min_dist[partition[point_id]], dist);
            result.frequency[partition[point_id]]++;
        }
        return result;
    }
};
