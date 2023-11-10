#pragma once

#include "points_io.h"
#include "metis_io.h"

#include "hnsw_router.h"
#include "../external/hnswlib/hnswlib/hnswlib.h"

class DistributedQueryBenchmark {
public:
    int rank;
    int comm_size;

    int num_shards;
    int dim;

    PointSet shard_points;
    std::vector<int> partition;

    #ifdef MIPS_DISTANCE
    using SpaceT = hnswlib::InnerProductSpace;
    #else
    using SpaceT = hnswlib::L2Space;
    #endif

    std::unique_ptr<SpaceT> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> shard_hnsw;
    std::unique_ptr<HNSWRouter> router;
    HNSWParameters hnsw_parameters;

    void LoadQueries(const std::string& query_file, const std::string& ground_truth_file) {
        PointSet queries = ReadPoints(query_file);
        auto ground_truth = ReadGroundTruth(ground_truth_file);
    }

    void LoadPartition(const std::string& partition_file) {
        partition = ReadMetisPartition(partition_file);
        num_shards = *std::max_element(partition.begin(), partition.end()) + 1;
    }

    void LoadShardPointSet(const std::string& point_set_file) {
        size_t num_points_in_shard = 0;
        for (const auto& part : partition) {
            if (part == rank) {
                num_points_in_shard++;
            }
        }

        uint32_t n, d;
        size_t offset = 0;

        std::ifstream in(point_set_file, std::ios::binary);
        in.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
        offset += sizeof(uint32_t);
        in.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
        offset += sizeof(uint32_t);

        shard_points.n = num_points_in_shard;
        shard_points.d = d;
        shard_points.coordinates.resize(shard_points.n * shard_points.d);
        dim = d;

        size_t coords_end = 0;
        for (uint32_t point_id = 0; point_id < n; ++point_id) {
            if (partition[point_id] == rank) {
                size_t begin = offset + point_id * d * sizeof(float);
                size_t range_length = 0;
                for ( ; point_id < n && partition[point_id] == rank; ++point_id) {
                    range_length += 1;
                }

                in.seekg(begin);
                in.read(reinterpret_cast<char*>(&shard_points.coordinates[coords_end]), range_length * d * sizeof(float));
                coords_end += range_length * d;
            }
        }
    }

    void BuildInShardIndex() {
        space = std::make_unique<SpaceT>(dim);
        hnsw = std::make_unique<hnswlib::HierarchicalNSW<float>>(space.get(), shard_points.n, hnsw_parameters.M, hnsw_parameters.ef_construction, /* random seed = */ 555);
        parlay::parallel_for(0, shard_points.n, [&](size_t i) { hnsw->addPoint(shard_points.GetPoint(i), i); });
        hnsw->setEf(hnsw_parameters.ef_search);
        shard_points.Drop();    // TODO we could do the HNSW insert during IO
    }

    void LoadRouter(const std::string& hnsw_router_file) {
        router = std::make_unique<HNSWRouter>(hnsw_router_file, dim, partition);
    }

    void ProcessQueries(std::vector<int>& queries) {
        // MPI barrier --> so that all ranks start at the same time
        MPI_Barrier(MPI_COMM_WORLD);

        // Design decisions
        // 1) Process routing queries first so that we can get everyone working as soon as possible
        //      Reduce #messages by grouping vectors going to the same shard
        //      Maybe send a couple messages already so the machines who run out of routing work get to do something
        // 2) Load balancing among replicas?    a) probe 2 random ones b) send to random one

        std::vector<int> waiting_for_return, incoming_requests;

        while (!queries.empty() && !waiting_for_return.empty() &&!incoming_requests.empty()) {
            // process routing queries first

            // listen for new requests

            // listen for potentially returned results


        }
    }
};
