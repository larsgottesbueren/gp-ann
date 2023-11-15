#pragma once

#include "points_io.h"
#include "metis_io.h"

#include "hnsw_router.h"
#include "../external/hnswlib/hnswlib/hnswlib.h"

#include "../external/message-queue/include/message-queue/buffered_queue.hpp"

class DistributedQueryBenchmark {
public:
    int rank;
    int comm_size;

    int num_shards;
    int dim;
    int num_neighbors;

    PointSet shard_points;
    std::vector<int> partition;

    #ifdef MIPS_DISTANCE
    using SpaceT = hnswlib::InnerProductSpace;
    #else
    using SpaceT = hnswlib::L2Space;
    #endif

    std::unique_ptr<SpaceT> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
    std::unique_ptr<HNSWRouter> router;
    HNSWParameters hnsw_parameters;

    DistributedQueryBenchmark() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    }

    void LoadPartition(const std::string& partition_file) {
        partition = ReadMetisPartition(partition_file);
        num_shards = NumPartsInPartition(partition);
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

    std::vector<int> Route(float* Q) {
        // TODO set the parameters during LoadRouter
        std::vector<int> probes = router->Query(Q, 250);
        // TODO trim down to desired number of probes
        return probes;
    }


    void ProcessQueries(const std::vector<int>& query_ids, PointSet& queries) {
        message_queue::FlushStrategy flush_strategy = message_queue::FlushStrategy::global;

        using RequestType = int;
        auto requests_queue =
                message_queue::make_buffered_queue<RequestType>();

        using ResponseType = std::pair<float, int>;
        auto responses_queue =
                message_queue::make_buffered_queue<ResponseType>();

        std::vector<int> local_requests;
        // TODO parallelize?
        for (int query_id : query_ids) {
            std::vector<int> probes = Route(queries.GetPoint(query_id));
            for (int shard_id : probes) {
                if (shard_id == rank) {
                    local_requests.push_back(query_id);
                } else {
                    requests_queue.post_message(query_id, shard_id);
                }
            }
        }

        auto return_neighbors = [&](message_queue::Envelope<RequestType> auto request_envelope) {
            for (const RequestType& query_id : request_envelope.message) {
                int original_sender = request_envelope.sender;
                auto result = hnsw->searchKnn(queries.GetPoint(query_id), num_neighbors);
                while (!result.empty()) {
                    auto next = result.top();
                    result.pop();
                    // TODO aggregate neighbors for the same query...
                    // TODO the concept doesn't work with recursively defined pairs of MPI_Datatypes --> atm query ID misses
                    responses_queue.post_message(next, original_sender);
                }
            }
        };

        auto accept_returned_neighbors = [&](message_queue::Envelope<ResponseType> auto response_envelope) {
            /* Do nothing here */
        };

        requests_queue.terminate(return_neighbors);
        responses_queue.terminate(accept_returned_neighbors);
    }
};
