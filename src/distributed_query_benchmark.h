#pragma once

#include "points_io.h"
#include "metis_io.h"

#include "hnsw_router.h"
#include "../external/hnswlib/hnswlib/hnswlib.h"

#include "../external/message-queue/include/message-queue/buffered_queue.hpp"

#include <parlay/primitives.h>

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
    int num_voting_neighbors = 250;
    int num_probes = 2;

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
        shard_points.Drop();    // TODO we could do the HNSW insert during IO --> halve the memory requirement
    }

    void LoadRouter(const std::string& hnsw_router_file) {
        router = std::make_unique<HNSWRouter>(hnsw_router_file, dim, partition);
    }

    std::vector<int> Route(float* Q) {
        std::vector<int> probes = router->Query(Q, num_voting_neighbors).RoutingQuery();
        probes.resize(std::min<int>(probes.size(), num_probes));
        return probes;
    }

    void ProcessQueries3(const std::vector<int>& query_ids, PointSet& queries) {
        message_queue::FlushStrategy flush_strategy = message_queue::FlushStrategy::global;

        struct Request {
            int query_id = -1;
            std::vector<float> coordinates;
        };
        auto merge_requests = [](auto& buf, message_queue::PEID buffer_destination, message_queue::PEID my_rank,
                    message_queue::Envelope auto msg) {
            buf.push_back(static_cast<float>(msg.message.query_id));
            for (float x : msg.message.coordinates) {
                buf.push_back(x);
            }
        };

        auto split_requests = [](message_queue::MPIBuffer<float> auto const& buf, message_queue::PEID buffer_origin,
                    message_queue::PEID my_rank) {
            std::vector<Request> incoming_requests;
            Request r;

            return incoming_requests;
        };

        auto requests_queue =
                message_queue::make_buffered_queue<Request, float>(MPI_COMM_WORLD, merge_requests, split_requests);


        struct Response {
            int query_id;
            std::vector<int> neighbors;
        };
        auto responses_queue =
                message_queue::make_buffered_queue<Response, int>();

        std::vector<int> local_requests;
        std::vector<std::vector<float>> request_buffers(comm_size);

        // we want to
        // a) process queries in parallel
        // b) dont lock message buffers to avoid overheads
        // c) get the first requests out as soon as possible so that a machine that runs out of routing
        //    work can get started on retrieving neighbors right away
        // The following performs multiple steps with 2^^i queries routed in step i, followed by sending their messages

        size_t step_size = 128;
        auto qq = parlay::make_slice(query_ids);
        for (size_t i = 0; i < query_ids.size(); i += step_size, step_size *= 2) {
            const auto queries_this_step = qq.cut(i, std::min(query_ids.size(), i + step_size));
            auto probes_nested = parlay::map(queries_this_step, [&](int query_id) { return Route(queries.GetPoint(query_id)); });
            //auto probes = parlay::flatten(probes_nested); // no point in doing the zip and flatten in parallel. we have to post messages sequentially anyway
            // well actually, merging the embedding streams in parallel could be nice
            for (size_t j = 0; j < queries_this_step.size(); ++j) {
                int query_id = queries_this_step[j];
                float* Q = queries.GetPoint(query_id);
                for (int shard_id : probes_nested[j]) {
                    if (shard_id == rank) {
                        local_requests.push_back(query_id);
                    } else {
                        auto& buf = request_buffers[shard_id];
                        // write the query ID to the stream as a float... oh well
                        buf.push_back(static_cast<float>(query_id));
                        // write the query vector
                        for (int k = 0; k < dim; ++k) {
                            buf.push_back(Q[k]);
                        }
                    }
                }
            }

            for (int j = 0; j < comm_size; ++j) {
                if (j != rank) {
                    requests_queue.post_message(std::move(request_buffers[j]), j);
                    // TODO flush the buffer queue
                    request_buffers[j].clear();
                }
            }
        }

        auto return_neighbors = [&](message_queue::Envelope<Request> auto request_envelope) {
            for (const Request& r : request_envelope.message) {
                int original_sender = request_envelope.sender;
                auto result = hnsw->searchKnn(queries.GetPoint(r.query_id), num_neighbors);
                Response response;
                response.query_id = r.query_id;
                while (!result.empty()) {
                    auto next = result.top();
                    response.neighbors.push_back(next.second);
                    result.pop();
                }
                responses_queue.post_message(std::move(response), original_sender);
            }
        };

        auto accept_returned_neighbors = [&](message_queue::Envelope<Response> auto response_envelope) {
            /* Do nothing here */
            // Dedup?
        };

        requests_queue.terminate(return_neighbors);
        responses_queue.terminate(accept_returned_neighbors);

        for (int query_id : local_requests) {
            auto result = hnsw->searchKnn(queries.GetPoint(r.query_id), num_neighbors);
        }
    }

};
