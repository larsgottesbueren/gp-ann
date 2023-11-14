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

    void ProcessQueries(const std::vector<int>& query_ids, PointSet& queries) {

    }

    void MessagePassingExample() {
        message_queue::FlushStrategy flush_strategy = message_queue::FlushStrategy::global;

        auto queue =
                message_queue::make_buffered_queue<int>(MPI_COMM_WORLD, message_queue::aggregation::AppendMerger{},
                                                        message_queue::aggregation::NoSplitter{},
                                                        message_queue::aggregation::NoOpCleaner{});
        std::mt19937 gen;
        std::uniform_int_distribution<int> dist(0, comm_size - 1);
        std::uniform_int_distribution<int> message_size_dist(1, 10);
        queue.global_threshold(10);
        queue.local_threshold(2);
        queue.flush_strategy(flush_strategy);
        for (auto i = 0; i < 50; ++i) {
            int destination = dist(gen);
            int message_size = message_size_dist(gen);
            auto message = std::vector<int>(message_size, 1);
            queue.post_message(std::move(message), destination, rank);
        }

        size_t zero_message_counter = 0;
        auto handler = [&](message_queue::Envelope<int> auto envelope) {
            message_queue::atomic_debug("Message...");
        };
        queue.terminate(handler);
    }
};
