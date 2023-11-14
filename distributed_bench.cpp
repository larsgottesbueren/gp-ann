#include <iostream>
#include <filesystem>
#include <map>

#include "points_io.h"
#include "metis_io.h"
#include "recall.h"
#include "route_search_combination.h"

#include <mpi.h>
#include "distributed_query_benchmark.h"

int main(int argc, const char* argv[]) {
    if (argc != 9) {
        std::cerr << "Usage ./QueryAttribution input-points queries ground-truth-file num_neighbors partition-file output-file partition_method requested-num-shards" << std::endl;
        std::abort();
    }

    std::string point_file = argv[1];
    std::string query_file = argv[2];
    std::string ground_truth_file = argv[3];
    std::string k_string = argv[4];
    std::string partition_file = argv[5];
    std::string output_file = argv[6];
    std::string part_method = argv[7];
    std::string router_file = argv[8];

    MPI_Init(nullptr, nullptr);

    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    DistributedQueryBenchmark bench;
    bench.LoadPartition(partition_file);
    bench.LoadShardPointSet(point_file);
    bench.BuildInShardIndex();
    bench.LoadRouter(router_file);

    PointSet queries = ReadPoints(query_file);
    std::vector<int> query_ids(queries.n);
    std::iota(query_ids.begin(), query_ids.end(), 0);
    #if false
    int seed = 555;
    std::mt19937 prng(seed);
    std::shuffle(query_ids.begin(), query_ids.end());
    #endif
    size_t chunk_size = ComputeChunkSize(query_ids.size(), comm_size);
    std::vector<int> my_query_ids(query_ids.begin() + rank * chunk_size, query_ids.begin() + std::min(query_ids.size(), (rank + 1) * chunk_size));

    MPI_Barrier(MPI_COMM_WORLD);

    double t1, t2;
    t1 = MPI_Wtime();
    bench.ProcessQueries(my_query_ids, queries);
    t2 = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);    // TODO is this necessary? the subsequent MPI_Reduce should enforce a sync
    double elapsed = t2 - t1;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "End-to-end time " << max_elapsed << std::endl;
    }

    MPI_Finalize();
    return 0;
}
