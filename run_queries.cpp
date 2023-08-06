#include <iostream>

#include "knn_graph.h"
#include "points_io.h"
#include "metis_io.h"
#include "recall.h"
#include "kmeans_tree.h"
#include "inverted_index.h"

int main(int argc, const char* argv[]) {


    if (argc != 5) {
        std::cerr << "Usage ./RunQueries input-points queries k partition" << std::endl;
        std::abort();
    }

    std::string point_file = argv[1];
    std::string query_file = argv[2];
    std::string k_string = argv[3];
    int k = std::stoi(k_string);
    std::string partition_file = argv[4];
    PointSet points = ReadPoints(point_file);
    PointSet queries = ReadPoints(query_file);
    std::vector<int> partition = ReadMetisPartition(partition_file);

    std::vector<float> distance_to_kth_neighbor = ComputeDistanceToKthNeighbor(points, queries, k);

    InvertedIndex inverted_index(points, partition, k);
    KMeansTreeRouter router;

    std::vector<std::vector<int>> buckets_to_probe_by_query(queries.n);
    std::vector<NNVec> neighbors_by_query(queries.n);

    auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < queries.n; ++i) {
        buckets_to_probe_by_query[i] = router.Query(queries.GetPoint(i));
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    int num_shards = *std::max_element(partition.begin(), partition.end()) + 1;
    std::vector<double> time_per_num_probes(num_shards, 0.0);
    std::vector<double> recall_per_num_probes(num_shards, 0.0);

    for (size_t num_probes = 1; num_probes <= num_shards; ++num_probes) {

        auto t3 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < queries.n; ++i) {
            float* Q = queries.GetPoint(i);
            neighbors_by_query[i] = inverted_index.Query(Q, buckets_to_probe_by_query[i], num_probes);
        }
        auto t4 = std::chrono::high_resolution_clock::now();

        double recall = Recall(neighbors_by_query, distance_to_kth_neighbor, k);
        double time = (t4-t3).count() / 1e6;
        std::cout << "Probing " << num_shards << " took " << time << "ms overall, and " << time / queries.n << " per query. Recall achieved " << recall << std::endl;

        recall_per_num_probes[num_probes-1] = recall;
        time_per_num_probes[num_probes-1] = time / queries.n;
    }

    // TODO parsable output


}
