#include <iostream>

#include "knn_graph.h"
#include "points_io.h"
#include "metis_io.h"
#include "recall.h"
#include "kmeans_tree.h"
#include "inverted_index.h"

void Normalize(PointSet& points) {
    for (size_t i = 0; i < points.n; ++i) {
        float* p = points.GetPoint(i);
        efanna2e::DistanceFastL2 dst;
        float norm = dst.norm(p, points.d);
        for (size_t j = 0; j < points.d; ++j) {
            p[j] /= norm;
        }
    }
}

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

    if (false) {
        Normalize(points);
        Normalize(queries);
    }

    std::vector<int> partition = ReadMetisPartition(partition_file);

    std::vector<float> distance_to_kth_neighbor = ComputeDistanceToKthNeighbor(points, queries, k);

    InvertedIndex inverted_index(points, partition, k);
    KMeansTreeRouter router;

    std::vector<std::vector<int>> buckets_to_probe_by_query(queries.n);
    std::vector<NNVec> neighbors_by_query(queries.n);

    auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < queries.n; ++i) {
        buckets_to_probe_by_query[i] = router.Query(queries.GetPoint(i), k);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double time_routing = (t2-t1).count() / 1e6;
    std::cout << "Routing took " << time_routing << " ms overall, and " << time_routing / queries.n << " per query";

    int num_shards = *std::max_element(partition.begin(), partition.end()) + 1;
    std::vector<double> time_per_num_probes(num_shards, 0.0);
    std::vector<double> recall_per_num_probes(num_shards, 0.0);

    for (int num_probes = 1; num_probes <= num_shards; ++num_probes) {

        auto t3 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < queries.n; ++i) {
            float* Q = queries.GetPoint(i);
            neighbors_by_query[i] = inverted_index.Query(Q, buckets_to_probe_by_query[i], num_probes);
        }
        auto t4 = std::chrono::high_resolution_clock::now();

        double recall = Recall(neighbors_by_query, distance_to_kth_neighbor, k);
        double time_probes = (t4-t3).count() / 1e6;
        std::cout << "Probing " << num_shards << " took " << time_probes << "ms overall, and " << time_probes / queries.n << " per query. Recall achieved " << recall << std::endl;

        recall_per_num_probes[num_probes-1] = recall;
        time_per_num_probes[num_probes-1] = time_probes / queries.n;
    }

    // TODO parsable output


}
