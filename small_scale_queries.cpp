#include <iostream>
#include <filesystem>

#include <parlay/primitives.h>

#include "points_io.h"
#include "metis_io.h"
#include "recall.h"
#include "kmeans_tree_router.h"
#include "hnsw_router.h"
#include "inverted_index.h"
#include "inverted_index_hnsw.h"

void L2Normalize(PointSet& points) {
    parlay::parallel_for(points.n, [&](size_t i) {
        L2Normalize(points.GetPoint(i), points.d);
    });
}

int main(int argc, const char* argv[]) {
    if (argc != 7) {
        std::cerr << "Usage ./RunQueries input-points queries ground-truth-file k partition normalize" << std::endl;
        std::abort();
    }

    std::string point_file = argv[1];
    std::string query_file = argv[2];
    std::string ground_truth_file = argv[3];
    std::string k_string = argv[4];
    int k = std::stoi(k_string);
    std::string partition_file = argv[5];
    PointSet points = ReadPoints(point_file);
    PointSet queries = ReadPoints(query_file);

    std::string str_normalize = argv[6];
    if (str_normalize == "True") {
        L2Normalize(points);
        L2Normalize(queries);
    }

    std::vector<NNVec> ground_truth;
    if (std::filesystem::exists(ground_truth_file)) {
        ground_truth = ReadGroundTruth(ground_truth_file);
    } else {
        std::cout << "start computing ground truth" << std::endl;
        ground_truth = ComputeGroundTruth(points, queries, k);
        std::cout << "computed ground truth" << std::endl;
    }

    Clusters clusters = ReadClusters(partition_file);
    int num_shards = clusters.size();

    std::vector<std::vector<int>> buckets_to_probe_by_query(queries.n);
    std::vector<NNVec> neighbors_by_query(queries.n);

    Timer timer;
    timer.Start();
    KMeansTreeRouterOptions options { .num_centroids = 32, .min_cluster_size = 200, .budget = 50000, .search_budget = 5000 };
    KMeansTreeRouter router;
    router.Train(points, clusters, options);
    std::cout << "Training KMTR took " << timer.Stop() << " seconds." << std::endl;

    auto [routing_points, routing_index_partition] = router.ExtractPoints();

    timer.Start();
    HNSWRouter hnsw_router(routing_points, num_shards, routing_index_partition, HNSWParameters{ .M = 32, .ef_construction = 200, .ef_search = 200 });
    hnsw_router.Train(routing_points);
    std::cout << "Training HNSW router took " << timer.Stop() << " seconds." << std::endl;

    timer.Start();
    auto buckets_to_probe_kmtr = parlay::tabulate(queries.n, [&](size_t q) {
        return router.Query(queries.GetPoint(q), options.search_budget);
    });
    double time = timer.Stop();
    std::cout << "KMTR routing took " << time << " seconds. That's " << 1000.0 * time / queries.n
              << "ms per query, or " << queries.n / time << " QPS" << std::endl;

    timer.Start();
    auto buckets_to_probe_hnsw = parlay::tabulate(queries.n, [&](size_t q) {
        return hnsw_router.Query(queries.GetPoint(q), 120).RoutingQuery();
    });
    time = timer.Stop();
    std::cout << "HSNW routing took " << time << " seconds. That's " << 1000.0 * time / queries.n
              << "ms per query, or " << queries.n / time << " QPS" << std::endl;


    InvertedIndexHNSW inverted_index(points, std::ranges::partition);

    std::cout << "Finished building inverted index" << std::endl;

    for (int num_probes = 1; num_probes <= num_shards; ++num_probes) {

        auto t3 = std::chrono::high_resolution_clock::now();
        parlay::parallel_for(0, queries.n, [&](size_t i) {
        // for (size_t i = 0; i < queries.n; ++i) {
            float* Q = queries.GetPoint(i);
            neighbors_by_query[i] = inverted_index.Query(Q, k, buckets_to_probe_by_query[i], num_probes);
        }
        );
        auto t4 = std::chrono::high_resolution_clock::now();
        std::cout << "finished query. now compute recall" << std::endl;
        double recall = Recall(neighbors_by_query, distance_to_kth_neighbor, k);
        double time_probes = (t4-t3).count() / 1e6;
        std::cout << "Probing " << num_probes << " took " << time_probes << "ms overall, and " << time_probes / queries.n << " per query. Recall achieved " << recall << std::endl;

        recall_per_num_probes[num_probes-1] = recall;
        time_per_num_probes[num_probes-1] = time_probes / queries.n;
    }

    // TODO parsable output

}
