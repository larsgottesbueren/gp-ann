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

int main(int argc, const char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage ./RunQueries input-points queries ground-truth-file k partition" << std::endl;
        std::abort();
    }

    std::string point_file = argv[1];
    std::string query_file = argv[2];
    std::string ground_truth_file = argv[3];
    std::string k_string = argv[4];
    int num_neighbors = std::stoi(k_string);
    std::string partition_file = argv[5];
    PointSet points = ReadPoints(point_file);
    PointSet queries = ReadPoints(query_file);
    Clusters clusters = ReadClusters(partition_file);

    std::vector<NNVec> ground_truth;
    if (std::filesystem::exists(ground_truth_file)) { ground_truth = ReadGroundTruth(ground_truth_file); } else {
        std::cout << "start computing ground truth" << std::endl;
        ground_truth = ComputeGroundTruth(points, queries, num_neighbors);
        std::cout << "computed ground truth" << std::endl;
    }

    int num_shards = clusters.size();

    Timer timer;
    timer.Start();
    KMeansTreeRouterOptions options{ .num_centroids = 32, .min_cluster_size = 200, .budget = 50000, .search_budget = 5000 };
    KMeansTreeRouter router;
    router.Train(points, clusters, options);
    std::cout << "Training KMTR took " << timer.Stop() << " seconds." << std::endl;

    auto [routing_points, routing_index_partition] = router.ExtractPoints();

    timer.Start();
    HNSWRouter hnsw_router(routing_points, num_shards, routing_index_partition, HNSWParameters{ .M = 32, .ef_construction = 200, .ef_search = 200 });
    hnsw_router.Train(routing_points);
    std::cout << "Training HNSW router took " << timer.Stop() << " seconds." << std::endl;

    timer.Start();
    auto buckets_to_probe_kmtr = parlay::tabulate(queries.n, [&](size_t q) { return router.Query(queries.GetPoint(q), options.search_budget); }, queries.n);
    double time = timer.Stop();
    std::cout << "KMTR routing took " << time << " seconds. That's " << 1000.0 * time / queries.n
            << "ms per query, or " << queries.n / time << " QPS" << std::endl;

    timer.Start();
    auto buckets_to_probe_hnsw = parlay::tabulate(queries.n, [&](size_t q) { return hnsw_router.Query(queries.GetPoint(q), 120).RoutingQuery(); }, queries.n);
    time = timer.Stop();
    std::cout << "HSNW routing took " << time << " seconds. That's " << 1000.0 * time / queries.n
            << "ms per query, or " << queries.n / time << " QPS" << std::endl;

    std::vector<std::pair<std::string, decltype(buckets_to_probe_kmtr)>> probes_v;
    probes_v.emplace_back(std::pair("KMTR", std::move(buckets_to_probe_kmtr)));
    probes_v.emplace_back(std::pair("HNSW", std::move(buckets_to_probe_hnsw)));


    // NOTE with IVF-HNSW and IVF deduplicating the resulting neighbors is not supported yet.
    // This is because the same top-K data structure is shared across multiple clusters.
    // We should build separate top-k data structures, remap and then mege
    timer.Start();
    InvertedIndexHNSW ivf_hnsw(points, clusters);
    std::cout << "Building IVF-HNSW took " << timer.Restart() << " seconds." << std::endl;
    InvertedIndex ivf(points, clusters);
    std::cout << "Building IVF took " << timer.Stop() << " seconds." << std::endl;

    std::cout << "Finished building IVFs" << std::endl;

    std::vector<float> distance_to_kth_neighbor = ConvertGroundTruthToDistanceToKthNeighbor(ground_truth, num_neighbors, points, queries);

    std::cout << "finished converting ground truth to distances" << std::endl;

    std::cout << "Start queries" << std::endl;

    for (const auto& [desc, probes] : probes_v) {
        for (int num_probes = 1; num_probes <= num_shards; ++num_probes) {
            timer.Start();
            std::vector<NNVec> neighbors(queries.n);
            parlay::parallel_for(0, queries.n, [&](size_t q) { neighbors[q] = ivf.Query(queries.GetPoint(q), num_neighbors, probes[q], num_probes); },
            queries.n);
            time = timer.Stop();
            double recall = Recall(neighbors, distance_to_kth_neighbor, num_neighbors);
            std::cout << "router = " << desc << " query = IVF " << "nprobes = " << num_probes << " recall = " << recall << " time = " << time << std::endl;

            timer.Start();
            parlay::parallel_for(0, queries.n, [&](size_t q) { neighbors[q] = ivf_hnsw.Query(queries.GetPoint(q), num_neighbors, probes[q], num_probes); },
            queries.n);
            time = timer.Stop();
            recall = Recall(neighbors, distance_to_kth_neighbor, num_neighbors);
            std::cout << "router = " << desc << " query = IVF-HNSW " << "nprobes = " << num_probes << " recall = " << recall << " time = " << time << std::endl;
        }
    }
}
