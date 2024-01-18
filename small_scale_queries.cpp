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


    // NOTE with IVF-HNSW and IVF deduplicating the resulting neighbors is not supported yet.
    // This is because the same top-K data structure is shared across multiple clusters.
    timer.Start();
    InvertedIndexHNSW ivf_hnsw(points, clusters);
    std::cout << "Building IVF-HNSW took " << timer.Restart() << " seconds." << std::endl;
    InvertedIndex ivf(points, clusters);
    std::cout << "Building IVF took " << timer.Stop() << " seconds." << std::endl;

    std::cout << "Finished building IVFs" << std::endl;

    for (int num_probes = 1; num_probes <= num_shards; ++num_probes) {
        parlay::parallel_for(0, queries.n, [&](size_t i) {
            float* Q = queries.GetPoint(i);
            neighbors_by_query[i] = ivf.Query(Q, k, buckets_to_probe_by_query[i], num_probes);
        });

        parlay::parallel_for(0, queries.n, [&](size_t i) {
            float* Q = queries.GetPoint(i);
            neighbors_by_query[i] = ivf_hnsw.Query(Q, k, buckets_to_probe_by_query[i], num_probes);
        });

    }
}
