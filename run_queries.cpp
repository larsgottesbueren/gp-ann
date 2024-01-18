#include <iostream>
#include <filesystem>

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
    // TODO parse parameters
    if (argc != 6 && argc != 10) {
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

    KMeansTreeRouterOptions options;
    if (argc != 6) {
        options.num_centroids = std::stoi(argv[6]);
        options.min_cluster_size = std::stoi(argv[7]);
        options.budget = std::stoi(argv[8]);
        options.search_budget = std::stoi(argv[9]);
    }

    std::string str_normalize = argv[8];
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

    KMeansTreeRouter router;
    router.Train(points, clusters, options);

    Timer timer;
    
    // TODO
    // run routing. KMTR router, HNSW router 
    
    // run in-shard searches brute-force (InvertedIndex) --> recall curve per shards probed

    parlay::parallel_for(0, queries.n, [&](size_t i) {
    //for (size_t i = 0; i < queries.n; ++i) {
        buckets_to_probe_by_query[i] = router.Query(queries.GetPoint(i), options.search_budget);
    }
    );
    auto t2 = std::chrono::high_resolution_clock::now();
    double time_routing = (t2-t1).count() / 1e6;
    std::cout << "Routing took " << time_routing << " ms overall, and " << time_routing / queries.n << " per query" << std::endl;


    std::vector<double> time_per_num_probes(num_shards, 0.0);
    std::vector<double> recall_per_num_probes(num_shards, 0.0);

    InvertedIndexHNSW inverted_index(points, partition);

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
