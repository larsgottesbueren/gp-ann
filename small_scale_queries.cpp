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

void DedupNeighbors(std::vector<NNVec> &neighbors, size_t num_neighbors)
{
    parlay::parallel_for(0, neighbors.size(), [&](size_t i)
                         {
       auto& n = neighbors[i];
        std::sort(n.begin(), n.end(), [](const auto& l, const auto& r) { return l.second < r.second; });
        n.erase(std::unique(n.begin(), n.end()), n.end());
        std::sort(n.begin(), n.end());
        n.resize(std::min(n.size(), num_neighbors)); });
}

int main(int argc, const char *argv[])
{
    if (argc != 8)
    {
        std::cerr << "Usage ./SmallScaleQueries input-points queries ground-truth-file num-neighbors partition part-method out-file" << std::endl;
        std::abort();
    }

    std::string point_file = argv[1];
    std::string query_file = argv[2];
    std::string ground_truth_file = argv[3];
    std::string k_string = argv[4];
    int num_neighbors = std::stoi(k_string);
    std::string partition_file = argv[5];
    std::string part_method = argv[6];
    std::string out_file = argv[7];

    PointSet points = ReadPoints(point_file);
    PointSet queries = ReadPoints(query_file);

    std::vector<NNVec> ground_truth;
    if (std::filesystem::exists(ground_truth_file))
    {
        ground_truth = ReadGroundTruth(ground_truth_file);
    }
    else
    {
        std::cout << "start computing ground truth" << std::endl;
        ground_truth = ComputeGroundTruth(points, queries, num_neighbors);
        std::cout << "computed ground truth" << std::endl;
        WriteGroundTruth(ground_truth_file, ground_truth);
        std::cout << "wrote ground truth to file " << ground_truth_file << std::endl;
    }
    std::vector<float> distance_to_kth_neighbor = ConvertGroundTruthToDistanceToKthNeighbor(ground_truth, num_neighbors, points, queries);
    std::cout << "finished converting ground truth to distances" << std::endl;

    if (!std::filesystem::exists(partition_file) || part_method == "None")
    {
        std::cout << "Not partitioned. --> Run HNSW directly on input" << std::endl;
        Timer timer;
        timer.Start();
        HNSWParameters hnsw_parameters;
#ifdef MIPS_DISTANCE
        hnswlib::InnerProductSpace space(points.d);
#else
        hnswlib::L2Space space(points.d);
#endif
        hnswlib::HierarchicalNSW<float> hnsw(&space, points.n, hnsw_parameters.M, hnsw_parameters.ef_construction, 555);
        parlay::parallel_for(0, points.n, [&](size_t i)
                             { hnsw.addPoint(points.GetPoint(i), i); });
        std::cout << "Building HNSW took " << timer.Stop() << " seconds." << std::endl;

        for (int ef : {20, 50, 80, 100, 120, 150, 200, 300, 400})
        {
            std::vector<NNVec> neighbors(queries.n);
            hnsw.setEf(ef);
            timer.Start();
            for (size_t q = 0; q < queries.n; ++q)
            {
                auto result_pq = hnsw.searchKnn(queries.GetPoint(q), num_neighbors);
                NNVec result;
                while (!result_pq.empty())
                {
                    result.emplace_back(result_pq.top());
                    result_pq.pop();
                }
                neighbors[q] = std::move(result);
            }
            double time = timer.Stop();
            double recall = Recall(neighbors, distance_to_kth_neighbor, num_neighbors);
            std::cout << "HNSW query with ef = " << ef << " took " << time << " seconds. recall = " << recall
                      << ". avg latency = " << 1000.0 * time / queries.n << " ms."
                      << " avg dist comps " << static_cast<double>(hnsw.metric_distance_computations) / queries.n
                      << std::endl;
        }

        return 0;
    }

    Clusters clusters = ReadClusters(partition_file);
    int num_shards = clusters.size();

    Timer timer;
    timer.Start();
    KMeansTreeRouterOptions options{.num_centroids = 32, .min_cluster_size = 200, .budget = 50000, .search_budget = 5000};
    KMeansTreeRouter router;
    router.Train(points, clusters, options);
    std::cout << "Training KMTR took " << timer.Stop() << " seconds." << std::endl;

    auto [routing_points, routing_index_partition] = router.ExtractPoints();

    timer.Start();
    HNSWRouter hnsw_router(routing_points, num_shards, routing_index_partition, HNSWParameters{.M = 16, .ef_construction = 200, .ef_search = 200});
    hnsw_router.Train(routing_points);
    std::cout << "Training HNSW router took " << timer.Stop() << " seconds." << std::endl;

    std::vector<std::tuple<std::string /*router*/, double /*routing time*/, std::vector<std::vector<int>> /*probes*/>> probes_v;

    std::vector<std::vector<int>> buckets_to_probe_kmtr(queries.n), buckets_to_probe_hnsw(queries.n);
    timer.Start();
    for (size_t q = 0; q < queries.n; ++q)
    {
        buckets_to_probe_kmtr[q] = router.Query(queries.GetPoint(q), options.search_budget);
    }
    double time = timer.Stop();
    std::cout << "KMTR routing took " << time << " seconds. That's " << 1000.0 * time / queries.n
              << "ms per query, or " << queries.n / time << " QPS" << std::endl;
    probes_v.emplace_back(std::tuple("KMTR", time, std::move(buckets_to_probe_kmtr)));

    timer.Start();
    for (size_t q = 0; q < queries.n; ++q)
    {
        buckets_to_probe_hnsw[q] = hnsw_router.Query(queries.GetPoint(q), 60).RoutingQuery();
    }
    time = timer.Stop();
    std::cout << "HSNW routing took " << time << " seconds. That's " << 1000.0 * time / queries.n
              << "ms per query, or " << queries.n / time << " QPS" << std::endl;
    probes_v.emplace_back(std::tuple("HNSW", time, std::move(buckets_to_probe_hnsw)));

    timer.Start();
    InvertedIndexHNSW ivf_hnsw(points);
    ivf_hnsw.hnsw_parameters = HNSWParameters{
        .M = 16,
        .ef_construction = 200,
        .ef_search = 120};
    ivf_hnsw.Build(points, clusters);
    std::cout << "Building IVF-HNSW took " << timer.Restart() << " seconds." << std::endl;
    InvertedIndex ivf(points, clusters);
    std::cout << "Building IVF took " << timer.Stop() << " seconds." << std::endl;

    std::cout << "Finished building IVFs" << std::endl;

    std::ofstream out(out_file);
    out << "partitioning,routing,shard query,probes,latency,routing latency, query latency,recall" << std::endl;

    std::cout << "Start queries" << std::endl;

    for (const auto &[desc, routing_time, probes] : probes_v)
    {
        std::vector<NNVec> neighbors(queries.n);
        time = 0;
        for (int num_probes = 1; num_probes <= num_shards; ++num_probes)
        {
            timer.Start();
            for (size_t q = 0; q < queries.n; ++q)
            {
                auto neighs = ivf.QueryBucket(queries.GetPoint(q), num_neighbors, probes[q][num_probes - 1]);
                neighbors[q].insert(neighbors[q].end(), neighs.begin(), neighs.end());
            }
            time += timer.Stop();
            DedupNeighbors(neighbors, num_neighbors);
            double recall = Recall(neighbors, distance_to_kth_neighbor, num_neighbors);
            double latency = (routing_time + time) / queries.n;
            std::cout << "router = " << desc << " query = IVF "
                      << "nprobes = " << num_probes << " recall = " << recall << " time = " << time
                      << " avg latency = " << 1000.0 * latency << " ms" << std::endl;
            out << part_method << "," << desc << ","
                << "BruteForce"
                << "," << num_probes << "," << latency << "," << routing_time / queries.n << ","
                << time / queries.n << "," << recall << std::endl;
        }

        for (auto &neighs : neighbors)
        {
            neighs.clear();
        }
        time = 0;
        for (int num_probes = 1; num_probes <= num_shards; ++num_probes)
        {
            timer.Start();
            for (size_t q = 0; q < queries.n; ++q)
            {
                auto neighs = ivf_hnsw.QueryBucket(queries.GetPoint(q), num_neighbors, probes[q][num_probes - 1]);
                neighbors[q].insert(neighbors[q].end(), neighs.begin(), neighs.end());
            }
            time += timer.Stop();
            DedupNeighbors(neighbors, num_neighbors);
            double recall = Recall(neighbors, distance_to_kth_neighbor, num_neighbors);
            double latency = (routing_time + time) / queries.n;
            std::cout << "router = " << desc << " query = IVF-HNSW "
                      << "nprobes = " << num_probes << " recall = " << recall << " time = " << time
                      << " avg latency = " << 1000.0 * latency << " ms" << std::endl;
            out << part_method << "," << desc << ","
                << "HNSW"
                << "," << num_probes << "," << latency << "," << routing_time / queries.n << ","
                << time / queries.n << "," << recall << std::endl;
        }
    }
}
