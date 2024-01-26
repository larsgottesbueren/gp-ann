#include <iostream>
#include <filesystem>
#include <recall.h>
#include <parlay/primitives.h>

#include "routes.h"
#include "points_io.h"
#include "metis_io.h"
#include "dist.h"


std::vector<double> RecallForIncreasingProbes(
    const std::vector<std::vector<int>>& buckets_to_probe, const Partition& partition, const std::vector<NNVec>& ground_truth, int num_neighbors, size_t
    num_shards) {
    size_t num_queries = ground_truth.size();
    std::vector<std::unordered_set<uint32_t>> neighbors(num_queries);
    std::vector<double> recall_values;
    size_t hits = 0;
    for (size_t probes = 0; probes < num_shards; ++probes) {
        hits += parlay::reduce(
            parlay::tabulate(num_queries, [&](size_t q) {
                int cluster = buckets_to_probe[q][probes];
                size_t my_new_hits = 0;
                for (int j = 0; j < num_neighbors; ++j) {
                    uint32_t neighbor = ground_truth[q][j].second;
                    // if we haven't seen the neighbor before
                    // and it's in the cluster we are looking at right now
                    if (!neighbors[q].contains(neighbor) && partition[neighbor] == cluster) {
                        neighbors[q].insert(neighbor);
                        my_new_hits++;
                    }
                }
                return my_new_hits;
            })
        );
        double recall = static_cast<double>(hits) / num_neighbors / num_queries;
        recall_values.push_back(recall);
    }
    return recall_values;
}

std::vector<std::vector<int>> BruteForceRouting(PointSet& queries, PointSet& points, const Partition& partition, size_t num_shards) {
    std::vector<std::vector<int>> probes(queries.n, std::vector<int>(num_shards));
    parlay::parallel_for(0, queries.n, [&](size_t q) {
        std::vector<float> min_dist(num_shards, std::numeric_limits<float>::max());
        for (size_t i = 0; i < points.n; ++i) {
            if (float dist = distance(points.GetPoint(i), queries.GetPoint(q), points.d); dist < min_dist[partition[i]]) {
                min_dist[partition[i]] = dist;
            }
        }
        auto& p = probes[q];
        std::iota(p.begin(), p.end(), 0);
        std::sort(p.begin(), p.end(), [&](int l, int r) { return min_dist[l] < min_dist[r]; });
    }, 1);
    return probes;
}

std::vector<std::vector<int>> FullDatasetRouting(
    const std::vector<NNVec>& ground_truth, PointSet& queries, PointSet& points, const Partition& partition, size_t num_shards) {
    std::vector<std::vector<float>> min_dist(queries.n, std::vector<float>(num_shards, std::numeric_limits<float>::max()));
    std::vector<uint32_t> non_covered_queries;
    for (size_t q = 0; q < queries.n; ++q) {
        for (const auto& [dist, neigh] : ground_truth[q]) { if (dist < min_dist[q][partition[neigh]]) { min_dist[q][partition[neigh]] = dist; } }
        if (std::any_of(min_dist[q].begin(), min_dist[q].end(), [](float x) { return x == std::numeric_limits<float>::max(); })) {
            non_covered_queries.push_back(q);
        }
    }

    std::cout << non_covered_queries.size() << " queries have unconvered shards from the ground-truth set" << std::endl;

#if false
    parlay::parallel_for(0, non_covered_queries.size(), [&](size_t j) {
        uint32_t q = non_covered_queries[j];
        for (size_t i = 0; i < points.n; ++i) {
            if (float dist = distance(points.GetPoint(i), queries.GetPoint(q), points.d); dist < min_dist[q][partition[i]]) {
                min_dist[q][partition[i]] = dist;
            }
        }
    });
#endif

    std::vector<std::vector<int>> probes(queries.n, std::vector<int>(num_shards));
    for (size_t q = 0; q < queries.n; ++q) {
        auto& p = probes[q];
        std::iota(p.begin(), p.end(), 0);
        std::sort(p.begin(), p.end(), [&](int l, int r) { return min_dist[q][l] < min_dist[q][r]; });
    }

    return probes;
}

std::vector<std::vector<int>> RouteUsingSingleCenter(PointSet& points, PointSet& queries, const Clusters& clusters) {
    PointSet centers;
    centers.d = points.d;
    centers.n = clusters.size();
    centers.Alloc();
    parlay::parallel_for(0, clusters.size(), [&](size_t c) {
        // avoid false sharing
        PointSet CC;
        CC.d = points.d;
        CC.n = 1;
        CC.Alloc();
        float* C = CC.GetPoint(0);

        double norm_sum = 0.0;
        for (uint32_t v : clusters[c]) {
            float* V = points.GetPoint(v);
#ifdef MIPS_DISTANCE
            double norm = vec_norm(V, centers.d);
            norm_sum += norm;
            float multiplier = 1.0f / std::sqrt(norm);
            for (size_t j = 0; j < centers.d; ++j) {
                C[j] += V[j] * multiplier;
            }
#else
            for (size_t j = 0; j < centers.d; ++j) { C[j] += V[j]; }
#endif
        }
#ifdef MIPS_DISTANCE
        float desired_norm = norm_sum / clusters[c].size();
        float current_norm = vec_norm(C, centers.d);
        float multiplier = std::sqrt(desired_norm / current_norm);
        for (size_t j = 0; j < centers.d; ++j) { C[j] *= multiplier; }
#else
        for (size_t j = 0; j < centers.d; ++j) { C[j] /= clusters[c].size(); }

        // copy over
        float* C2 = centers.GetPoint(c);
        for (size_t j = 0; j < centers.d; ++j) { C2[j] = C[j]; }
#endif
    }, 1);

    std::vector<std::vector<int>> probes(queries.n, std::vector<int>(clusters.size()));
    parlay::parallel_for(0, queries.n, [&](size_t q) {
        std::vector<float> min_dist;
        for (size_t c = 0; c < clusters.size(); ++c) { min_dist.push_back(distance(queries.GetPoint(q), centers.GetPoint(c), queries.d)); }
        auto& p = probes[q];
        std::iota(p.begin(), p.end(), 0);
        std::sort(p.begin(), p.end(), [&](int l, int r) { return min_dist[l] < min_dist[r]; });
    });
    return probes;
}

int main(int argc, const char* argv[]) {
    if (argc != 8) {
        std::cerr << "Usage ./AnalyzeApproximationLosses point-file query-file ground-truth-file num_neighbors partition-file part-method out-file" <<
                std::endl;
        std::abort();
    }

    std::string point_file = argv[1];
    std::string query_file = argv[2];
    std::string ground_truth_file = argv[3];
    std::string k_string = argv[4];
    std::string partition_file = argv[5];
    std::string part_method = argv[6];
    std::string out_file = argv[7];

    int num_neighbors = std::stoi(k_string);

#if false
    auto clusters = ReadClusters(partition_file);
    Cover cover = ConvertClustersToCover(clusters);
    size_t num_shards = clusters.size();
#else
    auto partition = ReadMetisPartition(partition_file);
    auto clusters = ConvertPartitionToClusters(partition);
    size_t num_shards = clusters.size();
#endif
    std::cout << "Finished reading partition file" << std::endl;

    std::vector<NNVec> ground_truth;
    if (std::filesystem::exists(ground_truth_file)) {
        ground_truth = ReadGroundTruth(ground_truth_file);
        std::cout << "Read ground truth file" << std::endl;
    } else { throw std::runtime_error("ground truth file doesnt exist"); }

    PointSet points = ReadPoints(point_file);
    PointSet queries = ReadPoints(query_file);
    ConvertGroundTruthToDistanceToKthNeighbor(ground_truth, 10, points, queries);


    auto single_center_probes = RouteUsingSingleCenter(points, queries, clusters);
    auto single_center_recall = RecallForIncreasingProbes(single_center_probes, partition, ground_truth, num_neighbors, num_shards);
    std::ofstream out2(out_file);
    out2 << "partitioning,num probes,recall,type" << std::endl; // header
    for (size_t j = 0; j < num_shards; ++j) {
        out2 << part_method << "," << j + 1 << "," << single_center_recall[j] << ",single center" << std::endl;
        std::cout << part_method << "," << j + 1 << "," << single_center_recall[j] << ",single center" << std::endl;
    }

    return 0;

    // --- Routing on full pointset --- //
    Timer timer;
    timer.Start();
    auto full_probes = FullDatasetRouting(ground_truth, queries, points, partition, num_shards);
    std::cout << "Finished full dataset routing. Took " << timer.Stop() << std::endl;
    std::vector<double> recall = RecallForIncreasingProbes(full_probes, partition, ground_truth, num_neighbors, num_shards);
    std::ofstream out(out_file);
    out << "partitioning,num probes,recall,type" << std::endl; // header
    for (size_t j = 0; j < num_shards; ++j) { out << part_method << "," << j + 1 << "," << recall[j] << ",full data" << std::endl; }

    // --- Routing on KMTR sample --- //
    timer.Start();
    KMeansTreeRouterOptions options;
    options.budget = 10000000;
    KMeansTreeRouter kmtr;
    kmtr.Train(points, clusters, options);
    auto [kmtr_points, kmtr_partition] = kmtr.ExtractPoints();
    std::cout << "Finished KMTR training. Took " << timer.Stop() << std::endl;
    std::cout << kmtr_points.n << " " << kmtr_partition.size() << " " << num_shards << std::endl;

    timer.Start();
    std::vector<std::vector<int>> kmtr_probes = BruteForceRouting(queries, kmtr_points, kmtr_partition, num_shards);
    std::cout << "brute force routing finished. took " << timer.Stop() << std::endl;
    recall = RecallForIncreasingProbes(kmtr_probes, partition, ground_truth, num_neighbors, num_shards);
    std::cout << "Finished KMTR sample brute force routing." << std::endl;

    for (size_t j = 0; j < num_shards; ++j) { out << part_method << "," << j + 1 << "," << recall[j] << ",kRt sample" << std::endl; }

    // --- Routing on uniform random sample --- //
    std::vector<uint32_t> iota(points.n);
    std::iota(iota.begin(), iota.end(), 0);
    std::mt19937 prng(420);
    std::vector<uint32_t> sample(options.budget);
    std::sample(iota.begin(), iota.end(), sample.begin(), sample.size(), prng);
    PointSet uf_points = ExtractPointsInBucket(sample, points);
    Partition uf_partition(sample.size());
    for (size_t i = 0; i < sample.size(); ++i) { uf_partition[i] = partition[sample[i]]; }
    auto uf_probes = BruteForceRouting(queries, uf_points, uf_partition, num_shards);
    recall = RecallForIncreasingProbes(uf_probes, partition, ground_truth, num_neighbors, num_shards);

    std::cout << "Finished UF sample brute force routing" << std::endl;

    for (size_t j = 0; j < num_shards; ++j) { out << part_method << "," << j + 1 << "," << recall[j] << ",uniform sample" << std::endl; }
}
