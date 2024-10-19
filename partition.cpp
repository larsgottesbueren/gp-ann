#include <fstream>
#include <iostream>
#include <random>

#include "kmeans.h"
#include "metis_io.h"
#include "overlapping_partitioning.h"
#include "partitioning.h"
#include "points_io.h"
#include "recall.h"

#include <parlay/primitives.h>

std::vector<int> BalancedKMeansCall(PointSet& points, int k, double eps) {
    PointSet centroids = RandomSample(points, k, 555);
    size_t max_cluster_size = points.n * (1.0 + eps) / k;
    Timer timer;
    timer.Start();
    auto result = BalancedKMeans(points, centroids, max_cluster_size);
    std::cout << "Balanced Kmeans took " << timer.Stop() << " seconds" << std::endl;
    return result;
}

std::vector<int> FlatKMeansCall(PointSet& points, int k, double eps) {
    PointSet centroids = RandomSample(points, k, 555);
    return KMeans(points, centroids);
}

void PrintImbalance(std::vector<int>& partition, int k) {
    auto histo = parlay::histogram_by_index(partition, k);
    auto max_part_size = *parlay::max_element(histo);
    std::cout << " max part size " << max_part_size << " " << partition.size() << " " << k << std::endl;
    double imbalance = double(max_part_size) / (partition.size() / k);
    std::cout << "imbalance " << imbalance << " max part size " << max_part_size << " perf balanced " << partition.size() / k << std::endl;
}

int main(int argc, const char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage ./Partition input-points query-points ground-truth" << std::endl;
    }
    std::string point_file = argv[1];
    std::string query_file = argv[2];
    std::string ground_truth_file = argv[3];
    int num_neighbors = 10;
    int num_shards = 40;
    double imbalance = 0.05;

    PointSet points = ReadPoints(point_file);
    PointSet queries = ReadPoints(query_file);
    std::vector<NNVec> ground_truth = ReadGroundTruth(ground_truth_file);

    Partition partition = PyramidPartitioning(points, num_shards, imbalance, /*imbalanced=*/true);
    std::cout << "Finished Pyramid" << std::endl;
    std::vector<int> cluster_sizes(num_shards, 0);
    for (int x : partition) {
        cluster_sizes[x]++;
    }
    std::cout << "Max Pyramid cluster size = " << *std::max_element(cluster_sizes.begin(), cluster_sizes.end()) << std::endl;

    { // Oracle
        std::vector<int> gt_right = GroundTruthRightEnd(ground_truth, num_neighbors);
        size_t hits = 0;
        std::vector<int> freq;
        for (size_t q = 0; q < queries.n; ++q) {
            const NNVec& nn = ground_truth[q];
            freq.assign(num_shards, 0);
            for (int j = 0; j < gt_right[q]; ++j) {
                int c = partition[nn[j].second];
                freq[c]++;
            }
            hits += std::min<int>(num_neighbors, *std::max_element(freq.begin(), freq.end()));
        }

        double recall = static_cast<double>(hits) / num_neighbors / queries.n;
        std::cout << "oracle recall. first shard " << recall << std::endl;
        
    }



    return 0;
#if false
    if (argc != 6 && argc != 7) {
        std::cerr << "Usage ./Partition input-points output-path num-clusters partitioning-method (default|strong) [overlap]" << std::endl;
        std::abort();
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];
    std::string k_str = argv[3];
    int k = std::stoi(k_str);
    std::string part_method = argv[4];
    std::string part_file = output_file + ".k=" + k_str + "." + part_method;

    std::string config = argv[5];
    bool strong = false;
    if (config == "strong") {
        strong = true;
    } else if (config != "default") {
        throw std::runtime_error("Unknown config: " + config);
    }

    double overlap = 0.0;
    if (argc == 7) {
        std::string overlap_str = argv[6];
        overlap = std::stod(overlap_str);
        part_file += ".o=" + overlap_str;
    }

    if (part_method == "Random") {
        uint32_t n;
        {
            std::ifstream in(input_file, std::ios::binary);
            in.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
        }
        std::vector<int> partition;
        partition.reserve(n);
        for (int b = 0; b < k; ++b) {
            partition.insert(partition.end(), n / k, b);
        }
        std::mt19937 prng(555);
        std::shuffle(partition.begin(), partition.end(), prng);
        WriteMetisPartition(partition, part_file);
        return 0;
    }

    PointSet points = ReadPoints(input_file);
    std::cout << "Finished reading points" << std::endl;

    if (part_method == "GP" && overlap != 0.0) {
        part_method = "OGP";
    }

    const double eps = 0.05;
    std::vector<int> partition;
    Clusters clusters;
    if (part_method == "GP") {
        partition = GraphPartitioning(points, k, eps, strong);
    } else if (part_method == "Pyramid") {
        partition = PyramidPartitioning(points, k, eps, part_file + ".pyramid_routing_index");
    } else if (part_method == "KMeans") {
        partition = KMeansPartitioning(points, k, eps);
    } else if (part_method == "BalancedKMeans") {
        partition = BalancedKMeansCall(points, k, eps);
    } else if (part_method == "FlatKMeans") {
        partition = FlatKMeansCall(points, k, eps);
    } else if (part_method == "RKM") {
        const size_t max_cluster_size = (1.0 + eps) * points.n / k;
        partition = RebalancingKMeansPartitioning(points, max_cluster_size, k);
    } else if (part_method == "ORKM") {
        const size_t max_cluster_size = (1.0 + eps) * points.n / k;
        int adjusted_num_clusters = std::ceil(k * (1.0 + overlap));
        auto rkm = RebalancingKMeansPartitioning(points, max_cluster_size, adjusted_num_clusters);
        clusters = OverlappingKMeansPartitioningSPANN(points, rkm, k, eps, overlap);
    } else if (part_method == "OurPyramid") {
        partition = OurPyramidPartitioning(points, k, eps, part_file + ".our_pyramid_routing_index", 0.02);
    } else if (part_method == "OGP") {
        clusters = OverlappingGraphPartitioning(points, k, eps, overlap, strong);
    } else if (part_method == "OGPS") {
        const size_t max_cluster_size = (1.0 + eps) * points.n / k;
        const size_t num_extra_assignments = overlap * points.n;
        const size_t num_total_assignments = points.n + num_extra_assignments;
        int adjusted_num_clusters = std::ceil(static_cast<double>(num_total_assignments) / max_cluster_size);
        auto kmp = GraphPartitioning(points, adjusted_num_clusters, eps, false);
        clusters = OverlappingKMeansPartitioningSPANN(points, kmp, k, eps, overlap);
    } else if (part_method == "OKM") {
        // leave the same num clusters, since k-means will use more than requested anyways
        Timer timer;
        timer.Start();
        auto kmp = KMeansPartitioning(points, k, eps);
        std::cout << "KM took " << timer.Stop() << " seconds" << std::endl;
        clusters = OverlappingKMeansPartitioningSPANN(points, kmp, k, eps, overlap);
    } else if (part_method == "OBKM") {
        int adjusted_num_clusters = std::ceil(k * (1.0 + overlap));
        // use adjusted num clusters for BKM call
        auto bkm = BalancedKMeansCall(points, adjusted_num_clusters, eps);
        // but use the original number for the overlap call, so that it chooses the correct max cluster size. The code can handle the case
        // that NumPartsInPartition(bkm) != k
        clusters = OverlappingKMeansPartitioningSPANN(points, bkm, k, eps, overlap);
    } else {
        std::cout << "Unsupported partitioning method " << part_method << " . The supported options are [GP, Pyramid, KMeans]" << std::endl;
        std::abort();
    }
    std::cout << "Finished partitioning" << std::endl;

    if (clusters.empty()) {
        clusters = ConvertPartitionToClusters(partition);
    }
    WriteClusters(clusters, part_file);
#endif
}
