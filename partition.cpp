#include <iostream>

#include "knn_graph.h"
#include "points_io.h"
#include "metis_io.h"
#include "partitioning.h"

std::vector<int> BalancedKMeansCall(PointSet& points, int k, double eps) {
    PointSet centroids = RandomSample(points, k, 555);
    size_t max_cluster_size = points.n * (1.0+eps) / k;
    return BalancedKMeans(points, centroids, max_cluster_size);
}

std::vector<int> FlatKMeansCall(PointSet& points, int k, double eps) {
    PointSet centroids = RandomSample(points, k, 555);
    return KMeans(points, centroids);
}

void PrintImbalance(std::vector<int>& partition, int k) {
    auto histo = parlay::histogram_by_index(partition, k);
    auto max_part_size = *parlay::max_element(histo);
    double imbalance = double(max_part_size) / (partition.size() / k);
    std::cout << "imbalance " << imbalance << " max part size " << max_part_size << " perf balanced " << partition.size() / k << std::endl;
}

int main(int argc, const char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage ./Partition input-points output-path num-clusters partitioning-method" << std::endl;
        std::abort();
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];
    std::string k_string = argv[3];
    int k = std::stoi(k_string);
    std::string part_method = argv[4];
    std::string part_file = output_file + ".k=" + std::to_string(k) + "." + part_method;

    if (part_method == "Random") {
        uint32_t n;
        {
            std::ifstream in(input_file, std::ios::binary);
            in.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
        }
        std::vector<int> partition;
        partition.reserve(n);
        for (int b = 0; b < k; ++b) {
            partition.insert(partition.end(), n/k, b);
        }
        std::mt19937 prng(555);
        std::shuffle(partition.begin(), partition.end(), prng);
        WriteMetisPartition(partition, part_file);
        return 0;
    }

    PointSet points = ReadPoints(input_file);
    std::cout << "Finished reading points" << std::endl;

    const double eps = 0.05;
    std::vector<int> partition;
    if (part_method == "GP") {
        partition = GraphPartitioning(points, k, eps);
    } else if (part_method == "Pyramid") {
        partition = PyramidPartitioning(points, k, eps, part_file + ".pyramid_routing_index");
    } else if (part_method == "KMeans") {
        partition = KMeansPartitioning(points, k, eps);
    } else if (part_method == "BalancedKMeans") {
        partition = BalancedKMeansCall(points, k, eps);
    } else if (part_method == "FlatKMeans") {
        partition = FlatKMeansCall(points, k, eps);
    } else if (part_method == "OurPyramid") {
        partition = OurPyramidPartitioning(points, k, eps, part_file + ".our_pyramid_routing_index", 0.02);
    } else {
        std::cout << "Unsupported partitioning method " << part_method << " . The supported options are [GP, Pyramid, KMeans]" << std::endl;
        std::abort();
    }
    std::cout << "Finished partitioning" << std::endl;
    PrintImbalance(partition, k);
    WriteMetisPartition(partition, part_file);
}
