#include <iostream>

#include "knn_graph.h"
#include "points_io.h"
#include "metis_io.h"
#include "partitioning.h"

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
        partition = RecursiveKMeansPartitioning(points, k, eps);
    } else if (part_method == "OurPyramid") {
        for (double coarsening_rate : { 0.005, 0.01, 0.015, 0.02 }) {
            std::string my_part_file = part_file + ".coarsen=" + std::to_string(coarsening_rate);
            std::vector<int> second_partition;
            partition = OurPyramidPartitioning(points, k, eps, second_partition, my_part_file + ".our_pyramid_routing_index", coarsening_rate);
            WriteMetisPartition(second_partition, my_part_file + ".hnsw_graph_part");
            WriteMetisPartition(partition, my_part_file);
        }
        std::cout << "Finished partitioning" << std::endl;
        return 0;
    } else {
        std::cout << "Unsupported partitioning method " << part_method << " . The supported options are [GP, Pyramid, KMeans]" << std::endl;
        std::abort();
    }
    std::cout << "Finished partitioning" << std::endl;
    WriteMetisPartition(partition, part_file);
}
