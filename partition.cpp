#include <iostream>

#include "knn_graph.h"
#include "points_io.h"
#include "metis_io.h"
#include "partitioning.h"

int main(int argc, const char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage ./Partition input-points output-path k" << std::endl;
        std::abort();
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];
    std::string k_string = argv[3];
    int k = std::stoi(k_string);
    PointSet points = ReadPoints(input_file);
    if (false) {
        Normalize(points);
    }

    std::vector<int> partition = GraphPartitioning(points, k, 0.05);
    std::cout << "Finished partitioning" << std::endl;
    WriteMetisPartition(partition, output_file);
}
