#include <iostream>

#include "knn_graph.h"
#include "points_io.h"
#include "metis_io.h"
#include "partitioning.h"

int main(int argc, const char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage ./Partition input-points output-path" << std::endl;
        std::abort();
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];
    PointSet points = ReadPoints(input_file);
    std::cout << "Finished reading points" << std::endl;
    #ifdef MIPS_DISTANCE
    Normalize(points);
    std::cout << "MIPS distance set --> Finished normalizing points" << std::endl;
    #endif

    std::vector<int> ks= { 10, 20, 40 };

    auto partitions = GraphPartitioning(points, ks, 0.05);
    std::cout << "Finished partitioning" << std::endl;
    for (size_t i = 0; i < ks.size(); ++i) {
        WriteMetisPartition(partitions[i], output_file + ".k=" + std::to_string(ks[i]));
    }
}
