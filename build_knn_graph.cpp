#include <iostream>

#include "knn_graph.h"
#include "points_io.h"
#include "metis_io.h"


int main(int argc, const char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage ./BuildKNNGraph input-points graph-output-path k" << std::endl;
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
    AdjGraph knn_graph = BuildKNNGraph(points, k);
    Symmetrize(knn_graph);
    WriteMetisGraph(output_file, knn_graph);
}
