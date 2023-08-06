#include <iostream>

#include "knn_graph.h"
#include "points_io.h"
#include "metis_io.h"

void Normalize(PointSet& points) {
    for (size_t i = 0; i < points.n; ++i) {
        float* p = points.GetPoint(i);
        bool all_zero = true;
        for (size_t j = 0; j < points.d; ++j) {
            if (p[j] != 0) {
                all_zero = false;
                break;
            }
        }
        if (all_zero) {
            std::cout << "Point " << i << " is fully zero --> delete" << std::endl;
        }
        efanna2e::DistanceFastL2 dst;
        float norm = dst.norm(p, points.d);
        if (norm < 1e-9) {
            std::cout << "tiny norm" << std::endl;
        }
        for (size_t j = 0; j < points.d; ++j) {
            p[j] /= norm;
        }
    }
    std::cout << "finished normalizing" << std::endl;
}

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
