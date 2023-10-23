#include <iostream>
#include "defs.h"
#include "dist.h"
#include "points_io.h"


int main(int argc, const char* argv[]) {
    std::vector<float> P;
    int dim = 100;
    for (int i = 0; i < dim; ++i) {
        P.push_back(i);
    }
    std::vector<float> Q = P;
    for (float& q : Q) q = q - 2.3f;

    int num_bogus_entries = 214;
    for (int i = 0; i < num_bogus_entries; ++i) {
        P.push_back(-555);
        Q.push_back(555);
    }

    float dist = distance(&P[0], &Q[0], dim);


    std::cout << "Dist = " << dist << std::endl;
    std::cout << "expected = " << dim * (2.3f * 2.3f) << std::endl;
    std::exit(0);

    std::string gtf = argv[1];
    auto ground_truth = ReadGroundTruth(gtf);

    #if false
    if (argc != 4) {
        std::cerr << "Usage ./Convert input-points output-points size" << std::endl;
        std::abort();
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];
    std::string k_string = argv[3];
    size_t n = std::stoi(k_string);

    PointSet points = ReadPoints(input_file, n);
    WritePoints(points, output_file);
    #endif
}
