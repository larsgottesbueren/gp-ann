#include <iostream>

#include "points_io.h"


int main(int argc, const char* argv[]) {
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
}
