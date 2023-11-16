#include <iostream>
#include "defs.h"
#include "points_io.h"
#include "recall.h"


int main(int argc, const char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage ./Convert input-points output-points size" << std::endl;
        std::abort();
    }

    std::string input_file = argv[1];
    std::string query_file = argv[2];
    std::string ground_truth_file = argv[3];
    std::string k_string = argv[4];
    int num_neighbors = std::stoi(k_string);

    PointSet points = ReadPoints(input_file);
    PointSet queries = ReadPoints(query_file);
    std::vector<NNVec> ground_truth = ReadGroundTruth(ground_truth_file);
    auto copy_gt = ground_truth;

    std::cout << "Unnormalized gt results" << std::endl;
    auto top_k_distances = ConvertGroundTruthToDistanceToKthNeighbor(ground_truth, num_neighbors, points, queries);
    std::cout << "finished. now normalize" << std::endl;
    ground_truth = std::move(copy_gt);
    Normalize(points);
    Normalize(queries);
    std::cout << "Normalized gt results" << std::endl;
    auto top_k_distances_2 = ConvertGroundTruthToDistanceToKthNeighbor(ground_truth, num_neighbors, points, queries);
    std::cout << "Done. Now?" << std::endl;
}
