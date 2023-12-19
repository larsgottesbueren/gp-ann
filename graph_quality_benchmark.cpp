#include <iostream>
#include <filesystem>

#include "points_io.h"
#include "recall.h"


int main(int argc, const char* argv[]) {
    std::string point_file = argv[1];
    std::string query_file = argv[2];
    std::string ground_truth_file = argv[3];
    std::string k_string = argv[4];
    int num_neighbors = std::stoi(k_string);

    PointSet points = ReadPoints(point_file);
    PointSet queries = ReadPoints(query_file);

    std::vector<NNVec> ground_truth;
    if (std::filesystem::exists(ground_truth_file)) {
        ground_truth = ReadGroundTruth(ground_truth_file);
        std::cout << "Read ground truth file" << std::endl;
    } else {
        std::cout << "start computing ground truth" << std::endl;
        ground_truth = ComputeGroundTruth(points, queries, num_neighbors);
        std::cout << "computed ground truth" << std::endl;
        WriteGroundTruth(ground_truth_file, ground_truth);
    }


}
