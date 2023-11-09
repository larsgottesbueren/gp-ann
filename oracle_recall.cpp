#include <iostream>
#include <filesystem>
#include <map>

#include "points_io.h"

#include "metis_io.h"
#include "recall.h"


int main(int argc, const char* argv[]) {
    if (argc != 4 && argc != 6) {
        std::cerr << "Usage ./OracleRecall ground-truth-file num_neighbors partition-file [point-file query-file]" << std::endl;
        std::abort();
    }

    std::string ground_truth_file = argv[1];
    std::string k_string = argv[2];
    int num_neighbors = std::stoi(k_string);
    std::string partition_file = argv[3];

    std::vector<NNVec> ground_truth;
    if (std::filesystem::exists(ground_truth_file)) {
        ground_truth = ReadGroundTruth(ground_truth_file);
        std::cout << "Read ground truth file" << std::endl;
    } else {
        std::string point_file = argv[4];
        std::string query_file = argv[5];
        PointSet points = ReadPoints(point_file);
        PointSet queries = ReadPoints(query_file);
        std::cout << "start computing ground truth" << std::endl;
        ground_truth = ComputeGroundTruth(points, queries, num_neighbors);
        std::cout << "computed ground truth" << std::endl;
    }

    auto partition = ReadMetisPartition(partition_file);
    std::cout << "Finished reading partition file" << std::endl;

    OracleRecall(ground_truth, partition, num_neighbors);
}
