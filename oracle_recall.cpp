#include <iostream>
#include <filesystem>

#include "points_io.h"

#include "metis_io.h"


int main(int argc, const char* argv[]) {
    if (argc != 4 && argc != 6) {
        std::cerr << "Usage ./OracleRecall ground-truth-file num_neighbors partition-file [point-file query-file]" << std::endl;
        std::abort();
    }

    std::string ground_truth_file = argv[1];
    std::string k_string = argv[2];
    int num_neighbors = std::stoi(k_string);
    std::string partition_file = argv[3];

    auto clusters = ReadClusters(partition_file);
    std::cout << "Finished reading partition file" << std::endl;
    Cover cover = ConvertClustersToCover(clusters);

    std::vector<NNVec> ground_truth;
    if (std::filesystem::exists(ground_truth_file)) {
        ground_truth = ReadGroundTruth(ground_truth_file);
        std::cout << "Read ground truth file" << std::endl;
    } else {
        throw std::runtime_error("ground truth file doesnt exist");
    }


    size_t hits = 0;
    std::vector<int> cluster_distribution(clusters.size(), 0);
    for (const NNVec& nn : ground_truth) {
        std::vector<int> freq(clusters.size(), 0);
        for (int j = 0; j < num_neighbors; ++j) {
            for (int c : cover[nn[j].second]) {
                cluster_distribution[c]++;
                freq[c]++;
            }
        }
        hits += *std::max_element(freq.begin(), freq.end());
    }
    std::cout << "First probe hits : " << hits << ". First probe recall " << static_cast<double>(hits) / ground_truth.size() / num_neighbors << std::endl;
    std::cout << "Cluster distribution ";
    for (int x : cluster_distribution) std::cout << x << " ";
    std::cout << std::endl;
}
