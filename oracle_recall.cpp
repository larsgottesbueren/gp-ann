#include <iostream>
#include <filesystem>
#include <parlay/primitives.h>

#include "routes.h"
#include "points_io.h"
#include "metis_io.h"


int main(int argc, const char* argv[]) {
    if (argc != 7) {
        std::cerr << "Usage ./OracleRecall ground-truth-file routes-file num_neighbors partition-file" << std::endl;
        std::abort();
    }

    std::string ground_truth_file = argv[1];
    std::string routes_file = argv[2];
    std::string k_string = argv[3];
    std::string partition_file = argv[4];

    int num_neighbors = std::stoi(k_string);

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

    size_t num_queries = ground_truth.size();
    size_t num_shards = clusters.size();
    std::vector<RoutingConfig> routes = DeserializeRoutes(routes_file);
    for (const RoutingConfig& route : routes) {
        std::vector<std::unordered_set<uint32_t>> neighbors(num_queries);
        std::vector<double> recall_values;
        size_t hits = 0;
        for (size_t probes = 0; probes < num_shards; ++probes) {
             hits += parlay::reduce(
                parlay::tabulate(num_queries, [&](size_t q) {
                    int cluster = route.buckets_to_probe[q][probes];
                    size_t my_new_hits = 0;
                    for (int j = 0; j < num_neighbors; ++j) {
                        uint32_t neighbor = ground_truth[q][j].second;
                        // if we haven't seen the neighbor before
                        // and it's in the cluster we are looking at right now
                        if (!neighbors[q].contains(neighbor) &&
                            std::find(cover[neighbor].begin(), cover[neighbor].end(), cluster) != cover[neighbor].end()) {
                            neighbors[q].insert(neighbor);
                            my_new_hits++;
                        }
                    }
                    return my_new_hits;
                })
            );
            double recall = static_cast<double>(hits) / num_neighbors;
        }
    }
}
