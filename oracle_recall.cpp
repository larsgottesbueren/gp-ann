#include <iostream>
#include <filesystem>
#include <parlay/primitives.h>

#include "routes.h"
#include "points_io.h"
#include "metis_io.h"


std::vector<double> RecallForIncreasingProbes(
    const std::vector<std::vector<int>>& buckets_to_probe, const Cover& cover, const std::vector<NNVec>& ground_truth, int num_neighbors, size_t num_shards) {
    size_t num_queries = ground_truth.size();
    std::vector<std::unordered_set<uint32_t>> neighbors(num_queries);
    std::vector<double> recall_values;
    size_t hits = 0;
    for (size_t probes = 0; probes < num_shards; ++probes) {
        hits += parlay::reduce(
           parlay::tabulate(num_queries, [&](size_t q) {
               int cluster = buckets_to_probe[q][probes];
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
        double recall = static_cast<double>(hits) / num_neighbors / num_queries;
        recall_values.push_back(recall);
    }
    return recall_values;
}

int main(int argc, const char* argv[]) {
    if (argc != 7) {
        std::cerr << "Usage ./OracleRecall ground-truth-file routes-file num_neighbors partition-file part-method out-file" << std::endl;
        std::abort();
    }

    std::string ground_truth_file = argv[1];
    std::string routes_file = argv[2];
    std::string k_string = argv[3];
    std::string partition_file = argv[4];
    std::string part_method = argv[5];
    std::string out_file = argv[6];

    int num_neighbors = std::stoi(k_string);

#if false
    auto clusters = ReadClusters(partition_file);
    Cover cover = ConvertClustersToCover(clusters);
    size_t num_shards = clusters.size();
#else
    auto partition = ReadMetisPartition(partition_file);
    size_t num_shards = NumPartsInPartition(partition);
    Cover cover = ConvertPartitionToCover(partition);
#endif
    std::cout << "Finished reading partition file" << std::endl;

    std::vector<NNVec> ground_truth;
    if (std::filesystem::exists(ground_truth_file)) {
        ground_truth = ReadGroundTruth(ground_truth_file);
        std::cout << "Read ground truth file" << std::endl;
    } else {
        throw std::runtime_error("ground truth file doesnt exist");
    }

    std::vector<RoutingConfig> routes = DeserializeRoutes(routes_file);

    auto rrv = parlay::map(routes, [&](const RoutingConfig& route) {
        return RecallForIncreasingProbes(route.buckets_to_probe, cover, ground_truth, num_neighbors, num_shards);
    });

    int best = 0;
    for (int i = 1; i < rrv.size(); ++i) {
        if (rrv[i][0] > rrv[best][0]) {
            best = i;
        }
    }

    std::cout << std::endl;
    std::cout << "best config " << best << " first shard recall " << rrv[best][0] << std::endl;

    std::ofstream out(out_file);
    // header
    out << "partitioning,num probes,recall,type" << std::endl;
    for (size_t j = 0; j < num_shards; ++j) {
        out << part_method << "," << j << "," << rrv[best][j] << ",brute-force-shard-search" << std::endl;
    }

    {   // Oracle
        std::vector<std::vector<int>> buckets_to_probe(ground_truth.size());
        parlay::parallel_for(0, ground_truth.size(), [&](size_t q) {
            const NNVec& nn = ground_truth[q];
            std::vector<int> freq(num_shards, 0);
            for (int j = 0; j < num_neighbors; ++j) {
                for (int c : cover[nn[j].second]) {
                    freq[c]++;
                }
            }
            std::vector<int> probes(num_shards);
            std::iota(probes.begin(), probes.end(), 0);
            // no update for found neighbors after each probe. just send it!
            std::sort(probes.begin(), probes.end(), [&](int l, int r) { return freq[l] > freq[r]; });
            buckets_to_probe[q] = std::move(probes);
        });

        auto oracle_recall_values = RecallForIncreasingProbes(buckets_to_probe, cover, ground_truth, num_neighbors, num_shards);
        std::cout << "oracle recall. first shard " << oracle_recall_values[0] << std::endl;
        for (size_t j = 0; j < num_shards; ++j) {
            out << part_method << "," << j << "," << oracle_recall_values[j] << ",oracle" << std::endl;
        }
    }
}
