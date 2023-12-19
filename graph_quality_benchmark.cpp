#include <iostream>
#include <filesystem>
#include <fstream>
#include <unordered_set>

#include <parlay/primitives.h>

#include "knn_graph.h"
#include "partitioning.h"
#include "points_io.h"
#include "recall.h"

std::vector<ApproximateKNNGraphBuilder> InstantiateGraphBuilders() {
    return {};
}

std::string Header() {
    return "";
}

std::string FormatOutput(const ApproximateKNNGraphBuilder& gb, double oracle_recall, double graph_recall, int num_neighbors) {
    return "";
}

using AdjHashGraph = parlay::sequence<std::unordered_set<int>>;

double GraphRecall(const AdjHashGraph& exact_graph, const AdjGraph& approximate_graph, int num_neighbors) {
    auto hits = parlay::delayed_tabulate(approximate_graph.size(), [&](size_t i) {
        const auto& exact_neighbors = exact_graph[i];
        const auto& neighbors = approximate_graph[i];
        int my_hits = 0;
        for (int j = 0; j < std::min<int>(num_neighbors, neighbors.size()); ++j) {
            if (exact_neighbors.contains(neighbors[j])) {
                my_hits++;
            }
        }
        return my_hits;
    });
    return static_cast<double>(parlay::reduce(hits)) / (approximate_graph.size() * num_neighbors);
}

double FirstShardOracleRecall(const std::vector<NNVec>& ground_truth, const Partition& partition, int num_query_neighbors) {
    int num_shards = NumPartsInPartition(partition);
    size_t hits = 0;
    for (const auto& neigh : ground_truth) {
        std::vector<int> freq(num_shards, 0);
        for (int i = 0; i < num_query_neighbors; ++i) {
            freq[partition[neigh[i].second]]++;
        }
        hits += *std::max_element(freq.begin(), freq.end());
    }
    return static_cast<double>(hits) / (ground_truth.size() * num_query_neighbors);
}

int main(int argc, const char* argv[]) {
    std::string point_file = argv[1];
    std::string query_file = argv[2];
    std::string ground_truth_file = argv[3];
    std::string output_file = argv[4];

    PointSet points = ReadPoints(point_file);
    PointSet queries = ReadPoints(query_file);

    if (!std::filesystem::exists(ground_truth_file)) {
        throw std::runtime_error("Ground truth file does not exist.");
    }
    std::vector<NNVec> ground_truth = ReadGroundTruth(ground_truth_file);
    std::cout << "Read ground truth file" << std::endl;


    int max_num_neighbors = 100;
    int num_query_neighbors = 10;
    int num_clusters = 16;
    double epsilon = 0.05;
    std::vector<int> num_neighbors_values = { 100, 80, 50, 20, 10, 8, 5, 3 };

    Timer timer;
    timer.Start();
    AdjGraph exact_graph = BuildExactKNNGraph(points, max_num_neighbors);
    std::cout << "Building exact graph took " << timer.Stop() << std::endl;

    timer.Start();
    auto exact_graph_hashes = parlay::map(num_neighbors_values, [&](int num_neighbors) {
        return parlay::map(exact_graph, [num_neighbors](const std::vector<int>& neighbors) {
            int degree = std::min<int>(num_neighbors, neighbors.size());
            return std::unordered_set<int>(neighbors.begin(), neighbors.begin() + degree);
        });
    });
    std::cout << "Convert to hash took " << timer.Stop() << std::endl;

    auto graph_builders = InstantiateGraphBuilders();

    auto output_lines = parlay::map(graph_builders, [&](ApproximateKNNGraphBuilder& graph_builder) -> std::string {
        AdjGraph approximate_graph = graph_builder.BuildApproximateNearestNeighborGraph(points, max_num_neighbors);

        std::stringstream stream;

        int nni = 0;
        for (int num_neighbors : num_neighbors_values) {
            double graph_recall = GraphRecall(exact_graph_hashes[nni], approximate_graph, num_neighbors);
            nni++;

            Partition partition = PartitionAdjListGraph(approximate_graph, num_clusters, epsilon, true);

            double oracle_recall = FirstShardOracleRecall(ground_truth, partition, num_query_neighbors);

            stream << FormatOutput(graph_builder, oracle_recall, graph_recall, num_neighbors) << "\n";
        }

        return stream.str();
    });

    std::ofstream out(output_file);
    out << Header() << "\n";
    for (const std::string& outputs : output_lines) {
        out << outputs;
    }
    out << std::flush;
}
