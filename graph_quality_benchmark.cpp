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
    ApproximateKNNGraphBuilder blueprint;
    blueprint.quiet = true;
    std::vector<ApproximateKNNGraphBuilder> configs;
    for (int reps : { 2, 3 }) {
        blueprint.REPETITIONS = reps;
        configs.push_back(blueprint);
    }
    auto copy = configs;
    configs.clear();
    for (int fanout : { 2, 3 }) {
        for (auto c : copy) {
            c.FANOUT = fanout;
            configs.push_back(c);
        }
    }
    return configs;
}

std::string Header() {
    return "fanout,repetitions,clustersize,degree,graph-recall,oracle-recall";
}

std::string FormatOutput(const ApproximateKNNGraphBuilder& gb, double oracle_recall, double graph_recall, int degree) {
    std::stringstream str;
    str << gb.FANOUT << "," << gb.REPETITIONS << "," << gb.MAX_CLUSTER_SIZE << ",";
    str << degree << "," << graph_recall << "," << oracle_recall;
    return str.str();
}

using AdjHashGraph = parlay::sequence<std::unordered_set<int>>;

double GraphRecall(const AdjHashGraph& exact_graph, const AdjGraph& approximate_graph, int degree) {
    auto hits = parlay::delayed_tabulate(approximate_graph.size(), [&](size_t i) {
        const auto& exact_neighbors = exact_graph[i];
        const auto& neighbors = approximate_graph[i];
        int my_hits = 0;
        for (int j = 0; j < std::min<int>(degree, neighbors.size()); ++j) {
            if (exact_neighbors.contains(neighbors[j])) {
                my_hits++;
            }
        }
        return my_hits;
    });
    return static_cast<double>(parlay::reduce(hits)) / (approximate_graph.size() * degree);
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

    points.Resize(10000);

    int max_degree = 100;
    int num_query_neighbors = 10;
    int num_clusters = 16;
    double epsilon = 0.05;
    std::vector<int> num_degree_values = { 100, 80, 50, 20, 10, 8, 5, 3 };

#if false
    if (!std::filesystem::exists(ground_truth_file)) {
        throw std::runtime_error("Ground truth file does not exist.");
    }
    std::vector<NNVec> ground_truth = ReadGroundTruth(ground_truth_file);
    std::cout << "Read ground truth file" << std::endl;
#else
    std::vector<NNVec> ground_truth = ComputeGroundTruth(points, queries, num_query_neighbors);
#endif

    Timer timer;
    timer.Start();
    AdjGraph exact_graph = BuildExactKNNGraph(points, max_degree);
    std::cout << "Building exact graph took " << timer.Stop() << std::endl;

    timer.Start();
    auto exact_graph_hashes = parlay::map(num_degree_values, [&](int degree) {
        return parlay::map(exact_graph, [degree](const std::vector<int>& neighbors) {
            return std::unordered_set<int>(neighbors.begin(), neighbors.begin() + std::min<int>(degree, neighbors.size()));
        });
    });
    std::cout << "Convert to hash took " << timer.Stop() << std::endl;

    auto graph_builders = InstantiateGraphBuilders();

    size_t total_num_configs = graph_builders.size() * num_degree_values.size();
    std::cout << "num configs " << total_num_configs << " num graph builders " << graph_builders.size() << std::endl;

    auto output_lines = parlay::map(graph_builders, [&](ApproximateKNNGraphBuilder& graph_builder) -> std::string {
        std::cout << "build graph" << std::endl;
        const AdjGraph approximate_graph = graph_builder.BuildApproximateNearestNeighborGraph(points, max_degree);
        std::cout << "build graph finished" << std::endl;
        std::stringstream stream;

        int nni = 0;
        for (int degree : num_degree_values) {
            std::cout << "graph recall. degree " << degree << std::endl;
            double graph_recall = GraphRecall(exact_graph_hashes[nni], approximate_graph, degree);
            std::cout << "graph recall. degree " << degree << " finished " << std::endl;
            nni++;

            std::cout << "partition " << std::endl;
            Partition partition = PartitionAdjListGraph(approximate_graph, num_clusters, epsilon, true);
            std::cout << "partition finished" << std::endl;

            std::cout << "first shard recall" << std::endl;
            double oracle_recall = FirstShardOracleRecall(ground_truth, partition, num_query_neighbors);
            std::cout << "first shard recall finished" << std::endl;

            stream << FormatOutput(graph_builder, oracle_recall, graph_recall, degree) << "\n";
            std::cout << "finish format" << std::endl;
        }

        return stream.str();
    });

    std::ofstream out(output_file);
    out << Header() << "\n";
    std::cout << Header() << std::endl;
    for (const std::string& outputs : output_lines) {
        out << outputs;
        std::cout << outputs << std::flush;
    }
    out << std::flush;
}
