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
    for (int reps : { 2, 3, 5, 8, 10}) {
        blueprint.REPETITIONS = reps;
        configs.push_back(blueprint);
    }
    auto copy = configs;
    configs.clear();
    for (int fanout : { 2, 3, 5, 8, 10 }) {
        for (auto c : copy) {
            c.FANOUT = fanout;
            configs.push_back(c);
        }
    }
    copy = configs;
    configs.clear();
    for (int cluster_size : { 500, 1000, 2000, 5000, 10000 }) {
        for (auto c : copy) {
            c.MAX_CLUSTER_SIZE = cluster_size;
            configs.push_back(c);
        }
    }
    return configs;
}

std::string Header() {
    return "approximate,fanout,repetitions,clustersize,degree,graph-recall,oracle-recall";
}

std::string FormatOutput(const ApproximateKNNGraphBuilder& gb, double oracle_recall, double graph_recall, int degree) {
    std::stringstream str;
    str << gb.FANOUT << "," << gb.REPETITIONS << "," << gb.MAX_CLUSTER_SIZE << ",";
    str << degree << "," << graph_recall << "," << oracle_recall;
    return str.str();
}

using AdjHashGraph = parlay::sequence<std::unordered_set<int>>;

double GraphRecall(const AdjHashGraph& exact_graph, const AdjGraph& approximate_graph) {
    auto hits = parlay::delayed_tabulate(approximate_graph.size(), [&](size_t i) {
        const auto& exact_neighbors = exact_graph[i];
        const auto& neighbors = approximate_graph[i];
        int my_hits = 0;
        for (int v : neighbors) {
            if (exact_neighbors.contains(v)) {
                my_hits++;
            }
        }
        return my_hits;
    });
    return static_cast<double>(parlay::reduce(hits)) / (approximate_graph.size() * exact_graph[0].size());
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

    int max_degree = 100;
    int num_query_neighbors = 10;
    int num_clusters = 16;
    double epsilon = 0.05;
    std::vector<int> num_degree_values = { 100, 80, 50, 20, 10, 8, 5, 3 };

    std::vector<NNVec> ground_truth;
    if (!std::filesystem::exists(ground_truth_file)) {
        ground_truth = ComputeGroundTruth(points, queries, num_query_neighbors);
    } else {
        ground_truth = ReadGroundTruth(ground_truth_file);
        std::cout << "Read ground truth file" << std::endl;
    }

    Timer timer;
    timer.Start();
    AdjGraph exact_graph = BuildExactKNNGraph(points, max_degree);
    std::cout << "Building exact graph took " << timer.Stop() << std::endl;

    timer.Start();
    auto exact_graph_hash =
        parlay::map(exact_graph, [num_query_neighbors](const std::vector<int>& neighbors) {
            return std::unordered_set<int>(neighbors.begin(),
                neighbors.begin() + std::min<int>(num_query_neighbors, neighbors.size()));
        });
    std::cout << "Convert to hash took " << timer.Stop() << std::endl;

    auto graph_builders = InstantiateGraphBuilders();

    size_t total_num_configs = graph_builders.size() * num_degree_values.size();
    std::cout << "num configs " << total_num_configs << " num graph builders " << graph_builders.size() << std::endl;
    size_t num_gb_configs_processed = 0;
    SpinLock cout_lock;

    timer.Start();
    auto output_lines = parlay::map(graph_builders, [&](ApproximateKNNGraphBuilder& graph_builder) -> std::string {
        const AdjGraph approximate_graph = graph_builder.BuildApproximateNearestNeighborGraph(points, max_degree);
        cout_lock.lock();
        num_gb_configs_processed++;
        std::cout << "Num GB configs finished " << num_gb_configs_processed << " / " << graph_builders.size() << std::endl;
        cout_lock.unlock();

        auto outputs = parlay::map(num_degree_values, [&](int degree) {
            AdjGraph degree_constrained_graph(approximate_graph.size());
            for (size_t i = 0; i < approximate_graph.size(); ++i) {
                degree_constrained_graph[i] = std::vector<int>(approximate_graph[i].begin(),
                    approximate_graph[i].begin() + std::min<int>(approximate_graph[i].size(), degree));
            }

            double graph_recall = GraphRecall(exact_graph_hash, degree_constrained_graph);
            Partition partition =
                PartitionAdjListGraph(degree_constrained_graph, num_clusters, epsilon, std::min<int>(parlay::num_workers(), 1), true);
            double oracle_recall = FirstShardOracleRecall(ground_truth, partition, num_query_neighbors);
            return "approximate," + FormatOutput(graph_builder, oracle_recall, graph_recall, degree);
        }, 1);


        std::stringstream stream;
        for (const std::string& o : outputs) stream << o << "\n";
        return stream.str();
    }, 1);
    std::cout << "All Approx builders took " << timer.Stop() << std::endl;

    auto exact_outputs = parlay::map(num_degree_values, [&](int degree) {
        AdjGraph degree_constrained_graph(exact_graph.size());
        for (size_t i = 0; i < exact_graph.size(); ++i) {
            degree_constrained_graph[i] =
                std::vector<int>(exact_graph[i].begin(), exact_graph[i].begin() + std::min<int>(exact_graph[i].size(), degree));
        }
        double graph_recall = GraphRecall(exact_graph_hash, degree_constrained_graph);
        Partition partition =
                PartitionAdjListGraph(degree_constrained_graph, num_clusters, epsilon, std::min<int>(parlay::num_workers(), 1), true);
        double oracle_recall = FirstShardOracleRecall(ground_truth, partition, num_query_neighbors);
        ApproximateKNNGraphBuilder gb;
        gb.FANOUT = 0;
        gb.REPETITIONS = 1;
        gb.MAX_CLUSTER_SIZE = exact_graph.size();
        return "exact," + FormatOutput(gb, oracle_recall, graph_recall, degree) + "\n";
    });

    output_lines.append(exact_outputs);

    std::ofstream out(output_file);
    out << Header() << "\n";
    std::cout << Header() << std::endl;
    for (const std::string& outputs : output_lines) {
        out << outputs;
        std::cout << outputs << std::flush;
    }
    out << std::flush;
}
