#include "metis_io.h"

#include <fstream>
#include <sstream>

std::vector<int> ReadMetisPartition(const std::string& path) {
    std::ifstream in(path);
    std::vector<int> partition;
    int part;
    while (in >> part) partition.push_back(part);

    if (!partition.empty()) {
        RemapPartitionIDs(partition);
    }
    return partition;
}

void WriteMetisPartition(const std::vector<int>& partition, const std::string& path) {
    std::ofstream out(path);
    for (int p : partition) {
        out << p << "\n";
    }
    out.close();
}

void WriteMetisGraph(const std::string& path, const AdjGraph& graph) {
    uint64_t num_edges = 0;
    for (const auto& n : graph) num_edges += n.size();
    if (num_edges % 2 != 0) throw std::runtime_error("Number of edges not even");
    num_edges /= 2;

    std::ofstream out(path);
    out << graph.size() << " " << num_edges << "\n";
    for (const auto& n : graph) {
        for (auto v : n) out << (v+1) << " ";
        out << "\n";
    }
}

Clusters ReadClusters(const std::string& path) {
    std::ifstream in(path);
    std::string line;
    Clusters clusters;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        std::vector<uint32_t> new_cluster;
        uint32_t id;
        while (iss >> id) {
            new_cluster.push_back(id);
        }
        clusters.emplace_back(std::move(new_cluster));
    }
    return clusters;
}

void WriteClusters(const Clusters& clusters, const std::string& path) {
    std::ofstream out(path);
    for (const auto& c : clusters) {
        for (const uint32_t id : c) {
            out << id << " ";
        }
        out << "\n";
    }
}
