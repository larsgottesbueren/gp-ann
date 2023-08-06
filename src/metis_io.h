#pragma once

#include <fstream>

#include "defs.h"

std::vector<int> ReadMetisPartition(const std::string& path) {
    std::ifstream in(path);
    std::vector<int> partition;
    int part;
    while (in >> part) partition.push_back(part);
    return partition;
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
