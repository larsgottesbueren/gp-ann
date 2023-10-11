#pragma once

#include "kmeans.h"
#include "knn_graph.h"

#include <kaminpar-shm/kaminpar.h>

std::vector<int> RecursiveKMeansPartitioning(PointSet& points, size_t max_cluster_size, int num_clusters = -1) {
    if (num_clusters < 0) {
        num_clusters = static_cast<int>(points.n / max_cluster_size);
    }
    if (num_clusters == 0) {
        return std::vector<int>(points.n, 0);
    }
    PointSet centroids = RandomSample(points, num_clusters, 555);
    std::vector<int> partition = KMeans(points, centroids);

    std::vector<size_t> cluster_sizes(num_clusters, 0);
    for (int part_id : partition) cluster_sizes[part_id]++;

    int next_part_id = num_clusters;
    for (int part_id = 0; part_id < int(cluster_sizes.size()); ++part_id) {
        if (cluster_sizes[part_id] > max_cluster_size) {
            // Determine nodes in the cluster (could do it for all clusters at once, be we assume that this happens for 1-2 clusters --> this is faster and uses less memory)
            std::vector<uint32_t> cluster;
            for (uint32_t point_id = 0; point_id < partition.size(); ++point_id) {
                if (partition[point_id] == part_id) {
                    cluster.push_back(point_id);
                }
            }

            // Set up the point subset of the cluster
            PointSet cluster_point_set;
            cluster_point_set.d = points.d;
            cluster_point_set.n = cluster.size();
            for (uint32_t point_id : cluster) {
                float* P = points.GetPoint(point_id);
                for (size_t d = 0; d < points.d; ++d) {
                    cluster_point_set.coordinates.push_back(P[d]);
                }
            }

            // Partition recursively
            std::vector<int> sub_partition = RecursiveKMeansPartitioning(cluster_point_set, max_cluster_size);

            // Translate partition IDs
            int max_sub_part_id = *std::max_element(sub_partition.begin(), sub_partition.end());
            for (uint32_t sub_point_id = 0; sub_point_id < cluster.size(); ++sub_point_id) {
                uint32_t point_id = cluster[sub_point_id];
                partition[point_id] = next_part_id + sub_partition[sub_point_id];
            }

            next_part_id += max_sub_part_id + 1;
        }
    }

    return partition;
}

std::vector<int> RecursiveKMeansPartitioning(PointSet& points, int num_clusters, double epsilon) {
    size_t max_cluster_size = points.n * (1+epsilon) / num_clusters;
    return RecursiveKMeansPartitioning(points, max_cluster_size, num_clusters);
}

struct CSR {
    CSR() : xadj(1, 0) {}
    std::vector<kaminpar::shm::EdgeID> xadj;
    std::vector<kaminpar::shm::NodeID> adjncy;
};

CSR ConvertAdjGraphToCSR(const AdjGraph& graph) {
    CSR csr;
    size_t num_edges = 0; for (const auto& n : graph) num_edges += n.size();
    csr.xadj.reserve(graph.size() + 1); csr.adjncy.reserve(num_edges);
    for (const auto& n : graph) {
        for (const int neighbor : n) {
            csr.adjncy.push_back(neighbor);
        }
        csr.xadj.push_back(csr.adjncy.size());
    }
    return csr;
}

std::vector<std::vector<int>> PartitionGraphWithKaMinPar(CSR& graph, std::vector<int>& ks, double epsilon) {
    size_t num_nodes = graph.xadj.size() - 1;
    std::vector<kaminpar::shm::BlockID> kaminpar_partition(num_nodes, -1);
    auto context = kaminpar::shm::create_default_context();
    context.partition.epsilon = epsilon;
    kaminpar::KaMinPar shm(std::thread::hardware_concurrency(), context);
    shm.take_graph(num_nodes, graph.xadj.data(), graph.adjncy.data(), /* vwgt = */ nullptr, /* adjwgt = */ nullptr);
    std::vector<std::vector<int>> results;
    for (int k : ks) {
        shm.compute_partition(555, k, kaminpar_partition.data());
        std::vector<int> partition(num_nodes);
        for (size_t i = 0; i < partition.size(); ++i) partition[i] = kaminpar_partition[i];     // convert unsigned int partition ID to signed int partition ID
        results.emplace_back(std::move(partition));
    }
    return results;
}

std::vector<std::vector<int>> GraphPartitioning(PointSet& points, std::vector<int>& num_clusters, double epsilon) {
    ApproximateKNNGraphBuilder graph_builder;
    AdjGraph knn_graph = graph_builder.BuildApproximateNearestNeighborGraph(points, 10);
    points.Drop();
    std::cout << "Built KNN graph" << std::endl;
    Symmetrize(knn_graph);
    CSR csr = ConvertAdjGraphToCSR(knn_graph);
    std::cout << "Symmetrized and converted graph" << std::endl;
    return PartitionGraphWithKaMinPar(csr, num_clusters, epsilon);
}

std::vector<int> PyramidPartitioning(PointSet& points, int num_clusters, double epsilon) {
    // Subsample points

    // Aggregate via k-means

    // Build kNN graph and partition

    // Assign points to the partition of the closest point in the aggregate set
    // Fix balance by assigning to the second closest etc.

    return { };
}
