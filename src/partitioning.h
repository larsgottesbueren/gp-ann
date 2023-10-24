#pragma once

#include "kmeans.h"
#include "knn_graph.h"
#include "metis_io.h"

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
    std::vector<std::vector<int>> results;
    for (int k : ks) {
        std::vector<kaminpar::shm::BlockID> kaminpar_partition(num_nodes, -1);
        auto context = kaminpar::shm::create_default_context();
        context.partition.epsilon = epsilon;
        kaminpar::KaMinPar shm(std::min<size_t>(32, std::thread::hardware_concurrency()), context);
        shm.take_graph(num_nodes, graph.xadj.data(), graph.adjncy.data(), /* vwgt = */ nullptr, /* adjwgt = */ nullptr);
        shm.compute_partition(555, k, kaminpar_partition.data());
        std::vector<int> partition(num_nodes);
        for (size_t i = 0; i < partition.size(); ++i) partition[i] = kaminpar_partition[i];     // convert unsigned int partition ID to signed int partition ID
        results.emplace_back(std::move(partition));
    }
    return results;
}

std::vector<std::vector<int>> GraphPartitioning(PointSet& points, std::vector<int>& num_clusters, double epsilon, const std::string& graph_output_path = "") {
    ApproximateKNNGraphBuilder graph_builder;
    AdjGraph knn_graph = graph_builder.BuildApproximateNearestNeighborGraph(points, 10);
    std::cout << "Built KNN graph" << std::endl;
    Symmetrize(knn_graph);
    if (!graph_output_path.empty()) {
        std::cout << "Writing knn graph file to " << graph_output_path << std::endl;
        WriteMetisGraph(graph_output_path, knn_graph);
    }
    CSR csr = ConvertAdjGraphToCSR(knn_graph);
    std::cout << "Symmetrized and converted graph" << std::endl;
    knn_graph.clear();
    knn_graph.shrink_to_fit();
    return PartitionGraphWithKaMinPar(csr, num_clusters, epsilon);
}

std::vector<int> PyramidPartitioning(PointSet& points, int num_clusters, double epsilon) {
    // Subsample points
    size_t num_subsample_points = 10000000;          // reasonable value. didn't make much difference
    PointSet subsample_points = RandomSample(points, num_subsample_points, 555);

    // Aggregate via k-means
    const size_t num_aggregate_points = 10000;      // from the paper
    PointSet aggregate_points = RandomSample(subsample_points, num_aggregate_points, 555);
    KMeans(subsample_points, aggregate_points);

    // Build kNN graph and partition
    std::vector<int> num_clusters_vec = { num_clusters };
    std::vector<int> aggregate_partition = GraphPartitioning(aggregate_points, num_clusters_vec, 0.05)[0];

    // Assign points to the partition of the closest point in the aggregate set
    // Fix balance by assigning to the second closest etc. if the first choice is overloaded
    size_t max_points_in_cluster = points.n * (1+epsilon) / num_clusters;
    std::vector<size_t> num_points_in_cluster(num_clusters, 0);
    std::vector<int> partition(points.n);

    SpinLock unfinished_points_lock;
    std::vector<uint32_t> unfinished_points;

    size_t num_leaders = 5;
    auto assign_point = [&](size_t i) {
        auto closest_leaders_top_k = ClosestLeaders(points, aggregate_points, i, num_leaders);
        auto closest_leaders = ConvertTopKToNNVec(closest_leaders_top_k);

        for (const auto& [dist, leader_id] : closest_leaders) {
            const int part = aggregate_partition[leader_id];
            if (num_points_in_cluster[part] < max_points_in_cluster) {
                __atomic_fetch_add(&num_points_in_cluster[part], 1, __ATOMIC_RELAXED);
                partition[i] = part;
                return; // from lambda
            }
        }

        // haven't found a candidate here --> go again in another round
        unfinished_points_lock.lock();
        unfinished_points.push_back(i);
        unfinished_points_lock.unlock();
    };

    parlay::parallel_for(0, points.n, assign_point);

    std::cout << "Main Pyramid assignment round finished. " << unfinished_points.size() << " still unassigned" << std::endl;
    size_t deadzone = (1-epsilon) * points.n / num_clusters;

    size_t num_extra_rounds = 0;
    while (!unfinished_points.empty()) {
        num_leaders = 1;    // switch to only picking the top choice

        // now we have to remove the points in aggregated_points associated with overloaded blocks
        std::vector<uint32_t> aggr_points_to_keep;
        std::vector<int> new_aggr_partition;
        for (uint32_t i = 0; i < aggregate_partition.size(); ++i) {
            if (num_points_in_cluster[aggregate_partition[i]] <= deadzone) {
                aggr_points_to_keep.push_back(i);
                new_aggr_partition.push_back(aggregate_partition[i]);
            }
        }
        aggregate_points = ExtractPointsInBucket(aggr_points_to_keep, aggregate_points);
        aggregate_partition = std::move(new_aggr_partition);

        std::vector<uint32_t> frontier;
        std::swap(unfinished_points, frontier);
        parlay::parallel_for(0, frontier.size(), [&](size_t i) { assign_point(frontier[i]); });
        std::cout << "Extra Pyramid assignment round " << ++num_extra_rounds << " finished. " << unfinished_points.size() << " still unassigned" << std::endl;
    }

    return partition;
}
