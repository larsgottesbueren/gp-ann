#pragma once

#include "kmeans.h"
#include "knn_graph.h"
#include "metis_io.h"

#include <kaminpar-shm/kaminpar.h>

#include "../external/hnswlib/hnswlib/hnswlib.h"
#include "hnsw_router.h"

std::vector<int> RecursiveKMeansPartitioning(PointSet& points, size_t max_cluster_size, int depth = 0, int num_clusters = -1) {
    if (num_clusters < 0) {
        num_clusters = static_cast<int>(ceil(double(points.n) / max_cluster_size));
    }
    if (num_clusters == 0) {
        return std::vector<int>(points.n, 0);
    }
    PointSet centroids = RandomSample(points, num_clusters, 555);

    Timer timer; timer.Start();
    std::vector<int> partition = KMeans(points, centroids);
    std::cout << "k-means at depth " << depth << " took " << timer.Stop() << " s" << std::endl;

    num_clusters = *std::max_element(partition.begin(), partition.end()) + 1;

    std::vector<size_t> cluster_sizes(num_clusters, 0);
    for (int part_id : partition) cluster_sizes[part_id]++;

    int num_overloaded_clusters = 0;
    for (int part_id = 0; part_id < num_clusters; ++part_id) {
        if (cluster_sizes[part_id] > max_cluster_size) num_overloaded_clusters++;
    }

    if (num_overloaded_clusters > 0) {
        std::cout << "At depth " << depth << " there are " << num_overloaded_clusters << " / " << num_clusters << " too heavy clusters. Refine them" << std::endl;
    }

    int next_part_id = num_clusters;
    for (int part_id = 0; part_id < int(cluster_sizes.size()); ++part_id) {
        if (cluster_sizes[part_id] > max_cluster_size) {
            std::cout << "Cluster " << part_id << " / " << num_clusters << " at depth " << depth << " is overloaded " << cluster_sizes[part_id] << " / " << max_cluster_size << std::endl;

            // Determine nodes in the cluster (could do it for all clusters at once, be we assume that this happens for 1-2 clusters --> this is faster and uses less memory)
            std::vector<uint32_t> cluster;
            for (uint32_t point_id = 0; point_id < partition.size(); ++point_id) {
                if (partition[point_id] == part_id) {
                    cluster.push_back(point_id);
                }
            }

            // Set up the point subset of the cluster
            PointSet cluster_point_set = ExtractPointsInBucket(cluster, points);

            // Partition recursively
            std::vector<int> sub_partition = RecursiveKMeansPartitioning(cluster_point_set, max_cluster_size, depth + 1);

            // Translate partition IDs
            int max_sub_part_id = *std::max_element(sub_partition.begin(), sub_partition.end());
            std::cout << "Cluster " << part_id << " / " << num_clusters << " at depth " << depth << " was overloaded " << cluster_sizes[part_id] << " / " << max_cluster_size
                      << " and got split into " << max_sub_part_id + 1 << " sub-clusters" << std::endl;
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
    return RecursiveKMeansPartitioning(points, max_cluster_size, 0, num_clusters);
}

struct CSR {
    CSR() : xadj(1, 0) {}
    std::vector<kaminpar::shm::EdgeID> xadj;
    std::vector<kaminpar::shm::NodeID> adjncy;
    std::vector<kaminpar::shm::NodeWeight> node_weights;
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

std::vector<int> PartitionGraphWithKaMinPar(CSR& graph, int k, double epsilon) {
    size_t num_nodes = graph.xadj.size() - 1;
    std::vector<kaminpar::shm::BlockID> kaminpar_partition(num_nodes, -1);
    auto context = kaminpar::shm::create_default_context();
    context.partition.epsilon = epsilon;
    kaminpar::KaMinPar shm(std::min<size_t>(32, std::thread::hardware_concurrency()), context);
    shm.take_graph(num_nodes, graph.xadj.data(), graph.adjncy.data(),
                   /* vwgt = */ graph.node_weights.empty() ? nullptr : graph.node_weights.data(),
                   /* adjwgt = */ nullptr);
    shm.compute_partition(555, k, kaminpar_partition.data());
    std::vector<int> partition(num_nodes);
    for (size_t i = 0; i < partition.size(); ++i) partition[i] = kaminpar_partition[i];     // convert unsigned int partition ID to signed int partition ID
    return partition;
}

std::vector<int> GraphPartitioning(PointSet& points, int num_clusters, double epsilon, const std::string& graph_output_path = "") {
    ApproximateKNNGraphBuilder graph_builder;
    Timer timer; timer.Start();
    AdjGraph knn_graph = graph_builder.BuildApproximateNearestNeighborGraph(points, 10);
    std::cout << "Built KNN graph. Took " << timer.Restart() << std::endl;
    points.Drop();
    std::cout << "Dealloc took " << timer.Restart() << std::endl;
    Symmetrize(knn_graph);
    std::cout << "symmetrize took " << timer.Restart() << std::endl;
    if (!graph_output_path.empty()) {
        std::cout << "Writing knn graph file to " << graph_output_path << std::endl;
        WriteMetisGraph(graph_output_path, knn_graph);
        std::cout << "Writing graph file took " << timer.Restart() << std::endl;
    }
    CSR csr = ConvertAdjGraphToCSR(knn_graph);
    std::cout << "Converting graph to CSR took " << timer.Restart() << std::endl;
    knn_graph.clear();
    knn_graph.shrink_to_fit();
    return PartitionGraphWithKaMinPar(csr, num_clusters, epsilon);
}

std::vector<int> PyramidPartitioning(PointSet& points, int num_clusters, double epsilon, const std::string& routing_index_path="") {
    // Subsample points
    size_t num_subsample_points = 10000000;          // reasonable value. didn't make much difference
    PointSet subsample_points = RandomSample(points, num_subsample_points, 555);

    // Aggregate via k-means
    const size_t num_aggregate_points = 10000;      // from the paper
    PointSet aggregate_points = RandomSample(subsample_points, num_aggregate_points, 555);
    std::vector<int> subsample_partition = KMeans(subsample_points, aggregate_points);

    if (!routing_index_path.empty()) {
        #ifdef MIPS_DISTANCE
        using SpaceType = hnswlib::InnerProductSpace;
        #else
        using SpaceType = hnswlib::L2Space;
        #endif
        SpaceType space(points.d);
        HNSWParameters hnsw_parameters;
        hnswlib::HierarchicalNSW<float> hnsw(&space, aggregate_points.n, hnsw_parameters.M, hnsw_parameters.ef_construction, 555);
        parlay::parallel_for(0, aggregate_points.n, [&](size_t i) { hnsw.addPoint(aggregate_points.GetPoint(i), i); }, 512);
        hnsw.saveIndex(routing_index_path);
    }

    // Build kNN graph
    ApproximateKNNGraphBuilder graph_builder;
    AdjGraph knn_graph = graph_builder.BuildApproximateNearestNeighborGraph(aggregate_points, 10);
    Symmetrize(knn_graph);
    CSR csr = ConvertAdjGraphToCSR(knn_graph);

    // assign node weights
    csr.node_weights.resize(num_aggregate_points, 0);
    for (int subsample_part_id : subsample_partition) csr.node_weights[subsample_part_id]++;

    // partition
    std::vector<int> aggregate_partition = PartitionGraphWithKaMinPar(csr, num_clusters, epsilon);

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
