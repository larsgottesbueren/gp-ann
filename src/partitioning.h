#pragma once

#include "kmeans.h"
#include "knn_graph.h"
#include "metis_io.h"

#include <kaminpar-shm/kaminpar.h>

#include "../external/hnswlib/hnswlib/hnswlib.h"
#include "hnsw_router.h"

#include <parlay/primitives.h>

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

    WriteMetisPartition(aggregate_partition, routing_index_path + ".routing_index_partition");

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

    size_t num_extra_rounds = 0;
    while (!unfinished_points.empty()) {
        num_leaders = 1;    // switch to only picking the top choice

        // now we have to remove the points in aggregated_points associated with overloaded blocks
        std::vector<uint32_t> aggr_points_to_keep;
        std::vector<int> new_aggr_partition;
        for (uint32_t i = 0; i < aggregate_partition.size(); ++i) {
            if (num_points_in_cluster[aggregate_partition[i]] < max_points_in_cluster) {
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

// want to extract only the leaf-level points here
// and the mapping of top-level points to leaf-level points
std::pair<std::vector<int>, PointSet>
HierarchicalKMeansParlayImpl(PointSet& points, double coarsening_ratio, int depth = 0) {
    int num_level_centroids = points.n * coarsening_ratio;
    if (num_level_centroids < 1) {
        num_level_centroids = 1;
    }
    bool finished = true;
    constexpr int MAX_LEVEL_CENTROIDS = 64;
    if (num_level_centroids > MAX_LEVEL_CENTROIDS) {
        num_level_centroids = MAX_LEVEL_CENTROIDS;
        finished = false;
    }

    Timer timer; timer.Start();
    PointSet level_centroids = RandomSample(points, num_level_centroids, 555);
    std::vector<int> level_partition = KMeans(points, level_centroids);
    double t = timer.Stop();
    if (depth < 2) {
        std::cout   << "KMeans on " << points.n << " points at depth " << depth << " with "
                    << level_centroids.n << " / " << num_level_centroids << " centroids took " << t << " s." << std::endl;
    }

    if (level_centroids.n == 1) {
        // also stop if we get down to a single centroid. this means we can't split the data any more
        // (for example near-duplicates)
        finished = true;
    }

    if (finished) {     // this is weird. it will always aggregate something, even if points is small...
        return std::make_pair(level_partition, level_centroids);
    }

    auto clusters = ConvertPartitionToBuckets(level_partition);

    auto recursion_results = parlay::map(clusters, [&](const auto& cluster) {
        PointSet cluster_points = ExtractPointsInBucket(cluster, points);
        return HierarchicalKMeansParlayImpl(cluster_points, coarsening_ratio, depth+1);
    }, depth < 2 ? clusters.size() : 1);

    auto part_id_offsets = parlay::map(recursion_results, [&](const auto& r) { return NumPartsInPartition(r.first); });
    size_t num_recursive_clusters = parlay::scan_inplace(part_id_offsets);

    PointSet centroids_from_recursion;
    centroids_from_recursion.d = points.d;
    auto point_offsets = parlay::map(recursion_results, [&](const auto& r) { return r.second.n; });
    centroids_from_recursion.n = parlay::scan_inplace(point_offsets);
    centroids_from_recursion.Alloc();

    if (centroids_from_recursion.n != num_recursive_clusters) {
        throw std::runtime_error("Num centroids from recursion != num recursive clusters");
    }

    parlay::for_each(
            parlay::zip(clusters, recursion_results, point_offsets, part_id_offsets),
            [&](const auto& z1) {
                const auto& [cluster, recursive_partition_and_points, point_offset, part_id_offset] = z1;

                // remap part IDs
                const auto& recursive_partition = recursive_partition_and_points.first;
                parlay::for_each(
                        parlay::zip(recursive_partition, cluster),
                        [&](const auto& z2) {
                            const auto& [rec_part_id, global_point_id] = z2;
                            level_partition[global_point_id] = rec_part_id + part_id_offset;
                        }
                );

                // merge points
                const auto& rec_points = recursive_partition_and_points.second;
                std::memcpy(centroids_from_recursion.GetPoint(point_offset), rec_points.coordinates.data(), rec_points.coordinates.size() * sizeof(float));
            }
    );

    return std::make_pair(level_partition, centroids_from_recursion);
}

std::pair<std::vector<int>, PointSet>
HierarchicalKMeans(PointSet& points, double coarsening_ratio, int depth = 0) {
    int num_level_centroids = points.n * coarsening_ratio;
    if (num_level_centroids < 1) {
        num_level_centroids = 1;
    }
    bool finished = true;
    constexpr int MAX_LEVEL_CENTROIDS = 64;
    if (num_level_centroids > MAX_LEVEL_CENTROIDS) {
        num_level_centroids = MAX_LEVEL_CENTROIDS;
        finished = false;
    }

    Timer timer; timer.Start();
    PointSet level_centroids = RandomSample(points, num_level_centroids, 555);
    std::vector<int> level_partition = KMeans(points, level_centroids);
    double t = timer.Stop();
    if (depth < 2) {
        std::cout   << "KMeans on " << points.n << " points at depth " << depth << " with "
                    << level_centroids.n << " / " << num_level_centroids << " centroids took " << t << " s." << std::endl;
    }

    if (level_centroids.n == 1) {
        // also stop if we get down to a single centroid. this means we can't split the data any more
        // (for example near-duplicates)
        finished = true;
    }

    if (finished) {     // this is weird. it will always aggregate something, even if points is small...
        return std::make_pair(level_partition, level_centroids);
    }

    auto clusters = ConvertPartitionToBuckets(level_partition);

    std::vector<std::pair<std::vector<int>, PointSet>> recursion_results(clusters.size());
    parlay::parallel_for(0, clusters.size(), [&](size_t i) {
        if (clusters[i].empty()) throw std::runtime_error("Cluster points empty. KMeans should remove empty cluster IDs");
        PointSet cluster_points = ExtractPointsInBucket(clusters[i], points);
        recursion_results[i] = HierarchicalKMeans(cluster_points, coarsening_ratio, depth+1);
    }, depth < 1 ? clusters.size() / 8 : 1);

    PointSet centroids_from_recursion;
    centroids_from_recursion.d = points.d;
    size_t num_rec_parts = 0;
    for (size_t i = 0; i < recursion_results.size(); ++i) {
        const auto& cluster = clusters[i];
        const auto& rec_part = recursion_results[i].first;
        auto& rec_points = recursion_results[i].second;

        for (const float coord : rec_points.coordinates) {
            centroids_from_recursion.coordinates.push_back(coord);
        }
        centroids_from_recursion.n += rec_points.n;
        if (centroids_from_recursion.n * centroids_from_recursion.d != centroids_from_recursion.coordinates.size()) {
            std::cout << i << " " << centroids_from_recursion.n << " " << centroids_from_recursion.d << " " << centroids_from_recursion.coordinates.size()
                        << " " << rec_points.n << " " << num_rec_parts << " " << depth << std::endl;
            throw std::runtime_error("Size of rec centroids is wrong");
        }

        for (size_t j = 0; j < cluster.size(); ++j) {
            level_partition[cluster[j]] = rec_part[j] + num_rec_parts;
        }
        num_rec_parts += NumPartsInPartition(rec_part);

        if (num_rec_parts != centroids_from_recursion.n) { throw std::runtime_error("Num rec parts doesnt match num centroids from recursion"); }
    }

    return std::make_pair(level_partition, centroids_from_recursion);
}

std::vector<int> OurPyramidPartitioning(PointSet& points, int num_clusters, double epsilon, std::vector<int>& second_partition,
                                        const std::string& routing_index_path, double coarsening_rate = 0.002) {
    std::cout << "Call OurPyramid with coarsening rate " << coarsening_rate << std::endl;
    Timer timer; timer.Start();
    auto [routing_clusters, routing_points] = HierarchicalKMeans(points, coarsening_rate);
    std::cout << "HierKMeans took " << timer.Restart() << std::endl;

    std::cout << "routing_clusters.size() = " << routing_clusters.size() << " num routing clusters = " << NumPartsInPartition(routing_clusters)
                << " num routing points = " << routing_points.n << std::endl;

    #ifdef MIPS_DISTANCE
    hnswlib::InnerProductSpace space(points.d);
    #else
    hnswlib::L2Space space(points.d);
    #endif
    HNSWParameters hnsw_parameters;
    hnswlib::HierarchicalNSW<float> hnsw(&space, routing_points.n, hnsw_parameters.M, hnsw_parameters.ef_construction, /* random seed = */ 500);
    parlay::parallel_for(0, routing_points.n, [&](size_t i) { hnsw.addPoint(routing_points.GetPoint(i), i); });
    std::cout << "Building HNSW took " << timer.Restart() << std::endl;
    hnsw.saveIndex(routing_index_path);

    AdjGraph hnsw_graph(routing_points.n);
    for (uint32_t i = 0; i < routing_points.n; ++i) {
        auto neighbors = hnsw.getConnectionsWithLock(i, 0);
        for (auto v : neighbors) {
            hnsw_graph[i].push_back(v);
        }
    }
    Symmetrize(hnsw_graph);
    CSR hnsw_csr = ConvertAdjGraphToCSR(hnsw_graph);
    std::cout << "Converting HNSW to AdjGraph + Symmetrize + AdjGraphToCsr took " << timer.Restart() << std::endl;

    ApproximateKNNGraphBuilder graph_builder;
    AdjGraph knn_graph = graph_builder.BuildApproximateNearestNeighborGraph(routing_points, 20);
    std::cout << "Build KNN graph took " << timer.Restart() << std::endl;
    Symmetrize(knn_graph);
    CSR knn_csr = ConvertAdjGraphToCSR(knn_graph);
    std::cout << "Symmetrize + Convert to CSR took " << timer.Restart() << std::endl;

    knn_csr.node_weights.resize(routing_points.n, 0);
    for (int cluster_id : routing_clusters) knn_csr.node_weights[cluster_id]++;
    hnsw_csr.node_weights = knn_csr.node_weights;

    std::vector<int> knn_partition = PartitionGraphWithKaMinPar(knn_csr, num_clusters, epsilon);
    std::vector<int> hnsw_partition = PartitionGraphWithKaMinPar(hnsw_csr, num_clusters, epsilon);

    WriteMetisPartition(knn_partition, routing_index_path + ".knn.routing_index_partition");
    WriteMetisPartition(hnsw_partition, routing_index_path + ".hnsw.routing_index_partition");

    // Project from coarse partition
    std::vector<int> full_knn_partition(points.n);
    std::vector<int> full_hnsw_partition(points.n);
    for (uint32_t i = 0; i < points.n; ++i) {
        full_knn_partition[i] = knn_partition[routing_clusters[i]];
        full_hnsw_partition[i] = hnsw_partition[routing_clusters[i]];
    }

    second_partition = std::move(full_hnsw_partition);

    return full_knn_partition;
}
