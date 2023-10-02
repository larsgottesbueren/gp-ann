#pragma once

#include "kmeans.h"
#include "knn_graph.h"

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
    for (int part_id = 0; part_id < cluster_sizes.size(); ++part_id) {
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
                for (int d = 0; d < points.d; ++d) {
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

std::vector<int> GraphPartitioning(PointSet& points, int num_clusters, double epsilon) {
    ApproximateKNNGraphBuilder graph_builder;
    AdjGraph knn_graph = graph_builder.BuildApproximateNearestNeighborGraph(points, 10);
    Symmetrize(knn_graph);

    // TODO call kaminpar
    std::vector<int> partition;

    return partition;
}
