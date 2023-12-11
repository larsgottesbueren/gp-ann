#include "overlapping_partitioning.h"

#include "partitioning.h"
#include "knn_graph.h"

#include <parlay/primitives.h>


Clusters OverlappingGraphPartitioning(PointSet& points, int num_clusters, double epsilon, double overlap) {
    ApproximateKNNGraphBuilder graph_builder;
    Timer timer;
    timer.Start();
    AdjGraph knn_graph = graph_builder.BuildApproximateNearestNeighborGraph(points, 10);
    std::cout << "Built KNN graph. Took " << timer.Restart() << std::endl;

    const size_t max_cluster_size = (1.0 + epsilon) * points.n / num_clusters;
    const int adjusted_num_clusters = num_clusters * (1.0 + overlap);
    const double adjusted_epsilon = (max_cluster_size * adjusted_num_clusters / static_cast<double>(points.n)) - 1.0;

    Partition partition = PartitionAdjListGraph(knn_graph, adjusted_num_clusters, adjusted_epsilon);

    // TODO now make it overlapping


    return { };
}

Clusters OverlappingKMeansPartitioningSPANN(PointSet& points, int num_clusters, double epsilon, double overlap) {
    return { };
}
