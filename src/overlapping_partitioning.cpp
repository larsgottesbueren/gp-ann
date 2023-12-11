#include "overlapping_partitioning.h"

#include "partitioning.h"
#include "knn_graph.h"

#include <parlay/primitives.h>

#include <parlay/worker_specific.h>

struct RatingMap {
    RatingMap(int num_clusters) : ratings(num_clusters, 0) { }
    std::vector<int> ratings;
    std::vector<int> slots;
};

// sequential histogram by index and select (or reduce by index) with pre-allocated histogram
std::pair<int, int> TopMove(uint32_t u, const std::vector<int>& neighbors, const Cover& cover, const Partition& partition,
    RatingMap& rating_map, const parlay::sequence<int>& cluster_sizes, int max_cluster_size) {
    for (uint32_t v : neighbors) {
        int part_v = partition[v];
        if (rating_map.ratings[part_v] == 0) {
            rating_map.slots.push_back(part_v);
        }
        rating_map.ratings[part_v] += 1;
    }

    int best_part = -1;
    int best_affinity = 0;
    auto is_target_valid = [&](int target) { return std::find(cover[u].begin(), cover[u].end(), target) == cover[u].end(); };
    for (int part : rating_map.slots) {
        const int affinity = rating_map.ratings[part];
        rating_map.ratings[part] = 0;
        if (cluster_sizes[part] < max_cluster_size && affinity > best_affinity && is_target_valid(part)) {
            best_affinity = affinity;
            best_part = part;
        }
    }
    rating_map.slots.clear();

    return std::make_pair(best_part, best_affinity);
}

Clusters OverlappingGraphPartitioning(PointSet& points, int num_clusters, double epsilon, double overlap) {
    ApproximateKNNGraphBuilder graph_builder;
    Timer timer;
    timer.Start();
    AdjGraph knn_graph = graph_builder.BuildApproximateNearestNeighborGraph(points, 10);
    std::cout << "Built KNN graph. Took " << timer.Restart() << std::endl;

    const size_t max_cluster_size = (1.0 + epsilon) * points.n / num_clusters;
    num_clusters = num_clusters * (1.0 + overlap);
    epsilon = (max_cluster_size * num_clusters / static_cast<double>(points.n)) - 1.0;

    Partition partition = PartitionAdjListGraph(knn_graph, num_clusters, epsilon);
    Cover cover = ConvertPartitionToCover(partition);
    Clusters clusters = ConvertPartitionToClusters(partition);

    auto cluster_sizes = parlay::histogram_by_index(partition, num_clusters);

    parlay::WorkerSpecific<RatingMap> rating_map_ets([&]() { return RatingMap(num_clusters); });

    auto nodes = parlay::iota<uint32_t>(points.n);

    while (true) {
        auto best_moves = parlay::map(nodes, [&](uint32_t u) {
           auto& rating_map = rating_map_ets.get();
            return TopMove(u, knn_graph[u], cover, partition, rating_map, cluster_sizes, max_cluster_size);
        });

        auto affinities = parlay::delayed_map(best_moves, [&](const auto& l) { return l.second; });

        int best_affinity = parlay::reduce(affinities, parlay::maxm<int>());

        if (best_affinity == 0) {
            break;
        }

        auto top_gain_nodes = parlay::filter(nodes, [&](uint32_t u) { return best_moves[u].second == best_affinity; });

        auto nodes_and_targets = parlay::delayed_map(top_gain_nodes, [&](uint32_t u) { return std::make_pair(best_moves[u].first, u); });

        auto moves_into_cluster = parlay::group_by_index(nodes_and_targets, num_clusters);

        parlay::parallel_for(0, num_clusters, [&](size_t cluster_id) {
            size_t num_moves_left = std::min(max_cluster_size - cluster_sizes[cluster_id], moves_into_cluster[cluster_id].size());
            cluster_sizes[cluster_id] += num_moves_left;
            // apply the first 'num_moves_left' from moves_into_cluster[cluster_id]
            parlay::parallel_for(0, num_moves_left, [&](size_t j) {
                uint32_t u = moves_into_cluster[cluster_id][j];
                cover[u].push_back(cluster_id);
            });

            // not parallel with std::vector unfortunately
            clusters[cluster_id].insert(
                clusters[cluster_id].end(), moves_into_cluster[cluster_id].begin(),
                moves_into_cluster[cluster_id].begin() + num_moves_left);
        }, 1);
    }

    return clusters;
}

Clusters OverlappingKMeansPartitioningSPANN(PointSet& points, int num_clusters, double epsilon, double overlap) {
    return { };
}
