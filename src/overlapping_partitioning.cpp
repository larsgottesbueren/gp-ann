#include "overlapping_partitioning.h"

#include <fstream>
#include <filesystem>

#include "kmeans_tree_router.h"

#include "partitioning.h"
#include "knn_graph.h"

#include <parlay/primitives.h>

#include <parlay/worker_specific.h>

template<typename RatingType>
struct RatingMap {
    explicit RatingMap(int num_clusters) : ratings(num_clusters, 0) { }
    std::vector<RatingType> ratings;
    std::vector<int> slots;
};

// sequential histogram by index and select (or reduce by index) with pre-allocated histogram
template<typename NeighborRange>
std::pair<int, int> TopMove(uint32_t u, const NeighborRange& neighbors, const Cover& cover, const Partition& partition,
    RatingMap<int>& rating_map, const parlay::sequence<int>& cluster_sizes, int max_cluster_size) {
    for (auto v : neighbors) {
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

Clusters OverlappingGraphPartitioning(PointSet& points, int num_clusters, double epsilon, double overlap, bool strong) {
    const size_t max_cluster_size = (1.0 + epsilon) * points.n / num_clusters;
    const size_t num_extra_assignments = overlap * points.n;
    // previously const size_t num_extra_assignments = (1.0 + epsilon) * n * (1.0 + overlap) - n
    size_t num_assignments_remaining = num_extra_assignments;
    const size_t num_total_assignments = points.n + num_extra_assignments;
    num_clusters = std::ceil(static_cast<double>(num_total_assignments) / max_cluster_size);

    std::cout << "max cluster size " << max_cluster_size << " num clusters " << num_clusters << " eps " << epsilon << " overlap " << overlap << std::endl;
    Timer timer;
    ApproximateKNNGraphBuilder graph_builder;
    if (strong) {
        graph_builder.FANOUT = 5;
        graph_builder.REPETITIONS = 5;
    }
    timer.Start();
    static constexpr int degree = 10;
    AdjGraph knn_graph = graph_builder.BuildApproximateNearestNeighborGraph(points, degree);
    std::cout << "Built KNN graph. Took " << timer.Stop() << std::endl;

    Partition partition = PartitionAdjListGraph(knn_graph, num_clusters, epsilon, std::min<int>(32, parlay::num_workers()), strong, false);
    Cover cover = ConvertPartitionToCover(partition);
    Clusters clusters = ConvertPartitionToClusters(partition);

    timer.Start();

    // the current implementation gives
    // extra_assignments = L_max * (k'-k) = L_max * k * overlap = (1+eps) * n/k * k * overlap = (1+eps)*n*overlap
    // instead of the expected 1 * n * overlap. that's fine, we just have to give the other method the same amount
    auto cluster_sizes = parlay::histogram_by_index(partition, num_clusters);

    parlay::WorkerSpecific<RatingMap<int>> rating_map_ets([&]() { return RatingMap<int>(num_clusters); });

    auto nodes = parlay::iota<uint32_t>(points.n);

    std::cout << "finished partitioning. start big loop" << std::endl;

    size_t reduction = 0;
    int iter = 0;
    while (num_assignments_remaining > 0) {
        auto best_moves = parlay::map(nodes, [&](uint32_t u) {
            auto& rating_map = rating_map_ets.get();
            // return TopMove(u, transpose[u], cover, partition, rating_map, cluster_sizes, max_cluster_size);
            return TopMove(u, knn_graph[u], cover, partition, rating_map, cluster_sizes, max_cluster_size);
        });
        auto affinities = parlay::delayed_map(best_moves, [&](const auto& l) { return l.second; });
        int best_affinity = parlay::reduce(affinities, parlay::maxm<int>());
        std::cout << "iter " << ++iter << " best affinity " << best_affinity << std::endl;
        if (best_affinity == 0) {
            break;
        }
        // if (best_affinity <= 2) { break; }
        auto top_gain_nodes = parlay::filter(nodes, [&](uint32_t u) { return best_moves[u].second == best_affinity; });
        auto nodes_and_targets = parlay::delayed_map(top_gain_nodes, [&](uint32_t u) { return std::make_pair(best_moves[u].first, u); });
        auto moves_into_cluster = parlay::group_by_index(nodes_and_targets, num_clusters);
        auto num_moves_into_cluster = parlay::tabulate(clusters.size(), [&](size_t cluster_id) {
            return std::min(max_cluster_size - cluster_sizes[cluster_id], moves_into_cluster[cluster_id].size());
        });
        size_t total_num_moves = parlay::reduce(num_moves_into_cluster);
        for (int cluster_id = 0; cluster_id < num_clusters; ++cluster_id) {
            num_moves_into_cluster[cluster_id] = std::min(num_assignments_remaining, num_moves_into_cluster[cluster_id]);
            num_assignments_remaining -= num_moves_into_cluster[cluster_id];
        }
        total_num_moves = parlay::reduce(num_moves_into_cluster);
        reduction += total_num_moves * std::min(best_affinity, degree);

        std::cout << "total num moves this round " << total_num_moves << std::endl;

        parlay::parallel_for(0, clusters.size(), [&](size_t cluster_id) {
            size_t num_moves = num_moves_into_cluster[cluster_id];
            cluster_sizes[cluster_id] += num_moves;
            parlay::parallel_for(0, num_moves, [&](size_t j) {
                uint32_t u = moves_into_cluster[cluster_id][j];
                cover[u].push_back(cluster_id);
            });
            clusters[cluster_id].insert(
                clusters[cluster_id].end(), moves_into_cluster[cluster_id].begin(),
                moves_into_cluster[cluster_id].begin() + num_moves);
        }, 1);
    }

    std::cout << "Finished loop. Total reduction " << reduction << " Num assignments remaining " << num_assignments_remaining << std::endl;
    #if false
    if (num_assignments_remaining > 0) {
        std::cout << "Fill up with distance-based overlap method" << std::endl;
        MakeOverlappingWithCentroids(points, clusters, max_cluster_size, num_assignments_remaining);
    }
    #endif

    std::cout << "Make GP overlapping took " <<  timer.Stop() << " seconds" << std::endl;
    return clusters;
}

void MakeOverlappingWithCentroids(PointSet& points, Clusters& clusters, size_t max_cluster_size, size_t num_extra_assignments) {
    auto it = std::remove_if(clusters.begin(), clusters.end(), [&](const auto& cluster) { return cluster.empty(); });
    clusters.erase(it, clusters.end());

    Cover cover = ConvertClustersToCover(clusters);

    Timer timer; timer.Start();
    // Step 1 build centroids and associations
    size_t num_centroids = 64;
    KMeansTreeRouterOptions kmtr_options {
        .num_centroids = num_centroids, .min_cluster_size = 350,
        .budget = static_cast<int64_t>(clusters.size() * num_centroids), .search_budget = 0};
    KMeansTreeRouter kmtr;
    kmtr.Train(points, clusters, kmtr_options);
    auto [sub_points, sub_part] = kmtr.ExtractPoints();
    std::cout << "Got reps. " << sub_points.n << " " << sub_part.size() << " Took " << timer.Restart() << " s" << std::endl;

    // Step 2 search closest centroid that is not in the own partition
    auto point_ids = parlay::iota<uint32_t>(points.n);

    struct Rating {
        float dist;
        int target_cluster;
        uint32_t point_id;
        bool operator<(const Rating& other) const { return dist < other.dist; }
    };

    auto cluster_rankings_per_point = parlay::map(point_ids, [&](uint32_t u) -> std::vector<Rating> {
        std::vector<float> min_dist(clusters.size(), std::numeric_limits<float>::max());
        // brute-force so we can more easily support the filter
        for (size_t j = 0; j < sub_points.n; ++j) {
            float dist = distance(points.GetPoint(u), sub_points.GetPoint(j), points.d);
            const int pv = sub_part[j];
            if (dist < min_dist[pv]) {
                min_dist[pv] = dist;
            }
        }

        // To prevent duplicate assignments, we set their distance to infinity, and filter the target list down
        for (int c : cover[u]) {
            min_dist[c] = std::numeric_limits<float>::max();
        }

        std::vector<int> targets;
        for (size_t c = 0; c < clusters.size(); ++c) {
            if (min_dist[c] != std::numeric_limits<float>::max() && clusters[c].size() < max_cluster_size) {
                targets.push_back(c);
            }
        }

        size_t num_keep = 5;
        std::sort(targets.begin(), targets.end(), [&](int l, int r) { return min_dist[l] < min_dist[r]; });
        targets.resize(std::min(targets.size(), num_keep));
        std::vector<Rating> ratings;
        for (int t : targets) {
            ratings.push_back(Rating{ .dist = min_dist[t], .target_cluster = t, .point_id = u });
        }
        return ratings;
    });

    std::cout  << "got cluster rankings. Took " << timer.Restart() << std::endl;

    auto cluster_rankings = parlay::flatten(cluster_rankings_per_point);

    parlay::sort_inplace(cluster_rankings);

    std::cout << "Flatten and sort took " << timer.Restart() << std::endl;

    size_t num_assignments_left = num_extra_assignments;
    size_t steps = 0;
    for (const Rating& r : cluster_rankings) {
        if (clusters[r.target_cluster].size() < max_cluster_size) {
            --num_assignments_left;
            clusters[r.target_cluster].push_back(r.point_id);
        }
        ++steps;
        if (num_assignments_left == 0) {
            break;
        }
    }

    std::cout << "Finished kmeans overlap partitioning. " << num_assignments_left << " possible assignments unused. Moves inspected: "
                << steps << " Time " << timer.Stop() << std::endl;
    std::cout << "Total time for overlap " << timer.total_duration.count() << " seconds" << std::endl;
}

Clusters OverlappingKMeansPartitioningSPANN(PointSet& points, const Partition& partition, int requested_num_clusters, double epsilon, double overlap) {
    const size_t num_extra_assignments = overlap * points.n;
    const size_t max_cluster_size = (1.0 + epsilon) * points.n / requested_num_clusters;
    Clusters clusters = ConvertPartitionToClusters(partition);
    MakeOverlappingWithCentroids(points, clusters, max_cluster_size, num_extra_assignments);
    return clusters;
}
