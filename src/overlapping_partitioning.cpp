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
std::pair<int, int> TopMove(uint32_t u, const std::vector<int>& neighbors, const Cover& cover, const Partition& partition,
    RatingMap<int>& rating_map, const parlay::sequence<int>& cluster_sizes, int max_cluster_size) {
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

#if false
void WriteGraph(AdjGraph& graph, const std::string& path) {
    std::ofstream out(path);
    out << graph.size() << "\n";
    for (const auto& neigh : graph) {
        for (int v : neigh) out << v << " ";
        out << "\n";
    }
}

AdjGraph ReadGraph(const std::string& path) {
    std::cout << "read graph" << std::endl;
    std::ifstream in(path);
    int num_nodes;
    std::string line;
    {
        std::getline(in, line);
        std::istringstream iss(line);
        iss >> num_nodes;
    }
    AdjGraph graph(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        std::getline(in, line);
        std::istringstream iss(line);
        int v;
        while (iss >> v) graph[i].push_back(v);
    }
    return graph;
}
#endif

Clusters OverlappingGraphPartitioning(PointSet& points, int num_clusters, double epsilon, double overlap) {
    const size_t max_cluster_size = (1.0 + epsilon) * points.n / num_clusters;
    num_clusters = std::ceil(num_clusters * (1.0 + overlap));

    std::cout << "max cluster size " << max_cluster_size << " num clusters " << num_clusters << " eps " << epsilon << " overlap " << overlap << std::endl;
#if false
    std::string dummy_file = "tmp.graph";
    if (!std::filesystem::exists(dummy_file)) {
        ApproximateKNNGraphBuilder graph_builder;
        Timer timer;
        timer.Start();
        AdjGraph knn_graph = graph_builder.BuildApproximateNearestNeighborGraph(points, 10);
        std::cout << "Built KNN graph. Took " << timer.Restart() << std::endl;

        WriteGraph(knn_graph, dummy_file);
    }
    AdjGraph knn_graph = ReadGraph(dummy_file);
#else
    ApproximateKNNGraphBuilder graph_builder;
    Timer timer;
    timer.Start();
    AdjGraph knn_graph = graph_builder.BuildApproximateNearestNeighborGraph(points, 10);
    std::cout << "Built KNN graph. Took " << timer.Restart() << std::endl;
#endif

    Partition partition = PartitionAdjListGraph(knn_graph, num_clusters, epsilon);
    Cover cover = ConvertPartitionToCover(partition);
    Clusters clusters = ConvertPartitionToClusters(partition);

    // the current implementation gives
    // extra_assignments = L_max * (k'-k) = L_max * k * overlap = (1+eps) * n/k * k * overlap = (1+eps)*n*overlap
    // instead of the expected 1 * n * overlap. that's fine, we just have to give the other method the same amount
    auto cluster_sizes = parlay::histogram_by_index(partition, num_clusters);

    parlay::WorkerSpecific<RatingMap<int>> rating_map_ets([&]() { return RatingMap<int>(num_clusters); });

    auto nodes = parlay::iota<uint32_t>(points.n);

    std::cout << "finished partitioning. start big loop" << std::endl;

    size_t reduction = 0;
    int iter = 0;
    while (true) {
        auto best_moves = parlay::map(nodes, [&](uint32_t u) {
            auto& rating_map = rating_map_ets.get();
            return TopMove(u, knn_graph[u], cover, partition, rating_map, cluster_sizes, max_cluster_size);
        });

        auto affinities = parlay::delayed_map(best_moves, [&](const auto& l) { return l.second; });

        int best_affinity = parlay::reduce(affinities, parlay::maxm<int>());
        std::cout << "iter " << ++iter << " best affinity " << best_affinity << std::endl;

        if (best_affinity == 0) {
            break;
        }

        auto top_gain_nodes = parlay::filter(nodes, [&](uint32_t u) { return best_moves[u].second == best_affinity; });

        auto nodes_and_targets = parlay::delayed_map(top_gain_nodes, [&](uint32_t u) { return std::make_pair(best_moves[u].first, u); });

        auto moves_into_cluster = parlay::group_by_index(nodes_and_targets, num_clusters);

        std::cout << "total num moves " << nodes_and_targets.size() << " num moves into cluster ";
        for (size_t i = 0; i  < num_clusters; ++i) {
            std::cout << moves_into_cluster[i].size() << " ";
        }
        std::cout << std::endl;

        size_t reduction_this_iteration = 0;
        for (int cluster_id = 0; cluster_id < num_clusters; ++cluster_id) {
            size_t num_moves_left = std::min(max_cluster_size - cluster_sizes[cluster_id], moves_into_cluster[cluster_id].size());
            reduction_this_iteration += best_affinity * num_moves_left;
        }
        reduction += reduction_this_iteration;
        std::cout << "Reduction this iteration " << reduction_this_iteration << std::endl;

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

    std::cout << "Total reduction " << reduction << std::endl;

    return clusters;
}

Clusters OverlappingKMeansPartitioningSPANN(PointSet& points, const Partition& partition, int requested_num_clusters, double epsilon, double overlap) {
    const size_t n = points.n;
    // usually it would be n * overlap, but because the way the GP overlap is implemented, we can make it (1+eps) * as much
    const size_t num_extra_assignments = (1.0 + epsilon) * n * overlap;
    const size_t max_cluster_size = (1.0 + epsilon) * points.n / requested_num_clusters;

    Clusters clusters = ConvertPartitionToClusters(partition);

    auto cluster_sizes = parlay::map(clusters, [&](const auto& c) { return c.size(); });

    // Step 1 build centroids and associations
    KMeansTreeRouterOptions kmtr_options {.num_centroids = 32, .min_cluster_size = 350, .budget = 10000, .search_budget = 0};
    KMeansTreeRouter kmtr;
    kmtr.Train(points, clusters, kmtr_options);
    auto [sub_points, sub_part] = kmtr.ExtractPoints();


    // Step 2 search closest centroid that is not in the own partition
    auto point_ids = parlay::iota<uint32_t>(points.n);
    parlay::WorkerSpecific<RatingMap<float>> rating_map_ets([&]() {
        RatingMap<float> rm(clusters.size());
        rm.ratings.assign(clusters.size(), std::numeric_limits<float>::max());
        return rm;
    });

    parlay::sequence<float> closest(n);

    auto cluster_rankings = parlay::map(point_ids, [&](uint32_t u) -> std::vector<std::pair<float, int>> {
        auto& rating_map = rating_map_ets.get();
        // brute-force so we can more easily support the filter
        for (size_t j = 0; j < sub_points.n; ++j) {
            float dist = distance(points.GetPoint(u), sub_points.GetPoint(j), points.d);
            const int pv = sub_part[j];
            auto& r = rating_map.ratings[pv];
            if (r == std::numeric_limits<float>::max()) {
                rating_map.slots.push_back(pv);
            }
            r = std::min(r, dist);
        }
        std::vector<std::pair<float, int>> result;
        float min_dist = std::numeric_limits<float>::max();
        for (int target : rating_map.slots) {
            min_dist = std::min(min_dist, rating_map.ratings[target]);
            if (target != partition[u]) {
                result.emplace_back(rating_map.ratings[target], target);
            }
            rating_map.ratings[target] = std::numeric_limits<float>::max();
        }
        closest[u] = min_dist;
        std::sort(result.begin(), result.end(), std::greater<>());  // will do pop-back later --> place closest at the end
        return result;
    });

    while (true) {
        auto points_and_targets = parlay::map_maybe(point_ids, [&](uint32_t u) -> std::optional<std::pair<int, int>> {
            auto& ranking = cluster_rankings[u];
            while (!ranking.empty()) {
                int target = ranking.back().second;
                ranking.pop_back();
                if (cluster_sizes[target] < max_cluster_size) {
                    return std::make_pair(u, target);
                }
            }
            return std::nullopt;
        });

        if (points_and_targets.empty()) {
            break;
        }

        auto moves_into_cluster = parlay::group_by_index(points_and_targets, clusters.size());

        for (size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
            size_t num_moves_left = std::min(max_cluster_size - cluster_sizes[cluster_id], moves_into_cluster[cluster_id].size());
            num_moves_left = std::min(num_moves_left, num_extra_assignments);
            num_extra_assignments -= num_moves_left;
            cluster_sizes[cluster_id] += num_moves_left;
            // apply the first 'num_moves_left' from moves_into_cluster[cluster_id]
            clusters[cluster_id].insert(
                clusters[cluster_id].end(), moves_into_cluster[cluster_id].begin(),
                moves_into_cluster[cluster_id].begin() + num_moves_left);
        }

    }

    return clusters;
}
