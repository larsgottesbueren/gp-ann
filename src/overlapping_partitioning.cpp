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

#if true
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

auto Transpose(const AdjGraph& graph) {
    auto rev = parlay::delayed_tabulate(graph.size(), [&](int i) {
            const auto& neighbors = graph[i];
            return parlay::delayed_map(neighbors, [i](int neigh) -> std::pair<int,int> {
                return std::make_pair(neigh, i);
            });
        });
    auto rev_edges = parlay::flatten(rev);
    return parlay::group_by_index(rev_edges, graph.size());
}

Clusters OverlappingGraphPartitioning(PointSet& points, int num_clusters, double epsilon, double overlap) {
    const size_t max_cluster_size = (1.0 + epsilon) * points.n / num_clusters;
    const size_t num_extra_assignments = overlap * points.n;
    // previously const size_t num_extra_assignments = (1.0 + epsilon) * n * (1.0 + overlap) - n
    size_t num_assignments_remaining = num_extra_assignments;
    const size_t num_total_assignments = points.n + num_extra_assignments;
    num_clusters = std::ceil(static_cast<double>(num_total_assignments) / max_cluster_size);

    std::cout << "max cluster size " << max_cluster_size << " num clusters " << num_clusters << " eps " << epsilon << " overlap " << overlap << std::endl;
    Timer timer;
#if false
    std::string dummy_file = "tmp.graph";
    if (!std::filesystem::exists(dummy_file)) {
        ApproximateKNNGraphBuilder graph_builder;
        timer.Start();
        AdjGraph knn_graph = graph_builder.BuildApproximateNearestNeighborGraph(points, 10);
        std::cout << "Built KNN graph. Took " << timer.Stop() << std::endl;

        WriteGraph(knn_graph, dummy_file);
    }
    AdjGraph knn_graph = ReadGraph(dummy_file);
#else
    ApproximateKNNGraphBuilder graph_builder;
    timer.Start();
    static constexpr int degree = 10;
    AdjGraph knn_graph = graph_builder.BuildApproximateNearestNeighborGraph(points, degree);
    std::cout << "Built KNN graph. Took " << timer.Restart() << std::endl;
    points.Drop();
    std::cout << "Dropping points took " << timer.Stop() << std::endl;
#endif
    // Idea so far.
    // place node with plurality of its neighbors. --> query for node finds the neighbors

    // instead. place neighbors with node? --> query for node finds the shard of the node and thus also the neighbors
    // was the write-up imprecise here? at some point we had more details, but I feel like they got cut

    // do this by minimizing cut on the transposed directed graph? how many edges are symmetric even?

    timer.Start();
    auto transpose = Transpose(knn_graph);
    std::cout << "transpose took " << timer.Stop() << std::endl;

    Partition partition = PartitionAdjListGraph(knn_graph, num_clusters, epsilon, std::min<int>(32, parlay::num_workers()), false);
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
    while (num_assignments_remaining > 0) {
        auto best_moves = parlay::map(nodes, [&](uint32_t u) {
            auto& rating_map = rating_map_ets.get();
            return TopMove(u, transpose[u], cover, partition, rating_map, cluster_sizes, max_cluster_size);
        });

        std::cout << "Computed best moves" << std::endl;

        auto affinities = parlay::delayed_map(best_moves, [&](const auto& l) { return l.second; });

        std::cout << "Computed affinities" << std::endl;

        int best_affinity = parlay::reduce(affinities, parlay::maxm<int>());
        std::cout << "iter " << ++iter << " best affinity " << best_affinity << std::endl;

        if (best_affinity > degree) {
            // in the beginning, let's accept more moves at once
            best_affinity = degree;
        }

        if (best_affinity == 0) {
            break;
        }

        // TODO we could apply all moves with affinity >= num_neighbors directly as well, which should save a ton of iterations in the transpose case
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
        reduction += total_num_moves * std::min(best_affinity, degree); // we're not counting some guys here, but that's fine

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

    std::cout << "Total reduction " << reduction << std::endl;

    return clusters;
}

Clusters OverlappingKMeansPartitioningSPANN(PointSet& points, const Partition& partition, int requested_num_clusters, double epsilon, double overlap) {
    const size_t n = points.n;
    const size_t num_extra_assignments = overlap * n;
    const size_t max_cluster_size = (1.0 + epsilon) * n / requested_num_clusters;

    Clusters clusters = ConvertPartitionToClusters(partition);

    // if there are any empty clusters, remove them
    auto it = std::remove_if(clusters.begin(), clusters.end(), [&](const auto& cluster) { return cluster.empty(); });
    clusters.erase(it, clusters.end());

    auto cluster_sizes = parlay::map(clusters, [&](const auto& c) { return c.size(); });

    std::cout << "num clusters = " << clusters.size() << " cluster sizes ";
    for (size_t cs : cluster_sizes) std::cout << cs << " ";
    std::cout << std::endl;

    Timer timer; timer.Start();
    // Step 1 build centroids and associations
    KMeansTreeRouterOptions kmtr_options {.num_centroids = 16, .min_cluster_size = 350, .budget = 16 * clusters.size(), .search_budget = 0};
    KMeansTreeRouter kmtr;
    kmtr.Train(points, clusters, kmtr_options);
    auto [sub_points, sub_part] = kmtr.ExtractPoints();

    std::cout << "Got reps. " << sub_points.n << " " << sub_part.size() << " Took " << timer.Restart() << " s" << std::endl;


    // Step 2 search closest centroid that is not in the own partition
    auto point_ids = parlay::iota<uint32_t>(points.n);
    parlay::WorkerSpecific<std::vector<float>> min_dist_ets([&]() {
        return std::vector<float>(clusters.size(), std::numeric_limits<float>::max());
    });

    auto cluster_rankings = parlay::map(point_ids, [&](uint32_t u) -> std::vector<int> {
        auto& min_dist = min_dist_ets.get();
        std::vector<int> targets;
        // brute-force so we can more easily support the filter
        for (size_t j = 0; j < sub_points.n; ++j) {
            float dist = distance(points.GetPoint(u), sub_points.GetPoint(j), points.d);
            const int pv = sub_part[j];
            if (min_dist[pv] == std::numeric_limits<float>::max()) {
                targets.push_back(pv);
            }
            if (dist < min_dist[pv]) {
                min_dist[pv] = dist;
            }
        }

        // first select top 5
        size_t num_keep = 5;
        std::sort(targets.begin(), targets.end(), [&](int l, int r) { return min_dist[l] < min_dist[r]; });
        // reset entries outside the top
        for (size_t i = num_keep; i < targets.size(); ++i) {
            min_dist[targets[i]] = std::numeric_limits<float>::max();
        }
        // keep only the top
        min_dist.resize(std::min(min_dist.size(), num_keep));
        // sort in descending order since we will use pop_back in the next step
        std::sort(targets.begin(), targets.end(), [&](int l, int r) { return min_dist[l] > min_dist[r]; });
        // reset entries from the top
        for (int t : targets) {
            min_dist[t] = std::numeric_limits<float>::max();
        }
        targets.shrink_to_fit();
        return targets;
    });

    std::cout << "got cluster rankings " << std::endl;
    std::cout << "assignment loop. num overlap assignments allowed " << num_extra_assignments << std::endl;

    size_t num_assignments_left = num_extra_assignments;
    size_t iter = 0;
    while (num_assignments_left > 0) {
        std::cout << "Iter " << ++iter << " num assignments left " << num_assignments_left << std::endl;
        auto targets_and_points = parlay::map_maybe(point_ids, [&](uint32_t u) -> std::optional<std::pair<int, int>> {
            auto& ranking = cluster_rankings[u];
            while (!ranking.empty()) {
                int target = ranking.back();
                ranking.pop_back();
                if (cluster_sizes[target] < max_cluster_size) {
                    return std::make_pair(target, u);
                }
            }
            return std::nullopt;
        });

        std::cout << "# primary moves " << targets_and_points.size() << std::endl;;

        if (targets_and_points.empty()) {
            break;
        }

        auto moves_into_cluster = parlay::group_by_index(targets_and_points, clusters.size());

        auto num_moves_into_cluster = parlay::tabulate(clusters.size(), [&](size_t cluster_id) {
            return std::min(max_cluster_size - cluster_sizes[cluster_id], moves_into_cluster[cluster_id].size());
        });

        size_t total_num_moves = parlay::reduce(num_moves_into_cluster);

        if (total_num_moves > num_assignments_left) {
            double fraction_to_keep = num_assignments_left / total_num_moves;
            num_moves_into_cluster = parlay::map(num_moves_into_cluster, [&](size_t num_moves) -> size_t {
                return std::floor(num_moves * fraction_to_keep);
            });
            total_num_moves = parlay::reduce(num_moves_into_cluster);
        }

        std::cout << "total num moves this round " << total_num_moves << std::endl;

        num_assignments_left -= total_num_moves;

        parlay::parallel_for(0, clusters.size(), [&](size_t cluster_id) {
            size_t num_moves = num_moves_into_cluster[cluster_id];
            cluster_sizes[cluster_id] += num_moves;
            clusters[cluster_id].insert(
                clusters[cluster_id].end(), moves_into_cluster[cluster_id].begin(),
                moves_into_cluster[cluster_id].begin() + num_moves);
        });
    }

    std::cout << "Finished" << std::endl;

    return clusters;
}
