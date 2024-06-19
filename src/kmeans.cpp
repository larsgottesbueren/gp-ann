#include "kmeans.h"

#include <numeric>
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <random>
#include "defs.h"
#include "dist.h"

namespace {
    int Top1Neighbor(PointSet& P, float* Q) {
        int best = -1;
        float best_dist = std::numeric_limits<float>::max();
        for (size_t i = 0; i < P.n; ++i) {
            float new_dist = distance(P.GetPoint(i), Q, P.d);
            if (new_dist < best_dist) {
                best_dist = new_dist;
                best = i;
            }
        }
        return best;
    }

    void NearestCenters(PointSet& P, PointSet& centroids, std::vector<int>& closest_center) {
        parlay::parallel_for(0, P.n, [&](size_t i) { closest_center[i] = Top1Neighbor(centroids, P.GetPoint(i)); });
    }

    void RemoveEmptyClusters(PointSet& centroids, std::vector<int>& closest_center, const std::vector<size_t>& cluster_size) {
        if (std::any_of(cluster_size.begin(), cluster_size.end(), [](size_t x) { return x == 0; })) {
            std::vector<int> remapped_cluster_ids(centroids.n, -1);
            size_t l = 0;
            for (size_t r = 0; r < centroids.n; ++r) {
                if (cluster_size[r] != 0) {
                    if (l != r) { // don't do the copy if not necessary
                        float* L = centroids.GetPoint(l);
                        float* R = centroids.GetPoint(r);
                        for (size_t j = 0; j < centroids.d; ++j) {
                            L[j] = R[j];
                        }
                    }
                    remapped_cluster_ids[r] = l;
                    l++;
                }
            }
            centroids.n = l;
            centroids.coordinates.resize(centroids.n * centroids.d);
            parlay::parallel_for(0, closest_center.size(), [&](size_t i) { closest_center[i] = remapped_cluster_ids[closest_center[i]]; });
        }
    }

    void atomic_fetch_add_float(float* addr, float x) {
        float expected;
        __atomic_load(addr, &expected, __ATOMIC_RELAXED);
        float desired = expected + x;
        while (!__atomic_compare_exchange(addr, &expected, &desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
            desired = expected + x;
        }
    }

#ifdef MIPS_DISTANCE

    void atomic_fetch_add_double(double* addr, double x) {
        double expected;
        __atomic_load(addr, &expected, __ATOMIC_RELAXED);
        double desired = expected + x;
        while (!__atomic_compare_exchange(addr, &expected, &desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
            desired = expected + x;
        }
    }

    void NormalizeCentroidsIP(PointSet& centroids, const std::vector<size_t>& cluster_size, const std::vector<float>& norm_sums) {
        for (size_t c = 0; c < centroids.n; ++c) {
            float* C = centroids.GetPoint(c);
            if (cluster_size[c] == 0) {
                continue;
            }
            float desired_norm = norm_sums[c] / cluster_size[c];
            float current_norm = vec_norm(C, centroids.d);
            float multiplier = std::sqrt(desired_norm / current_norm);
            for (size_t j = 0; j < centroids.d; ++j) {
                C[j] *= multiplier;
            }
        }
    }

    void SumPointsInClustersIP(PointSet& P, PointSet& centroids, std::vector<int>& closest_center, std::vector<size_t>& cluster_size,
                               const parlay::sequence<float>& vector_sqrt_norms, std::vector<float>& norm_sums, size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            int c = closest_center[i];
            cluster_size[c]++;
            float* C = centroids.GetPoint(c);
            float* Pi = P.GetPoint(i);
            norm_sums[c] += vector_sqrt_norms[i] * vector_sqrt_norms[i];
            float multiplier = 1.0f / vector_sqrt_norms[i];
            for (size_t j = 0; j < P.d; ++j) {
                C[j] += Pi[j] * multiplier;
            }
        }
    }

#else

    void NormalizeCentroidsL2(PointSet& centroids, const std::vector<size_t>& cluster_size) {
        for (size_t c = 0; c < centroids.n; ++c) {
            float* C = centroids.GetPoint(c);
            if (cluster_size[c] == 0) {
                continue;
            }
            for (size_t j = 0; j < centroids.d; ++j) {
                C[j] /= cluster_size[c];
            }
        }
    }

    void SumPointsInClustersL2(PointSet& P, PointSet& centroids, std::vector<int>& closest_center, std::vector<size_t>& cluster_size, size_t start,
                               size_t end) {
        for (size_t i = start; i < end; ++i) {
            int c = closest_center[i];
            cluster_size[c]++;
            float* C = centroids.GetPoint(c);
            float* Pi = P.GetPoint(i);
            for (size_t j = 0; j < P.d; ++j) {
                C[j] += Pi[j];
            }
        }
    }

#endif

    std::vector<size_t> AggregateClusters(PointSet& P, PointSet& centroids, std::vector<int>& closest_center, const parlay::sequence<float>& vector_sqrt_norms,
                                          bool normalize = true) {
        centroids.coordinates.assign(centroids.coordinates.size(), 0.f);
        std::vector<size_t> cluster_size(centroids.n, 0);
#ifdef MIPS_DISTANCE
        std::vector<float> norm_sums(centroids.n, 0.0);
        SumPointsInClustersIP(P, centroids, closest_center, cluster_size, vector_sqrt_norms, norm_sums, 0, closest_center.size());
        if (normalize) {
            NormalizeCentroidsIP(centroids, cluster_size, norm_sums);
        }
#else
        SumPointsInClustersL2(P, centroids, closest_center, cluster_size, 0, closest_center.size());
        if (normalize) {
            NormalizeCentroidsL2(centroids, cluster_size);
        }
#endif
        RemoveEmptyClusters(centroids, closest_center, cluster_size);
        return cluster_size;
    }

    std::vector<size_t> AggregateClustersParallel(PointSet& P, PointSet& centroids, std::vector<int>& closest_center,
                                                  const parlay::sequence<float>& vector_sqrt_norms, bool normalize = true) {
        size_t block_size = std::max<size_t>(5000000, centroids.coordinates.size() * 200);
        if (P.n <= block_size)
            return AggregateClusters(P, centroids, closest_center, vector_sqrt_norms, normalize);

        centroids.coordinates.assign(centroids.coordinates.size(), 0.f);
        std::vector<size_t> cluster_size(centroids.n, 0);

#ifdef MIPS_DISTANCE
        std::vector<float> norm_sums(centroids.n, 0.f);
#endif

        // This is what a distributed implementation would do... Not great but at least it can get some speedups
        parlay::internal::sliced_for(P.n, block_size, [&](size_t block_id, size_t start, size_t end) {
            PointSet b_centroids;
            b_centroids.n = centroids.n;
            b_centroids.d = centroids.d;
            b_centroids.Alloc();
            std::vector<size_t> b_cluster_size(centroids.n, 0);

#ifdef MIPS_DISTANCE
            std::vector<float> b_norm_sums(centroids.n, 0.f);
            SumPointsInClustersIP(P, b_centroids, closest_center, b_cluster_size, vector_sqrt_norms, b_norm_sums, start, end);
#else
            SumPointsInClustersL2(P, b_centroids, closest_center, b_cluster_size, start, end);
#endif

            for (size_t i = 0; i < cluster_size.size(); ++i) {
                __atomic_fetch_add(&cluster_size[i], b_cluster_size[i], __ATOMIC_RELAXED);
#ifdef MIPS_DISTANCE
                atomic_fetch_add_float(&norm_sums[i], b_norm_sums[i]);
#endif
            }
            for (size_t i = 0; i < centroids.n; ++i) {
                float* BC = b_centroids.GetPoint(i);
                float* C = centroids.GetPoint(i);
                for (size_t j = 0; j < centroids.d; ++j) {
                    atomic_fetch_add_float(&C[j], BC[j]);
                }
            }
        });

        if (normalize) {
#ifdef MIPS_DISTANCE
            NormalizeCentroidsIP(centroids, cluster_size, norm_sums);
#else
            NormalizeCentroidsL2(centroids, cluster_size);
#endif
        }

        RemoveEmptyClusters(centroids, closest_center, cluster_size);
        return cluster_size;
    }
} // namespace

PointSet RandomSample(PointSet& points, size_t num_samples, int seed) {
    PointSet centroids;
    centroids.n = num_samples;
    centroids.d = points.d;

    std::vector<int> iota(points.n);
    std::iota(iota.begin(), iota.end(), 0);

    std::mt19937 prng(seed);
    std::vector<int> sample(num_samples);
    std::sample(iota.begin(), iota.end(), sample.begin(), num_samples, prng);

    for (int i : sample) {
        float* p = points.GetPoint(i);
        for (size_t j = 0; j < points.d; ++j) {
            centroids.coordinates.push_back(p[j]);
        }
    }
    return centroids;
}

std::vector<int> KMeans(PointSet& P, PointSet& centroids) {
    if (centroids.n < 1) {
        throw std::runtime_error("KMeans #centroids < 1");
    }
    std::vector<int> closest_center(P.n, -1);
    parlay::sequence<float> vector_sqrt_norms;
#ifdef MIPS_DISTANCE
    // precompute norms and sqrts since it slowed down centroid calculation
    vector_sqrt_norms = parlay::tabulate(P.n, [&](size_t i) -> float { return std::sqrt(vec_norm(P.GetPoint(i), P.d)); });
#endif
    static constexpr size_t NUM_ROUNDS = 20;
    for (size_t r = 0; r < NUM_ROUNDS; ++r) {
        NearestCenters(P, centroids, closest_center);
        AggregateClustersParallel(P, centroids, closest_center, vector_sqrt_norms);
    }
    return closest_center;
}

double ObjectiveValue(PointSet& points, PointSet& centroids, const std::vector<int>& closest_center) {
    return parlay::reduce(parlay::delayed_tabulate(
            points.n, [&](size_t i) -> double { return pos_distance(points.GetPoint(i), centroids.GetPoint(closest_center[i]), points.d); }));
}

double square(double x) { return x * x; }

std::vector<int> BalancedKMeans(PointSet& points, PointSet& centroids, size_t max_cluster_size) {
    std::vector<int> closest_center = KMeans(points, centroids);

    // precompute norms and sqrts since it slowed down centroid calculation
    parlay::sequence<float> vector_sqrt_norms =
            parlay::tabulate(points.n, [&](size_t i) -> float { return std::sqrt(vec_norm(points.GetPoint(i), points.d)); });

    PointSet cluster_coordinate_sums = centroids;
    std::vector<size_t> cluster_sizes = AggregateClustersParallel(points, cluster_coordinate_sums, closest_center, vector_sqrt_norms, false);

    std::cout << "Objective " << ObjectiveValue(points, centroids, closest_center) << std::endl;


    auto d_vec_sqrt_norms = parlay::delayed_map(vector_sqrt_norms, [](float x) -> double { return x; });
    auto assignment_and_norm = parlay::zip(closest_center, d_vec_sqrt_norms);
    auto cluster_norm_sums = parlay::reduce_by_index(assignment_and_norm, centroids.n);

    std::cout << "cluster norm sums ";
    for (size_t j = 0; j < cluster_norm_sums.size(); ++j) {
        std::cout << cluster_norm_sums[j] << " ";
    }
    std::cout << std::endl;
    std::cout << "total norm sums " << std::accumulate(cluster_norm_sums.begin(), cluster_norm_sums.end(), 0.0) << std::endl;

    auto is_balanced = [&] { return parlay::all_of(cluster_sizes, [&](size_t cluster_size) { return cluster_size <= max_cluster_size; }); };

    if (is_balanced())
        return closest_center;

    auto print_cluster_sizes = [&] {
        size_t max_size = *parlay::max_element(cluster_sizes);
        size_t min_size = *parlay::min_element(cluster_sizes);
        size_t perfect_balance = points.n / centroids.n;
        double overshot = double(max_size) / max_cluster_size;
        double imbalance = double(max_size) / perfect_balance;
        double imbalance2 = double(min_size) / perfect_balance;

        std::cout << "overshot " << overshot << " imbalance " << imbalance << " min imb " << imbalance2 << " largest cluster size " << max_size
                  << " constraint " << max_cluster_size << std::endl;
    };

    // taken from the BKM+ paper and implementation.
    auto penalty_function_iter = [](int round) -> double {
        if (round > 100) {
            return 1.01;
        } else {
            return 1.5009 - 0.0009 * round;
        }
    };

    int round = 0;
    constexpr int MAX_ROUNDS = 150;
    double round_penalty = 0.0;

    std::vector<int> best_partition = closest_center;
    double best_objective = std::numeric_limits<double>::max();

    parlay::sequence<double> penalties_needed(points.n);

    print_cluster_sizes();

    auto atomic_move = [&](uint32_t point_id, int old_cluster, int new_cluster) {
        if (new_cluster == old_cluster) {
            return;
        }
        closest_center[point_id] = new_cluster;
        __atomic_fetch_sub(&cluster_sizes[old_cluster], 1, __ATOMIC_RELAXED); // HORRIBLE contention...
        __atomic_fetch_add(&cluster_sizes[new_cluster], 1, __ATOMIC_RELAXED);

        float* coords_best = cluster_coordinate_sums.GetPoint(new_cluster);
        float* coords_old = cluster_coordinate_sums.GetPoint(old_cluster);
        float multiplier = 1.0f;
#ifdef MIPS_DISTANCE
        atomic_fetch_add_double(&cluster_norm_sums[old_cluster], -square(vector_sqrt_norms[point_id]));
        atomic_fetch_add_double(&cluster_norm_sums[new_cluster], square(vector_sqrt_norms[point_id]));
        multiplier = 1.0f / vector_sqrt_norms[point_id];
#endif
        float* p = points.GetPoint(point_id);
        for (size_t j = 0; j < points.d; ++j) {
            atomic_fetch_add_float(coords_best + j, p[j] * multiplier);
            atomic_fetch_add_float(coords_old + j, -p[j] * multiplier);
        }
    };


    auto update_centroids = [&] {
// update centroids phase
#ifdef MIPS_DISTANCE
        for (size_t c = 0; c < centroids.n; ++c) {
            float* C = centroids.GetPoint(c);
            float* C2 = cluster_coordinate_sums.GetPoint(c);
            if (cluster_sizes[c] == 0) {
                for (size_t j = 0; j < centroids.d; ++j) {
                    C[j] = 0.0f;
                }
            } else {
                float desired_norm = cluster_norm_sums[c] / cluster_sizes[c];
                float current_norm = vec_norm(C2, centroids.d);
                float multiplier = std::sqrt(desired_norm / current_norm);
                for (size_t j = 0; j < centroids.d; ++j) {
                    C[j] = C2[j] * multiplier;
                }
            }
        }
#else
        for (size_t c = 0; c < centroids.n; ++c) {
            float* C = centroids.GetPoint(c);
            float* C2 = cluster_coordinate_sums.GetPoint(c);
            for (size_t j = 0; j < centroids.d; ++j) {
                if (cluster_sizes[c] == 0) {
                    C[j] = 0.0;
                } else {
                    C[j] = C2[j] / cluster_sizes[c];
                }
            }
        }
#endif
    };

    while (round++ <= MAX_ROUNDS) {
        std::cout << "round = " << round << " penalty " << round_penalty << std::endl;

#if false
        while (parlay::any_of(cluster_sizes, [&](size_t cs) { return cs == 0; })) {
            int largest = 0;
            int smallest = -1;
            for (size_t i = 0; i < cluster_sizes.size(); ++i) {
                if (cluster_sizes[i] == 0) {
                    smallest = i;
                }
                if (cluster_sizes[i] > cluster_sizes[largest]) {
                    largest = i;
                }
            }
            std::cout << "Rebalance from " << largest << " to " << smallest << ". current sizes " << cluster_sizes[largest] << " " << cluster_sizes[smallest]
                      << std::endl;
            size_t half = cluster_sizes[largest] / 2;
            uint32_t point_id = 0;
            while (half > 0) {
                while (closest_center[point_id] != largest) {
                    ++point_id;
                }
                atomic_move(point_id, largest, smallest);
                --half;
            }
            update_centroids();
        }
#endif

        // mini-batch cluster moves and updates
        auto perm = parlay::random_shuffle(parlay::iota<uint32_t>(points.n), parlay::random(round));
        size_t num_subrounds = 1000;
        size_t n = points.n;
        size_t chunk_size = idiv_ceil(n, num_subrounds);
        for (size_t sub_round = 0; sub_round < num_subrounds; ++sub_round) {
            auto [start, end] = bounds(sub_round, n, chunk_size);

            // moving phase
            parlay::parallel_for(start, end, [&](size_t i) {
                uint32_t point_id = perm[i];
                float* p = points.GetPoint(point_id);
                const int old_cluster = closest_center[point_id];
                const float old_cluster_dist = pos_distance(centroids.GetPoint(old_cluster), p, points.d);

                const size_t old_cluster_size = cluster_sizes[old_cluster];

                int best = old_cluster;
                double best_score = std::numeric_limits<double>::max();
                double min_penalty_needed = std::numeric_limits<double>::max();

                for (int j = 0; j < int(centroids.n); ++j) {
                    const size_t cluster_size = cluster_sizes[j];
                    const float dist = pos_distance(centroids.GetPoint(j), p, points.d);
                    const double score = dist + round_penalty * cluster_size;
                    int denom = old_cluster_size - cluster_size;
                    if (denom == 0) {
                        denom = 1;
                    }

                    const double penalty_needed = (dist - old_cluster_dist) / denom;
                    if (old_cluster_size > cluster_size) {
                        if (round_penalty < penalty_needed) {
                            if (penalty_needed < min_penalty_needed) {
                                min_penalty_needed = penalty_needed;
                            }
                        } else {
                            if (score < best_score) {
                                best = j;
                                best_score = score;
                            }
                        }
                    } else {
                        if (round_penalty < penalty_needed && score < best_score) {
                            best = j;
                            best_score = score;
                        }
                    }
                }

                penalties_needed[point_id] = min_penalty_needed;

                atomic_move(point_id, old_cluster, best);
            });

            update_centroids();
        }


        print_cluster_sizes();
        const double next_penalty = *parlay::min_element(penalties_needed);
        const double objective = ObjectiveValue(points, centroids, closest_center);
        std::cout << "objective " << objective << " next penalty " << next_penalty << std::endl;
        if (is_balanced()) {
            if (objective < best_objective) {
                best_objective = objective;
                best_partition = closest_center;
            } else {
                // no improvement but balanced --> let's quit
                break;
            }
        } else {
            // adjust penalty
            round_penalty = penalty_function_iter(round) * next_penalty;
        }
    }

    cluster_sizes = AggregateClustersParallel(points, centroids, best_partition, vector_sqrt_norms, true);
    int num_clusters = cluster_sizes.size();

    print_cluster_sizes();

    int num_overloaded_clusters = 0;
    for (int part_id = 0; part_id < num_clusters; ++part_id) {
        if (cluster_sizes[part_id] > max_cluster_size) {
            num_overloaded_clusters++;
        }
    }
    if (num_overloaded_clusters == 0) {
        return best_partition;
    }

    std::cout << "There are " << num_overloaded_clusters << " / " << num_clusters << " too heavy clusters. Rebalance stuff" << std::endl;
    Clusters clusters = ConvertPartitionToClusters(best_partition);
    for (int c = 0; c < num_clusters; ++c) {
        while (clusters[c].size() > max_cluster_size) {
            // remigrate points -- just skip updating the centroids
            uint32_t v = clusters[c].back();
            float min_dist = std::numeric_limits<float>::max();
            int target = -1;
            for (size_t j = 0; j < clusters.size(); ++j) {
                if (clusters[j].size() < max_cluster_size) {
                    if (float dist = distance(points.GetPoint(v), centroids.GetPoint(j), points.d); dist < min_dist) {
                        min_dist = dist;
                        target = j;
                    }
                }
            }
            assert(target != -1);
            clusters[target].push_back(v);
            best_partition[v] = target;
            clusters[c].pop_back();
        }
    }

    for (int c = 0; c < num_clusters; ++c) {
        cluster_sizes[c] = clusters[c].size();
    }

    print_cluster_sizes();

    return best_partition;
}
