#pragma once

#include "dist.h"
#include "defs.h"
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <numeric>
#include <random>

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
                    for (size_t j = 0; j < centroids.d; ++j) { L[j] = R[j]; }
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

void NormalizeCentroidsIP(PointSet& centroids, const std::vector<size_t>& cluster_size, const std::vector<float>& norm_sums) {
    for (size_t c = 0; c < centroids.n; ++c) {
        float* C = centroids.GetPoint(c);
        if (cluster_size[c] == 0) { continue; }
        float desired_norm = norm_sums[c] / cluster_size[c];
        float current_norm = vec_norm(C, centroids.d);
        float multiplier = std::sqrt(desired_norm / current_norm);
        for (size_t j = 0; j < centroids.d; ++j) { C[j] *= multiplier; }
    }
}

void NormalizeCentroidsL2(PointSet& centroids, const std::vector<size_t>& cluster_size) {
    for (size_t c = 0; c < centroids.n; ++c) {
        float* C = centroids.GetPoint(c);
        if (cluster_size[c] == 0) { continue; }
        for (size_t j = 0; j < centroids.d; ++j) { C[j] /= cluster_size[c]; }
    }
}

void SumPointsInClustersL2(PointSet& P, PointSet& centroids, std::vector<int>& closest_center, std::vector<size_t>& cluster_size, size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
        int c = closest_center[i];
        cluster_size[c]++;
        float* C = centroids.GetPoint(c);
        float* Pi = P.GetPoint(i);
        for (size_t j = 0; j < P.d; ++j) { C[j] += Pi[j]; }
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
        for (size_t j = 0; j < P.d; ++j) { C[j] += Pi[j] * multiplier; }
    }
}

std::vector<size_t> AggregateClusters(PointSet& P, PointSet& centroids, std::vector<int>& closest_center,
                                      const parlay::sequence<float>& vector_sqrt_norms, bool normalize = true) {
    centroids.coordinates.assign(centroids.coordinates.size(), 0.f);
    std::vector<size_t> cluster_size(centroids.n, 0);
#ifdef MIPS_DISTANCE
	std::vector<float> norm_sums(centroids.n, 0.0);
	SumPointsInClustersIP(P, centroids, closest_center, cluster_size, vector_sqrt_norms, norm_sums, 0, closest_center.size());
    if (normalize) { NormalizeCentroidsIP(centroids, cluster_size, norm_sums); }
#else
    SumPointsInClustersL2(P, centroids, closest_center, cluster_size, 0, closest_center.size());
    if (normalize) { NormalizeCentroidsL2(centroids, cluster_size); }
#endif
    RemoveEmptyClusters(centroids, closest_center, cluster_size);
    return cluster_size;
}

inline void atomic_fetch_add_float(float* addr, float x) {
    float expected;
    __atomic_load(addr, &expected, __ATOMIC_RELAXED);
    float desired = expected + x;
    while (!__atomic_compare_exchange(addr, &expected, &desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) { desired = expected + x; }
}

inline void atomic_fetch_add_double(double* addr, double x) {
    double expected;
    __atomic_load(addr, &expected, __ATOMIC_RELAXED);
    double desired = expected + x;
    while (!__atomic_compare_exchange(addr, &expected, &desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) { desired = expected + x; }
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
            for (size_t j = 0; j < centroids.d; ++j) { atomic_fetch_add_float(&C[j], BC[j]); }
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
        for (size_t j = 0; j < points.d; ++j) { centroids.coordinates.push_back(p[j]); }
    }
    return centroids;
}

void NearestCentersAccelerated(PointSet& P, PointSet& centroids, std::vector<int>& closest_center) {
#ifdef MIPS_DISTANCE
    using SpaceType = hnswlib::InnerProductSpace;
#else
    using SpaceType = hnswlib::L2Space;
#endif
    SpaceType space(P.d);
    HNSWParameters hnsw_parameters;
    hnsw_parameters.M = 16;
    std::cout << "Num centroids " << centroids.n << std::endl;
    Timer timer;
    timer.Start();

    hnswlib::HierarchicalNSW<float> hnsw(&space, centroids.n, hnsw_parameters.M, hnsw_parameters.ef_construction, 555);

    size_t seq_insertion = std::min(1UL << 11, centroids.n);
    for (size_t i = 0; i < seq_insertion; ++i) { hnsw.addPoint(centroids.GetPoint(i), i); }
    parlay::parallel_for(seq_insertion, centroids.n, [&](size_t i) { hnsw.addPoint(centroids.GetPoint(i), i); }, 512);
    std::cout << "Build centroid HNSW took " << timer.Restart() << std::endl;

    hnsw.setEf(128);

    parlay::parallel_for(0, P.n, [&](size_t i) {
        auto res = hnsw.searchKnn(P.GetPoint(i), 1);
        closest_center[i] = res.top().second;
    }, 1024);
    std::cout << "HNSW closest center assignment took " << timer.Restart() << std::endl;
}

std::vector<int> KMeans(PointSet& P, PointSet& centroids) {
    if (centroids.n < 1) { throw std::runtime_error("KMeans #centroids < 1"); }
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
    return parlay::reduce(
        parlay::delayed_tabulate(points.n, [&](size_t i) -> double {
            return pos_distance(points.GetPoint(i), centroids.GetPoint(closest_center[i]), points.d);
        })
    );
}

std::vector<int> BalancedKMeans(PointSet& points, PointSet& centroids, size_t max_cluster_size) {
    std::vector<int> closest_center = KMeans(points, centroids);

    PointSet cluster_coordinate_sums = centroids;
    std::vector<size_t> cluster_sizes =
            AggregateClustersParallel(points, cluster_coordinate_sums, closest_center, /*TODO mips*/{ }, false);

    std::cout << "Objective " << ObjectiveValue(points, centroids, closest_center) << std::endl;


    // precompute norms and sqrts since it slowed down centroid calculation
    parlay::sequence<double> vector_sqrt_norms = parlay::tabulate(points.n, [&](size_t i) -> double {
        return std::sqrt(vec_norm(points.GetPoint(i), points.d));
    });

    parlay::sequence<double> cluster_norm_sums = parlay::reduce_by_index(
        parlay::zip(closest_center, vector_sqrt_norms),
        centroids.n);

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
    auto penalty_function_iter = [](int round) -> double { if (round > 100) { return 1.01; } else { return 1.5009 - 0.0009 * round; } };

    int round = 0;
    constexpr int MAX_ROUNDS = 500;
    double round_penalty = 0.0;

    std::vector<int> best_partition = closest_center;
    double best_objective = std::numeric_limits<double>::max();

    parlay::sequence<double> penalties_needed(points.n);

    print_cluster_sizes();

    while (round++ <= MAX_ROUNDS) {
        std::cout << "round = " << round << " penalty " << round_penalty << std::endl;

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
                    if (denom == 0) { denom = 1; }

                    const double penalty_needed = (dist - old_cluster_dist) / denom;
                    if (old_cluster_size > cluster_size) {
                        if (round_penalty < penalty_needed) {
                            if (penalty_needed < min_penalty_needed) { min_penalty_needed = penalty_needed; }
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

                if (best != old_cluster) {
                    closest_center[point_id] = best;
                    __atomic_fetch_sub(&cluster_sizes[old_cluster], 1, __ATOMIC_RELAXED); // HORRIBLE contention...
                    __atomic_fetch_add(&cluster_sizes[best], 1, __ATOMIC_RELAXED);

                    // TODO would a second pass that groups by cluster IDs be faster, or even the distributed style version that builds deltas for each cluster
                    float* coords_best = cluster_coordinate_sums.GetPoint(best);
                    for (int j = 0; j < points.d; ++j) { atomic_fetch_add_float(coords_best + j, p[j]); }

                    float* coords_old = cluster_coordinate_sums.GetPoint(old_cluster);
                    for (int j = 0; j < points.d; ++j) { atomic_fetch_add_float(coords_old + j, -p[j]); }

#ifdef MIPS_DISTANCE
                    atomic_fetch_add_double(&cluster_norm_sums[old_cluster], -vector_sqrt_norms[point_id]);
                    atomic_fetch_add_double(&cluster_norm_sums[best], vector_sqrt_norms[point_id]);
#endif
                }
            });

            // update centroids phase
#ifdef MIPS_DISTANCE
            for (int c = 0; c < centroids.n; ++c) {
                float* C = centroids.GetPoint(c);
                float* C2 = cluster_coordinate_sums.GetPoint(c);
                if (cluster_sizes[c] == 0) {
                    for (int j = 0; j < centroids.d; ++j) {
                        C[j] = 0.0f;
                    }
                } else {
                    float desired_norm = cluster_norm_sums[c] / cluster_sizes[c];
                    float current_norm = vec_norm(C, centroids.d);
                    float multiplier = std::sqrt(desired_norm / current_norm);
                    for (int j = 0; j < centroids.d; ++j) {
                        C[j] = C2[j] * multiplier;
                    }
                }
            }
#else
            for (int c = 0; c < centroids.n; ++c) {
                float* C = centroids.GetPoint(c);
                float* C2 = cluster_coordinate_sums.GetPoint(c);
                for (int j = 0; j < centroids.d; ++j) {
                    if (cluster_sizes[c] == 0) {
                        C[j] = 0.0;
                    } else {
                        C[j] = C2[j] / cluster_sizes[c];
                    }
                }
            }
#endif
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

    return best_partition;
}
