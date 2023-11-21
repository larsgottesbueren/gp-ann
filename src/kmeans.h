#pragma once

#include "dist.h"
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <numeric>
#include <random>

void NearestCenters(PointSet& P, PointSet& centroids, std::vector<int>& closest_center) {
	parlay::parallel_for(0, P.n, [&](size_t i) {
		closest_center[i] = Top1Neighbor(centroids, P.GetPoint(i));
	});
}

void RemoveEmptyClusters(PointSet& centroids, std::vector<int>& closest_center, const std::vector<size_t>& cluster_size) {
    if (std::any_of(cluster_size.begin(), cluster_size.end(), [](size_t x) { return x == 0; })) {
        std::vector<int> remapped_cluster_ids(centroids.n, -1);
        size_t l = 0;
        for (size_t r = 0; r < centroids.n; ++r) {
            if (cluster_size[r] != 0) {
                if (l != r) {       // don't do the copy if not necessary
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
        parlay::parallel_for(0, closest_center.size(), [&](size_t i) {
            closest_center[i] = remapped_cluster_ids[closest_center[i]];
        });
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

void SumPointsInClustersL2(PointSet& P, PointSet& centroids, std::vector<int>& closest_center, std::vector<size_t>& cluster_size, size_t start, size_t end) {
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

std::vector<size_t> AggregateClusters(PointSet& P, PointSet& centroids, std::vector<int>& closest_center,
                                      const parlay::sequence<float>& vector_sqrt_norms) {
    centroids.coordinates.assign(centroids.coordinates.size(), 0.f);
	std::vector<size_t> cluster_size(centroids.n, 0);
    #ifdef MIPS_DISTANCE
	std::vector<float> norm_sums(centroids.n, 0.0);
	SumPointsInClustersIP(P, centroids, closest_center, cluster_size, vector_sqrt_norms, norm_sums, 0, closest_center.size());
    NormalizeCentroidsIP(centroids, cluster_size, norm_sums);
    #else
    SumPointsInClustersL2(P, centroids, closest_center, cluster_size, 0, closest_center.size());
	NormalizeCentroidsL2(centroids, cluster_size);
    #endif
	RemoveEmptyClusters(centroids, closest_center, cluster_size);
	return cluster_size;
}

void atomic_fetch_add_float(float* addr, float x) {
    float expected;
    __atomic_load(addr, &expected, __ATOMIC_RELAXED);
    float desired = expected + x;
    while (!__atomic_compare_exchange(addr, &expected, &desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
        desired = expected + x;
    }
}

std::vector<size_t> AggregateClustersParallel(PointSet& P, PointSet& centroids, std::vector<int>& closest_center, const parlay::sequence<float>& vector_sqrt_norms) {
    size_t block_size = std::max<size_t>(5000000, centroids.coordinates.size() * 200);
    if (P.n <= block_size) return AggregateClusters(P, centroids, closest_center, vector_sqrt_norms);

    centroids.coordinates.assign(centroids.coordinates.size(), 0.f);
    std::vector<size_t> cluster_size(centroids.n, 0);

    #ifdef MIPS_DISTANCE
    std::vector<float> norm_sums(centroids.n, 0.f);
    #endif

    // This is what a distributed implementation would do... Not great but at least it can get some speedups
    parlay::internal::sliced_for(P.n, block_size, [&](size_t block_id, size_t start, size_t end) {
        PointSet b_centroids;
        b_centroids.n = centroids.n; b_centroids.d = centroids.d; b_centroids.Alloc();
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
            float* BC = b_centroids.GetPoint(i); float* C = centroids.GetPoint(i);
            for (size_t j = 0; j < centroids.d; ++j) {
                atomic_fetch_add_float(&C[j], BC[j]);
            }
        }
    });

    #ifdef MIPS_DISTANCE
    NormalizeCentroidsIP(centroids, cluster_size, norm_sums);
    #else
    NormalizeCentroidsL2(centroids, cluster_size);
    #endif
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
        for (size_t j = 0; j < points.d; ++j) {
            centroids.coordinates.push_back(p[j]);
        }
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
    Timer timer; timer.Start();

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

void KMeansRebalancing(PointSet& points, PointSet& centroids, size_t max_cluster_size, std::vector<size_t>& cluster_sizes_ext,
                       std::vector<int>& closest_center, std::vector<float>& influence) {
    auto centroid_distances = parlay::tabulate(points.n, [&](size_t i) {
        return parlay::tabulate(centroids.n, [&](size_t j) {
           return distance(points.GetPoint(i), centroids.GetPoint(j), points.d);
        });
    });



    for (int r = 0; r < 5; ++r) {
        parlay::parallel_for(0, points.n, [&](size_t i) {
            int best = -1; float best_dist = std::numeric_limits<float>::max();
            for (size_t j = 0; j < centroid_distances[i].size(); ++j) {
                float eff_dist = centroid_distances[i][j] * influence[j];
                if (eff_dist < best_dist) best_dist = eff_dist, best = j;
            }
            closest_center[i] = best;
        });

        auto cluster_sizes = parlay::histogram_by_index(closest_center, centroids.n);

        if (parlay::all_of(cluster_sizes, [&](size_t x) { return x <= max_cluster_size; })) {
            return;
        }

        for (size_t j = 0; j < centroids.n; ++j) {
            double gamma = static_cast<double>(max_cluster_size) / cluster_sizes[j];
            gamma = std::pow(gamma, 1.0/6.0);
            std::cout << cluster_sizes[j] << " " << influence[j] << " " << gamma << std::endl;
            // cluster larger than max size --> smaller gamma --> larger influence param --> effective distance is worse
            influence[j] = influence[j] / gamma;
        }
        std::cout << "----------------------------" << std::endl;

    }


}

std::vector<int> BalancedKMeans(PointSet& points, PointSet& centroids, size_t max_cluster_size) {
    static constexpr size_t NUM_ROUNDS = 20;
    static constexpr size_t NUM_ROUNDS_WITH_IMBALANCE_ALLOWED = 8;  // first build some solid clustering, then start balancing it.
    std::vector<int> closest_center(points.n, -1);
    parlay::sequence<float> vector_sqrt_norms;
    std::vector<float> influence(centroids.n, 1.f);
    #ifdef MIPS_DISTANCE
    // precompute norms and sqrts since it slowed down centroid calculation
    vector_sqrt_norms = parlay::tabulate(points.n, [&](size_t i) -> float { return std::sqrt(vec_norm(points.GetPoint(i), points.d)); });
    #endif
    std::vector<size_t> cluster_sizes;
    for (size_t r = 0; r < NUM_ROUNDS_WITH_IMBALANCE_ALLOWED; ++r) {
        NearestCenters(points, centroids, closest_center);
        cluster_sizes = AggregateClusters(points, centroids, closest_center, vector_sqrt_norms);
    }
    for (size_t r = NUM_ROUNDS_WITH_IMBALANCE_ALLOWED; r < NUM_ROUNDS; ++r) {
        if (std::all_of(cluster_sizes.begin(), cluster_sizes.end(), [&](size_t cs) { return cs <= max_cluster_size; })) {
            break;
        }
        // TODO
        // trace imbalance over time. if it drifts too heavily, reset influence?
        // feels a bit like luck if it can finish
        KMeansRebalancing(points, centroids, max_cluster_size, cluster_sizes, closest_center, influence);
        AggregateClusters(points, centroids, closest_center, vector_sqrt_norms);
    }
    return closest_center;
}
