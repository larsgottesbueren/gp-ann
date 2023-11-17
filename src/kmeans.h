#pragma once

#include "dist.h"
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <numeric>
#include <random>

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
	parlay::parallel_for(0, P.n, [&](size_t i) {
		closest_center[i] = Top1Neighbor(centroids, P.GetPoint(i));
	});
}

std::vector<size_t> AggregateClusters(PointSet& P, PointSet& centroids, std::vector<int>& closest_center) {
    centroids.coordinates.assign(centroids.coordinates.size(), 0.f);
	std::vector<size_t> cluster_size(centroids.n, 0);
    #ifdef MIPS_DISTANCE
	std::vector<double> norm_sums(centroids.n, 0.0);
    #endif
	for (size_t i = 0; i < closest_center.size(); ++i) {
		int c = closest_center[i];
		cluster_size[c]++;
		float* C = centroids.GetPoint(c);
		float* Pi = P.GetPoint(i);
        #ifdef MIPS_DISTANCE
        double norm = vec_norm(Pi, centroids.d);
        norm_sums[c] += norm;
        float multiplier = 1.0f / std::sqrt(norm);
        for (size_t j = 0; j < P.d; ++j) {
            C[j] += Pi[j] * multiplier;
        }
        #else
        for (size_t j = 0; j < P.d; ++j) {
            C[j] += Pi[j];
        }
        #endif
	}

	bool any_zero = false;
	for (size_t c = 0; c < centroids.n; ++c) {
		float* C = centroids.GetPoint(c);
		if (cluster_size[c] == 0) {
		    any_zero = true;
		    continue;
		}
        #ifdef MIPS_DISTANCE
		double desired_norm = norm_sums[c] / cluster_size[c];
		double current_norm = vec_norm(C, centroids.d);
		double multiplier = std::sqrt(desired_norm / current_norm);
		for (size_t j = 0; j < P.d; ++j) {
		    C[j] *= multiplier;
		}
        #else
		for (size_t j = 0; j < P.d; ++j) {
			C[j] /= cluster_size[c];
		}
        #endif
	}

	if (any_zero) {
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
	    for (int& cluster_id : closest_center) {
            cluster_id = remapped_cluster_ids[cluster_id];
            if (cluster_id == -1) throw std::runtime_error("ClusterID -1");
	    }
	}

	return cluster_size;
}

void AggregateClustersParallel(PointSet& P, PointSet& centroids, std::vector<int>& closest_center) {
    auto clusters = parlay::group_by_index(
            parlay::delayed_tabulate(
                    closest_center.size(),
                    [&](size_t i) { return std::make_pair(closest_center[i], i); }
            ), centroids.n);

    centroids.coordinates.assign(centroids.coordinates.size(), 0.f);

    parlay::parallel_for(0, clusters.size(), [&](int c) {
        const auto& cluster = clusters[c];
        float* C = centroids.GetPoint(c);
        for (auto u : cluster) {
            float* Pu = P.GetPoint(u);
            for (size_t j = 0; j < P.d; ++j) {
                C[j] += Pu[j];
            }
        }
        if (!clusters.empty()) {
            for (size_t j = 0; j < P.d; ++j) {
                C[j] /= cluster.size();
            }
        }
    }, 1);

    bool any_zero = parlay::any_of(clusters, [&](const auto& C) { return C.empty(); });

    if (any_zero) {
        std::vector<int> remapped_cluster_ids(centroids.n, -1);
        size_t l = 0;
        for (size_t r = 0; r < centroids.n; ++r) {
            if (!clusters[r].empty()) {
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
        for (int& cluster_id : closest_center) {
            cluster_id = remapped_cluster_ids[cluster_id];
            if (cluster_id == -1) throw std::runtime_error("ClusterID -1");
        }
    }

    #ifdef MIPS_DISTANCE
    Normalize(centroids);
    #endif
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
	static constexpr size_t NUM_ROUNDS = 20;
	for (size_t r = 0; r < NUM_ROUNDS; ++r) {
		NearestCenters(P, centroids, closest_center);
        AggregateClusters(P, centroids, closest_center);
	}
	return closest_center;
}

std::vector<int> BalancedKMeans(PointSet& points, PointSet& centroids, size_t max_cluster_size) {
    std::vector<int> closest_center(points.n, -1);
    static constexpr size_t NUM_ROUNDS = 20;
    std::vector<size_t> cluster_sizes;
    for (size_t r = 0; r < NUM_ROUNDS; ++r) {
        NearestCenters(points, centroids, closest_center);
        cluster_sizes = AggregateClusters(points, centroids, closest_center);
    }
}
