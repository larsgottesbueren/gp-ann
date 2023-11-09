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

PointSet AggregateClusters(PointSet& P, PointSet& centroids, std::vector<int>& closest_center) {
    PointSet new_centroids = centroids;
    new_centroids.coordinates.assign(centroids.coordinates.size(), 0.f);
	std::vector<size_t> cluster_size(centroids.n, 0);
	for (size_t i = 0; i < closest_center.size(); ++i) {
		int c = closest_center[i];
		cluster_size[c]++;
		float* C = new_centroids.GetPoint(c);
		float* Pi = P.GetPoint(i);
		for (size_t j = 0; j < P.d; ++j) {
			C[j] += Pi[j];
		}
	}

	bool any_zero = false;
	for (size_t i = 0; i < new_centroids.n; ++i) {
		float* C = new_centroids.GetPoint(i);
		if (cluster_size[i] == 0) {
		    any_zero = true;
		    continue;
		}
		for (size_t j = 0; j < P.d; ++j) {
			C[j] /= cluster_size[i];
		}
	}

	if (any_zero) {
	    std::vector<int> remapped_cluster_ids(new_centroids.n, -1);
	    size_t l = 0;
	    for (size_t r = 0; r < new_centroids.n; ++r) {
	        if (cluster_size[r] != 0) {
	            if (l != r) {       // don't do the copy if not necessary
                    float* L = new_centroids.GetPoint(l);
                    float* R = new_centroids.GetPoint(r);
                    for (size_t j = 0; j < new_centroids.d; ++j) {
                        L[j] = R[j];
                    }
	            }
	            remapped_cluster_ids[r] = l;
                l++;
            }
	    }
	    // std::cout << "Removed " << (centroids.n - l) << " empty clusters" << std::endl;
	    if (l <= 10) {
	        std::cout << "<= 10 clusters left -.- num clusters left = " << l << " prev num clusters " << new_centroids.n << std::endl;
	    }
        new_centroids.n = l;
	    for (int& cluster_id : closest_center) {
            cluster_id = remapped_cluster_ids[cluster_id];
	    }
	}

    #ifdef MIPS_DISTANCE
	Normalize(new_centroids);
    #endif
	return new_centroids;
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


std::vector<int> KMeans(PointSet& P, PointSet& centroids, double eps = 5e-4) {
    if (centroids.n < 1) { throw std::runtime_error("KMeans #centroids < 1"); }
	std::vector<int> closest_center(P.n, -1);
	static constexpr size_t NUM_ROUNDS = 25;
	bool finished = false;
	for (size_t r = 0; !finished && r < NUM_ROUNDS; ++r) {
		NearestCenters(P, centroids, closest_center);
		PointSet new_centroids = AggregateClusters(P, centroids, closest_center);
		if (new_centroids.n == centroids.n) {
            finished = parlay::all_of(
                                parlay::tabulate(centroids.n, [&](size_t i) -> float {
                                    return distance(centroids.GetPoint(i), new_centroids.GetPoint(i), centroids.d);
                                }),
                                [&](float dist) {
                                    return dist < eps;
                                }
                            );

        }
		centroids = std::move(new_centroids);
	}
	return closest_center;
}
