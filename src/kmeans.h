#pragma once

#include "dist.h"
#include <parlay/parallel.h>
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

void AggregateClusters(PointSet& P, PointSet& centroids, std::vector<int>& closest_center) {
	centroids.coordinates.assign(centroids.coordinates.size(), 0.f);
	std::vector<size_t> cluster_size(centroids.n, 0);
	for (size_t i = 0; i < closest_center.size(); ++i) {
		int c = closest_center[i];
		cluster_size[c]++;
		float* C = centroids.GetPoint(c);
		float* Pi = P.GetPoint(i);
		for (size_t j = 0; j < P.d; ++j) {
			C[j] += Pi[j];
		}
	}

	bool any_zero = false;
	for (size_t i = 0; i < centroids.n; ++i) {
		float* C = centroids.GetPoint(i);
		if (cluster_size[i] == 0) {
		    any_zero = true;
		    continue;
		}
		for (size_t j = 0; j < P.d; ++j) {
			C[j] /= cluster_size[i];
		}
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
	    // std::cout << "Removed " << (centroids.n - l) << " empty clusters" << std::endl;
	    if (l <= 10) {
	        std::cout << "<= 10 clusters left -.- num clusters left = " << l << " prev num clusters " << centroids.n << std::endl;
	    }
        centroids.n = l;
	    for (int& cluster_id : closest_center) {
            cluster_id = remapped_cluster_ids[cluster_id];
	    }
	}
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

std::vector<int> KMeans(PointSet& P, PointSet& centroids) {
	std::vector<int> closest_center(P.n, -1);
	static constexpr size_t NUM_ROUNDS = 11;
	for (size_t r = 0; r < NUM_ROUNDS; ++r) {
		NearestCenters(P, centroids, closest_center);
		AggregateClusters(P, centroids, closest_center);
		// TODO stop early?
	}
	return closest_center;
}
