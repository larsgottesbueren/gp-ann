#pragma once

#include "dist.h"
#include <parlay/parallel.h>

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
	for (size_t i = 0; i < centroids.n; ++i) {
		float* C = centroids.GetPoint(i);
		for (size_t j = 0; j < P.d; ++j) {
			C[j] /= cluster_size[i];
		}
	}
}

void KMeans(PointSet& P, PointSet& centroids, size_t d, size_t n) {
	centroids.coordinates.assign(centroids.coordinates.size(), 0.f);
	std::vector<int> closest_center(P.n, -1);
	static constexpr size_t NUM_ROUNDS = 11;
	for (size_t r = 0; r < NUM_ROUNDS; ++r) {
		NearestCenters(P, centroids, closest_center);
		AggregateClusters(P, centroids, closest_center);
		// TODO stop early?
	}
}
