#pragma once

#include "kmeans.h"
#include "inverted_index.h"

struct KMeansTreeNode {
	std::vector<KMeansTreeNode> children;
	PointSet centroids;
};

struct Options {
    int num_centroids = 64;
    int min_cluster_size = 250;
};

struct KMeansTreeRouter {
    std::vector<KMeansTreeNode> roots;
    int num_shards;

    void Train(InvertedIndex& inv_ind, const Options& options) {
        const auto& offsets = inv_ind.offsets;
        auto& P = inv_ind.reordered_P;
        num_shards = offsets.size() - 1;

        for (int b = 0; b < num_shards; ++b) {
            size_t shard_size = offsets[b+1] - offsets[b];
            PointSet points_in_shard;
            points_in_shard.coordinates_begin = P.GetPoint(offsets[b]);
            points_in_shard.d = P.d;
            points_in_shard.n = shard_size;
        }


    }

	std::vector<int> Query(float* Q, int num_shards) {
	    std::vector<int> probes(num_shards);
	    for (int i = 0; i < num_shards; ++i) { probes[i] = i; }
	    return probes;
	}
};
