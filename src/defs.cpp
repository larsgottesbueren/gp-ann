#include "defs.h"

#include <vector>
#include <cstdint>
#include <chrono>
#include <algorithm>

#include "topn.h"
#include "dist.h"

PointSet ExtractPointsInBucket(const std::vector<uint32_t>& bucket, PointSet& points) {
    PointSet ps;
    ps.n = bucket.size();
    ps.d = points.d;
    ps.coordinates.reserve(ps.n * ps.d);
    for (auto u : bucket) {
        float* p = points.GetPoint(u);
        for (size_t j = 0; j < points.d; ++j) {
            ps.coordinates.push_back(p[j]);
        }
    }
    return ps;
}

int NumPartsInPartition(const std::vector<int>& partition) {
    if (partition.empty()) return 0;
    return *std::max_element(partition.begin(), partition.end()) + 1;
}

std::vector<std::vector<uint32_t>> ConvertPartitionToClusters(const std::vector<int>& partition) {
    int num_buckets = NumPartsInPartition(partition);
    std::vector<std::vector<uint32_t>> buckets(num_buckets);
    for (uint32_t u = 0; u < partition.size(); ++u) {
        buckets[partition[u]].push_back(u);
    }
    return buckets;
}

void RemapPartitionIDs(std::vector<int>& partition) {
    if (partition.empty()) return;
    int num_shards = NumPartsInPartition(partition);
    std::vector<size_t> num_points_in_shard(num_shards, 0);
    for (int x : partition) { num_points_in_shard[x]++; }
    if (std::any_of(num_points_in_shard.begin(), num_points_in_shard.end(), [](const auto& n) { return n == 0; })) {
        std::vector<int> remapped(num_shards, 0);
        int l = 0;
        for (int r = 0; r < num_shards; ++r) {
            if (num_points_in_shard[r] > 0) {
                remapped[r] = l++;
            }
        }
        for (int& x : partition) { x = remapped[x]; }
    }
}

NNVec ConvertTopKToNNVec(TopN& top_k) {
    NNVec res = top_k.Take();
    std::reverse(res.begin(), res.end());
    return res;
}

TopN ClosestLeaders(PointSet& points, PointSet& leader_points, uint32_t my_id, int k) {
    TopN top_k(k);
    float* Q = points.GetPoint(my_id);
    for (uint32_t j = 0; j < leader_points.n; ++j) {
        float* P = leader_points.GetPoint(j);
        float dist = distance(P, Q, points.d);
        top_k.Add(std::make_pair(dist, j));
    }
    return top_k;
}
