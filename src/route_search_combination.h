#pragma once

#include "routes.h"
#include "shard_searches.h"

struct EmitResult {
    std::vector<double> local_work;
    size_t total_hits;
    double n_probes;
};

void AttributeRecallAndQueryTimeIncreasingNumProbes(const RoutingConfig& route, const ShardSearch& search, size_t num_queries, size_t num_shards,
    int num_neighbors, std::function<void(EmitResult)>& emit);

void AttributeRecallAndQueryTimeVariableNumProbes(const RoutingConfig& route, const ShardSearch& search,
    size_t num_queries, size_t num_shards, int num_neighbors, std::function<void(EmitResult)>& emit);

void MaxShardSearchRecall(const std::vector<ShardSearch>& shard_searches, int num_neighbors, int num_queries, int num_shards, int num_requested_shards);

void MaxRoutingRecall(const std::vector<RoutingConfig>& routes, const std::vector<NNVec>& ground_truth, int num_neighbors,
    const std::vector<int>& partition, int num_shards);

void PrintCombinationsOfRoutesAndSearches(const std::vector<RoutingConfig>& routes, const std::vector<ShardSearch>& shard_searches, const std::string& output_file,
                                          int num_neighbors, int num_queries, int num_shards, int num_requested_shards, const std::string& part_method);
