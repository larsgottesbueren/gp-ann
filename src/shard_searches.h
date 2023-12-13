#pragma once

#include "defs.h"

struct ShardSearch {
    ShardSearch() { };

    void Init(size_t ef_search, int num_shards, size_t num_queries) {
        this->ef_search = ef_search;
        // query_hits_in_shard.assign(num_shards, std::vector<int>(num_queries, 0));
        time_query_in_shard.assign(num_shards, std::vector<double>(num_queries, 0.0));
        neighbors.assign(num_shards, std::vector<NNVec>(num_queries));
    }

    size_t ef_search = 0;
    // std::vector<std::vector<int>> query_hits_in_shard;
    std::vector<std::vector<NNVec>> neighbors;
    std::vector<std::vector<double>> time_query_in_shard;

    std::string Serialize() const;

    static ShardSearch Deserialize(std::ifstream& in);
};

void SerializeShardSearches(const std::vector<ShardSearch>& shard_searches, const std::string& output_file);

std::vector<ShardSearch> DeserializeShardSearches(const std::string& input_file);


std::vector<ShardSearch> RunInShardSearches(PointSet& points, PointSet& queries, HNSWParameters hnsw_parameters, int num_neighbors,
                                            const Clusters& clusters, int num_shards, const std::vector<float>& distance_to_kth_neighbor);
