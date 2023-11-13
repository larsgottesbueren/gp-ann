#pragma once

#include "routes.h"
#include "shard_searches.h"

struct EmitResult {
    std::vector<double> local_work;
    size_t total_hits;
    double n_probes;
};

void AttributeRecallAndQueryTimeIncreasingNumProbes(const RoutingConfig& route, const ShardSearch& search, size_t num_queries, size_t num_shards, int num_neighbors, std::function<void(EmitResult)>& emit) {
    size_t total_hits = 0;
    std::vector<int> hits_per_query(num_queries, 0);
    std::vector<double> local_work(num_shards, 0.0);
    for (int n_probes = 1; n_probes <= num_shards; ++n_probes) {
        for (size_t q = 0; q < num_queries; ++q) {
            int b = route.buckets_to_probe[q][n_probes - 1];
            int diff = std::min(search.query_hits_in_shard[b][q], num_neighbors - hits_per_query[q]);
            hits_per_query[q] += diff;
            total_hits += diff;
            local_work[b] += search.time_query_in_shard[b][q];
        }
        emit(EmitResult{
                .local_work = local_work,
                .total_hits = total_hits,
                .n_probes = static_cast<double>(n_probes),
        });
    }
}

void AttributeRecallAndQueryTimeVariableNumProbes(const RoutingConfig& route, const ShardSearch& search, size_t num_queries, size_t num_shards, int num_neighbors, std::function<void(EmitResult)>& emit) {
    std::vector<double> local_work(num_shards, 0.0);
    size_t total_hits = 0;
    size_t total_num_probes = 0;
    for (size_t q = 0; q < num_queries; ++q) {
        total_num_probes += route.buckets_to_probe[q].size();
        int hits = 0;
        for (int b : route.buckets_to_probe[q]) {
            hits += search.query_hits_in_shard[b][q];
            local_work[b] += search.time_query_in_shard[b][q];
        }
        hits = std::min(hits, num_neighbors);
        total_hits += hits;
    }
    emit(EmitResult{
            .local_work = local_work,
            .total_hits = total_hits,
            .n_probes = double(total_num_probes) / num_queries,
    });
}

void MaxShardSearchRecall(const std::vector<ShardSearch>& shard_searches, int num_neighbors, int num_queries, int num_shards, int num_requested_shards) {
    for (const auto& search : shard_searches) {
        size_t total_hits = 0;
        for (int q = 0; q < num_queries; ++q) {
            int hits = 0;
            for (const auto& s : search.query_hits_in_shard) {
                hits += s[q];
            }
            total_hits += std::min(num_neighbors, hits);
        }
        double recall = double(total_hits) / double(num_queries);
        std::cout << "Search with ef_search = " << search.ef_search << " scored " << recall << " total recall" << std::endl;
    }
}

void MaxRoutingRecall(const std::vector<RoutingConfig>& routes, const std::vector<NNVec>& ground_truth, int num_neighbors, const std::vector<int>& partition, int num_shards) {
    for (const auto& route : routes) {
        std::vector<size_t> hits(num_shards, 0);
        for (size_t q = 0; q < route.buckets_to_probe.size(); ++q) {
            std::vector<int> overlap(num_shards, 0);
            for (int nb = 0; nb < num_neighbors; ++nb) {
                overlap[partition[ground_truth[q][nb].second]]++;
            }

            for (int nprobes = 0; nprobes < std::min<int>(num_shards, route.buckets_to_probe[q].size()); ++nprobes) {
                hits[nprobes] += overlap[route.buckets_to_probe[q][nprobes]];
            }
        }


        std::cout << "Route with " << route.index_trainer << " " << route.routing_algorithm << " " << route.hnsw_num_voting_neighbors << std::endl;
        size_t hit_sum = 0;
        for (int nprobes = 0; nprobes < num_shards; ++nprobes) {
            hit_sum += hits[nprobes];
            if (route.try_increasing_num_shards) {
                double recall = double(hit_sum) / num_neighbors / ground_truth.size();
                std::cout << "Nprobes = " << nprobes + 1 << " recall = " << recall << "\t";
            }
        }
        if (!route.try_increasing_num_shards) {
            double recall = double(hit_sum) / num_neighbors / ground_truth.size();
            std::cout << "recall = " recall;
        }
        std::cout << std::endl;
    }
}

void PrintCombinationsOfRoutesAndSearches(const std::vector<RoutingConfig>& routes, const std::vector<ShardSearch>& shard_searches, const std::string& output_file,
                                          int num_neighbors, int num_queries, int num_shards, int num_requested_shards, const std::string& part_method) {

    std::ofstream out(output_file);
    // header
    std::string header = "partitioning,shard query,routing query,routing index,ef-search-shard,num voting points,routing time,num probes,recall,QPS,QPS per host,"
                         "QPS without routing, QPS without routing per host,num hosts,num shards,requested num shards\n";
    out << header;

    struct Desc {
        std::string format_string;
        double recall;
        double QPS_per_host;
    };

    std::map<std::string, std::vector<Desc>> outputs;

    for (const auto& route : routes) {
        for (const auto& search : shard_searches) {
            std::function<void(EmitResult)> format_output = [&](const EmitResult& r) -> void {
                double recall = static_cast<double>(r.total_hits) / static_cast<double>(num_neighbors * num_queries);

                auto lwr = r.local_work;
                std::vector<size_t> assigned_hosts(num_shards, 1);
                size_t num_hosts = num_shards;

                for (size_t extra_hosts = 0; extra_hosts < 21; ++extra_hosts, ++num_hosts) {
                    const size_t max_shard = std::distance(lwr.begin(), std::max_element(lwr.begin(), lwr.end()));
                    const double max_latency = lwr[max_shard];

                    {   // output and formatting bits
                        double QPS_without_routing = num_queries / max_latency;
                        double QPS_without_routing_per_host = QPS_without_routing / num_hosts;

                        double total_time = max_latency + (route.routing_time / num_hosts);
                        double QPS = num_queries / total_time;
                        double QPS_per_host = QPS / num_hosts;

                        std::stringstream str;
                        str << part_method << ",HNSW," << route.routing_algorithm << "," << route.index_trainer << ","
                            << search.ef_search << "," << route.hnsw_num_voting_neighbors
                            << "," << route.routing_time / num_queries
                            << "," << r.n_probes << "," << recall << "," << QPS << "," << QPS_per_host
                            << "," << QPS_without_routing << "," << QPS_without_routing_per_host
                            << "," << num_hosts << "," << num_shards << "," << num_requested_shards << "\n";
                        out << str.str() << std::flush;
                        // std::cout << str.str() << std::flush;
                        outputs[route.routing_algorithm].push_back(Desc{ .format_string = str.str(), .recall = recall, .QPS_per_host = QPS_per_host });
                    }

                    // assign one more replica to the slowest shard
                    assigned_hosts[max_shard]++;
                    lwr[max_shard] = r.local_work[max_shard] / assigned_hosts[max_shard];
                }
            };
            if (route.try_increasing_num_shards) {
                AttributeRecallAndQueryTimeIncreasingNumProbes(route, search, num_queries, num_shards, num_neighbors, format_output);
            } else {
                AttributeRecallAndQueryTimeVariableNumProbes(route, search, num_queries, num_shards, num_neighbors, format_output);
            }
        }
    }

    std::ofstream pareto_out(output_file + ".pareto");
    pareto_out << header;
    for (auto& [routing_algo, configs] : outputs) {
        if (configs.empty()) continue;

        auto dominates = [](const Desc& l, const Desc& r) -> bool { return l.recall < r.recall && l.QPS_per_host < r.QPS_per_host; };

        std::vector<Desc> pareto;
        for (const auto& c : configs) {
            bool insert_new = true;
            for (int64_t i = 0; i < pareto.size(); ++i) {
                if (dominates(pareto[i], c)) {  // remove pareto[i]
                    pareto[i] = std::move(pareto.back());
                    pareto.pop_back();
                    --i;
                } else if (dominates(c, pareto[i])) {
                    insert_new = false;
                    break;
                }
            }
            if (insert_new) {
                pareto.push_back(c);
            }
        }

        for (const auto& c : pareto) {
            pareto_out << c.format_string << std::endl;
            // std::cout << c.format_string << std::endl;
        }
    }
}
