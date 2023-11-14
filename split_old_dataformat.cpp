#include <iostream>

#include "shard_searches.h"
#include "routes.h"
#include "route_search_combination.h"

void OldSerialize(const std::vector<RoutingConfig>& routes, const std::vector<ShardSearch>& shard_searches, const std::string& output_file) {
    std::ofstream out(output_file);
    out << routes.size() << " " << shard_searches.size() << std::endl;
    for (const RoutingConfig& r : routes) {
        out << "R" << std::endl;
        out << r.Serialize();
    }
    for (const ShardSearch& search : shard_searches) {
        out << "S" << std::endl;
        out << search.Serialize();
    }
}

void OldDeserialize(std::vector<RoutingConfig>& routes, std::vector<ShardSearch>& shard_searches, const std::string& input_file) {
    std::ifstream in(input_file);
    size_t num_routes, num_searches;
    std::string header;
    std::getline(in, header);
    std::istringstream iss(header);
    iss >> num_routes >> num_searches;
    std::cout << "nr=" << num_routes << " ns=" << num_searches << std::endl;
    for (size_t i = 0; i < num_routes; ++i) {
        std::getline(in, header);
        std::cout << "i = " << i << " for routes " << std::endl;
        if (header != "R") std::cout << "routing config doesn't start with marker R. Instead: " << header << std::endl;
        RoutingConfig r = RoutingConfig::Deserialize(in);
        routes.push_back(std::move(r));
    }
    for (size_t i = 0; i < num_searches; ++i) {
        std::getline(in, header);
        if (header != "S") std::cout << "search config doesn't start with marker S. Instead: " << header << std::endl;
        ShardSearch s = ShardSearch::Deserialize(in);
        shard_searches.push_back(std::move(s));
    }
}

int main2(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage ./QueryAttribution output-file" << std::endl;
        std::abort();
    }

    std::string output_file = argv[1];

    std::vector<ShardSearch> shard_searches;
    std::vector<RoutingConfig> routes;

    OldDeserialize(routes, shard_searches, output_file + "routes_and_searches.txt");

    SerializeRoutes(routes, output_file + ".searches");
    SerializeShardSearches(shard_searches, output_file + ".searches");
}

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage ./QueryAttribution output-file" << std::endl;
        std::abort();
    }

    std::string output_file = argv[1];

    std::vector<ShardSearch> shard_searches = DeserializeShardSearches(output_file);
    MaxShardSearchRecall(shard_searches, 10, shard_searches[0].query_hits_in_shard[0].size(), 40, shard_searches[0].query_hits_in_shard.size());

}
