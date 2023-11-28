#include <iostream>

#include "route_search_combination.h"

int main(int argc, const char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage ./Convert routes searches output part-method num-actual-shards" << std::endl;
        std::abort();
    }

    std::string routes_file = argv[1];
    auto routes = DeserializeRoutes(routes_file);

    std::string searches_file = argv[2];
    auto searches = DeserializeShardSearches(searches_file);

    std::cout << "num routes " << routes.size() << " num searches " << searches.size() << std::endl;

    std::string output_file = argv[3];
    std::string part_method = argv[4];
    std::string num_actual_shards_str = argv[5];
    int num_actual_shards = std::stoi(num_actual_shards_str);
    PrintCombinationsOfRoutesAndSearches(routes, searches, output_file,
    10, 100000, num_actual_shards, 40, part_method);
}
