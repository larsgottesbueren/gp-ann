#include <iostream>

#include "route_search_combination.h"

int main(int argc, const char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage ./Convert routes searches" << std::endl;
        std::abort();
    }

    std::string routes_file = argv[1];
    auto routes = DeserializeRoutes(routes_file);

    std::string searches_file = argv[2];
    auto searches = DeserializeShardSearches(searches_file);

    std::cout << "num routes " << routes.size() << " num searches " << searches.size() << std::endl;
}
