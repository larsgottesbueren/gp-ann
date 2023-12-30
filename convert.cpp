#include <iostream>
#include <parlay/primitives.h>

#include "points_io.h"
#include "metis_io.h"
#include "route_search_combination.h"

int main(int argc, const char* argv[]) {
#if false
    std::string file = argv[1];
    Clusters clusters = ReadClusters(file);
    for (size_t c = 0; c < clusters.size(); ++c) {
        size_t old_size = clusters[c].size();
        auto uniques = parlay::remove_duplicates(clusters[c]);
        size_t dupes = old_size - uniques.size();
        std::cout << "Cluster " << c << " has " << dupes << " duplicates out of " << old_size << " total entries." << std::endl;
    }
#endif
#if false
    std::string file = argv[1];
    Partition partition = ReadMetisPartition(file);
    Clusters clusters = ConvertPartitionToClusters(partition);
    std::string out_file = argv[2];
    WriteClusters(clusters, out_file);
#endif

#if false
    std::string file = argv[1];
    std::string str_num_points = argv[2];
    int num_points = std::stoi(str_num_points);
    PointSet points = ReadPoints(file, num_points);

    std::string out_file = argv[3];
    WritePoints(points, out_file);
#endif

#if true
    if (argc != 6) {
        std::cerr << "Usage ./Convert routes searches output part-method query-file" << std::endl;
        std::abort();
    }

    std::string searches_file = argv[2];
    auto searches = DeserializeShardSearches(searches_file);

    std::string routes_file = argv[1];
    auto routes = DeserializeRoutes(routes_file);

    std::cout << "num routes " << routes.size() << " num searches " << searches.size() << std::endl;

    std::string output_file = argv[3];
    std::string part_method = argv[4];
    std::string part_file = argv[5];
    std::string query_file = argv[6];

    int num_actual_shards = searches.front().neighbors.size();
    std::cout << "num actual shards = " << num_actual_shards << std::endl;

    auto queries = ReadPoints(query_file);
    int num_queries = queries.n;

    PrintCombinationsOfRoutesAndSearches(routes, searches, output_file, 10, num_queries, num_actual_shards, 40, part_method);
#endif
}
