#include <iostream>

#include "knn_graph.h"
#include "points_io.h"
#include "metis_io.h"
#include "recall.h"
#include "kmeans_tree.h"
#include "inverted_index.h"

int main(int argc, const char* argv[]) {


    if (argc != 5) {
        std::cerr << "Usage ./RunQueries input-points queries k partition" << std::endl;
        std::abort();
    }

    std::string point_file = argv[1];
    std::string query_file = argv[2];
    std::string k_string = argv[3];
    int k = std::stoi(k_string);
    std::string partition_file = argv[4];
    PointSet points = ReadPoints(point_file);
    PointSet queries = ReadPoints(query_file);
    std::vector<int> partition = ReadMetisPartition(partition_file);

    std::vector<float> distance_to_kth_neighbor = ComputeDistanceToKthNeighbor(points, queries, k);

    InvertedIndex inverted_index(points, partition, k);
    KMeansTreeRouter router;


}
