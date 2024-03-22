#include "points_io.h"

#include <cstdint>
#include <fstream>

#include "defs.h"

#include <parlay/parallel.h>

size_t ComputeChunkSize(size_t a, size_t b) { return (a + b - 1) / b; }

namespace internal {

    PointSet ReadPoints(const std::string& path, int64_t size) {
        uint32_t n, d;
        size_t offset = 0;
        {
            std::ifstream in(path, std::ios::binary);
            in.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
            offset += sizeof(uint32_t);
            in.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
            offset += sizeof(uint32_t);
        }

        if (size != -1) {
            n = size;
        }

        std::cout << n << " " << d << std::endl;

        PointSet points;
        points.n = n;
        points.d = d;

        Timer timer;
        timer.Start();

        points.coordinates.resize(points.n * points.d);

        std::cout << "alloc + touch done. Took " << timer.Restart() << std::endl;

        size_t num_chunks = parlay::num_workers();
        size_t chunk_size = ComputeChunkSize(points.coordinates.size(), num_chunks);
        parlay::parallel_for(
                0, num_chunks,
                [&](size_t i) {
                    std::ifstream in(path, std::ios::binary);
                    size_t begin = i * chunk_size;
                    size_t end = std::min(points.coordinates.size(), (i + 1) * chunk_size);

                    in.seekg(offset + begin * sizeof(float));

                    in.read(reinterpret_cast<char*>(&points.coordinates[begin]), (end - begin) * sizeof(float));
                },
                1);


        std::cout << "Read took " << timer.Stop() << std::endl;

        return points;
    }

    template<typename CoordinateType>
    PointSet ReadBytes(const std::string& path) {
        uint32_t n, d;
        size_t offset = 0;
        {
            std::ifstream in(path, std::ios::binary);
            in.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
            offset += sizeof(uint32_t);
            in.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
            offset += sizeof(uint32_t);
        }

        std::cout << n << " " << d << std::endl;

        PointSet points;
        points.n = n;
        points.d = d;

        Timer timer;
        timer.Start();

        points.coordinates.resize(points.n * points.d);

        std::cout << "alloc + touch done. Took " << timer.Restart() << std::endl;

        size_t num_chunks = parlay::num_workers();
        size_t chunk_size = ComputeChunkSize(points.coordinates.size(), num_chunks);
        parlay::parallel_for(
                0, num_chunks,
                [&](size_t i) {
                    std::ifstream in(path, std::ios::binary);
                    size_t begin = i * chunk_size;
                    size_t end = std::min(points.coordinates.size(), (i + 1) * chunk_size);

                    in.seekg(offset + begin * sizeof(CoordinateType));

                    std::vector<CoordinateType> buffer(end - begin);
                    in.read(reinterpret_cast<char*>(&buffer[0]), (end - begin) * sizeof(CoordinateType));
                    for (size_t i = 0; i < buffer.size(); ++i) {
                        points.coordinates[begin++] = static_cast<float>(buffer[i]);
                    }
                },
                1);


        std::cout << "Read took " << timer.Stop() << std::endl;

        return points;
    }
} // namespace internal

PointSet ReadPoints(const std::string& path, int64_t size) {
    if (path.ends_with(".fbin")) {
        return internal::ReadPoints(path, size);
    } else if (path.ends_with(".u8bin")) {
        return internal::ReadBytes<uint8_t>(path);
    } else if (path.ends_with(".i8bin")) {
        return internal::ReadBytes<int8_t>(path);
    } else {
        throw std::runtime_error("Invalid file ending for the pointset path. Valid options are [.fbin, .u8bin, .i8bin]");
    }
}

void WritePoints(PointSet& points, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    uint32_t n = points.n, d = points.d;
    std::cout << n << " " << d << std::endl;
    out.write(reinterpret_cast<const char*>(&n), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&d), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&points.coordinates[0]), points.coordinates.size() * sizeof(float));
}


std::vector<NNVec> ReadGroundTruth(const std::string& path) {
    uint32_t num_queries = 0;
    uint32_t num_neighbors = 0;
    std::ifstream in(path, std::ios::binary);
    in.read(reinterpret_cast<char*>(&num_queries), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&num_neighbors), sizeof(uint32_t));
    std::cout << "num queries = " << num_queries << " num neighbors = " << num_neighbors << std::endl;

    std::vector<NNVec> gt(num_queries, NNVec(num_neighbors));

    for (uint32_t q = 0; q < num_queries; ++q) {
        for (uint32_t j = 0; j < num_neighbors; ++j) {
            in.read(reinterpret_cast<char*>(&gt[q][j].second), sizeof(uint32_t));
        }
    }

    for (uint32_t q = 0; q < num_queries; ++q) {
        for (uint32_t j = 0; j < num_neighbors; ++j) {
            in.read(reinterpret_cast<char*>(&gt[q][j].first), sizeof(float));
        }
    }

    return gt;
}


void WriteGroundTruth(const std::string& path, const std::vector<NNVec>& ground_truth) {
    uint32_t num_queries = ground_truth.size();
    uint32_t num_neighbors = ground_truth[0].size();
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&num_queries), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&num_neighbors), sizeof(uint32_t));
    for (uint32_t q = 0; q < num_queries; ++q) {
        for (uint32_t j = 0; j < num_neighbors; ++j) {
            out.write(reinterpret_cast<const char*>(&ground_truth[q][j].second), sizeof(uint32_t));
        }
    }

    for (uint32_t q = 0; q < num_queries; ++q) {
        for (uint32_t j = 0; j < num_neighbors; ++j) {
            out.write(reinterpret_cast<const char*>(&ground_truth[q][j].first), sizeof(float));
        }
    }
}
