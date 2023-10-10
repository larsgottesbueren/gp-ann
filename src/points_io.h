#pragma once

#include <fstream>
#include <cstdint>

#include "defs.h"

#include <parlay/parallel.h>

size_t ComputeChunkSize(size_t a, size_t b) {
    return (a+b-1) / b;
}

PointSet ReadPoints(const std::string& path, int64_t size = -1) {
    uint32_t n, d;
    size_t offset = 0;
    {
        std::ifstream in(path, std::ios::binary);
        if (size == -1) {
            in.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
            offset += sizeof(uint32_t);
        } else {
            offset += sizeof(uint32_t);     // TODO this is for testing currently
            n = size;
        }
        in.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
        offset += sizeof(uint32_t);
    }

    std::cout << n << " " << d << std::endl;

    PointSet points;
    points.n = n; points.d = d;

    Timer timer; timer.Start();

    points.coordinates.resize(points.n * points.d);

    std::cout << "alloc + touch done. Took " << timer.Restart() << std::endl;

    size_t num_chunks = 256;
    size_t chunk_size = ComputeChunkSize(points.coordinates.size(), num_chunks);
    parlay::parallel_for(0, num_chunks, [&](size_t i) {
        std::ifstream in(path, std::ios::binary);
        size_t begin = i * chunk_size;
        size_t end = std::min(points.coordinates.size(), (i+1) * chunk_size);

        in.seekg(offset + begin * sizeof(float));

        in.read(reinterpret_cast<char*>(&points.coordinates[begin]), (end-begin) * sizeof(float));
    }, 1);


    std::cout << "Read took " << timer.Stop() << std::endl;

    return points;
}

void WritePoints(PointSet& points, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    uint32_t n = points.n, d = points.d;
    std::cout << n << " " << d << std::endl;
    out.write(reinterpret_cast<const char*>(&n), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&d), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&points.coordinates[0]), points.coordinates.size()*sizeof(float));
}
