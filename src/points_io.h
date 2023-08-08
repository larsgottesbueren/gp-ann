#pragma once

#include <fstream>
#include <cstdint>

#include "defs.h"

PointSet ReadPoints(const std::string& path, int64_t size = -1) {
    std::ifstream in(path, std::ios::binary);
    uint32_t n, d;
    if (size == -1) {
        in.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
    } else {
        n = size;
    }
    in.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));

    std::cout << n << " " << d << std::endl;

    PointSet points;
    points.n = n; points.d = d;
    points.coordinates.resize(points.n * points.d);
    in.read(reinterpret_cast<char*>(&points.coordinates[0]), points.n * points.d * sizeof(float));
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
