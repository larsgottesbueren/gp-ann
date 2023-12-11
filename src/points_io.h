#pragma once

#include "defs.h"

PointSet ReadPoints(const std::string& path, int64_t size = -1);

void WritePoints(PointSet& points, const std::string& path);

std::vector<NNVec> ReadGroundTruth(const std::string& path);
