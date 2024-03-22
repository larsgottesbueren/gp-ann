#pragma once

#include "defs.h"

PointSet ReadPoints(const std::string& path, int64_t size = -1);

PointSet ReadBytePoints(const std::string& path, bool is_signed);

void WritePoints(PointSet& points, const std::string& path);

std::vector<NNVec> ReadGroundTruth(const std::string& path);

void WriteGroundTruth(const std::string& path, const std::vector<NNVec>& ground_truth);
