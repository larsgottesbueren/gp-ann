#pragma once

#include "defs.h"

std::vector<int> ReadMetisPartition(const std::string& path);

void WriteMetisPartition(const std::vector<int>& partition, const std::string& path);

void WriteMetisGraph(const std::string& path, const AdjGraph& graph);
