#pragma once

#include "defs.h"
#include "dist.h"

struct HNSWRouter {
    HNSWRouter(PointSet& routing_points, const std::vector<int>& partition_offsets) {

    }

    std::vector<int> Query(float* Q, int /*budget*/) {

        return { }
    }
};
