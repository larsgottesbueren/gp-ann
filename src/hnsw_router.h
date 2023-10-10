#pragma once

#include "defs.h"
#include "dist.h"

struct HNSWRouter {
    PointSet routing_points;
    std::vector<int> partition_offsets

    HNSWRouter(PointSet routing_points, std::vector<int> partition_offsets) :
        routing_points(std::move(routing_points)),
        partition_offsets(std::move(partition_offsets))
    {

    }

    std::vector<int> Query(float* Q, int /*budget*/) {

        return { }
    }
};
