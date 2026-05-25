#pragma once

#include <cstdint>


struct CombineGroupProgress {
    uint32_t groupId = 0;
    uint32_t tileId = 0;
    uint32_t phase = 0;
    uint32_t completedTasks = 0;
    uint32_t publishedEpoch = 0;
};

