#pragma once

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"

#ifndef V4_FORCE_INLINE_AICORE
#define V4_FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__
#endif

struct GmmTileRuntimeMeta {
    uint32_t rowBegin = 0;
    uint32_t rowEnd = 0;
    uint32_t kBegin = 0;
    uint32_t kEnd = 0;
    uint32_t nBegin = 0;
    uint32_t nEnd = 0;
    uint32_t localExpert = 0;
    uint32_t expertRangeIndex = 0;
};

struct Gmm2TileRuntimeMeta {
    uint32_t rowBegin = 0;
    uint32_t rowEnd = 0;
    uint32_t k2Begin = 0;
    uint32_t k2End = 0;
    uint32_t n2Begin = 0;
    uint32_t n2End = 0;
    uint32_t localExpert = 0;
    uint32_t expertRangeIndex = 0;
};

#endif
