#ifndef MOE_SWIGLU_SEGMENT_HPP
#define MOE_SWIGLU_SEGMENT_HPP

#include <cstdint>

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#include "kernel_operator.h"
#define MOE_SWIGLU_SEGMENT_FN AICORE inline
#else
#define MOE_SWIGLU_SEGMENT_FN inline constexpr
#endif

MOE_SWIGLU_SEGMENT_FN uint32_t MoeSwigluSegmentNum(uint32_t expertPerRank)
{
    return expertPerRank <= 1U ? expertPerRank : 2U;
}

MOE_SWIGLU_SEGMENT_FN uint32_t MoeSwigluEpilogueGranularity(uint32_t expertPerRank)
{
    if (expertPerRank <= 1U) {
        return expertPerRank;
    }
    return expertPerRank <= 4U ? expertPerRank - 1U : expertPerRank - 3U;
}

MOE_SWIGLU_SEGMENT_FN uint32_t MoeSwigluSegmentStartExpert(uint32_t expertPerRank, uint32_t segmentIdx)
{
    return segmentIdx == 0U ? 0U : MoeSwigluEpilogueGranularity(expertPerRank);
}

MOE_SWIGLU_SEGMENT_FN uint32_t MoeSwigluSegmentEndExpert(uint32_t expertPerRank, uint32_t segmentIdx)
{
    return segmentIdx == 0U && MoeSwigluSegmentNum(expertPerRank) == 2U ? MoeSwigluEpilogueGranularity(expertPerRank) :
                                                                          expertPerRank;
}

#undef MOE_SWIGLU_SEGMENT_FN

#endif // MOE_SWIGLU_SEGMENT_HPP
