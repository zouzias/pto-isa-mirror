#pragma once

#include <stddef.h>
#include <stdint.h>

#include "op_kernel/profile_debug_config.h"

#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
constexpr size_t kDispatchFFNCombineProfileEntryBytes = 8192U;
#else
constexpr size_t kDispatchFFNCombineProfileEntryBytes = 4096U;
#endif
constexpr size_t kDispatchFFNCombineProfileEntriesPerBlock = 3U; // cube + two vec subblocks
constexpr size_t kDispatchFFNCombineProfileBytesPerBlock =
    kDispatchFFNCombineProfileEntryBytes * kDispatchFFNCombineProfileEntriesPerBlock;

constexpr size_t kDispatchFFNCombineProfileKernelStart = 0U;
constexpr size_t kDispatchFFNCombineProfileKernelEnd = 1U;
constexpr size_t kDispatchFFNCombineProfileStageBase = 2U;
constexpr size_t kDispatchFFNCombineProfileStageCount = 7U;
constexpr size_t kDispatchFFNCombineProfileFrontDoneBase =
    kDispatchFFNCombineProfileStageBase + kDispatchFFNCombineProfileStageCount * 2U;
constexpr size_t kDispatchFFNCombineProfileFrontDoneStepCount = 4U;
constexpr size_t kDispatchFFNCombineProfileReadyStageBase =
    kDispatchFFNCombineProfileFrontDoneBase + kDispatchFFNCombineProfileFrontDoneStepCount * 2U;
constexpr size_t kDispatchFFNCombineProfileReadyStageCount = 4U;
constexpr size_t kDispatchFFNCombineProfileCombineDetailBase =
    kDispatchFFNCombineProfileReadyStageBase + kDispatchFFNCombineProfileReadyStageCount;
constexpr size_t kDispatchFFNCombineProfileCombineDetailFieldCount = 4U;
constexpr size_t kDispatchFFNCombineProfileCombineDetailCount = 20U;
constexpr size_t kDispatchFFNCombineProfileGmm1DetailBase =
    kDispatchFFNCombineProfileCombineDetailBase +
    kDispatchFFNCombineProfileCombineDetailCount * kDispatchFFNCombineProfileCombineDetailFieldCount;
constexpr size_t kDispatchFFNCombineProfileGmm1DetailFieldCount = 4U;
constexpr size_t kDispatchFFNCombineProfileGmm1DetailMaxExperts = 16U;
constexpr size_t kDispatchFFNCombineProfileFrontDetailBase =
    kDispatchFFNCombineProfileGmm1DetailBase +
    kDispatchFFNCombineProfileGmm1DetailMaxExperts * kDispatchFFNCombineProfileGmm1DetailFieldCount;
constexpr size_t kDispatchFFNCombineProfileFrontDetailFieldCount = 4U;
constexpr size_t kDispatchFFNCombineProfileFrontDetailCount = 34U;
constexpr size_t kDispatchFFNCombineProfileDispatchDetailBase =
    kDispatchFFNCombineProfileFrontDetailBase +
    kDispatchFFNCombineProfileFrontDetailCount * kDispatchFFNCombineProfileFrontDetailFieldCount;
constexpr size_t kDispatchFFNCombineProfileDispatchDetailFieldCount = 4U;
constexpr size_t kDispatchFFNCombineProfileDispatchDetailCount = 18U;
constexpr size_t kDispatchFFNCombineProfileUnpermuteDetailBase =
    kDispatchFFNCombineProfileDispatchDetailBase +
    kDispatchFFNCombineProfileDispatchDetailCount * kDispatchFFNCombineProfileDispatchDetailFieldCount;
constexpr size_t kDispatchFFNCombineProfileUnpermuteDetailFieldCount = 4U;
constexpr size_t kDispatchFFNCombineProfileUnpermuteDetailCount = 17U;
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
constexpr size_t kDispatchFFNCombineProfileExpertWallFieldCount = 2U;
constexpr size_t kDispatchFFNCombineProfileCombineExpertWallBase =
    kDispatchFFNCombineProfileUnpermuteDetailBase +
    kDispatchFFNCombineProfileUnpermuteDetailCount * kDispatchFFNCombineProfileUnpermuteDetailFieldCount;
constexpr size_t kDispatchFFNCombineProfileGmm2ExpertWallBase =
    kDispatchFFNCombineProfileCombineExpertWallBase +
    kDispatchFFNCombineProfileGmm1DetailMaxExperts * kDispatchFFNCombineProfileExpertWallFieldCount;
constexpr size_t kDispatchFFNCombineProfileGmm1WaitDetailBase =
    kDispatchFFNCombineProfileGmm2ExpertWallBase +
    kDispatchFFNCombineProfileGmm1DetailMaxExperts * kDispatchFFNCombineProfileExpertWallFieldCount;
constexpr size_t kDispatchFFNCombineProfileGmm1WaitDetailFieldCount = 4U;
constexpr size_t kDispatchFFNCombineProfileGmm1WaitDetailCount = 2U;
constexpr size_t kDispatchFFNCombineProfileDispatchV2CTraceBase =
    kDispatchFFNCombineProfileGmm1WaitDetailBase +
    kDispatchFFNCombineProfileGmm1WaitDetailCount * kDispatchFFNCombineProfileGmm1WaitDetailFieldCount;
constexpr size_t kDispatchFFNCombineProfileDispatchV2CTraceFieldCount = 4U;
constexpr size_t kDispatchFFNCombineProfileDispatchV2CTraceGroupCount = 16U;
constexpr size_t kDispatchFFNCombineProfileDispatchV2CTraceCount = kDispatchFFNCombineProfileDispatchV2CTraceGroupCount;
constexpr size_t kDispatchFFNCombineProfileGmm2ToCombineTraceBase =
    kDispatchFFNCombineProfileDispatchV2CTraceBase +
    kDispatchFFNCombineProfileDispatchV2CTraceCount * kDispatchFFNCombineProfileDispatchV2CTraceFieldCount;
constexpr size_t kDispatchFFNCombineProfileGmm2ToCombineTraceFieldCount = 4U;
constexpr size_t kDispatchFFNCombineProfileGmm2ToCombineTraceGroupCount = 16U;
constexpr size_t kDispatchFFNCombineProfileGmm2ToCombineTraceCount =
    kDispatchFFNCombineProfileGmm2ToCombineTraceGroupCount;
constexpr size_t kDispatchFFNCombineProfileGmm2CvDetailBase =
    kDispatchFFNCombineProfileGmm2ToCombineTraceBase +
    kDispatchFFNCombineProfileGmm2ToCombineTraceCount * kDispatchFFNCombineProfileGmm2ToCombineTraceFieldCount;
constexpr size_t kDispatchFFNCombineProfileGmm2CvDetailFieldCount = 4U;
constexpr size_t kDispatchFFNCombineProfileGmm2CvDetailCount = 1U;
#endif
constexpr size_t kDispatchFFNCombineProfileEntryU64Count = kDispatchFFNCombineProfileEntryBytes / sizeof(uint64_t);

enum DispatchFFNCombineProfileStage : size_t
{
    DISPATCH_FFN_COMBINE_PROFILE_STAGE_FRONT = 0U,
    DISPATCH_FFN_COMBINE_PROFILE_STAGE_DISPATCH = 1U,
    DISPATCH_FFN_COMBINE_PROFILE_STAGE_GMM1 = 2U,
    DISPATCH_FFN_COMBINE_PROFILE_STAGE_SWIGLU = 3U,
    DISPATCH_FFN_COMBINE_PROFILE_STAGE_GMM2 = 4U,
    DISPATCH_FFN_COMBINE_PROFILE_STAGE_COMBINE = 5U,
    DISPATCH_FFN_COMBINE_PROFILE_STAGE_UNPERMUTE = 6U,
};

enum DispatchFFNCombineProfileFrontDoneStep : size_t
{
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DONE_STEP7_ENTRY_SYNC = 0U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DONE_STEP8_NOTIFY = 1U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DONE_STEP9_WAIT = 2U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DONE_STEP10_EXIT_SYNC = 3U,
};

enum DispatchFFNCombineProfileReadyStage : size_t
{
    DISPATCH_FFN_COMBINE_PROFILE_READY_STAGE_GMM1 = 0U,
    DISPATCH_FFN_COMBINE_PROFILE_READY_STAGE_SWIGLU = 1U,
    DISPATCH_FFN_COMBINE_PROFILE_READY_STAGE_GMM2 = 2U,
    DISPATCH_FFN_COMBINE_PROFILE_READY_STAGE_COMBINE = 3U,
};

enum DispatchFFNCombineProfileCombineDetail : size_t
{
    DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_DIRECT_SMALL_TOTAL = 0U,
    DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_SMALL_GROUP = 1U,
    DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_SMALL_TILE = 2U,
    DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_SMALL_WAIT_GMM2 = 3U,
    DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_SMALL_DEQUANT_TOTAL = 4U,
    DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_SMALL_DEQUANT_SCALE_MUL = 5U,
    DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_SMALL_RANK_INTERSECT = 6U,
    DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_DIRECT_LARGE_TOTAL = 7U,
    DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_LARGE_GROUP = 8U,
    DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_LARGE_WAIT_GMM2_FLAG = 9U,
    DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_LARGE_WAIT_GMM2_SYNC = 10U,
    DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_LARGE_SEGMENT = 11U,
    DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_LARGE_ROW = 12U,
    DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_LARGE_STORE = 13U,
    DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_FINAL_BOUNDARY = 14U,
    DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_FINAL_SYNC = 15U,
    DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_FINAL_CROSS_RANK_SYNC = 16U,
    DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_CV_POP_WAIT = 17U,
    DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_CV_SCALE_LOAD = 18U,
    DISPATCH_FFN_COMBINE_PROFILE_COMBINE_DETAIL_CV_FREE = 19U,
};

enum DispatchFFNCombineProfileFrontDetail : size_t
{
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_FULL_LOAD_TOTAL = 0U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_SINGLE_TOTAL = 1U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_MULTI_TOTAL = 2U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_FULL_LOAD_SORT = 3U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_FULL_LOAD_COPY_EXPANDED = 4U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_FULL_LOAD_COUNT = 5U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_PRODUCER_BOUNDARY = 6U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_SINGLE_SORT = 7U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_SORT_VBS = 8U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_SORT_LOCAL_MERGE = 9U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_SORT_MIDDLE_MERGE = 10U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_SORT_OUT = 11U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_METADATA = 12U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_SRC_TO_DST = 13U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_QUANT = 14U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_EXCHANGE_COUNTS = 15U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_EXCHANGE_PREPARE = 16U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_EXCHANGE_PUBLISH = 17U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_EXCHANGE_WAIT_RESTORE = 18U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_CUMSUM = 19U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_DONE_NOTIFY = 20U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_ROUTE_SYNC = 21U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_EXCHANGE_PRE_PUBLISH_SYNC = 22U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_EXCHANGE_RESTORE_TOTAL = 23U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_EXCHANGE_WAIT_MARKER = 24U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_EXCHANGE_RESTORE_COPY = 25U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_EXCHANGE_PRESUM = 26U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_EXCHANGE_STORE_PRESUM = 27U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_EXCHANGE_POST_RESTORE_SYNC = 28U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_EXCHANGE_FINAL_DCCI = 29U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_CUMSUM_INIT = 30U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_CUMSUM_ROWS = 31U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_CUMSUM_EXPERT_NUMS = 32U,
    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DETAIL_CUMSUM_FINAL_SYNC = 33U,
};

enum DispatchFFNCombineProfileDispatchDetail : size_t
{
    DISPATCH_FFN_COMBINE_PROFILE_DISPATCH_DETAIL_TOTAL = 0U,
    DISPATCH_FFN_COMBINE_PROFILE_DISPATCH_DETAIL_SETUP_FLAGS = 1U,
    DISPATCH_FFN_COMBINE_PROFILE_DISPATCH_DETAIL_INITIAL_READY = 2U,
    DISPATCH_FFN_COMBINE_PROFILE_DISPATCH_DETAIL_PREV_SUM_INIT = 3U,
    DISPATCH_FFN_COMBINE_PROFILE_DISPATCH_DETAIL_GROUP = 4U,
    DISPATCH_FFN_COMBINE_PROFILE_DISPATCH_DETAIL_GROUP_METADATA = 5U,
    DISPATCH_FFN_COMBINE_PROFILE_DISPATCH_DETAIL_SOURCE_RANK = 6U,
    DISPATCH_FFN_COMBINE_PROFILE_DISPATCH_DETAIL_FETCH_TOTAL = 7U,
    DISPATCH_FFN_COMBINE_PROFILE_DISPATCH_DETAIL_FETCH_BATCH = 8U,
    DISPATCH_FFN_COMBINE_PROFILE_DISPATCH_DETAIL_FETCH_LOAD = 9U,
    DISPATCH_FFN_COMBINE_PROFILE_DISPATCH_DETAIL_FETCH_STORE = 10U,
    DISPATCH_FFN_COMBINE_PROFILE_DISPATCH_DETAIL_STORE_WAIT = 11U,
    DISPATCH_FFN_COMBINE_PROFILE_DISPATCH_DETAIL_STORE_ROWS = 12U,
    DISPATCH_FFN_COMBINE_PROFILE_DISPATCH_DETAIL_STORE_SCALES = 13U,
    DISPATCH_FFN_COMBINE_PROFILE_DISPATCH_DETAIL_GROUP_SYNC = 14U,
    DISPATCH_FFN_COMBINE_PROFILE_DISPATCH_DETAIL_GROUP_READY = 15U,
    DISPATCH_FFN_COMBINE_PROFILE_DISPATCH_DETAIL_FINAL_WAIT_EVENTS = 16U,
    DISPATCH_FFN_COMBINE_PROFILE_DISPATCH_DETAIL_FINAL_SYNC = 17U,
};

enum DispatchFFNCombineProfileUnpermuteDetail : size_t
{
    DISPATCH_FFN_COMBINE_PROFILE_UNPERMUTE_DETAIL_TOTAL = 0U,
    DISPATCH_FFN_COMBINE_PROFILE_UNPERMUTE_DETAIL_SET_FLAGS = 1U,
    DISPATCH_FFN_COMBINE_PROFILE_UNPERMUTE_DETAIL_METADATA_PREFETCH = 2U,
    DISPATCH_FFN_COMBINE_PROFILE_UNPERMUTE_DETAIL_METADATA_WAIT = 3U,
    DISPATCH_FFN_COMBINE_PROFILE_UNPERMUTE_DETAIL_BATCH = 4U,
    DISPATCH_FFN_COMBINE_PROFILE_UNPERMUTE_DETAIL_TOKEN = 5U,
    DISPATCH_FFN_COMBINE_PROFILE_UNPERMUTE_DETAIL_FILL_ACC = 6U,
    DISPATCH_FFN_COMBINE_PROFILE_UNPERMUTE_DETAIL_TOPK_LOOP = 7U,
    DISPATCH_FFN_COMBINE_PROFILE_UNPERMUTE_DETAIL_LOAD_D = 8U,
    DISPATCH_FFN_COMBINE_PROFILE_UNPERMUTE_DETAIL_ACCUMULATE = 9U,
    DISPATCH_FFN_COMBINE_PROFILE_UNPERMUTE_DETAIL_STORE_OUTPUT = 10U,
    DISPATCH_FFN_COMBINE_PROFILE_UNPERMUTE_DETAIL_STORE_WAIT = 11U,
    DISPATCH_FFN_COMBINE_PROFILE_UNPERMUTE_DETAIL_STORE_CAST = 12U,
    DISPATCH_FFN_COMBINE_PROFILE_UNPERMUTE_DETAIL_STORE_TSTORE = 13U,
    DISPATCH_FFN_COMBINE_PROFILE_UNPERMUTE_DETAIL_FINALIZE_PIPE = 14U,
    DISPATCH_FFN_COMBINE_PROFILE_UNPERMUTE_DETAIL_IDLE_FINALIZE = 15U,
    DISPATCH_FFN_COMBINE_PROFILE_UNPERMUTE_DETAIL_DEBUG_DONE = 16U,
};

#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
enum DispatchFFNCombineProfileGmm1WaitDetail : size_t
{
    DISPATCH_FFN_COMBINE_PROFILE_GMM1_WAIT_GROUP_V2C = 0U,
    DISPATCH_FFN_COMBINE_PROFILE_GMM1_WAIT_METADATA_READ = 1U,
};

enum DispatchFFNCombineProfileDispatchV2CTrace : size_t
{
    DISPATCH_FFN_COMBINE_PROFILE_DISPATCH_V2C_TRACE_GROUP_BASE = 0U,
};

enum DispatchFFNCombineProfileGmm2ToCombineTrace : size_t
{
    DISPATCH_FFN_COMBINE_PROFILE_GMM2_TO_COMBINE_TRACE_GROUP_BASE = 0U,
};

enum DispatchFFNCombineProfileGmm2CvDetail : size_t
{
    DISPATCH_FFN_COMBINE_PROFILE_GMM2_CV_DETAIL_STORE = 0U,
};
#endif

constexpr size_t DispatchFFNCombineProfileStageStartIndex(size_t stage)
{
    return kDispatchFFNCombineProfileStageBase + stage * 2U;
}

constexpr size_t DispatchFFNCombineProfileStageEndIndex(size_t stage)
{
    return DispatchFFNCombineProfileStageStartIndex(stage) + 1U;
}

constexpr size_t DispatchFFNCombineProfileFrontDoneStartIndex(size_t step)
{
    return kDispatchFFNCombineProfileFrontDoneBase + step * 2U;
}

constexpr size_t DispatchFFNCombineProfileFrontDoneEndIndex(size_t step)
{
    return DispatchFFNCombineProfileFrontDoneStartIndex(step) + 1U;
}

constexpr size_t DispatchFFNCombineProfileReadyStageStartIndex(size_t stage)
{
    return kDispatchFFNCombineProfileReadyStageBase + stage;
}

constexpr size_t DispatchFFNCombineProfileCombineDetailStartIndex(size_t detail)
{
    return kDispatchFFNCombineProfileCombineDetailBase + detail * kDispatchFFNCombineProfileCombineDetailFieldCount;
}

constexpr size_t DispatchFFNCombineProfileCombineDetailEndIndex(size_t detail)
{
    return DispatchFFNCombineProfileCombineDetailStartIndex(detail) + 1U;
}

constexpr size_t DispatchFFNCombineProfileCombineDetailTotalIndex(size_t detail)
{
    return DispatchFFNCombineProfileCombineDetailStartIndex(detail) + 2U;
}

constexpr size_t DispatchFFNCombineProfileCombineDetailCountIndex(size_t detail)
{
    return DispatchFFNCombineProfileCombineDetailStartIndex(detail) + 3U;
}

static_assert(DispatchFFNCombineProfileCombineDetailCountIndex(kDispatchFFNCombineProfileCombineDetailCount - 1U) <
              kDispatchFFNCombineProfileEntryU64Count);

constexpr size_t DispatchFFNCombineProfileGmm1DetailStartIndex(size_t expert)
{
    return kDispatchFFNCombineProfileGmm1DetailBase + expert * kDispatchFFNCombineProfileGmm1DetailFieldCount;
}

constexpr size_t DispatchFFNCombineProfileGmm1DetailEndIndex(size_t expert)
{
    return DispatchFFNCombineProfileGmm1DetailStartIndex(expert) + 1U;
}

constexpr size_t DispatchFFNCombineProfileGmm1DetailTotalIndex(size_t expert)
{
    return DispatchFFNCombineProfileGmm1DetailStartIndex(expert) + 2U;
}

constexpr size_t DispatchFFNCombineProfileGmm1DetailCountIndex(size_t expert)
{
    return DispatchFFNCombineProfileGmm1DetailStartIndex(expert) + 3U;
}

static_assert(DispatchFFNCombineProfileGmm1DetailCountIndex(kDispatchFFNCombineProfileGmm1DetailMaxExperts - 1U) <
              kDispatchFFNCombineProfileEntryU64Count);

constexpr size_t DispatchFFNCombineProfileFrontDetailStartIndex(size_t detail)
{
    return kDispatchFFNCombineProfileFrontDetailBase + detail * kDispatchFFNCombineProfileFrontDetailFieldCount;
}

constexpr size_t DispatchFFNCombineProfileFrontDetailEndIndex(size_t detail)
{
    return DispatchFFNCombineProfileFrontDetailStartIndex(detail) + 1U;
}

constexpr size_t DispatchFFNCombineProfileFrontDetailTotalIndex(size_t detail)
{
    return DispatchFFNCombineProfileFrontDetailStartIndex(detail) + 2U;
}

constexpr size_t DispatchFFNCombineProfileFrontDetailCountIndex(size_t detail)
{
    return DispatchFFNCombineProfileFrontDetailStartIndex(detail) + 3U;
}

static_assert(DispatchFFNCombineProfileFrontDetailCountIndex(kDispatchFFNCombineProfileFrontDetailCount - 1U) <
              kDispatchFFNCombineProfileEntryU64Count);

constexpr size_t DispatchFFNCombineProfileDispatchDetailStartIndex(size_t detail)
{
    return kDispatchFFNCombineProfileDispatchDetailBase + detail * kDispatchFFNCombineProfileDispatchDetailFieldCount;
}

constexpr size_t DispatchFFNCombineProfileDispatchDetailEndIndex(size_t detail)
{
    return DispatchFFNCombineProfileDispatchDetailStartIndex(detail) + 1U;
}

constexpr size_t DispatchFFNCombineProfileDispatchDetailTotalIndex(size_t detail)
{
    return DispatchFFNCombineProfileDispatchDetailStartIndex(detail) + 2U;
}

constexpr size_t DispatchFFNCombineProfileDispatchDetailCountIndex(size_t detail)
{
    return DispatchFFNCombineProfileDispatchDetailStartIndex(detail) + 3U;
}

static_assert(DispatchFFNCombineProfileDispatchDetailCountIndex(kDispatchFFNCombineProfileDispatchDetailCount - 1U) <
              kDispatchFFNCombineProfileEntryU64Count);

constexpr size_t DispatchFFNCombineProfileUnpermuteDetailStartIndex(size_t detail)
{
    return kDispatchFFNCombineProfileUnpermuteDetailBase + detail * kDispatchFFNCombineProfileUnpermuteDetailFieldCount;
}

constexpr size_t DispatchFFNCombineProfileUnpermuteDetailEndIndex(size_t detail)
{
    return DispatchFFNCombineProfileUnpermuteDetailStartIndex(detail) + 1U;
}

constexpr size_t DispatchFFNCombineProfileUnpermuteDetailTotalIndex(size_t detail)
{
    return DispatchFFNCombineProfileUnpermuteDetailStartIndex(detail) + 2U;
}

constexpr size_t DispatchFFNCombineProfileUnpermuteDetailCountIndex(size_t detail)
{
    return DispatchFFNCombineProfileUnpermuteDetailStartIndex(detail) + 3U;
}

static_assert(DispatchFFNCombineProfileUnpermuteDetailCountIndex(kDispatchFFNCombineProfileUnpermuteDetailCount - 1U) <
              kDispatchFFNCombineProfileEntryU64Count);

#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
constexpr size_t DispatchFFNCombineProfileCombineExpertWallStartIndex(size_t expert)
{
    return kDispatchFFNCombineProfileCombineExpertWallBase + expert * kDispatchFFNCombineProfileExpertWallFieldCount;
}

constexpr size_t DispatchFFNCombineProfileCombineExpertWallEndIndex(size_t expert)
{
    return DispatchFFNCombineProfileCombineExpertWallStartIndex(expert) + 1U;
}

constexpr size_t DispatchFFNCombineProfileGmm2ExpertWallStartIndex(size_t expert)
{
    return kDispatchFFNCombineProfileGmm2ExpertWallBase + expert * kDispatchFFNCombineProfileExpertWallFieldCount;
}

constexpr size_t DispatchFFNCombineProfileGmm2ExpertWallEndIndex(size_t expert)
{
    return DispatchFFNCombineProfileGmm2ExpertWallStartIndex(expert) + 1U;
}

static_assert(DispatchFFNCombineProfileGmm2ExpertWallEndIndex(kDispatchFFNCombineProfileGmm1DetailMaxExperts - 1U) <
              kDispatchFFNCombineProfileEntryU64Count);

constexpr size_t DispatchFFNCombineProfileGmm1WaitDetailStartIndex(size_t detail)
{
    return kDispatchFFNCombineProfileGmm1WaitDetailBase + detail * kDispatchFFNCombineProfileGmm1WaitDetailFieldCount;
}

constexpr size_t DispatchFFNCombineProfileGmm1WaitDetailEndIndex(size_t detail)
{
    return DispatchFFNCombineProfileGmm1WaitDetailStartIndex(detail) + 1U;
}

constexpr size_t DispatchFFNCombineProfileGmm1WaitDetailTotalIndex(size_t detail)
{
    return DispatchFFNCombineProfileGmm1WaitDetailStartIndex(detail) + 2U;
}

constexpr size_t DispatchFFNCombineProfileGmm1WaitDetailCountIndex(size_t detail)
{
    return DispatchFFNCombineProfileGmm1WaitDetailStartIndex(detail) + 3U;
}

static_assert(DispatchFFNCombineProfileGmm1WaitDetailCountIndex(kDispatchFFNCombineProfileGmm1WaitDetailCount - 1U) <
              kDispatchFFNCombineProfileEntryU64Count);

constexpr size_t DispatchFFNCombineProfileDispatchV2CTraceStartIndex(size_t event)
{
    return kDispatchFFNCombineProfileDispatchV2CTraceBase +
           event * kDispatchFFNCombineProfileDispatchV2CTraceFieldCount;
}

constexpr size_t DispatchFFNCombineProfileDispatchV2CTraceEndIndex(size_t event)
{
    return DispatchFFNCombineProfileDispatchV2CTraceStartIndex(event) + 1U;
}

constexpr size_t DispatchFFNCombineProfileDispatchV2CTraceFlagIndex(size_t event)
{
    return DispatchFFNCombineProfileDispatchV2CTraceStartIndex(event) + 2U;
}

constexpr size_t DispatchFFNCombineProfileDispatchV2CTraceCountIndex(size_t event)
{
    return DispatchFFNCombineProfileDispatchV2CTraceStartIndex(event) + 3U;
}

static_assert(DispatchFFNCombineProfileDispatchV2CTraceCountIndex(kDispatchFFNCombineProfileDispatchV2CTraceCount -
                                                                  1U) < kDispatchFFNCombineProfileEntryU64Count);

constexpr size_t DispatchFFNCombineProfileGmm2ToCombineTraceStartIndex(size_t event)
{
    return kDispatchFFNCombineProfileGmm2ToCombineTraceBase +
           event * kDispatchFFNCombineProfileGmm2ToCombineTraceFieldCount;
}

constexpr size_t DispatchFFNCombineProfileGmm2ToCombineTraceEndIndex(size_t event)
{
    return DispatchFFNCombineProfileGmm2ToCombineTraceStartIndex(event) + 1U;
}

constexpr size_t DispatchFFNCombineProfileGmm2ToCombineTraceFlagIndex(size_t event)
{
    return DispatchFFNCombineProfileGmm2ToCombineTraceStartIndex(event) + 2U;
}

constexpr size_t DispatchFFNCombineProfileGmm2ToCombineTraceCountIndex(size_t event)
{
    return DispatchFFNCombineProfileGmm2ToCombineTraceStartIndex(event) + 3U;
}

static_assert(DispatchFFNCombineProfileGmm2ToCombineTraceCountIndex(kDispatchFFNCombineProfileGmm2ToCombineTraceCount -
                                                                    1U) < kDispatchFFNCombineProfileEntryU64Count);

constexpr size_t DispatchFFNCombineProfileGmm2CvDetailStartIndex(size_t detail)
{
    return kDispatchFFNCombineProfileGmm2CvDetailBase + detail * kDispatchFFNCombineProfileGmm2CvDetailFieldCount;
}

constexpr size_t DispatchFFNCombineProfileGmm2CvDetailEndIndex(size_t detail)
{
    return DispatchFFNCombineProfileGmm2CvDetailStartIndex(detail) + 1U;
}

constexpr size_t DispatchFFNCombineProfileGmm2CvDetailTotalIndex(size_t detail)
{
    return DispatchFFNCombineProfileGmm2CvDetailStartIndex(detail) + 2U;
}

constexpr size_t DispatchFFNCombineProfileGmm2CvDetailCountIndex(size_t detail)
{
    return DispatchFFNCombineProfileGmm2CvDetailStartIndex(detail) + 3U;
}

static_assert(DispatchFFNCombineProfileGmm2CvDetailCountIndex(kDispatchFFNCombineProfileGmm2CvDetailCount - 1U) <
              kDispatchFFNCombineProfileEntryU64Count);
#endif

struct DispatchFFNCombineLaunchArgs {
    void *ffts = nullptr;
    void *x = nullptr;
    void *weight1 = nullptr;
    void *weight2 = nullptr;
    void *expert_idx = nullptr;
    void *scale1 = nullptr;
    void *scale2 = nullptr;
    void *probs = nullptr;
    void *x_active_mask = nullptr;
    void *out = nullptr;
    void *expert_token_nums = nullptr;
    void *workspace = nullptr;
    void *tiling = nullptr;
    void *profile_data = nullptr;
    uint32_t block_dim = 1;
    uint32_t stage_profile = 0;
    uint32_t start_sync_debug = 0;
};

void launchDispatchFFNCombine(const DispatchFFNCombineLaunchArgs &args, void *stream);
