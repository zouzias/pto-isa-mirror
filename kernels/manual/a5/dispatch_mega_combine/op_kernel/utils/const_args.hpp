/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef CONST_ARGS_HPP
#define CONST_ARGS_HPP

#include <cstdint>

#include "pto/common/buffer_limits.hpp"

constexpr static uint64_t MB_SIZE = 1024 * 1024UL;
constexpr static int32_t UB_ALIGN = 32;

struct AtlasA5 {
    static constexpr uint32_t BIAS_SIZE = PTO_BIAS_SIZE_BYTES;
    static constexpr uint32_t FIXBUF_SIZE = PTO_FBUF_SIZE_BYTES;
    static constexpr uint32_t UB_SIZE = PTO_UBUF_SIZE_BYTES;
    static constexpr uint32_t L1_SIZE = PTO_CBUF_SIZE_BYTES;
    static constexpr uint32_t L0A_SIZE = PTO_L0A_SIZE_BYTES;
    static constexpr uint32_t L0B_SIZE = PTO_L0B_SIZE_BYTES;
    static constexpr uint32_t L0C_SIZE = PTO_L0C_SIZE_BYTES;
};

// Maximum A5 topology supported by this binary. The selected 28/32/36-core role
// split is carried in MegaMoeFixedGroupTiling at runtime. The constants below
// are only compile-time capacities for synchronization and remote-window layouts.
struct MegaMoeA5Config {
    static constexpr uint32_t MAX_PHYSICAL_AIC_NUM = 36U;
    static constexpr uint32_t AIV_SUBBLOCKS_PER_PHYSICAL_BLOCK = 2U;
    static constexpr uint32_t MAX_GMM1_AIC_NUM = 24U;
    static constexpr uint32_t MAX_GMM2_AIC_NUM = 16U;
    static constexpr uint32_t EXPERT_PROGRESS_MAX_RANKS = 16U;
    static constexpr uint32_t UB_SYNC_RESERVE_BYTES = 40U * 1024U;
    static constexpr uint16_t SECOND_AIV_SUBBLOCK_FLAG_OFFSET = 16U;
};

constexpr uint32_t kMegaMoeFixedPhysicalAicNum = MegaMoeA5Config::MAX_PHYSICAL_AIC_NUM;
constexpr uint32_t kMegaMoeFixedAivSubblocksPerPhysicalBlock = MegaMoeA5Config::AIV_SUBBLOCKS_PER_PHYSICAL_BLOCK;
constexpr uint32_t kMegaMoeFixedPhysicalAivNum =
    kMegaMoeFixedPhysicalAicNum * kMegaMoeFixedAivSubblocksPerPhysicalBlock;
constexpr uint32_t kMegaMoeFixedGmm1GroupSize = MegaMoeA5Config::MAX_GMM1_AIC_NUM;
constexpr uint32_t kMegaMoeFixedGmm2GroupSize = MegaMoeA5Config::MAX_GMM2_AIC_NUM;
constexpr uint32_t kMegaMoeFixedDispatchGroupSize = kMegaMoeFixedGmm1GroupSize;
// Every physical AIC can produce a CV tile during the full-AIC GMM1 waves;
// reserve the SwiGLU synchronization lanes for the complete AIC set.
constexpr uint32_t kMegaMoeFixedSwigluGroupSize = kMegaMoeFixedPhysicalAicNum;
constexpr uint32_t kMegaMoeFixedUnpermuteGroupSize = kMegaMoeFixedPhysicalAivNum;
constexpr uint32_t kMegaMoeFixedInitialUnpermuteAiv0WorkerCount = 32U;
constexpr uint32_t kMegaMoeExpertProgressMaxRanks = MegaMoeA5Config::EXPERT_PROGRESS_MAX_RANKS;
constexpr uint32_t kMegaMoeFrontMaskCountRecordBytes = 32U;
constexpr uint32_t kMegaMoeRouteMetaFields = 8U;
constexpr uint32_t kMegaMoeCombineHbmBatchRows = 2U;
constexpr uint32_t kMegaMoeMxGroupSize = 32U;
constexpr uint32_t kMegaMoeMxDataAlignmentBytes = 256U;
constexpr uint32_t kMegaMoeMxScaleAlignmentBytes = 32U;
constexpr uint32_t kMegaMoeMxScalePrefetchK = 4096U;
constexpr uint32_t kMegaMoeGmmTileM = 128U;
constexpr uint32_t kMegaMoeGmmTileN = 256U;
constexpr uint32_t kMegaMoeReadyCountSlotBytes = 64U;
constexpr uint32_t kMegaMoeGmm2CombineCvTileRows = 128U;
constexpr uint32_t kMegaMoeGmm2CombineCvTileCols = 256U;
constexpr uint32_t kMegaMoeGmm2CombineCvFifoDepth = 3U;
constexpr uint32_t kMegaMoeGmm2CombineCvSlotBytes =
    kMegaMoeGmm2CombineCvTileRows * kMegaMoeGmm2CombineCvTileCols * sizeof(uint16_t);
constexpr uint32_t kMegaMoeGmm2CombineCvBufferBytes =
    kMegaMoeGmm2CombineCvFifoDepth * kMegaMoeGmm2CombineCvSlotBytes;
constexpr uint32_t kMegaMoeGmm2CombineMetadataSlotBytes =
    kMegaMoeGmm2CombineCvTileRows * kMegaMoeRouteMetaFields * sizeof(int32_t);
constexpr uint32_t kMegaMoeGmm2CombineRequiredUbBytes =
    kMegaMoeGmm2CombineCvBufferBytes +
    kMegaMoeGmm2CombineCvFifoDepth * kMegaMoeGmm2CombineMetadataSlotBytes;

constexpr uint16_t kMegaMoeFixedSecondAivSubblockFlagOffset = MegaMoeA5Config::SECOND_AIV_SUBBLOCK_FLAG_OFFSET;

constexpr uint32_t A5_UB_SYNC_RESERVE_BYTES = MegaMoeA5Config::UB_SYNC_RESERVE_BYTES;
constexpr uint32_t A5_MAIN_UB_SIZE = AtlasA5::UB_SIZE - A5_UB_SYNC_RESERVE_BYTES;

static_assert(kMegaMoeFixedAivSubblocksPerPhysicalBlock == 2U);
static_assert(kMegaMoeFixedGmm1GroupSize < kMegaMoeFixedPhysicalAicNum);
static_assert(kMegaMoeFixedGmm2GroupSize < kMegaMoeFixedPhysicalAicNum);
static_assert(kMegaMoeFixedDispatchGroupSize == kMegaMoeFixedGmm1GroupSize);
static_assert(kMegaMoeFixedSwigluGroupSize == kMegaMoeFixedPhysicalAicNum);
static_assert(kMegaMoeFixedUnpermuteGroupSize == kMegaMoeFixedPhysicalAivNum);
static_assert(kMegaMoeFixedInitialUnpermuteAiv0WorkerCount <= kMegaMoeFixedPhysicalAicNum);
static_assert(AtlasA5::UB_SIZE == 256U * 1024U);
static_assert(A5_MAIN_UB_SIZE == 216U * 1024U);
static_assert(kMegaMoeGmm2CombineCvSlotBytes == 64U * 1024U);
static_assert(kMegaMoeGmm2CombineRequiredUbBytes <= A5_MAIN_UB_SIZE);

#endif // CONST_ARGS_HPP
