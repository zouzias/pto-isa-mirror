/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_DISPATCH_LAYOUT_H_
#define MOE_DISPATCH_LAYOUT_H_

#include "common.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace moe_dispatch {

inline uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    if (alignment == 0) {
        throw std::invalid_argument("alignment must be nonzero");
    }
    const uint64_t rem = value % alignment;
    if (rem == 0) {
        return value;
    }
    const uint64_t add = alignment - rem;
    if (value > std::numeric_limits<uint64_t>::max() - add) {
        throw std::overflow_error("AlignUp overflow");
    }
    return value + add;
}

inline uint64_t CheckedMul(uint64_t a, uint64_t b, const char *label)
{
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::overflow_error(std::string(label) + " overflow");
    }
    return a * b;
}

inline uint64_t AppendField(uint64_t *offset, uint64_t bytes, uint64_t alignment = 64)
{
    *offset = AlignUp(*offset, alignment);
    const uint64_t fieldOffset = *offset;
    if (*offset > std::numeric_limits<uint64_t>::max() - bytes) {
        throw std::overflow_error("layout byte overflow");
    }
    *offset += bytes;
    return fieldOffset;
}

inline uint64_t EffectiveAivBlocks(const MoeDispatchShape &shape)
{
    return shape.aivBlocks == 0 ? 1 : shape.aivBlocks;
}

inline uint64_t ExpertNumPadded(const MoeDispatchShape &shape)
{
    return AlignUp(shape.expertNum, shape.metadataPad);
}

inline uint64_t ExpandedRows(const MoeDispatchShape &shape)
{
    return CheckedMul(shape.m, shape.topK, "expanded rows");
}

inline uint64_t PackedABytes(const MoeDispatchShape &shape)
{
    return CheckedMul(CheckedMul(shape.maxOutputSize, shape.k, "packedA elems"), 2, "packedA bytes");
}

inline uint64_t ExpandedRowIdxBytes(const MoeDispatchShape &shape)
{
    return CheckedMul(ExpandedRows(shape), 4, "expandedRowIdx bytes");
}

inline uint64_t TokenPerExpertBytes(const MoeDispatchShape &shape)
{
    return CheckedMul(CheckedMul(shape.ep, ExpertNumPadded(shape), "tokenPerExpert elems"), 4, "tokenPerExpert bytes");
}

inline MoeDispatchWorkspaceLayout ComputeWorkspaceLayout(const MoeDispatchShape &shape)
{
    constexpr uint64_t kI32 = 4;
    constexpr uint64_t kHalf = 2;
    const uint64_t aivBlocks = EffectiveAivBlocks(shape);
    const uint64_t expertNumPadded = ExpertNumPadded(shape);
    const uint64_t routes = ExpandedRows(shape);
    uint64_t offset = 0;

    MoeDispatchWorkspaceLayout layout{};
    layout.localTokenPerExpert = AppendField(&offset, CheckedMul(expertNumPadded, kI32, "localTokenPerExpert"));
    layout.blockTokenPerExpert =
        AppendField(&offset, CheckedMul(CheckedMul(aivBlocks, expertNumPadded, "blockTokenPerExpert elems"), kI32,
                                        "blockTokenPerExpert bytes"));
    layout.blockPrefixPerExpert =
        AppendField(&offset, CheckedMul(CheckedMul(aivBlocks, expertNumPadded, "blockPrefixPerExpert elems"), kI32,
                                        "blockPrefixPerExpert bytes"));
    layout.routeExpert = AppendField(&offset, CheckedMul(routes, kI32, "routeExpert bytes"));
    layout.routePackedRow = AppendField(&offset, CheckedMul(routes, kI32, "routePackedRow bytes"));
    const uint64_t signalSlots = AlignUp(CheckedMul(aivBlocks, kMoeDispatchSignalStrideI32, "localSync slots"), 16);
    layout.localSync = AppendField(&offset, CheckedMul(signalSlots, kI32, "localSync bytes"));
    layout.tileScratch = AppendField(
        &offset, CheckedMul(CheckedMul(aivBlocks, shape.tileCols, "tileScratch elems"), kHalf, "tileScratch bytes"));
    layout.totalBytes = AlignUp(offset, 64);
    return layout;
}

inline MoeDispatchWindowLayout ComputeGuardedWindowLayout(const MoeDispatchShape &shape, uint64_t windowBytes)
{
    if (windowBytes <= kMoeDispatchTailControlBytes + kMoeDispatchWindowHeadGuardBytes) {
        throw std::runtime_error("HCCL remote window is too small for guarded moe_dispatch layout");
    }

    MoeDispatchWindowLayout layout{};
    layout.mode = MoeDispatchWindowLayoutMode::GuardedDispatchOnly;
    layout.headGuard = 0;
    layout.tokenPerExpert = windowBytes - kMoeDispatchTailControlBytes;
    layout.signal = windowBytes - kMoeDispatchSignalBytes;
    layout.totalVisibleBytes = windowBytes;

    uint64_t offset = kMoeDispatchWindowHeadGuardBytes;
    layout.packedA = AppendField(&offset, PackedABytes(shape), kMoeDispatchWindowAlignBytes);
    layout.expandedRowIdx = AppendField(&offset, ExpandedRowIdxBytes(shape), kMoeDispatchWindowAlignBytes);
    layout.reservedScratch = AlignUp(offset, kMoeDispatchWindowAlignBytes);
    return layout;
}

inline void ValidateWindowLayout(const MoeDispatchShape &shape, const MoeDispatchWindowLayout &layout)
{
    if (layout.mode != MoeDispatchWindowLayoutMode::GuardedDispatchOnly) {
        throw std::runtime_error("only guarded moe_dispatch window layout is implemented");
    }
    if (layout.packedA < kMoeDispatchWindowHeadGuardBytes) {
        throw std::runtime_error("packedA overlaps A5 HCCL window head guard");
    }
    const uint64_t liveEnd = layout.reservedScratch;
    if (liveEnd > layout.tokenPerExpert) {
        throw std::runtime_error("HCCL remote window dispatch payload overlaps token-count region");
    }
    if (layout.tokenPerExpert + TokenPerExpertBytes(shape) > layout.signal) {
        throw std::runtime_error("HCCL remote window token-count region overlaps signal region");
    }
    if (layout.signal + kMoeDispatchSignalBytes > layout.totalVisibleBytes) {
        throw std::runtime_error("HCCL remote window signal region exceeds window size");
    }
}

inline uint64_t EstimateHcclBuffSizeMb(const MoeDispatchShape &shape)
{
    uint64_t liveBytes = kMoeDispatchWindowHeadGuardBytes;
    liveBytes = AlignUp(liveBytes, kMoeDispatchWindowAlignBytes) + PackedABytes(shape);
    liveBytes = AlignUp(liveBytes, kMoeDispatchWindowAlignBytes) + ExpandedRowIdxBytes(shape);
    const uint64_t bytes =
        AlignUp(liveBytes, kMoeDispatchWindowAlignBytes) + kMoeDispatchTailControlBytes + 64ULL * kMoeDispatchMiB;
    return AlignUp(bytes, kMoeDispatchMiB) / kMoeDispatchMiB;
}

} // namespace moe_dispatch

#endif // MOE_DISPATCH_LAYOUT_H_
