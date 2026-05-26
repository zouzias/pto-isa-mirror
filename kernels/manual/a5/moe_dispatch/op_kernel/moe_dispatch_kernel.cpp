/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>

#ifndef PIPE_FIX
#define PIPE_FIX static_cast<pipe_t>(10)
#endif

#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

#include "common.h"
#include "kernel_launchers.h"
#include "op_kernel/utils/const_args.hpp"
#include "op_kernel/utils/hccl_window.hpp"

using moe_dispatch::MoeDispatchShape;
using moe_dispatch::PtoRemoteWindow;

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t *
#endif

PTO_SYNCALL_AIV_KERNEL_META(MoeDispatchKernel);

namespace {

constexpr int32_t kDefaultTileCols = 1024;
constexpr uint64_t kPingUbAddr = 0x0;
constexpr uint64_t kPongUbAddr = 0x1000;

using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;

template <typename T>
using GlobalNd = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

template <typename T, int kCols = kDefaultTileCols>
using VecTile = pto::Tile<pto::TileType::Vec, T, 1, kCols, pto::BLayout::RowMajor, -1, -1>;

AICORE inline uint64_t AlignUpDevice(uint64_t value, uint64_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

AICORE inline uint64_t PackedABytesDevice(MoeDispatchShape shape)
{
    return static_cast<uint64_t>(shape.maxOutputSize) * shape.k * sizeof(half);
}

AICORE inline uint64_t ExpandedRowIdxBytesDevice(MoeDispatchShape shape)
{
    return static_cast<uint64_t>(shape.m) * shape.topK * sizeof(int32_t);
}

struct GuardedWindowLayoutDevice {
    uint64_t packedA;
    uint64_t expandedRowIdx;
    uint64_t tokenPerExpert;
    uint64_t signal;
};

AICORE inline GuardedWindowLayoutDevice MakeGuardedWindowLayoutDevice(MoeDispatchShape shape, uint64_t windowBytes)
{
    GuardedWindowLayoutDevice layout{};
    uint64_t offset = MOE_DISPATCH_WINDOW_HEAD_GUARD_BYTES;
    layout.packedA = AlignUpDevice(offset, MOE_DISPATCH_WINDOW_ALIGN_BYTES);
    offset = layout.packedA + PackedABytesDevice(shape);
    layout.expandedRowIdx = AlignUpDevice(offset, MOE_DISPATCH_WINDOW_ALIGN_BYTES);
    layout.tokenPerExpert = windowBytes - MOE_DISPATCH_TAIL_CONTROL_BYTES;
    layout.signal = windowBytes - MOE_DISPATCH_SIGNAL_BYTES;
    return layout;
}

template <typename T>
AICORE inline GlobalNd<T> MakeGlobal2D(__gm__ T *ptr, int32_t rows, int32_t cols, int32_t rowStride)
{
    ShapeDyn shape(1, 1, 1, rows, cols);
    StrideDyn stride(rowStride * rows, rowStride * rows, rowStride * rows, rowStride, 1);
    return GlobalNd<T>(ptr, shape, stride);
}

AICORE inline uint32_t TokenShardBegin(uint32_t totalTokens, uint32_t blockId, uint32_t blockNum)
{
    uint32_t base = totalTokens / blockNum;
    uint32_t rem = totalTokens % blockNum;
    return blockId * base + (blockId < rem ? blockId : rem);
}

AICORE inline uint32_t TokenShardEnd(uint32_t totalTokens, uint32_t blockId, uint32_t blockNum)
{
    return TokenShardBegin(totalTokens, blockId + 1, blockNum);
}

AICORE inline void PublishDispatchReady(PtoRemoteWindow &remoteWindow, uint32_t blockId, uint32_t blockNum)
{
    for (uint32_t peer = blockId; peer < static_cast<uint32_t>(remoteWindow.RankSize()); peer += blockNum) {
        remoteWindow.NotifyRemoteTokenReady(static_cast<int32_t>(peer));
    }
}

AICORE inline void DispatchRows(MoeDispatchShape shape, uint32_t myRank, GM_ADDR inputA, __gm__ int32_t *expertIdx,
                                PtoRemoteWindow &remoteWindow, GM_ADDR peerWindow, uint32_t blockId, uint32_t blockNum)
{
    GuardedWindowLayoutDevice layout = MakeGuardedWindowLayoutDevice(shape, remoteWindow.SegmentSize());
    __gm__ half *input = reinterpret_cast<__gm__ half *>(inputA);
    __gm__ half *localPackedA = reinterpret_cast<__gm__ half *>(peerWindow + layout.packedA);
    __gm__ int32_t *localExpandedRowIdx = reinterpret_cast<__gm__ int32_t *>(peerWindow + layout.expandedRowIdx);
    __gm__ int32_t *localTokenPerExpert = reinterpret_cast<__gm__ int32_t *>(peerWindow + layout.tokenPerExpert);

    VecTile<half, kDefaultTileCols> ping(1, kDefaultTileCols);
    VecTile<half, kDefaultTileCols> pong(1, kDefaultTileCols);
    TASSIGN(ping, kPingUbAddr);
    TASSIGN(pong, kPongUbAddr);

    const uint32_t tokenBegin = TokenShardBegin(shape.m, blockId, blockNum);
    const uint32_t tokenEnd = TokenShardEnd(shape.m, blockId, blockNum);
    for (uint32_t token = tokenBegin; token < tokenEnd; ++token) {
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            const uint32_t route = token * shape.topK + slot;
            int32_t expert = expertIdx[route];
            if (expert < 0 || static_cast<uint32_t>(expert) >= shape.expertNum) {
                continue;
            }
            const uint32_t owner = static_cast<uint32_t>(expert) / shape.expertPerRank;
            const uint32_t packedRow = route;
            if (packedRow >= shape.maxOutputSize) {
                continue;
            }
            localExpandedRowIdx[route] = static_cast<int32_t>(packedRow);
            localTokenPerExpert[expert] = localTokenPerExpert[expert] + 1;

            __gm__ half *dstBase = localPackedA;
            if (owner != myRank) {
                dstBase = reinterpret_cast<__gm__ half *>(remoteWindow(static_cast<int64_t>(layout.packedA), owner));
            }

            for (uint32_t col = 0; col < shape.k; col += kDefaultTileCols) {
                const int32_t cols =
                    static_cast<int32_t>((shape.k - col) < kDefaultTileCols ? (shape.k - col) : kDefaultTileCols);
                GlobalNd<half> srcGlobal =
                    MakeGlobal2D(input + static_cast<uint64_t>(token) * shape.k + col, 1, cols, shape.k);
                GlobalNd<half> dstGlobal =
                    MakeGlobal2D(dstBase + static_cast<uint64_t>(packedRow) * shape.k + col, 1, cols, shape.k);
                if (owner == myRank) {
                    TLOAD(ping, srcGlobal);
                    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                    TSTORE(dstGlobal, ping);
                    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                } else {
                    pto::comm::TPUT(dstGlobal, srcGlobal, ping, pong);
                }
            }
        }
    }
}

} // namespace

extern "C" __global__ AICORE void MoeDispatchKernel(MoeDispatchShape shape, uint32_t myRank, GM_ADDR inputA,
                                                    __gm__ int32_t *expertIdx, GM_ADDR peerWindow,
                                                    GM_ADDR remoteWindowContext, GM_ADDR workspace)
{
    (void)workspace;
    const uint32_t blockId = static_cast<uint32_t>(block_idx);
    const uint32_t blockNum = shape.aivBlocks == 0 ? 1 : shape.aivBlocks;
    if (shape.ep == 0 || shape.m == 0 || shape.k == 0 || shape.topK == 0 || shape.expertPerRank == 0 ||
        shape.expertNum == 0 || blockId >= blockNum) {
        return;
    }

    PtoRemoteWindow remoteWindow;
    remoteWindow.Init(remoteWindowContext);
    DispatchRows(shape, myRank, inputA, expertIdx, remoteWindow, peerWindow, blockId, blockNum);
    PublishDispatchReady(remoteWindow, blockId, blockNum);
}

namespace moe_dispatch {

void LaunchMoeDispatchKernel(MoeDispatchShape shape, uint32_t myRank, uint8_t *inputA, int32_t *expertIdx,
                             uint8_t *peerWindow, uint8_t *remoteWindowContext, uint8_t *workspace, void *stream,
                             uint32_t launchBlockCount)
{
    MoeDispatchKernel<<<launchBlockCount, nullptr, stream>>>(shape, myRank, inputA, expertIdx, peerWindow,
                                                             remoteWindowContext, workspace);
}

} // namespace moe_dispatch
