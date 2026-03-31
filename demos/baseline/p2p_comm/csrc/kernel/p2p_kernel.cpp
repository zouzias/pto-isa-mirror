/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstddef>
#include <cstdint>
#include <iostream>

#include <pto/pto-inst.hpp>
#include "pto/common/pto_tile.hpp"
#include "common.hpp"

// ============================================================================
// Demo Constants
// ============================================================================
static constexpr size_t ELEM_COUNT = 256;
static constexpr size_t UB_OFFSET_A = 0x0;
static constexpr size_t UB_OFFSET_B = 0x10000;
static constexpr size_t SYNC_BUF_BYTES = 64 * sizeof(int32_t);

using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using GlobalI32 = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;
using VecTile = pto::Tile<pto::TileType::Vec, int32_t, 1, ELEM_COUNT, pto::BLayout::RowMajor, -1, -1>;

// ============================================================================
// TPUT Demo Kernel
//
// Phase 0 (Rank 0 only): Load src -> send_shmem (via UB), then TPUT to Rank 1.
// Phase 1 (Rank 1 only): Read recv_shmem -> dst (via UB).
// ============================================================================
__global__ AICORE void TPutP2PKernel(__gm__ int32_t *dst, __gm__ int32_t *src, __gm__ int32_t *shmem, int nranks,
                                     __gm__ HcclDeviceContext *hcclCtx, int phase)
{
    if (nranks < 2)
        return;

    ShapeDyn shape(1, 1, 1, 1, ELEM_COUNT);
    StrideDyn stride(ELEM_COUNT, ELEM_COUNT, ELEM_COUNT, ELEM_COUNT, 1);
    int myRank = static_cast<int>(hcclCtx->rankId);

    __gm__ int32_t *base = reinterpret_cast<__gm__ int32_t *>(
        reinterpret_cast<__gm__ uint8_t *>(shmem) + SYNC_BUF_BYTES);
    __gm__ int32_t *sendBuf = base;
    __gm__ int32_t *recvBuf = base + ELEM_COUNT;

    if (phase == 0 && myRank == 0) {
        GlobalI32 srcG(src, shape, stride);
        GlobalI32 sendG(sendBuf, shape, stride);
        VecTile tile(1, ELEM_COUNT);
        TASSIGN(tile, UB_OFFSET_A);

        TLOAD(tile, srcG);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(sendG, tile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

        __gm__ int32_t *remoteRecv = HcclRemotePtr(hcclCtx, recvBuf, 1);
        GlobalI32 remoteG(remoteRecv, shape, stride);
        pto::comm::TPUT(remoteG, sendG, tile);
        pipe_barrier(PIPE_ALL);
    } else if (phase == 1 && myRank == 1) {
        GlobalI32 dstG(dst, shape, stride);
        GlobalI32 recvG(recvBuf, shape, stride);
        VecTile tile(1, ELEM_COUNT);
        TASSIGN(tile, UB_OFFSET_A);

        TLOAD(tile, recvG);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstG, tile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
}

// ============================================================================
// TGET Demo Kernel
//
// Phase 0 (Rank 0 only): Load src -> send_shmem to make data readable.
// Phase 1 (Rank 1 only): TGET from Rank 0's send_shmem -> local recv_shmem,
//                         then copy recv_shmem -> dst.
// ============================================================================
__global__ AICORE void TGetP2PKernel(__gm__ int32_t *dst, __gm__ int32_t *src, __gm__ int32_t *shmem, int nranks,
                                     __gm__ HcclDeviceContext *hcclCtx, int phase)
{
    if (nranks < 2)
        return;

    ShapeDyn shape(1, 1, 1, 1, ELEM_COUNT);
    StrideDyn stride(ELEM_COUNT, ELEM_COUNT, ELEM_COUNT, ELEM_COUNT, 1);
    int myRank = static_cast<int>(hcclCtx->rankId);

    __gm__ int32_t *base = reinterpret_cast<__gm__ int32_t *>(
        reinterpret_cast<__gm__ uint8_t *>(shmem) + SYNC_BUF_BYTES);
    __gm__ int32_t *sendBuf = base;
    __gm__ int32_t *recvBuf = base + ELEM_COUNT;

    if (phase == 0 && myRank == 0) {
        GlobalI32 srcG(src, shape, stride);
        GlobalI32 sendG(sendBuf, shape, stride);
        VecTile tile(1, ELEM_COUNT);
        TASSIGN(tile, UB_OFFSET_A);

        TLOAD(tile, srcG);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(sendG, tile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    } else if (phase == 1 && myRank == 1) {
        GlobalI32 recvG(recvBuf, shape, stride);
        VecTile staging(1, ELEM_COUNT);
        TASSIGN(staging, UB_OFFSET_A);

        __gm__ int32_t *remoteSend = HcclRemotePtr(hcclCtx, sendBuf, 0);
        GlobalI32 remoteG(remoteSend, shape, stride);
        pto::comm::TGET(recvG, remoteG, staging);
        pipe_barrier(PIPE_ALL);

        VecTile result(1, ELEM_COUNT);
        TASSIGN(result, UB_OFFSET_B);
        GlobalI32 dstG(dst, shape, stride);
        TLOAD(result, recvG);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstG, result);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
}

// ============================================================================
// Host-side helpers
// ============================================================================
static bool VerifyOutput(const int32_t *host, size_t count, int32_t startVal, const char *tag)
{
    for (size_t i = 0; i < count; ++i) {
        int32_t expected = static_cast<int32_t>(i) + startVal;
        if (host[i] != expected) {
            std::cerr << "[" << tag << " FAIL] index " << i << ": expected " << expected << ", got " << host[i]
                      << std::endl;
            return false;
        }
    }
    return true;
}

// ============================================================================
// RunTPutDemo: TPUT point-to-point (Rank 0 -> Rank 1)
// ============================================================================
static bool RunTPutDemoKernel(int rankId, int nRanks, int nDevices, int firstDeviceId, const HcclRootInfo *rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo))
        return false;

    void *inputDev = nullptr;
    void *outputDev = nullptr;
    if (aclrtMalloc(&inputDev, ELEM_COUNT * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST) != 0 ||
        aclrtMalloc(&outputDev, ELEM_COUNT * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
        std::cerr << "[ERROR] aclrtMalloc failed" << std::endl;
        return false;
    }

    int32_t *inputHost = nullptr;
    int32_t *outputHost = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&inputHost), ELEM_COUNT * sizeof(int32_t)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&outputHost), ELEM_COUNT * sizeof(int32_t)) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed" << std::endl;
        return false;
    }

    // Rank 0 fills input with [0, 1, 2, ...]; Rank 1 leaves it zeroed.
    for (size_t i = 0; i < ELEM_COUNT; ++i) {
        inputHost[i] = (rankId == 0) ? static_cast<int32_t>(i) : 0;
        outputHost[i] = -1;
    }
    aclrtMemcpy(inputDev, ELEM_COUNT * sizeof(int32_t), inputHost, ELEM_COUNT * sizeof(int32_t),
                ACL_MEMCPY_HOST_TO_DEVICE);

    uint64_t winBase = ctx.hostCtx.windowsIn[rankId];
    size_t winOff = 0;
    void *shmem = WindowAlloc(winBase, winOff, SYNC_BUF_BYTES + 4 * ELEM_COUNT * sizeof(int32_t));

    HcclHostBarrier(ctx.comm, ctx.stream);

    // Phase 0: Rank 0 sends
    TPutP2PKernel<<<1, nullptr, ctx.stream>>>(
        (int32_t *)outputDev, (int32_t *)inputDev, (int32_t *)shmem, nRanks, ctx.deviceCtx, 0);
    aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    // Phase 1: Rank 1 reads received data
    TPutP2PKernel<<<1, nullptr, ctx.stream>>>(
        (int32_t *)outputDev, (int32_t *)inputDev, (int32_t *)shmem, nRanks, ctx.deviceCtx, 1);
    aclrtSynchronizeStream(ctx.stream);

    bool ok = true;
    if (rankId == 1) {
        aclrtMemcpy(outputHost, ELEM_COUNT * sizeof(int32_t), outputDev, ELEM_COUNT * sizeof(int32_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
        ok = VerifyOutput(outputHost, ELEM_COUNT, 0, "TPUT");
        if (ok) {
            std::cout << "[TPUT PASS] Rank 1 received Rank 0's data. First 5: [";
            for (size_t i = 0; i < 5; ++i)
                std::cout << (i ? ", " : "") << outputHost[i];
            std::cout << ", ...]" << std::endl;
        }
    }

    aclrtFreeHost(inputHost);
    aclrtFreeHost(outputHost);
    aclrtFree(inputDev);
    aclrtFree(outputDev);

    return ctx.Finalize() && ok;
}

bool RunTPutDemo(int nRanks, int firstRankId, int firstDeviceId)
{
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunTPutDemoKernel(rankId, nRanks, nRanks, firstDeviceId, rootInfo);
        });
}

// ============================================================================
// RunTGetDemo: TGET point-to-point (Rank 1 pulls from Rank 0)
// ============================================================================
static bool RunTGetDemoKernel(int rankId, int nRanks, int nDevices, int firstDeviceId, const HcclRootInfo *rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo))
        return false;

    void *inputDev = nullptr;
    void *outputDev = nullptr;
    if (aclrtMalloc(&inputDev, ELEM_COUNT * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST) != 0 ||
        aclrtMalloc(&outputDev, ELEM_COUNT * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
        std::cerr << "[ERROR] aclrtMalloc failed" << std::endl;
        return false;
    }

    int32_t *inputHost = nullptr;
    int32_t *outputHost = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&inputHost), ELEM_COUNT * sizeof(int32_t)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&outputHost), ELEM_COUNT * sizeof(int32_t)) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed" << std::endl;
        return false;
    }

    for (size_t i = 0; i < ELEM_COUNT; ++i) {
        inputHost[i] = (rankId == 0) ? static_cast<int32_t>(i + 1000) : 0;
        outputHost[i] = -1;
    }
    aclrtMemcpy(inputDev, ELEM_COUNT * sizeof(int32_t), inputHost, ELEM_COUNT * sizeof(int32_t),
                ACL_MEMCPY_HOST_TO_DEVICE);

    uint64_t winBase = ctx.hostCtx.windowsIn[rankId];
    size_t winOff = 0;
    void *shmem = WindowAlloc(winBase, winOff, SYNC_BUF_BYTES + 4 * ELEM_COUNT * sizeof(int32_t));

    HcclHostBarrier(ctx.comm, ctx.stream);

    // Phase 0: Rank 0 prepares readable data in shmem
    TGetP2PKernel<<<1, nullptr, ctx.stream>>>(
        (int32_t *)outputDev, (int32_t *)inputDev, (int32_t *)shmem, nRanks, ctx.deviceCtx, 0);
    aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    // Phase 1: Rank 1 pulls data via TGET
    TGetP2PKernel<<<1, nullptr, ctx.stream>>>(
        (int32_t *)outputDev, (int32_t *)inputDev, (int32_t *)shmem, nRanks, ctx.deviceCtx, 1);
    aclrtSynchronizeStream(ctx.stream);

    bool ok = true;
    if (rankId == 1) {
        aclrtMemcpy(outputHost, ELEM_COUNT * sizeof(int32_t), outputDev, ELEM_COUNT * sizeof(int32_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
        ok = VerifyOutput(outputHost, ELEM_COUNT, 1000, "TGET");
        if (ok) {
            std::cout << "[TGET PASS] Rank 1 pulled Rank 0's data. First 5: [";
            for (size_t i = 0; i < 5; ++i)
                std::cout << (i ? ", " : "") << outputHost[i];
            std::cout << ", ...]" << std::endl;
        }
    }

    aclrtFreeHost(inputHost);
    aclrtFreeHost(outputHost);
    aclrtFree(inputDev);
    aclrtFree(outputDev);

    return ctx.Finalize() && ok;
}

bool RunTGetDemo(int nRanks, int firstRankId, int firstDeviceId)
{
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunTGetDemoKernel(rankId, nRanks, nRanks, firstDeviceId, rootInfo);
        });
}
