/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#define ENABLE_DEBUG_PRINT 1

#include <cstddef>
#include <cstdint>
#include <iostream>

#include <pto/pto-inst.hpp>
#include "pto/comm/comm_types.hpp"
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"

static constexpr size_t HCCL_WIN_SYNC_PREFIX = 64 * sizeof(int32_t);

template <typename T>
__global__ AICORE void WindowMemCopyIn(__gm__ T *winDst, __gm__ T *devSrc, int count)
{
    for (int i = 0; i < count; ++i) {
        winDst[i] = devSrc[i];
    }
    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// TALL_TO_ALL Test Kernel
// Tests the TALL_TO_ALL collective — full AllToAll exchange among all ranks.
// Each rank's input has nranks slices (one destined for each peer).
// After the op, each rank's output[srcRank * sliceSize .. (srcRank+1)*sliceSize-1]
// contains the data that srcRank sent to this rank.
// ============================================================================
template <typename T, size_t sliceSize>
__global__ AICORE void TAllToAllKernelImpl(__gm__ T *input, __gm__ T *output, int nranks,
                                           __gm__ HcclDeviceContext *hcclCtx)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, sliceSize, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(hcclCtx->rankId);
    size_t totalCount = static_cast<size_t>(nranks) * sliceSize;

    ShapeDyn shape(1, 1, 1, nranks, sliceSize);
    StrideDyn stride(totalCount, totalCount, totalCount, sliceSize, 1);

    Global outputG(output, shape, stride);

    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ T *remoteInput = HcclRemotePtr(hcclCtx, input, i);
        tensors[i] = Global(remoteInput, shape, stride);
    }

    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, my_rank);

    TileData stagingTile(1, sliceSize);
    TASSIGN(stagingTile, 0x0);

    pto::comm::TALL_TO_ALL(pg, outputG, stagingTile);

    pipe_barrier(PIPE_ALL);
}

template <typename T, size_t sliceSize>
bool RunAllToAllKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo *rootInfo)
{
    size_t totalCount = static_cast<size_t>(n_ranks) * sliceSize;

    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo)) {
        return false;
    }

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    if (n_ranks > 1) {
        WindowAlloc(localWinBase, winOffset, HCCL_WIN_SYNC_PREFIX);
    }
    void *input_ptr = WindowAlloc(localWinBase, winOffset, totalCount * sizeof(T));

    T *input_host = nullptr;
    T *output_host = nullptr;
    T *output_device = nullptr;
    T *staging = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&input_host), totalCount * sizeof(T)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&output_host), totalCount * sizeof(T)) != 0 ||
        aclrtMalloc(reinterpret_cast<void **>(&output_device), totalCount * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) !=
            0 ||
        aclrtMalloc(reinterpret_cast<void **>(&staging), totalCount * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost/aclrtMalloc failed!" << std::endl;
        return false;
    }

    // Fill entire input uniformly with value (rank_id + 1)
    for (size_t i = 0; i < totalCount; ++i) {
        input_host[i] = static_cast<T>(rank_id + 1);
    }

    aclrtMemcpy(staging, totalCount * sizeof(T), input_host, totalCount * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    WindowMemCopyIn<T><<<1, nullptr, ctx.stream>>>((T *)input_ptr, staging, static_cast<int>(totalCount));
    aclrtSynchronizeStream(ctx.stream);
    aclrtFree(staging);

    HcclHostBarrier(ctx.comm, ctx.stream);

    TAllToAllKernelImpl<T, sliceSize>
        <<<1, nullptr, ctx.stream>>>((T *)input_ptr, (T *)output_device, n_ranks, ctx.deviceCtx);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    // All ranks verify: output[srcRank * sliceSize .. (srcRank+1)*sliceSize - 1] == (srcRank + 1)
    bool is_ok = true;
    aclrtMemcpy(output_host, totalCount * sizeof(T), output_device, totalCount * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    for (int srcRank = 0; srcRank < n_ranks; ++srcRank) {
        T expected = static_cast<T>(srcRank + 1);
        for (size_t j = 0; j < sliceSize; ++j) {
            size_t idx = static_cast<size_t>(srcRank) * sliceSize + j;
            T actual = output_host[idx];
            if (actual != expected) {
                std::cout << "Rank " << rank_id << " validation failed at index " << idx << " (srcRank=" << srcRank
                          << ", offset=" << j << "): expected " << (float)expected << ", got " << (float)actual
                          << std::endl;
                is_ok = false;
                break;
            }
        }
        if (!is_ok)
            break;
    }

#if ENABLE_DEBUG_PRINT
    if (is_ok) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "[DEBUG] Rank " << rank_id << ": TALL_TO_ALL SUCCESSFUL!" << std::endl;
        std::cout << "Summary: AllToAll with " << n_ranks << " ranks, sliceSize " << sliceSize << " elements."
                  << std::endl;
        std::cout << "Sample Output (First " << (totalCount > 5 ? 5 : totalCount) << " elements): [ ";
        for (size_t i = 0; i < (totalCount > 5 ? 5 : totalCount); ++i) {
            std::cout << (float)output_host[i] << " ";
        }
        if (totalCount > 5)
            std::cout << "... ";
        std::cout << "]" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }
#endif

    aclrtFreeHost(input_host);
    aclrtFreeHost(output_host);
    aclrtFree(output_device);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t sliceSize>
bool RunAllToAll(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunAllToAllKernel<T, sliceSize>(rankId, n_ranks, n_devices, first_device_id, rootInfo);
        });
}

// Explicit instantiations
template bool RunAllToAll<int32_t, 256>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Non-template wrappers for test main.cpp
bool RunAllToAllInt32_256_2Ranks(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunAllToAll<int32_t, 256>(n_ranks, n_devices, first_rank_id, first_device_id);
}

bool RunAllToAllInt32_256_4Ranks(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunAllToAll<int32_t, 256>(n_ranks, n_devices, first_rank_id, first_device_id);
}
