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
// TALL_GATHER Test Kernel
// Tests the TALL_GATHER collective — all ranks contribute a slice, all get the
// full concatenated result.
//
// parallelGroup[r] = rank r's source buffer (per-rank shape: 1×count).
// dstGlobalData    = output buffer (shape: nranks×count).
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TAllGatherKernelImpl(__gm__ T *input, __gm__ T *output, int nranks,
                                            __gm__ HcclDeviceContext *hcclCtx)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    using TileData = pto::Tile<pto::TileType::Vec, T, 1, count, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(hcclCtx->rankId);

    ShapeDyn srcShape(1, 1, 1, 1, count);
    StrideDyn srcStride(count, count, count, count, 1);

    ShapeDyn dstShape(1, 1, 1, nranks, count);
    StrideDyn dstStride(static_cast<int>(nranks * count), static_cast<int>(nranks * count),
                        static_cast<int>(nranks * count), count, 1);

    Global outputG(output, dstShape, dstStride);

    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ T *remoteInput = HcclRemotePtr(hcclCtx, input, i);
        tensors[i] = Global(remoteInput, srcShape, srcStride);
    }

    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, my_rank);

    TileData stagingTile(1, count);
    TASSIGN(stagingTile, 0x0);

    pto::comm::TALL_GATHER(pg, outputG, stagingTile);

    pipe_barrier(PIPE_ALL);
}

template <typename T, size_t count>
static bool VerifyAllGatherOutput(T *output_host, int rank_id, int n_ranks)
{
    const size_t total_output = static_cast<size_t>(n_ranks) * count;
    for (int r = 0; r < n_ranks; ++r) {
        for (size_t i = 0; i < count; ++i) {
            const T expected = static_cast<T>((r + 1) * 10 + static_cast<int>(i));
            T actual = output_host[static_cast<size_t>(r) * count + i];
            if (actual != expected) {
                std::cout << "Rank " << rank_id << " validation failed at output[" << r << "*" << count << "+" << i
                          << "] = [" << (static_cast<size_t>(r) * count + i) << "]: expected " << (float)expected
                          << ", got " << (float)actual << std::endl;
                return false;
            }
        }
    }
#if ENABLE_DEBUG_PRINT
    std::cout << "\n================================================================" << std::endl;
    std::cout << "[DEBUG] Rank " << rank_id << ": TALL_GATHER SUCCESSFUL!" << std::endl;
    std::cout << "Summary: AllGather " << n_ranks << " ranks, " << count << " elements/rank, total " << total_output
              << " elements." << std::endl;
    std::cout << "Sample Result (First 5 elements): [ ";
    for (size_t i = 0; i < (total_output > 5 ? 5 : total_output); ++i) {
        std::cout << (float)output_host[i] << " ";
    }
    if (total_output > 5)
        std::cout << "... ";
    std::cout << "]" << std::endl;
    std::cout << "================================================================\n" << std::endl;
#endif
    return true;
}

template <typename T, size_t count>
bool RunAllGatherKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo *rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo)) {
        return false;
    }

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    if (n_ranks > 1) {
        WindowAlloc(localWinBase, winOffset, HCCL_WIN_SYNC_PREFIX);
    }
    void *input_ptr = WindowAlloc(localWinBase, winOffset, count * sizeof(T));

    const size_t total_output = static_cast<size_t>(n_ranks) * count;

    T *input_host = nullptr;
    T *output_host = nullptr;
    T *output_device = nullptr;
    T *staging = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&input_host), count * sizeof(T)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&output_host), total_output * sizeof(T)) != 0 ||
        aclrtMalloc(reinterpret_cast<void **>(&output_device), total_output * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) !=
            0 ||
        aclrtMalloc(reinterpret_cast<void **>(&staging), count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost/aclrtMalloc failed!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < count; ++i) {
        input_host[i] = static_cast<T>((rank_id + 1) * 10 + static_cast<int>(i));
    }

    aclrtMemcpy(staging, count * sizeof(T), input_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    WindowMemCopyIn<T><<<1, nullptr, ctx.stream>>>((T *)input_ptr, staging, static_cast<int>(count));
    aclrtSynchronizeStream(ctx.stream);
    aclrtFree(staging);

    HcclHostBarrier(ctx.comm, ctx.stream);

    TAllGatherKernelImpl<T, count>
        <<<1, nullptr, ctx.stream>>>((T *)input_ptr, (T *)output_device, n_ranks, ctx.deviceCtx);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    aclrtMemcpy(output_host, total_output * sizeof(T), output_device, total_output * sizeof(T),
                ACL_MEMCPY_DEVICE_TO_HOST);

    bool is_ok = VerifyAllGatherOutput<T, count>(output_host, rank_id, n_ranks);

    aclrtFreeHost(input_host);
    aclrtFreeHost(output_host);
    aclrtFree(output_device);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t count>
bool RunAllGather(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunAllGatherKernel<T, count>(rankId, n_ranks, n_devices, first_device_id, rootInfo);
        });
}

// Explicit instantiations
template bool RunAllGather<int32_t, 256>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Non-template wrappers for test main.cpp
bool RunAllGatherInt32_256(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunAllGather<int32_t, 256>(n_ranks, n_devices, first_rank_id, first_device_id);
}
