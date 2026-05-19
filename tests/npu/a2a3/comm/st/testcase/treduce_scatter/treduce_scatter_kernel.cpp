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
// TREDUCE_SCATTER Test Kernel
// All ranks collectively reduce, each rank keeps its own slice of the result.
// Input per rank: nranks * sliceSize elements
// Output per rank: sliceSize elements (the slice corresponding to selfIdx)
// ============================================================================
template <typename T, size_t sliceSize, pto::comm::ReduceOp op>
__global__ AICORE void TReduceScatterKernelImpl(__gm__ T *input, __gm__ T *output, int nranks,
                                                __gm__ HcclDeviceContext *hcclCtx)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    using TileData = pto::Tile<pto::TileType::Vec, T, 1, sliceSize, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(hcclCtx->rankId);
    size_t totalCount = static_cast<size_t>(nranks) * sliceSize;

    // Input shape: (1,1,1,nranks,sliceSize) — DIM_3 = nranks so IMPL can slice along rows
    ShapeDyn inputShape(1, 1, 1, nranks, sliceSize);
    StrideDyn inputStride(totalCount, totalCount, totalCount, sliceSize, 1);

    // Output: (1,1,1,1,sliceSize) — this rank's single slice
    ShapeDyn outputShape(1, 1, 1, 1, sliceSize);
    StrideDyn outputStride(sliceSize, sliceSize, sliceSize, sliceSize, 1);

    Global outputG(output, outputShape, outputStride);

    // ParallelGroup: tensors[r] points to rank r's full input buffer
    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ T *remoteInput = HcclRemotePtr(hcclCtx, input, i);
        tensors[i] = Global(remoteInput, inputShape, inputStride);
    }

    // rootIdx = selfIdx: determines which slice this rank keeps
    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, my_rank);

    TileData accTile(1, sliceSize);
    TileData recvTile(1, sliceSize);

    TASSIGN(accTile, 0x0);
    TASSIGN(recvTile, 0x10000);

    // All ranks participate in TREDUCE_SCATTER
    pto::comm::TREDUCE_SCATTER(pg, outputG, accTile, recvTile, op);

    pipe_barrier(PIPE_ALL);
}

template <typename T>
T ReduceScatterExpected(T base, int selfIdx, int n_ranks, size_t sliceSize, size_t elemIdx, pto::comm::ReduceOp op)
{
    // For element i in selfIdx's slice:
    //   input[r][selfIdx * sliceSize + i] = (selfIdx * sliceSize + i) + r * 100
    // Reduce across all ranks r:
    T expected = static_cast<T>(selfIdx * static_cast<int>(sliceSize) + static_cast<int>(elemIdx));
    for (int r = 1; r < n_ranks; ++r) {
        const T val = static_cast<T>(selfIdx * static_cast<int>(sliceSize) + static_cast<int>(elemIdx) + r * 100);
        switch (op) {
            case pto::comm::ReduceOp::Sum:
                expected = static_cast<T>(expected + val);
                break;
            case pto::comm::ReduceOp::Max:
                expected = (val > expected) ? val : expected;
                break;
            case pto::comm::ReduceOp::Min:
                expected = (val < expected) ? val : expected;
                break;
        }
    }
    return expected;
}

template <typename T, size_t sliceSize, pto::comm::ReduceOp op>
bool RunReduceScatterKernel(int rank_id, int n_ranks, int n_devices, int first_device_id,
                            const HcclRootInfo *rootInfo)
{
    constexpr size_t totalCount = 16 * sliceSize;  // max 16 ranks
    size_t actualTotalCount = static_cast<size_t>(n_ranks) * sliceSize;

    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo)) {
        return false;
    }

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    if (n_ranks > 1) {
        WindowAlloc(localWinBase, winOffset, HCCL_WIN_SYNC_PREFIX);
    }
    void *input_ptr = WindowAlloc(localWinBase, winOffset, actualTotalCount * sizeof(T));

    T *input_host = nullptr;
    T *output_host = nullptr;
    T *output_device = nullptr;
    T *staging = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&input_host), actualTotalCount * sizeof(T)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&output_host), sliceSize * sizeof(T)) != 0 ||
        aclrtMalloc(reinterpret_cast<void **>(&output_device), sliceSize * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) !=
            0 ||
        aclrtMalloc(reinterpret_cast<void **>(&staging), actualTotalCount * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) !=
            0) {
        std::cerr << "[ERROR] aclrtMallocHost/aclrtMalloc failed!" << std::endl;
        return false;
    }

    // Each rank fills its full input buffer: input[i] = i + rank_id * 100
    for (size_t i = 0; i < actualTotalCount; ++i) {
        input_host[i] = static_cast<T>(i + rank_id * 100);
    }

    aclrtMemcpy(staging, actualTotalCount * sizeof(T), input_host, actualTotalCount * sizeof(T),
                ACL_MEMCPY_HOST_TO_DEVICE);
    WindowMemCopyIn<T><<<1, nullptr, ctx.stream>>>((T *)input_ptr, staging, static_cast<int>(actualTotalCount));
    aclrtSynchronizeStream(ctx.stream);
    aclrtFree(staging);

    HcclHostBarrier(ctx.comm, ctx.stream);

    TReduceScatterKernelImpl<T, sliceSize, op>
        <<<1, nullptr, ctx.stream>>>((T *)input_ptr, (T *)output_device, n_ranks, ctx.deviceCtx);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    // Every rank verifies its own slice
    bool is_ok = true;
    aclrtMemcpy(output_host, sliceSize * sizeof(T), output_device, sliceSize * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    for (size_t i = 0; i < sliceSize; ++i) {
        const T expected = ReduceScatterExpected<T>(static_cast<T>(0), rank_id, n_ranks, sliceSize, i, op);
        T actual = output_host[i];
        if (actual != expected) {
            std::cout << "Rank " << rank_id << " validation failed at index " << i << ": expected " << (float)expected
                      << ", got " << (float)actual << std::endl;
            is_ok = false;
            break;
        }
    }

#if ENABLE_DEBUG_PRINT
    if (is_ok) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "[DEBUG] Rank " << rank_id << ": TREDUCE_SCATTER SUCCESSFUL!" << std::endl;
        std::cout << "Summary: ReduceScatter " << n_ranks << " ranks, sliceSize " << sliceSize << " elements."
                  << std::endl;
        std::cout << "Sample Result (First 5 elements): [ ";
        for (size_t i = 0; i < (sliceSize > 5 ? 5 : sliceSize); ++i) {
            std::cout << (float)output_host[i] << " ";
        }
        if (sliceSize > 5)
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

template <typename T, size_t sliceSize, pto::comm::ReduceOp op>
bool RunReduceScatter(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunReduceScatterKernel<T, sliceSize, op>(rankId, n_ranks, n_devices, first_device_id, rootInfo);
        });
}

// Explicit instantiations
template bool RunReduceScatter<int32_t, 256, pto::comm::ReduceOp::Sum>(int n_ranks, int n_devices, int first_rank_id,
                                                                       int first_device_id);

// Non-template wrappers for test main.cpp
bool RunReduceScatterInt32_256_Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunReduceScatter<int32_t, 256, pto::comm::ReduceOp::Sum>(n_ranks, n_devices, first_rank_id, first_device_id);
}
