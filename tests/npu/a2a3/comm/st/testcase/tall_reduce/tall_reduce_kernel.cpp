/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "../comm_kernel_common.hpp"

// ============================================================================
// TALL_REDUCE Test Kernel
// Tests the TALL_REDUCE collective - all ranks reduce and all get the result
// ============================================================================
template <typename T, size_t count, pto::comm::ReduceOp op>
__global__ AICORE void TAllReduceKernelImpl(__gm__ T *input, __gm__ T *output, int nranks,
                                            __gm__ HcclDeviceContext *hcclCtx)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    using TileData = pto::Tile<pto::TileType::Vec, T, 1, count, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(hcclCtx->rankId);

    ShapeDyn shape(1, 1, 1, 1, count);
    StrideDyn stride(count, count, count, count, 1);

    Global outputG(output, shape, stride);

    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ T *remoteInput = HcclRemotePtr(hcclCtx, input, i);
        tensors[i] = Global(remoteInput, shape, stride);
    }

    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, my_rank);

    TileData accTile(1, count);
    TileData recvTile(1, count);

    TASSIGN(accTile, 0x0);
    TASSIGN(recvTile, 0x10000);

    pto::comm::TALL_REDUCE(pg, outputG, accTile, recvTile, op);

    pipe_barrier(PIPE_ALL);
}

template <typename T>
T ReduceExpected(T base, int n_ranks, pto::comm::ReduceOp op)
{
    T expected = base;
    for (int r = 1; r < n_ranks; ++r) {
        const T val = static_cast<T>(base + r * 100);
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

template <typename T, size_t count, pto::comm::ReduceOp op>
static bool VerifyAllReduceOutput(T *output_host, int rank_id, int n_ranks)
{
    for (size_t i = 0; i < count; ++i) {
        const T expected = ReduceExpected(static_cast<T>(i), n_ranks, op);
        T actual = output_host[i];
        if (actual != expected) {
            std::cout << "Rank " << rank_id << " validation failed at index " << i << ": expected " << (float)expected
                      << ", got " << (float)actual << std::endl;
            return false;
        }
    }
#if ENABLE_DEBUG_PRINT
    std::cout << "\n================================================================" << std::endl;
    std::cout << "[DEBUG] Rank " << rank_id << ": TALL_REDUCE SUCCESSFUL!" << std::endl;
    std::cout << "Summary: AllReduced " << n_ranks << " segments, result size " << count << " elements." << std::endl;
    std::cout << "Sample Result (First 5 elements): [ ";
    for (size_t i = 0; i < (count > 5 ? 5 : count); ++i) {
        std::cout << (float)output_host[i] << " ";
    }
    if (count > 5)
        std::cout << "... ";
    std::cout << "]" << std::endl;
    std::cout << "================================================================\n" << std::endl;
#endif
    return true;
}

template <typename T, size_t count, pto::comm::ReduceOp op>
bool RunAllReduceKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo *rootInfo)
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

    T *input_host = nullptr;
    T *output_host = nullptr;
    T *output_device = nullptr;
    T *staging = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&input_host), count * sizeof(T)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&output_host), count * sizeof(T)) != 0 ||
        aclrtMalloc(reinterpret_cast<void **>(&output_device), count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) != 0 ||
        aclrtMalloc(reinterpret_cast<void **>(&staging), count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost/aclrtMalloc failed!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < count; ++i) {
        input_host[i] = static_cast<T>(i + rank_id * 100);
    }

    aclrtMemcpy(staging, count * sizeof(T), input_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    WindowMemCopyIn<T><<<1, nullptr, ctx.stream>>>((T *)input_ptr, staging, static_cast<int>(count));
    aclrtSynchronizeStream(ctx.stream);
    aclrtFree(staging);

    HcclHostBarrier(ctx.comm, ctx.stream);

    TAllReduceKernelImpl<T, count, op>
        <<<1, nullptr, ctx.stream>>>((T *)input_ptr, (T *)output_device, n_ranks, ctx.deviceCtx);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    aclrtMemcpy(output_host, count * sizeof(T), output_device, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    bool is_ok = VerifyAllReduceOutput<T, count, op>(output_host, rank_id, n_ranks);

    aclrtFreeHost(input_host);
    aclrtFreeHost(output_host);
    aclrtFree(output_device);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t count, pto::comm::ReduceOp op>
bool RunAllReduce(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunAllReduceKernel<T, count, op>(rankId, n_ranks, n_devices, first_device_id, rootInfo);
        });
}

// Explicit instantiations
template bool RunAllReduce<int32_t, 256, pto::comm::ReduceOp::Sum>(int n_ranks, int n_devices, int first_rank_id,
                                                                   int first_device_id);
template bool RunAllReduce<int32_t, 256, pto::comm::ReduceOp::Max>(int n_ranks, int n_devices, int first_rank_id,
                                                                   int first_device_id);

// Non-template wrappers for test main.cpp
bool RunAllReduceInt32_256_Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunAllReduce<int32_t, 256, pto::comm::ReduceOp::Sum>(n_ranks, n_devices, first_rank_id, first_device_id);
}

bool RunAllReduceInt32_256_Max(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunAllReduce<int32_t, 256, pto::comm::ReduceOp::Max>(n_ranks, n_devices, first_rank_id, first_device_id);
}

// ============================================================================
// Large Shape Chunked Test Kernel
// Tests TALL_REDUCE with GlobalTensor shape > UB tile capacity (forces chunked path)
// GlobalTensor per rank: (1, 1, 1, total_rows, cols), Tile: (tile_rows, cols)
// where total_rows > tile_rows, triggering automatic chunking in TALL_REDUCE_IMPL
// ============================================================================
template <typename T, size_t total_rows, size_t cols, size_t tile_rows, pto::comm::ReduceOp op>
__global__ AICORE void TAllReduceLargeShapeKernelImpl(__gm__ T *input, __gm__ T *output, int nranks,
                                                      __gm__ HcclDeviceContext *hcclCtx)
{
    constexpr size_t total_count = total_rows * cols;
    static_assert(total_rows > tile_rows, "total_rows must exceed tile_rows to test chunking");
    static_assert(total_rows % tile_rows == 0, "total_rows must be divisible by tile_rows for static tile");

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, T, tile_rows, cols, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(hcclCtx->rankId);

    ShapeDyn fullShape(1, 1, 1, total_rows, cols);
    StrideDyn fullStride(total_count, total_count, total_count, cols, 1);

    Global outputG(output, fullShape, fullStride);

    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ T *remoteInput = HcclRemotePtr(hcclCtx, input, i);
        tensors[i] = Global(remoteInput, fullShape, fullStride);
    }

    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, my_rank);

    TileData accTile(tile_rows, cols);
    TileData recvTile(tile_rows, cols);

    TASSIGN(accTile, 0x0);
    TASSIGN(recvTile, 0x10000);

    pto::comm::TALL_REDUCE(pg, outputG, accTile, recvTile, op);

    pipe_barrier(PIPE_ALL);
}

template <typename T, size_t total_rows, size_t cols, size_t tile_rows, pto::comm::ReduceOp op>
static bool VerifyAllReduceLargeOutput(T *output_host, int rank_id, int n_ranks)
{
    constexpr size_t total_count = total_rows * cols;
    for (size_t i = 0; i < total_count; ++i) {
        const T expected = ReduceExpected(static_cast<T>(i), n_ranks, op);
        T actual = output_host[i];
        if (actual != expected) {
            std::cout << "Rank " << rank_id << " validation failed at index " << i << " (row=" << (i / cols)
                      << ", col=" << (i % cols) << ")"
                      << ": expected " << (float)expected << ", got " << (float)actual << std::endl;
            return false;
        }
    }
#if ENABLE_DEBUG_PRINT
    std::cout << "\n================================================================" << std::endl;
    std::cout << "[DEBUG] Rank " << rank_id << ": TALL_REDUCE LargeShape SUCCESSFUL! (" << total_rows << "x" << cols
              << ", tile=" << tile_rows << "x" << cols << ", chunks=" << (total_rows / tile_rows) << ")" << std::endl;
    std::cout << "Sample Result (First 5 elements): [ ";
    for (size_t i = 0; i < (total_count > 5 ? 5 : total_count); ++i) {
        std::cout << (float)output_host[i] << " ";
    }
    if (total_count > 5)
        std::cout << "... ";
    std::cout << "]" << std::endl;
    std::cout << "================================================================\n" << std::endl;
#endif
    return true;
}

template <typename T, size_t total_rows, size_t cols, size_t tile_rows, pto::comm::ReduceOp op>
bool RunAllReduceLargeShapeKernel(int rank_id, int n_ranks, int n_devices, int first_device_id,
                                  const HcclRootInfo *rootInfo)
{
    constexpr size_t total_count = total_rows * cols;

    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo)) {
        return false;
    }

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    if (n_ranks > 1) {
        WindowAlloc(localWinBase, winOffset, HCCL_WIN_SYNC_PREFIX);
    }
    void *input_ptr = WindowAlloc(localWinBase, winOffset, total_count * sizeof(T));

    T *input_host = nullptr;
    T *output_host = nullptr;
    T *output_device = nullptr;
    T *staging = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&input_host), total_count * sizeof(T)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&output_host), total_count * sizeof(T)) != 0 ||
        aclrtMalloc(reinterpret_cast<void **>(&output_device), total_count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) !=
            0 ||
        aclrtMalloc(reinterpret_cast<void **>(&staging), total_count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost/aclrtMalloc failed!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < total_count; ++i) {
        input_host[i] = static_cast<T>(i + rank_id * 100);
    }

    aclrtMemcpy(staging, total_count * sizeof(T), input_host, total_count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    WindowMemCopyIn<T><<<1, nullptr, ctx.stream>>>((T *)input_ptr, staging, static_cast<int>(total_count));
    aclrtSynchronizeStream(ctx.stream);
    aclrtFree(staging);

    HcclHostBarrier(ctx.comm, ctx.stream);

    TAllReduceLargeShapeKernelImpl<T, total_rows, cols, tile_rows, op>
        <<<1, nullptr, ctx.stream>>>((T *)input_ptr, (T *)output_device, n_ranks, ctx.deviceCtx);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    aclrtMemcpy(output_host, total_count * sizeof(T), output_device, total_count * sizeof(T),
                ACL_MEMCPY_DEVICE_TO_HOST);
    bool is_ok = VerifyAllReduceLargeOutput<T, total_rows, cols, tile_rows, op>(output_host, rank_id, n_ranks);

    aclrtFreeHost(input_host);
    aclrtFreeHost(output_host);
    aclrtFree(output_device);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t total_rows, size_t cols, size_t tile_rows, pto::comm::ReduceOp op>
bool RunAllReduceLargeShape(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(n_ranks, first_rank_id, first_device_id,
                                      [&](int rankId, const HcclRootInfo *rootInfo) {
                                          return RunAllReduceLargeShapeKernel<T, total_rows, cols, tile_rows, op>(
                                              rankId, n_ranks, n_devices, first_device_id, rootInfo);
                                      });
}

// Explicit instantiations for large shape tests
// int32: 128 rows x 32 cols, tile 16 rows → 8 chunks, Sum
template bool RunAllReduceLargeShape<int32_t, 128, 32, 16, pto::comm::ReduceOp::Sum>(int, int, int, int);

// Non-template wrappers for large shape tests
bool RunAllReduceLargeShape_Int32_128x32_tile16_Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunAllReduceLargeShape<int32_t, 128, 32, 16, pto::comm::ReduceOp::Sum>(n_ranks, n_devices, first_rank_id,
                                                                                  first_device_id);
}
