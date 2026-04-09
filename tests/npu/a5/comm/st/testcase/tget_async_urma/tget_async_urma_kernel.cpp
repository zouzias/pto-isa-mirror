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
#include "../common.hpp"

// ============================================================================
// TGET_ASYNC via URMA — device kernel.
// ============================================================================

template <typename T, size_t count>
__global__ AICORE void TGetAsyncUrmaKernelImpl(__gm__ T *localBuf, int nranks, int my_rank, int root_rank,
                                               int elem_offset, int elem_count, __gm__ uint64_t *remoteAddrs,
                                               __gm__ uint8_t *urmaWorkspace)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    if (elem_count <= 0 || elem_offset < 0 || elem_offset + elem_count > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    ShapeDyn shape(1, 1, 1, 1, elem_count);
    StrideDyn stride(elem_count, elem_count, elem_count, elem_count, 1);

    constexpr size_t kDataOffset = 64 * sizeof(int32_t);

    __gm__ T *sendBuf = reinterpret_cast<__gm__ T *>(reinterpret_cast<__gm__ uint8_t *>(localBuf) + kDataOffset);
    __gm__ T *recvBuf = sendBuf + count;

    pipe_barrier(PIPE_ALL);

    if (my_rank == root_rank) {
#ifdef PTO_URMA_SUPPORTED
        for (int target_rank = 0; target_rank < nranks; ++target_rank) {
            if (target_rank == root_rank) {
                continue;
            }
            __gm__ T *remoteSendBuf =
                reinterpret_cast<__gm__ T *>(remoteAddrs[target_rank] + kDataOffset) + elem_offset;
            __gm__ T *localRecvBuf = recvBuf + target_rank * count + elem_offset;
            Global remoteSendG(remoteSendBuf, shape, stride);
            Global localRecvG(localRecvBuf, shape, stride);

            pto::comm::AsyncSession session;
            pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, static_cast<uint32_t>(target_rank),
                                                                     session);
            auto event = pto::comm::TGET_ASYNC<pto::comm::DmaEngine::URMA>(localRecvG, remoteSendG, session);
            event.Wait(session);
        }
#endif
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Host-side runner.
// ============================================================================
template <typename T, size_t count>
bool RunGetAsyncUrmaRootGetKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, int root_rank)
{
    const size_t recv_elems = static_cast<size_t>(n_ranks) * count;
    size_t commBytesNeeded = 64 * sizeof(int32_t) + (static_cast<size_t>(n_ranks) + 1) * count * sizeof(T);

    UrmaTestContext ctx;
    if (!ctx.Setup(rank_id, n_ranks, n_devices, first_device_id, root_rank, commBytesNeeded)) {
        return false;
    }

    uint8_t *input_host = nullptr;
    uint8_t *output_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&input_host), count * sizeof(T));
    aclrtMallocHost(reinterpret_cast<void **>(&output_host), recv_elems * sizeof(T));
    if (!input_host || !output_host) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        ctx.Cleanup();
        return false;
    }

    for (size_t i = 0; i < count; ++i) {
        reinterpret_cast<T *>(input_host)[i] = static_cast<T>(i + rank_id * 10000);
    }
    for (size_t i = 0; i < recv_elems; ++i) {
        reinterpret_cast<T *>(output_host)[i] = static_cast<T>(-1);
    }

    constexpr size_t kDataOffset = 64 * sizeof(int32_t);
    uint8_t *commBytes = reinterpret_cast<uint8_t *>(ctx.devBuf);
    T *sendBuf = reinterpret_cast<T *>(commBytes + kDataOffset);
    T *recvBuf = sendBuf + count;

    aclrtMemcpy(sendBuf, count * sizeof(T), input_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(recvBuf, recv_elems * sizeof(T), output_host, recv_elems * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    CommMpiBarrier();

    TGetAsyncUrmaKernelImpl<T, count><<<1, nullptr, ctx.stream>>>(
        reinterpret_cast<T *>(ctx.devBuf), n_ranks, rank_id, root_rank, 0, static_cast<int>(count),
        reinterpret_cast<uint64_t *>(ctx.devRemoteAddrs), reinterpret_cast<uint8_t *>(ctx.urmaMgr.GetWorkspaceAddr()));
    int syncRet = aclrtSynchronizeStream(ctx.stream);

    CommMpiBarrier();

    bool is_ok = true;
    if (rank_id == root_rank) {
        aclrtMemcpy(output_host, recv_elems * sizeof(T), recvBuf, recv_elems * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

        for (int src_rank = 0; src_rank < n_ranks && is_ok; ++src_rank) {
            if (src_rank == root_rank) {
                continue;
            }
            const size_t base = static_cast<size_t>(src_rank) * count;
            for (size_t i = 0; i < count; ++i) {
                T value = reinterpret_cast<T *>(output_host)[base + i];
                T expected = static_cast<T>(i + src_rank * 10000);
                if (value != expected) {
                    std::cerr << "Rank " << rank_id << " Device " << ctx.deviceId << " SyncRet " << syncRet
                              << " Expected: " << (float)expected << " Actual: " << (float)value << std::endl;
                    is_ok = false;
                    break;
                }
            }
        }
    }

    aclrtFreeHost(input_host);
    aclrtFreeHost(output_host);
    ctx.Cleanup();

    return is_ok;
}

// ============================================================================
// MPI-based multi-rank launch.
// ============================================================================
template <typename T, size_t count>
bool RunGetAsyncUrmaRootGet(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunUrmaTestMpiLaunch(n_ranks, n_devices, first_rank_id, first_device_id,
                                RunGetAsyncUrmaRootGetKernel<T, count>);
}

// Explicit instantiations
template bool RunGetAsyncUrmaRootGet<float, 256>(int, int, int, int);
template bool RunGetAsyncUrmaRootGet<int32_t, 4096>(int, int, int, int);
template bool RunGetAsyncUrmaRootGet<uint8_t, 512>(int, int, int, int);
template bool RunGetAsyncUrmaRootGet<float, 524288>(int, int, int, int);     // MR = 8MB (>2MB)
template bool RunGetAsyncUrmaRootGet<int32_t, 67108864>(int, int, int, int); // MR ≈ 770MB (>512MB)
