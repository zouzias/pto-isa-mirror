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

#include "../common.hpp"
#include "pto/common/pto_tile.hpp"
#ifdef PTO_URMA_SUPPORTED
#include "pto/comm/async/urma/urma_async_intrin.hpp"
#endif

// ============================================================================
// TPUT_ASYNC via URMA — one device kernel for single- and multi-AIV.
// Shard = get_block_idx() * (count / get_block_num()). With <<<1>>> this is the full buffer.
// ============================================================================

template <typename T, size_t count>
__global__ AICORE void TPutAsyncUrmaKernelImpl(__gm__ T *localBuf, int nranks, int my_rank, int first_rank_id,
                                               int root_rank, __gm__ uint8_t *urmaWorkspace)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    const int blockNum = static_cast<int>(get_block_num());
    const int coreIdx = static_cast<int>(get_block_idx());
    if (blockNum <= 0 || coreIdx < 0 || coreIdx >= blockNum || static_cast<int>(count) % blockNum != 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    const int elemCount = static_cast<int>(count) / blockNum;
    const int elemOffset = coreIdx * elemCount;
    constexpr size_t kDataOffset = 64 * sizeof(int32_t);

    ShapeDyn shape(1, 1, 1, 1, elemCount);
    StrideDyn stride(elemCount, elemCount, elemCount, elemCount, 1);

    __gm__ T *sendBuf = reinterpret_cast<__gm__ T *>(reinterpret_cast<__gm__ uint8_t *>(localBuf) + kDataOffset);
    Global sendG(sendBuf + elemOffset, shape, stride);

    pipe_barrier(PIPE_ALL);

    if (my_rank == root_rank) {
#ifdef PTO_URMA_SUPPORTED
        const int myPeer = my_rank - first_rank_id;
        for (int targetPeer = 0; targetPeer < nranks; ++targetPeer) {
            if (targetPeer == myPeer) {
                continue;
            }
            uint64_t peerBase = pto::comm::urma::UrmaPeerMrBaseAddr(urmaWorkspace, static_cast<uint32_t>(targetPeer));
            __gm__ T *remoteRecvBuf = reinterpret_cast<__gm__ T *>(peerBase + kDataOffset) + count + elemOffset;
            Global remoteRecvG(remoteRecvBuf, shape, stride);

            pto::comm::AsyncSession session;
            pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, static_cast<uint32_t>(targetPeer),
                                                                     session);
            auto event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(remoteRecvG, sendG, session);
            event.Wait(session);
        }
#endif
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Host-side runner. Same path for single-/multi-AIV: qpNum = blockDim, launch <<<blockDim>>>.
// ============================================================================
template <typename T, size_t count, int blockDim>
bool RunPutAsyncUrmaRootPutKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, int first_rank_id,
                                  int root_rank)
{
    static_assert(blockDim >= 1, "blockDim must be >= 1");
    static_assert(count % static_cast<size_t>(blockDim) == 0, "count must be divisible by blockDim");

    size_t commBytesNeeded = 64 * sizeof(int32_t) + 2 * count * sizeof(T);

    UrmaTestContext ctx;
    if (!ctx.Setup(rank_id, n_ranks, n_devices, first_device_id, root_rank, commBytesNeeded,
                   static_cast<uint32_t>(blockDim))) {
        return false;
    }

    if (ctx.urmaMgr.GetQpNum() != static_cast<uint32_t>(blockDim)) {
        std::cerr << "[ERROR] expected qpNum=" << blockDim << " got " << ctx.urmaMgr.GetQpNum() << std::endl;
        ctx.Cleanup();
        return false;
    }

    uint8_t *input_host = nullptr;
    uint8_t *output_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&input_host), count * sizeof(T));
    aclrtMallocHost(reinterpret_cast<void **>(&output_host), count * sizeof(T));
    if (!input_host || !output_host) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        if (input_host)
            aclrtFreeHost(input_host);
        if (output_host)
            aclrtFreeHost(output_host);
        ctx.Cleanup();
        return false;
    }

    for (size_t i = 0; i < count; ++i) {
        reinterpret_cast<T *>(input_host)[i] = static_cast<T>(i + rank_id * 10000);
        reinterpret_cast<T *>(output_host)[i] = static_cast<T>(-1);
    }

    constexpr size_t kDataOffset = 64 * sizeof(int32_t);
    uint8_t *commBytes = reinterpret_cast<uint8_t *>(ctx.devBuf);
    T *sendBuf = reinterpret_cast<T *>(commBytes + kDataOffset);
    T *recvBuf = sendBuf + count;

    aclrtMemcpy(sendBuf, count * sizeof(T), input_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(recvBuf, count * sizeof(T), output_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    CommMpiBarrier();

    TPutAsyncUrmaKernelImpl<T, count>
        <<<blockDim, nullptr, ctx.stream>>>(reinterpret_cast<T *>(ctx.devBuf), n_ranks, rank_id, first_rank_id,
                                            root_rank, reinterpret_cast<uint8_t *>(ctx.urmaMgr.GetWorkspaceAddr()));
    int syncRet = aclrtSynchronizeStream(ctx.stream);

    CommMpiBarrier();

    aclrtMemcpy(output_host, count * sizeof(T), recvBuf, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    bool is_ok = true;
    if (rank_id != root_rank) {
        for (size_t i = 0; i < count; ++i) {
            T value = reinterpret_cast<T *>(output_host)[i];
            T expected = static_cast<T>(i + root_rank * 10000);
            if (value != expected) {
                std::cerr << "Rank " << rank_id << " Device " << ctx.deviceId << " SyncRet " << syncRet << " idx=" << i
                          << " Expected: " << static_cast<float>(expected) << " Actual: " << static_cast<float>(value)
                          << " blockDim=" << blockDim << std::endl;
                is_ok = false;
                break;
            }
        }
    }

    if (rank_id == root_rank && is_ok && blockDim > 1) {
        std::cerr << "[PASS] count=" << count << " blockDim=" << blockDim << " qpNum=" << ctx.urmaMgr.GetQpNum()
                  << std::endl;
    }

    aclrtFreeHost(input_host);
    aclrtFreeHost(output_host);
    ctx.Cleanup();
    return is_ok;
}

template <typename T, size_t count, int blockDim>
bool RunPutAsyncUrmaRootPut(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunUrmaTestMpiLaunch(n_ranks, n_devices, first_rank_id, first_device_id,
                                RunPutAsyncUrmaRootPutKernel<T, count, blockDim>);
}

// Explicit instantiations (blockDim=1 for single-AIV, >1 for multi-AIV)
template bool RunPutAsyncUrmaRootPut<float, 256, 1>(int, int, int, int);
template bool RunPutAsyncUrmaRootPut<int32_t, 4096, 1>(int, int, int, int);
template bool RunPutAsyncUrmaRootPut<uint8_t, 512, 1>(int, int, int, int);
template bool RunPutAsyncUrmaRootPut<uint8_t, 64, 1>(int, int, int, int);
template bool RunPutAsyncUrmaRootPut<float, 64, 1>(int, int, int, int);
template bool RunPutAsyncUrmaRootPut<float, 524288, 1>(int, int, int, int);     // MR = 6MB (>2MB)
template bool RunPutAsyncUrmaRootPut<int32_t, 67108864, 1>(int, int, int, int); // MR ≈ 514MB (>512MB)
template bool RunPutAsyncUrmaRootPut<float, 1024, 4>(int, int, int, int);
template bool RunPutAsyncUrmaRootPut<int32_t, 4096, 8>(int, int, int, int);
template bool RunPutAsyncUrmaRootPut<float, 4096, 64>(int, int, int, int);
