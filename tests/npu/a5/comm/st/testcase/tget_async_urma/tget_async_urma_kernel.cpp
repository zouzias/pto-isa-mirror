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
// TGET_ASYNC via URMA — one device kernel for single- and multi-AIV.
// Shard = get_block_idx() * (count / get_block_num()). With <<<1>>> this is the full buffer.
// ============================================================================

template <typename T, size_t count>
__global__ AICORE void TGetAsyncUrmaKernelImpl(__gm__ T *localBuf, int nranks, int my_rank, int first_rank_id,
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
    __gm__ T *recvBuf = sendBuf + count;

    pipe_barrier(PIPE_ALL);

    if (my_rank == root_rank) {
#ifdef PTO_URMA_SUPPORTED
        const int myPeer = my_rank - first_rank_id;
        for (int targetPeer = 0; targetPeer < nranks; ++targetPeer) {
            if (targetPeer == myPeer) {
                continue;
            }
            uint64_t peerBase = pto::comm::urma::UrmaPeerMrBaseAddr(urmaWorkspace, static_cast<uint32_t>(targetPeer));
            __gm__ T *remoteSendBuf = reinterpret_cast<__gm__ T *>(peerBase + kDataOffset) + elemOffset;
            __gm__ T *localRecvBuf = recvBuf + targetPeer * count + elemOffset;
            Global remoteSendG(remoteSendBuf, shape, stride);
            Global localRecvG(localRecvBuf, shape, stride);

            pto::comm::AsyncSession session;
            pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, static_cast<uint32_t>(targetPeer),
                                                                     session);
            auto event = pto::comm::TGET_ASYNC<pto::comm::DmaEngine::URMA>(localRecvG, remoteSendG, session);
            event.Wait(session);
        }
#endif
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Verify root-get results: each remote rank's data should match pattern i + rank * 10000.
// ============================================================================
template <typename T, size_t count>
bool VerifyRootGetResults(const uint8_t *output_host, int n_ranks, int first_rank_id, int root_rank, int rank_id,
                          int deviceId, int syncRet)
{
    const int root_peer = root_rank - first_rank_id;
    for (int src_peer = 0; src_peer < n_ranks; ++src_peer) {
        if (src_peer == root_peer)
            continue;
        const int src_logical = first_rank_id + src_peer;
        const size_t base = static_cast<size_t>(src_peer) * count;
        for (size_t i = 0; i < count; ++i) {
            T value = reinterpret_cast<const T *>(output_host)[base + i];
            T expected = static_cast<T>(i + src_logical * 10000);
            if (value != expected) {
                std::cerr << "Rank " << rank_id << " Device " << deviceId << " SyncRet " << syncRet
                          << " Expected: " << (float)expected << " Actual: " << (float)value << std::endl;
                return false;
            }
        }
    }
    return true;
}

// ============================================================================
// Host-side helpers / runner. Same path for single-/multi-AIV: qpNum = blockDim.
// ============================================================================
template <typename T, size_t count, int blockDim>
bool SetupUrmaCtxForBlocks(UrmaTestContext &ctx, int rank_id, int n_ranks, int n_devices, int first_device_id,
                           int root_rank, size_t commBytesNeeded)
{
    static_assert(blockDim >= 1, "blockDim must be >= 1");
    static_assert(count % static_cast<size_t>(blockDim) == 0, "count must be divisible by blockDim");
    if (!ctx.Setup(rank_id, n_ranks, n_devices, first_device_id, root_rank, commBytesNeeded,
                   static_cast<uint32_t>(blockDim))) {
        return false;
    }
    if (ctx.urmaMgr.GetQpNum() != static_cast<uint32_t>(blockDim)) {
        std::cerr << "[ERROR] expected qpNum=" << blockDim << " got " << ctx.urmaMgr.GetQpNum() << std::endl;
        ctx.Cleanup();
        return false;
    }
    return true;
}

template <typename T, size_t count>
bool AllocAndFillGetHostBuffers(uint8_t *&input_host, uint8_t *&output_host, int rank_id, int n_ranks)
{
    const size_t recv_elems = static_cast<size_t>(n_ranks) * count;
    aclrtMallocHost(reinterpret_cast<void **>(&input_host), count * sizeof(T));
    aclrtMallocHost(reinterpret_cast<void **>(&output_host), recv_elems * sizeof(T));
    if (!input_host || !output_host) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        if (input_host)
            aclrtFreeHost(input_host);
        if (output_host)
            aclrtFreeHost(output_host);
        input_host = nullptr;
        output_host = nullptr;
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        reinterpret_cast<T *>(input_host)[i] = static_cast<T>(i + rank_id * 10000);
    }
    for (size_t i = 0; i < recv_elems; ++i) {
        reinterpret_cast<T *>(output_host)[i] = static_cast<T>(-1);
    }
    return true;
}

template <typename T, size_t count, int blockDim>
bool RunGetAsyncUrmaRootGetKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, int first_rank_id,
                                  int root_rank)
{
    const size_t recv_elems = static_cast<size_t>(n_ranks) * count;
    size_t commBytesNeeded = 64 * sizeof(int32_t) + (static_cast<size_t>(n_ranks) + 1) * count * sizeof(T);

    UrmaTestContext ctx;
    if (!SetupUrmaCtxForBlocks<T, count, blockDim>(ctx, rank_id, n_ranks, n_devices, first_device_id, root_rank,
                                                   commBytesNeeded)) {
        return false;
    }

    uint8_t *input_host = nullptr;
    uint8_t *output_host = nullptr;
    if (!AllocAndFillGetHostBuffers<T, count>(input_host, output_host, rank_id, n_ranks)) {
        ctx.Cleanup();
        return false;
    }

    constexpr size_t kDataOffset = 64 * sizeof(int32_t);
    T *sendBuf = reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(ctx.devBuf) + kDataOffset);
    T *recvBuf = sendBuf + count;
    aclrtMemcpy(sendBuf, count * sizeof(T), input_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(recvBuf, recv_elems * sizeof(T), output_host, recv_elems * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    CommMpiBarrier();
    TGetAsyncUrmaKernelImpl<T, count>
        <<<blockDim, nullptr, ctx.stream>>>(reinterpret_cast<T *>(ctx.devBuf), n_ranks, rank_id, first_rank_id,
                                            root_rank, reinterpret_cast<uint8_t *>(ctx.urmaMgr.GetWorkspaceAddr()));
    int syncRet = aclrtSynchronizeStream(ctx.stream);
    CommMpiBarrier();

    bool is_ok = true;
    if (rank_id == root_rank) {
        aclrtMemcpy(output_host, recv_elems * sizeof(T), recvBuf, recv_elems * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
        is_ok = VerifyRootGetResults<T, count>(output_host, n_ranks, first_rank_id, root_rank, rank_id, ctx.deviceId,
                                               syncRet);
        if (is_ok && blockDim > 1) {
            std::cerr << "[PASS] count=" << count << " blockDim=" << blockDim << " qpNum=" << ctx.urmaMgr.GetQpNum()
                      << std::endl;
        }
    }

    aclrtFreeHost(input_host);
    aclrtFreeHost(output_host);
    ctx.Cleanup();
    return is_ok;
}

template <typename T, size_t count, int blockDim>
bool RunGetAsyncUrmaRootGet(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunUrmaTestMpiLaunch(n_ranks, n_devices, first_rank_id, first_device_id,
                                RunGetAsyncUrmaRootGetKernel<T, count, blockDim>);
}

// Explicit instantiations (blockDim=1 for single-AIV, >1 for multi-AIV)
template bool RunGetAsyncUrmaRootGet<float, 256, 1>(int, int, int, int);
template bool RunGetAsyncUrmaRootGet<int32_t, 4096, 1>(int, int, int, int);
template bool RunGetAsyncUrmaRootGet<uint8_t, 512, 1>(int, int, int, int);
template bool RunGetAsyncUrmaRootGet<float, 524288, 1>(int, int, int, int);     // MR = 8MB (>2MB)
template bool RunGetAsyncUrmaRootGet<int32_t, 67108864, 1>(int, int, int, int); // MR ≈ 770MB (>512MB)
template bool RunGetAsyncUrmaRootGet<float, 1024, 4>(int, int, int, int);
template bool RunGetAsyncUrmaRootGet<int32_t, 4096, 8>(int, int, int, int);
template bool RunGetAsyncUrmaRootGet<float, 4096, 64>(int, int, int, int);
