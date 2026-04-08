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
#include <vector>

#include <pto/pto-inst.hpp>
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"

// ============================================================================
// TGET_ASYNC via UDMA — standalone test (no HCCL dependency).
// ============================================================================

template <typename T, size_t count>
__global__ AICORE void TGetAsyncUdmaKernelImpl(__gm__ T *localBuf, int nranks, int my_rank, int root_rank,
                                               int elem_offset, int elem_count,
                                               __gm__ uint64_t *remoteAddrs,
                                               __gm__ uint8_t *udmaWorkspace)
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
            pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(
                udmaWorkspace, static_cast<uint32_t>(target_rank), session);
            auto event =
                pto::comm::TGET_ASYNC<pto::comm::DmaEngine::URMA>(localRecvG, remoteSendG, session);
            event.Wait(session);
        }
#endif
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Host-side runner — no HCCL.
// ============================================================================
template <typename T, size_t count>
bool RunGetAsyncUdmaRootGetKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, int root_rank)
{
    int deviceId = rank_id % n_devices + first_device_id;

    aclError aErr = aclrtSetDevice(deviceId);
    if (aErr != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtSetDevice(" << deviceId << ") failed: " << aErr << std::endl;
        return false;
    }
    rtStream_t stream = nullptr;
    if (rtStreamCreate(&stream, RT_STREAM_PRIORITY_DEFAULT) != 0) {
        std::cerr << "[ERROR] rtStreamCreate failed" << std::endl;
        return false;
    }

    const size_t recv_elems = static_cast<size_t>(n_ranks) * count;
    size_t commBytesNeeded = 64 * sizeof(int32_t) + (static_cast<size_t>(n_ranks) + 1) * count * sizeof(T);

    // UDMA MR registration requires huge-page memory (CANN ADXL doc: "如通过
    // HCCS传输，则内存分配规则需配置为ACL_MEM_MALLOC_HUGE_ONLY").
    // ACL_MEM_MALLOC_HUGE_FIRST silently falls back to normal 4KB pages when
    // size <= 1MB, causing RaCtxLmemRegister to fail with 528101.
    // Huge-page granularity is 2MB, so round up to 2MB minimum.
    constexpr size_t kHugePageSize = 2UL * 1024 * 1024;
    size_t allocSize = (commBytesNeeded + kHugePageSize - 1) & ~(kHugePageSize - 1);
    if (allocSize < kHugePageSize) {
        allocSize = kHugePageSize;
    }

    void *devBuf = nullptr;
    aErr = aclrtMalloc(&devBuf, allocSize, ACL_MEM_MALLOC_HUGE_ONLY);
    if (aErr != ACL_SUCCESS || devBuf == nullptr) {
        std::cerr << "[ERROR] aclrtMalloc(" << allocSize << ") failed: " << aErr << std::endl;
        return false;
    }
    aclrtMemset(devBuf, allocSize, 0, allocSize);
    COMM_LOG("[INFO] Rank " << rank_id << " devBuf=" << devBuf << " size=" << allocSize);

    // Exchange device buffer addresses via MPI allgather
    std::vector<uint64_t> allDevAddrs(n_ranks);
    uint64_t myAddr = reinterpret_cast<uint64_t>(devBuf);
    MpiAllgatherWrapper(&myAddr, allDevAddrs.data(), sizeof(uint64_t), nullptr);

    // Copy remote address table to device memory
    void *devRemoteAddrs = nullptr;
    aErr = aclrtMalloc(&devRemoteAddrs, n_ranks * sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aErr != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMalloc for remoteAddrs failed" << std::endl;
        aclrtFree(devBuf);
        return false;
    }
    aclrtMemcpy(devRemoteAddrs, n_ranks * sizeof(uint64_t), allDevAddrs.data(), n_ranks * sizeof(uint64_t),
                ACL_MEMCPY_HOST_TO_DEVICE);

    // UDMA workspace init — RaInit is first caller, no HCCL conflict
    UdmaBootstrapHandle bootstrap{MpiAllgatherWrapper, MpiBarrierWrapper, nullptr};
    UdmaWorkspaceManager udmaMgr;
    if (!udmaMgr.Init(static_cast<uint32_t>(deviceId), static_cast<uint32_t>(rank_id),
                      static_cast<uint32_t>(n_ranks), devBuf, allocSize, bootstrap)) {
        std::cerr << "[ERROR] UdmaWorkspaceManager Init failed!" << std::endl;
        aclrtFree(devRemoteAddrs);
        aclrtFree(devBuf);
        return false;
    }

    // Prepare host data
    uint8_t *input_host = nullptr;
    uint8_t *output_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&input_host), count * sizeof(T));
    aclrtMallocHost(reinterpret_cast<void **>(&output_host), recv_elems * sizeof(T));
    if (!input_host || !output_host) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < count; ++i) {
        reinterpret_cast<T *>(input_host)[i] = static_cast<T>(i + rank_id * 10000);
    }
    for (size_t i = 0; i < recv_elems; ++i) {
        reinterpret_cast<T *>(output_host)[i] = static_cast<T>(-1);
    }

    // Copy send data to device; recv area is already zeroed
    uint8_t *commBytes = reinterpret_cast<uint8_t *>(devBuf);
    T *sendBuf = reinterpret_cast<T *>(commBytes + 64 * sizeof(int32_t));
    T *recvBuf = sendBuf + count;

    aclrtMemcpy(sendBuf, count * sizeof(T), input_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(recvBuf, recv_elems * sizeof(T), output_host, recv_elems * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    CommMpiBarrier();

    // Launch kernel
    TGetAsyncUdmaKernelImpl<T, count><<<1, nullptr, stream>>>(
        reinterpret_cast<T *>(devBuf), n_ranks, rank_id, root_rank, 0, static_cast<int>(count),
        reinterpret_cast<uint64_t *>(devRemoteAddrs), reinterpret_cast<uint8_t *>(udmaMgr.GetWorkspaceAddr()));
    int syncRet = aclrtSynchronizeStream(stream);

    CommMpiBarrier();

    // Verify results on root rank
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
                    std::cerr << "Rank " << rank_id << " Device " << deviceId << " SyncRet " << syncRet
                              << " Expected: " << (float)expected << " Actual: " << (float)value << std::endl;
                    is_ok = false;
                    break;
                }
            }
        }
    }

    aclrtFreeHost(input_host);
    aclrtFreeHost(output_host);
    udmaMgr.Finalize();
    aclrtFree(devRemoteAddrs);
    aclrtFree(devBuf);
    rtStreamDestroy(stream);

    return is_ok;
}

// ============================================================================
// MPI-based multi-rank launch.
// ============================================================================
template <typename T, size_t count>
bool RunGetAsyncUdmaRootGet(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    int mpiRank = CommMpiRank();
    int mpiSize = CommMpiSize();
    int rankId = first_rank_id + mpiRank;
    int root_rank = first_rank_id;

    if (mpiSize != n_ranks) {
        if (mpiRank == 0) {
            std::cerr << "[ERROR] MPI world size (" << mpiSize << ") != expected nRanks (" << n_ranks
                      << "). Launch with: mpirun -n " << n_ranks << " ./test_binary" << std::endl;
        }
        return false;
    }

    int deviceCount = GetAvailableDeviceCount();
    if (deviceCount < n_ranks + first_device_id) {
        if (mpiRank == 0) {
            std::cerr << "[SKIP] Need " << (n_ranks + first_device_id) << " NPU(s), have " << deviceCount << std::endl;
        }
        return true;
    }

    constexpr int kAclRepeatInit = 100002;
    aclError aclRet = aclInit(nullptr);
    if (aclRet != ACL_SUCCESS && static_cast<int>(aclRet) != kAclRepeatInit) {
        std::cerr << "[ERROR] aclInit failed: " << static_cast<int>(aclRet) << std::endl;
        return false;
    }

    bool result = RunGetAsyncUdmaRootGetKernel<T, count>(rankId, n_ranks, n_devices, first_device_id, root_rank);

    aclFinalize();
    return result;
}

// Explicit instantiations
template bool RunGetAsyncUdmaRootGet<float, 256>(int, int, int, int);
template bool RunGetAsyncUdmaRootGet<int32_t, 4096>(int, int, int, int);
template bool RunGetAsyncUdmaRootGet<uint8_t, 512>(int, int, int, int);
