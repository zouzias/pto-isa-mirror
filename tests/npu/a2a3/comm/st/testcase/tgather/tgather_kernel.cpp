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

#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <iostream>

#include "pto/comm/pto_comm_inst.hpp"
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"

#include <pto/pto-inst.hpp>

#define ENABLE_DEBUG_PRINT 1

// ============================================================================
// TGATHER Test Kernel
// Tests the TGATHER collective - root gathers data from all ranks
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TGatherKernelImpl(__gm__ T *dst, __gm__ T *src, int nranks)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    // UB Tile definition
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, count, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = shmem_my_pe();

    // Source shape: each rank has [1, 1, 1, 1, count] elements
    ShapeDyn srcShape(1, 1, 1, 1, count);
    StrideDyn srcStride(count, count, count, count, 1);

    // Destination shape: root collects [1, 1, 1, nranks, count] elements
    ShapeDyn dstShape(1, 1, 1, nranks, count);
    StrideDyn dstStride(nranks * count, nranks * count, nranks * count, count, 1);
    Global dstG(dst, dstShape, dstStride);

    // Create ParallelGroup: each tensor in the group is the source buffer on that rank
    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ T *remoteSrc = ShmemPtr(src, i);
        tensors[i] = Global(remoteSrc, srcShape, srcStride);
    }
    
    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, my_rank);
    
    // Allocate UB tile for staging data
    TileData ubTile(1, count);
    TASSIGN(ubTile, 0x0);
    
    // Only root (rank 0) executes TGATHER
    if (my_rank == 0) {
        pto::comm::TGATHER(pg, dstG, ubTile);
    }
    
    ShmemDeviceQuiet();
    ShmemDeviceBarrierAll();
}

template <typename T, size_t count>
bool RunGatherKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, uint64_t local_mem_size)
{
    // Initialize shmem TLS configuration
    int32_t ret = ShmemSetConfStoreTls(false, nullptr, 0);
    if (ret != 0) {
        std::cerr << "[ERROR] Failed to init shmem tls\n";
        return false;
    }

    if (n_devices <= 0 || n_ranks <= 0) {
        std::cerr << "[ERROR] n_devices and n_ranks must be > 0\n";
        return false;
    }
    const int32_t device_id = rank_id % n_devices + first_device_id;
    int status = 0;
    aclrtStream stream = nullptr;

    status |= aclInit(nullptr);
    status |= aclrtSetDevice(device_id);
    status |= aclrtCreateStream(&stream);

    // Initialize shmem symmetric heap
    ShmemEnv env;
    const char *ip = "tcp://127.0.0.1:8770";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;
    env.heapBytes = local_mem_size;
    
    if (!ShmemInitFromEnv(env)) {
        std::cerr << "[ERROR] ShmemInitFromEnv failed!" << std::endl;
        return false;
    }

    // Allocate symmetric heap memory for source (each rank's contribution)
    size_t src_size = count * sizeof(T);
    size_t dst_size = n_ranks * count * sizeof(T);
    void* src_ptr = ShmemMalloc(src_size);
    void* dst_ptr = nullptr;
    
    // Only root allocates destination buffer
    if (rank_id == 0) {
        dst_ptr = ShmemMalloc(dst_size);
    } else {
        dst_ptr = ShmemMalloc(src_size);  // Dummy allocation for symmetry
    }

    if (src_ptr == nullptr || dst_ptr == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    T *src_host, *dst_host;
    aclrtMallocHost(reinterpret_cast<void**>(&src_host), src_size);
    aclrtMallocHost(reinterpret_cast<void**>(&dst_host), dst_size);

    // Initialize source data: each rank has its unique range
    for (size_t i = 0; i < count; ++i) {
        src_host[i] = static_cast<T>(i + rank_id * 10000);
    }
    for (size_t i = 0; i < n_ranks * count; ++i) {
        dst_host[i] = static_cast<T>(-1);
    }

    aclrtMemcpy(src_ptr, src_size, src_host, src_size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (rank_id == 0) {
        aclrtMemcpy(dst_ptr, dst_size, dst_host, dst_size, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    // Barrier to ensure all ranks have initialized their data
    ShmemBarrierAll();

    TGatherKernelImpl<T, count><<<1, nullptr, stream>>>((T*)dst_ptr, (T*)src_ptr, n_ranks);
    status = aclrtSynchronizeStream(stream);

    // Barrier after kernel execution
    ShmemBarrierAll();

    // Only root verifies result
    bool is_ok = true;
    if (rank_id == 0) {
        aclrtMemcpy(dst_host, dst_size, dst_ptr, dst_size, ACL_MEMCPY_DEVICE_TO_HOST);

        for (int r = 0; r < n_ranks; ++r) {
            for (size_t i = 0; i < count; ++i) {
                T expected = static_cast<T>(i + r * 10000);
                T actual = dst_host[r * count + i];
                if (actual != expected) {
                    std::cout << "Rank " << rank_id << " validation failed at rank " << r << " index " << i 
                              << ": expected " << (float)expected << ", got " << (float)actual << std::endl;
                    is_ok = false;
                    break;
                }
            }
            if (!is_ok) break;
        }

#if ENABLE_DEBUG_PRINT
        if (is_ok) {
            std::cout << "\n================================================================" << std::endl;
            std::cout << "[DEBUG] Rank 0: TGATHER SUCCESSFUL!" << std::endl;
            std::cout << "Summary: Gathered " << n_ranks << " segments, each with " << count << " elements." << std::endl;
            std::cout << "Detailed View (First 5 elements of each rank's contribution):" << std::endl;
            for (int r = 0; r < n_ranks; ++r) {
                std::cout << "  - Segment from Rank " << r << ": [ ";
                for (size_t i = 0; i < (count > 5 ? 5 : count); ++i) {
                    std::cout << (float)dst_host[r * count + i] << " ";
                }
                if (count > 5) std::cout << "... ";
                std::cout << "]" << std::endl;
            }
            std::cout << "================================================================\n" << std::endl;
        }
#endif
    }

    aclrtFreeHost(src_host);
    aclrtFreeHost(dst_host);
    ShmemFree(src_ptr);
    ShmemFree(dst_ptr);

    ShmemFinalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

template <typename T, size_t count>
bool RunGather(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    std::vector<pid_t> pids;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) { // child
            const bool ok = RunGatherKernel<T, count>(first_rank_id + r, n_ranks, n_devices, first_device_id, local_mem_size);
            _exit(ok ? 0 : 1);
        } else if (pid > 0) {
            pids.push_back(pid);
        } else {
            return false;
        }
    }
    bool success = true;
    for (pid_t p : pids) {
        int status = 0;
        waitpid(p, &status, 0);
        if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0)) success = false;
    }
    return success;    
}

// Explicit instantiations
template bool RunGather<float, 256>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunGather<int32_t, 4096>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunGather<uint8_t, 512>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
