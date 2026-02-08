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
// TSCATTER Test Kernel
// Tests the TSCATTER collective - root scatters data to all ranks
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TScatterKernelImpl(__gm__ T *src, __gm__ T *dst, int nranks)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    // UB Tile definition
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, count, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = shmem_my_pe();

    // Source shape: root has [1, 1, 1, nranks, count] elements
    ShapeDyn srcShape(1, 1, 1, nranks, count);
    StrideDyn srcStride(nranks * count, nranks * count, nranks * count, count, 1);
    Global srcG(src, srcShape, srcStride);

    // Destination shape: each rank receives [1, 1, 1, 1, count] elements
    ShapeDyn dstShape(1, 1, 1, 1, count);
    StrideDyn dstStride(count, count, count, count, 1);

    // Create ParallelGroup: each tensor in the group is the destination buffer on that rank
    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ T *remoteDst = ShmemPtr(dst, i);
        tensors[i] = Global(remoteDst, dstShape, dstStride);
    }
    
    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, my_rank);
    
    // Allocate UB tile for staging data
    TileData ubTile(1, count);
    TASSIGN(ubTile, 0x0);
    
    // Only root (rank 0) executes TSCATTER
    if (my_rank == 0) {
        pto::comm::TSCATTER(pg, srcG, ubTile);
    }
    
    ShmemDeviceQuiet();
    ShmemDeviceBarrierAll();
}

template <typename T, size_t count>
bool RunScatterKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, uint64_t local_mem_size)
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
    const char *ip = "tcp://127.0.0.1:8771";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;
    env.heapBytes = local_mem_size;
    
    if (!ShmemInitFromEnv(env)) {
        std::cerr << "[ERROR] ShmemInitFromEnv failed!" << std::endl;
        return false;
    }

    // Allocate symmetric heap memory
    size_t dst_size = count * sizeof(T);
    size_t src_size = n_ranks * count * sizeof(T);
    void* dst_ptr = ShmemMalloc(dst_size);
    void* src_ptr = nullptr;
    
    // Only root allocates source buffer
    if (rank_id == 0) {
        src_ptr = ShmemMalloc(src_size);
    } else {
        src_ptr = ShmemMalloc(dst_size);  // Dummy allocation for symmetry
    }

    if (src_ptr == nullptr || dst_ptr == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    T *src_host, *dst_host;
    aclrtMallocHost(reinterpret_cast<void**>(&src_host), src_size);
    aclrtMallocHost(reinterpret_cast<void**>(&dst_host), dst_size);

    // Root initializes source data: data for rank r at offset r*count
    if (rank_id == 0) {
        for (int r = 0; r < n_ranks; ++r) {
            for (size_t i = 0; i < count; ++i) {
                src_host[r * count + i] = static_cast<T>(i + r * 10000);
            }
        }
        aclrtMemcpy(src_ptr, src_size, src_host, src_size, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    
    // Initialize destination to -1
    for (size_t i = 0; i < count; ++i) {
        dst_host[i] = static_cast<T>(-1);
    }
    aclrtMemcpy(dst_ptr, dst_size, dst_host, dst_size, ACL_MEMCPY_HOST_TO_DEVICE);

    // Barrier to ensure all ranks have initialized their data
    ShmemBarrierAll();

    TScatterKernelImpl<T, count><<<1, nullptr, stream>>>((T*)src_ptr, (T*)dst_ptr, n_ranks);
    status = aclrtSynchronizeStream(stream);

    // Barrier after kernel execution
    ShmemBarrierAll();

    // All ranks verify their received data
    aclrtMemcpy(dst_host, dst_size, dst_ptr, dst_size, ACL_MEMCPY_DEVICE_TO_HOST);

    bool is_ok = true;
    for (size_t i = 0; i < count; ++i) {
        T expected = static_cast<T>(i + rank_id * 10000);
        T actual = dst_host[i];
        if (actual != expected) {
            std::cout << "Rank " << rank_id << " validation failed at index " << i 
                      << ": expected " << (float)expected << ", got " << (float)actual << std::endl;
            is_ok = false;
            break;
        }
    }

#if ENABLE_DEBUG_PRINT
    if (is_ok && rank_id == 0) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "[DEBUG] Rank 0: TSCATTER SUCCESSFUL!" << std::endl;
        std::cout << "Summary: Scattered " << n_ranks << " segments, each with " << count << " elements." << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }
#endif

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
bool RunScatter(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    std::vector<pid_t> pids;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) { // child
            const bool ok = RunScatterKernel<T, count>(first_rank_id + r, n_ranks, n_devices, first_device_id, local_mem_size);
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
template bool RunScatter<float, 256>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunScatter<int32_t, 4096>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunScatter<uint8_t, 512>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
