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

template <typename T, size_t count>
__global__ AICORE void TAllGatherKernelImpl(__gm__ T *dst, __gm__ T *src, int nranks)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    // UB Tile definition: must be pre-allocated for native implementation
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, count, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = shmem_my_pe();

    ShapeDyn srcShape(1, 1, 1, 1, count);
    StrideDyn srcStride(count, count, count, count, 1);

    ShapeDyn dstShape(1, 1, 1, 1, count * nranks);
    StrideDyn dstStride(count * nranks, count * nranks, count * nranks, count * nranks, 1);
    Global dstG(dst, dstShape, dstStride);
    dstG.SetRank(my_rank);

    // Create ParallelGroup: each tensor in the group represents the input buffer on a different rank.
    Global baseSrcG(src, srcShape, srcStride);
    Global tensors[16];
    Global *tensorPtrs[16];
    
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        tensors[i] = baseSrcG;
        tensors[i].SetRank(i);  // Each tensor corresponds to rank i's memory region
        tensorPtrs[i] = &tensors[i];
    }
    
    pto::comm::ParallelGroup<Global> pg(tensorPtrs, actual_nranks, my_rank);
    
    // Allocate UB tile for staging data
    TileData ubTile(1, count);
    TASSIGN(ubTile, 0x0);
    
    // Call TALLGATHER with UB tile
    pto::comm::TALLGATHER(pg, dstG, ubTile);
    pto::comm::TQUIET();
}

template <typename T, size_t count>
bool RunAllGatherKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, uint64_t local_mem_size){
    
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
    const char *ip = "tcp://127.0.0.1:8767";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;
    env.heapBytes = local_mem_size;
    
    if (!ShmemInitFromEnv(env)) {
        std::cerr << "[ERROR] ShmemInitFromEnv failed!" << std::endl;
        return false;
    }

    // Use ShmemMalloc for both input and output to ensure they are accessible by all ranks
    size_t input_size = count * sizeof(T);
    size_t output_size = n_ranks * count * sizeof(T);
    void* input_ptr = ShmemMalloc(input_size);
    void* output_ptr = ShmemMalloc(output_size);

    if (input_ptr == nullptr || output_ptr == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    T *input_host, *output_host;
    aclrtMallocHost(reinterpret_cast<void**>(&input_host), input_size);
    aclrtMallocHost(reinterpret_cast<void**>(&output_host), output_size);

    // Initialize input data: each rank has its unique range
    for (size_t i = 0; i < count; ++i) {
        input_host[i] = static_cast<T>(i + rank_id * 10000);
    }
    for (size_t i = 0; i < n_ranks * count; ++i) {
        output_host[i] = static_cast<T>(-1);
    }

    aclrtMemcpy(input_ptr, input_size, input_host, input_size, ACL_MEMCPY_HOST_TO_DEVICE);

    // Barrier to ensure all ranks have initialized their data
    ShmemBarrierAll();

    TAllGatherKernelImpl<T, count><<<1, nullptr, stream>>>((T*)output_ptr, (T*)input_ptr, n_ranks);
    status = aclrtSynchronizeStream(stream);

    // Barrier after kernel execution
    ShmemBarrierAll();

    aclrtMemcpy(output_host, output_size, output_ptr, output_size, ACL_MEMCPY_DEVICE_TO_HOST);

    bool is_ok = true;
    for (int r = 0; r < n_ranks; ++r) {
        for (size_t i = 0; i < count; ++i) {
            T expected = static_cast<T>(i + r * 10000);
            T actual = output_host[r * count + i];
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
    if (is_ok && rank_id == 0) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "[DEBUG] Rank 0: TALLGATHER SUCCESSFUL!" << std::endl;
        std::cout << "Summary: Gathered " << n_ranks << " segments, each with " << count << " elements." << std::endl;
        std::cout << "Detailed View (First 5 elements of each rank's contribution):" << std::endl;
        for (int r = 0; r < n_ranks; ++r) {
            std::cout << "  - Segment from Rank " << r << ": [ ";
            for (size_t i = 0; i < (count > 5 ? 5 : count); ++i) {
                std::cout << (float)output_host[r * count + i] << " ";
            }
            if (count > 5) std::cout << "... ";
            std::cout << "]" << std::endl;
        }
        std::cout << "================================================================\n" << std::endl;
    }
#endif

    aclrtFreeHost(input_host);
    aclrtFreeHost(output_host);
    ShmemFree(input_ptr);
    ShmemFree(output_ptr);

    ShmemFinalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

template <typename T, size_t count>
bool RunAllGather(int n_ranks, int n_devices, int first_rank_id, int first_device_id){
    std::vector<pid_t> pids;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    for (int r = 0; r < n_ranks; ++r){
        pid_t pid = fork();
        if (pid == 0) { // child
            const bool ok = RunAllGatherKernel<T, count>(first_rank_id + r, n_ranks, n_devices, first_device_id, local_mem_size);
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
template bool RunAllGather<float, 256>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunAllGather<int32_t, 4096>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunAllGather<uint8_t, 512>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
