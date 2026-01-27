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
__global__ AICORE void TAllReduceKernelImpl(__gm__ T *input, __gm__ T *output, int nranks)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    // UB Tile definition: must be pre-allocated for native implementation
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, count, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = shmem_my_pe();

    ShapeDyn shape(1, 1, 1, 1, count);
    StrideDyn stride(count, count, count, count, 1);
    
    Global tempG(input, shape, stride);
    Global outputG(output, shape, stride);
    Global *tensorPtrs[16];
    Global tensors[16];
    
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        tensors[i] = tempG;
        tensors[i].SetRank(i);
        tensorPtrs[i] = &tensors[i];
    }
    
    pto::comm::ParallelGroup<Global> pg(tensorPtrs, actual_nranks, my_rank);
    
    // Allocate UB tiles for ping-pong double buffering
    // These must be pre-allocated and passed to the instruction
    TileData accTile(1, count);
    TileData pingTile(1, count);
    TileData pongTile(1, count);
    
    // Assign UB addresses (compiler would do this in real usage)
    TASSIGN(accTile, 0x0);
    TASSIGN(pingTile, 0x10000);
    TASSIGN(pongTile, 0x20000);
    
    // Call TALLREDUCE with UB tiles
    pto::comm::TALLREDUCE(pg, outputG, accTile, pingTile, pongTile);
    pto::comm::TQUIET();
}

template <typename T, size_t count>
bool RunAllReduceKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, uint64_t local_mem_size){
    
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
    const char *ip = "tcp://127.0.0.1:8766";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;
    env.heapBytes = local_mem_size;
    
    if (!ShmemInitFromEnv(env)) {
        std::cerr << "[ERROR] ShmemInitFromEnv failed!" << std::endl;
        return false;
    }

    // Allocate symmetric heap memory for input (shared across ranks)
    void *input_ptr = ShmemMalloc(count * sizeof(T));
    if (input_ptr == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    T *input_host;
    aclrtMallocHost(reinterpret_cast<void**>(&input_host), count * sizeof(T));

    T* output_host;
    aclrtMallocHost(reinterpret_cast<void**>(&output_host), count * sizeof(T));

    T* output_device;
    aclrtMalloc(reinterpret_cast<void**>(&output_device), count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
    
    // Initialize input data: Rank R has data i + R * 100
    for (size_t i = 0; i < count; ++i) {
        input_host[i] = static_cast<T>(i + rank_id * 100);
    }

    aclrtMemcpy(input_ptr, count * sizeof(T), input_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

#if ENABLE_DEBUG_PRINT
    std::cout << "[DEBUG] Rank " << rank_id << " input_ptr: " << input_ptr << std::endl;
#endif

    // Barrier to ensure all ranks have initialized their data
    ShmemBarrierAll();

    TAllReduceKernelImpl<T, count><<<1, nullptr, stream>>>((T*)input_ptr, (T*)output_device, n_ranks);
    status = aclrtSynchronizeStream(stream);

    // Barrier after kernel execution
    ShmemBarrierAll();

    aclrtMemcpy(output_host, count * sizeof(T), output_device, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    // Verify: Expected result is Sum_{R=0}^{n_ranks-1} (i + R * 100) 
    bool is_ok = true;
    for (size_t i = 0; i < count; ++i) {
        T expected = 0;
        for (int r = 0; r < n_ranks; ++r) {
            expected += static_cast<T>(i + r * 100);
        }
        T actual = output_host[i];
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
        std::cout << "[DEBUG] Rank 0: TALLREDUCE SUCCESSFUL!" << std::endl;
        std::cout << "Summary: Reduced " << n_ranks << " segments, result size " << count << " elements." << std::endl;
        std::cout << "Sample Result (First 5 elements): [ ";
        for (size_t i = 0; i < (count > 5 ? 5 : count); ++i) {
            std::cout << (float)output_host[i] << " ";
        }
        if (count > 5) std::cout << "... ";
        std::cout << "]" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }
#endif

    aclrtFreeHost(input_host);
    aclrtFreeHost(output_host);
    aclrtFree(output_device);
    ShmemFree(input_ptr);

    ShmemFinalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

template <typename T, size_t count>
bool RunAllReduce(int n_ranks, int n_devices, int first_rank_id, int first_device_id){
    std::vector<pid_t> pids;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    for (int r = 0; r < n_ranks; ++r){
        pid_t pid = fork();
        if (pid == 0) { // child
            const bool ok = RunAllReduceKernel<T, count>(first_rank_id + r, n_ranks, n_devices, first_device_id, local_mem_size);
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
template bool RunAllReduce<float, 256>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunAllReduce<int32_t, 4096>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunAllReduce<int32_t, 512>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
