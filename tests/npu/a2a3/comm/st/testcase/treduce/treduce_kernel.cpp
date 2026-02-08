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


// ============================================================================
// TREDUCE Test Kernel
// Tests the TREDUCE collective - root gathers and reduces data from all ranks
// ============================================================================
template <typename T, size_t count, pto::comm::ReduceOp op>
__global__ AICORE void TReduceKernelImpl(__gm__ T *input, __gm__ T *output, int nranks)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    // UB Tile definition
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, count, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = shmem_my_pe();

    ShapeDyn shape(1, 1, 1, 1, count);
    StrideDyn stride(count, count, count, count, 1);
    
    Global outputG(output, shape, stride);

    // Create ParallelGroup: each tensor in the group is the input buffer on that rank
    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ T *remoteInput = ShmemPtr(input, i);
        tensors[i] = Global(remoteInput, shape, stride);
    }
    
    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, my_rank);
    
    // Allocate UB tiles for accumulation and receiving
    TileData accTile(1, count);
    TileData recvTile(1, count);
    
    TASSIGN(accTile, 0x0);
    TASSIGN(recvTile, 0x10000);
    
    // Only root (rank 0) executes TREDUCE
    if (my_rank == 0) {
        pto::comm::TREDUCE(pg, outputG, accTile, recvTile, op);
    }
    
    ShmemDeviceQuiet();
    ShmemDeviceBarrierAll();
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
bool RunReduceKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, uint64_t local_mem_size,
                     const ShmemUniqueId *uid)
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
    const char *ip = "tcp://127.0.0.1:8772";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;
    env.heapBytes = local_mem_size;
    
    if (!ShmemInitFromEnvWithUniqueId(env, uid)) {
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

    // Barrier to ensure all ranks have initialized their data
    ShmemBarrierAll();

    TReduceKernelImpl<T, count, op><<<1, nullptr, stream>>>((T*)input_ptr, (T*)output_device, n_ranks);
    status = aclrtSynchronizeStream(stream);

    // Barrier after kernel execution
    ShmemBarrierAll();

    // Only root verifies result
    bool is_ok = true;
    if (rank_id == 0) {
        aclrtMemcpy(output_host, count * sizeof(T), output_device, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

        // Verify expected result based on ReduceOp
        for (size_t i = 0; i < count; ++i) {
            const T expected = ReduceExpected(static_cast<T>(i), n_ranks, op);
            T actual = output_host[i];
            if (actual != expected) {
                std::cout << "Rank " << rank_id << " validation failed at index " << i 
                          << ": expected " << (float)expected << ", got " << (float)actual << std::endl;
                is_ok = false;
                break;
            }
        }

#if ENABLE_DEBUG_PRINT
        if (is_ok) {
            std::cout << "\n================================================================" << std::endl;
            std::cout << "[DEBUG] Rank 0: TREDUCE SUCCESSFUL!" << std::endl;
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
    }

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

template <typename T, size_t count, pto::comm::ReduceOp op>
bool RunReduce(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    std::vector<pid_t> pids;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
#if defined(CANN_SHMEM)
    ShmemUniqueId uid;
    ShmemUniqueId *uid_ptr = nullptr;
    if (aclshmemx_get_uniqueid(&uid) == 0) {
        uid_ptr = &uid;
    }
#else
    ShmemUniqueId *uid_ptr = nullptr;
#endif
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) { // child
            const bool ok = RunReduceKernel<T, count, op>(first_rank_id + r, n_ranks, n_devices, first_device_id,
                                                         local_mem_size, uid_ptr);
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
template bool RunReduce<float, 256, pto::comm::ReduceOp::Sum>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunReduce<int32_t, 4096, pto::comm::ReduceOp::Sum>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunReduce<int32_t, 512, pto::comm::ReduceOp::Sum>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunReduce<int32_t, 256, pto::comm::ReduceOp::Max>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunReduce<int32_t, 256, pto::comm::ReduceOp::Min>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Non-template wrappers for test main.cpp
bool RunReduceFloat256Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id) {
    return RunReduce<float, 256, pto::comm::ReduceOp::Sum>(n_ranks, n_devices, first_rank_id, first_device_id);
}

bool RunReduceInt32_4096_Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id) {
    return RunReduce<int32_t, 4096, pto::comm::ReduceOp::Sum>(n_ranks, n_devices, first_rank_id, first_device_id);
}

bool RunReduceInt32_512_Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id) {
    return RunReduce<int32_t, 512, pto::comm::ReduceOp::Sum>(n_ranks, n_devices, first_rank_id, first_device_id);
}

bool RunReduceInt32_256_Max(int n_ranks, int n_devices, int first_rank_id, int first_device_id) {
    return RunReduce<int32_t, 256, pto::comm::ReduceOp::Max>(n_ranks, n_devices, first_rank_id, first_device_id);
}

bool RunReduceInt32_256_Min(int n_ranks, int n_devices, int first_rank_id, int first_device_id) {
    return RunReduce<int32_t, 256, pto::comm::ReduceOp::Min>(n_ranks, n_devices, first_rank_id, first_device_id);
}
