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
#include <algorithm>

#include "pto/comm/pto_comm_inst.hpp"
#include "pto/comm/sdma_types.hpp"
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"

#include <pto/pto-inst.hpp>

#define ENABLE_DEBUG_PRINT 1

// ============================================================================
// 1D Vector Tile Test Kernel (TPUT_ASYNC)
// Root rank puts to all other ranks (non-ring).
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TPutAsyncKernelImpl(__gm__ T *shmem, int nranks, int root_rank)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, count);
    StrideDyn stride(count, count, count, count, 1);

    int my_rank = shmem_my_pe();

    __gm__ T *shmem_data = reinterpret_cast<__gm__ T *>(shmem);
    __gm__ T *send_shmem = shmem_data;
    __gm__ T *recv_shmem = shmem_data + count;

    Global sendG(send_shmem, shape, stride);

    if (my_rank == root_rank) {
        constexpr int kEventSlots = pto::comm::sdma::SDMA_EVENT_SLOT_COUNT;
        pto::comm::AsyncEvent events[kEventSlots];
        int issued = 0;
        for (int target_rank = 0; target_rank < nranks; ++target_rank) {
            if (target_rank == root_rank) {
                continue;
            }
            __gm__ T *remote_recv_shmem = ShmemPtr(recv_shmem, target_rank);
            Global remoteRecvG(remote_recv_shmem, shape, stride);
            if (issued >= kEventSlots) {
                events[issued % kEventSlots].Wait();
            }
            events[issued % kEventSlots] = pto::comm::TPUT_ASYNC(remoteRecvG, sendG);
            issued++;
        }
        const int pending = (issued < kEventSlots) ? issued : kEventSlots;
        for (int i = 0; i < pending; ++i) {
            events[i].Wait();
        }
        ShmemDeviceQuiet();
    }

    ShmemDeviceBarrierAll();
}

template <typename T, size_t count>
bool RunPutAsyncRootPutKernel(int rank_id, int n_ranks, int n_devices, int first_device_id,
                              uint64_t local_mem_size, int root_rank)
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
    
    if (ShmemInitForSdma(env) != 0) {
        std::cerr << "[ERROR] ShmemInitForSdma failed!" << std::endl;
        return false;
    }

    uint8_t *input_host, *output_host;
    aclrtMallocHost(reinterpret_cast<void**>(&input_host), count * sizeof(T));
    aclrtMallocHost(reinterpret_cast<void**>(&output_host), count * sizeof(T));

    // Initialize Input/Output Host
    for (size_t i = 0; i < count; ++i) {
        reinterpret_cast<T*>(input_host)[i] = static_cast<T>(i + rank_id * 10000);
        reinterpret_cast<T*>(output_host)[i] = static_cast<T>(-1);
    }

    // Allocate symmetric heap memory for shared buffer (send + recv)
    void* shmem_ptr = ShmemMalloc(2 * count * sizeof(T));
    if (shmem_ptr == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    T *shmem_data = reinterpret_cast<T *>(shmem_ptr);
    T *send_shmem = shmem_data;
    T *recv_shmem = shmem_data + count;

    // Copy host input to shmem send buffer, init recv buffer to -1
    aclrtMemcpy(send_shmem, count * sizeof(T), input_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(recv_shmem, count * sizeof(T), output_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    // Barrier to ensure all ranks have initialized
    ShmemBarrierAll();

    TPutAsyncKernelImpl<T, count><<<1, nullptr, stream>>>((T*)shmem_ptr, n_ranks, root_rank);
    status = aclrtSynchronizeStream(stream);

    // Barrier after kernel execution
    ShmemBarrierAll();

    aclrtMemcpy(output_host, count * sizeof(T), recv_shmem, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    // Verify: Each rank should receive data from root rank
    bool is_ok = true;
    if (rank_id != root_rank) {
        for (int i = 0; i < count; ++i) {
            T value = reinterpret_cast<T*>(output_host)[i];
            T expected = static_cast<T>(i + root_rank * 10000);
            if (value != expected) {
                std::cout << "Rank " << rank_id << " Device " << device_id << " Status " << status << std::endl;
                std::cout << "Expected value: " << (float)expected << std::endl;
                std::cout << "Actual value: " << (float)value << std::endl;
                is_ok = false;
                break;
            }
        }
    }

#if ENABLE_DEBUG_PRINT
    if (is_ok && rank_id != root_rank) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "[DEBUG] Rank " << rank_id << ": TPUT_ASYNC Root-Put SUCCESSFUL!" << std::endl;
        std::cout << "Sample Result (First 5 elements): [ ";
        for (size_t i = 0; i < (count > 5 ? 5 : count); ++i) {
            std::cout << (float)reinterpret_cast<T*>(output_host)[i] << " ";
        }
        if (count > 5) std::cout << "... ";
        std::cout << "]" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }
#endif

    status |= aclrtFreeHost(input_host);
    status |= aclrtFreeHost(output_host);
    ShmemFree(shmem_ptr);

    ShmemFinalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

template <typename T, size_t count>
bool RunPutAsyncRootPut(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    std::vector<pid_t> pids;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    const int root_rank = first_rank_id;
    for (int r = 0; r < n_ranks; ++r){
        pid_t pid = fork();
        if (pid == 0) { // child
            const bool ok = RunPutAsyncRootPutKernel<T, count>(first_rank_id + r, n_ranks, n_devices,
                                                              first_device_id, local_mem_size, root_rank);
            _exit(ok ? 0 : 1);
        } else if (pid > 0) {
            pids.push_back(pid);
        } else {
            return 1; // fork failed
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

// Explicit instantiations for 1D tests
template bool RunPutAsyncRootPut<float, 256>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunPutAsyncRootPut<int32_t, 4096>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunPutAsyncRootPut<uint8_t, 512>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
