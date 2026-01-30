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
#include "pto/comm/sdma/sdma.hpp"
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"

#include <pto/pto-inst.hpp>

#define ENABLE_DEBUG_PRINT 1

// ============================================================================
// Device-side Debug Print Support
// Enable by compiling with DEBUG_MODE=ON (adds --cce-enable-print flag)
// ============================================================================
#ifdef _DEBUG
    #define KERNEL_DEBUG_PRINT(...) AscendC::printf(__VA_ARGS__)
#else
    #define KERNEL_DEBUG_PRINT(...) ((void)0)
#endif

// ============================================================================
// 1D Vector Test Kernel - SDMA version
// TPUT_SDMA: Asynchronous remote write using SDMA engine (direct GM to GM)
// 
// Data flow (Ring communication):
//   Rank i sends srcG directly to Rank (i-1)'s recv buffer
//   Rank i receives data from Rank (i+1) into its recv buffer
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TPutSdmaKernelImpl(__gm__ T *dst, __gm__ T *src, __gm__ T *shmem, int nranks)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    KERNEL_DEBUG_PRINT("[TPutSdmaKernel] ======== Kernel Start ========\n");
    KERNEL_DEBUG_PRINT("[TPutSdmaKernel] dst=%p, src=%p, shmem=%p, nranks=%d\n", dst, src, shmem, nranks);
    KERNEL_DEBUG_PRINT("[TPutSdmaKernel] count=%lu, sizeof(T)=%lu\n", count, sizeof(T));

    ShapeDyn shape(1, 1, 1, 1, count);
    StrideDyn stride(count, count, count, count, 1);

    int my_rank = shmem_my_pe();
    int prev_rank = (my_rank + nranks - 1) % nranks;
    
    KERNEL_DEBUG_PRINT("[TPutSdmaKernel] my_rank=%d, prev_rank=%d\n", my_rank, prev_rank);

    // Shared memory layout: only need recv buffer (no send buffer needed)
    __gm__ T *shmem_data = (__gm__ T *)((__gm__ T *)shmem + 64 * sizeof(int32_t));
    __gm__ T *recv_shmem = shmem_data;
    
    KERNEL_DEBUG_PRINT("[TPutSdmaKernel] shmem_data=%p, recv_shmem=%p\n", shmem_data, recv_shmem);

    Global srcG(src, shape, stride);
    Global dstG(dst, shape, stride);
    Global recvG(recv_shmem, shape, stride);

    KERNEL_DEBUG_PRINT("[TPutSdmaKernel] GlobalTensors created, calling ShmemDeviceBarrierAll...\n");
    
    // Synchronize to ensure all ranks are ready
    ShmemDeviceBarrierAll();
    
    KERNEL_DEBUG_PRINT("[TPutSdmaKernel] First barrier done\n");

    // Get remote PE's recv buffer address
    __gm__ T *remote_recv_shmem = ShmemPtr(recv_shmem, prev_rank);
    Global remoteRecvG(remote_recv_shmem, shape, stride);
    
    KERNEL_DEBUG_PRINT("[TPutSdmaKernel] remote_recv_shmem=%p (target pe=%d)\n", remote_recv_shmem, prev_rank);
    
    // TPUT_SDMA: Direct transfer from local srcG to remote recvG
    // No intermediate local buffer needed
    KERNEL_DEBUG_PRINT("[TPutSdmaKernel] Calling TPUT_SDMA (local->remote)...\n");
    auto put_event = pto::comm::TPUT_SDMA(remoteRecvG, srcG);
    
    KERNEL_DEBUG_PRINT("[TPutSdmaKernel] TPUT_SDMA returned, waiting for completion...\n");
    
    // Wait for SDMA transfer completion
    pto::comm::sdma::SDMA::wait(put_event);
    
    KERNEL_DEBUG_PRINT("[TPutSdmaKernel] SDMA wait done, calling ShmemDeviceQuiet...\n");
    
    // Ensure all remote operations from this PE are complete
    ShmemDeviceQuiet();
    
    KERNEL_DEBUG_PRINT("[TPutSdmaKernel] Quiet done, calling second barrier...\n");
    
    // Then synchronize all PEs
    ShmemDeviceBarrierAll();
    
    KERNEL_DEBUG_PRINT("[TPutSdmaKernel] Second barrier done, copying result to output...\n");

    // Copy result from local recv buffer to output using SDMA
    auto result_event = pto::comm::TPUT_SDMA(dstG, recvG);
    pto::comm::sdma::SDMA::wait(result_event);
    
    KERNEL_DEBUG_PRINT("[TPutSdmaKernel] ======== Kernel Complete ========\n");
}

template <typename T, size_t count>
bool RunPutSdmaRingKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, uint64_t local_mem_size)
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

    void *input_ptr, *output_ptr;
    aclrtMalloc(&input_ptr, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&output_ptr, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);

    uint8_t *input_host, *output_host;
    aclrtMallocHost(reinterpret_cast<void**>(&input_host), count * sizeof(T));
    aclrtMallocHost(reinterpret_cast<void**>(&output_host), count * sizeof(T));

    // Initialize Input/Output Host
    for (size_t i = 0; i < count; ++i) {
        reinterpret_cast<T*>(input_host)[i] = static_cast<T>(i + rank_id * 10000);
        reinterpret_cast<T*>(output_host)[i] = static_cast<T>(-1);
    }
    
    aclrtMemcpy(input_ptr, count * sizeof(T), input_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    // Allocate symmetric heap memory for shared buffer (sync buffer + data buffer)
    void* shmem_ptr = ShmemMalloc(64 * sizeof(int32_t) + 4 * count * sizeof(T));
    if (shmem_ptr == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    // Barrier to ensure all ranks have initialized
    ShmemBarrierAll();

    TPutSdmaKernelImpl<T, count><<<1, nullptr, stream>>>((T*)output_ptr, (T*)input_ptr, (T*)shmem_ptr, n_ranks);
    status = aclrtSynchronizeStream(stream);

    // Barrier after kernel execution
    ShmemBarrierAll();

    aclrtMemcpy(output_host, count * sizeof(T), output_ptr, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    // Verify: Each rank should receive data from next rank
    bool is_ok = true;
    for(int i = 0; i < count; ++i){
        T value = reinterpret_cast<T*>(output_host)[i];
        T expected = static_cast<T>(i + (rank_id + 1) % n_ranks * 10000);
        if(value != expected){
            std::cout << "Rank " << rank_id << " Device " << device_id << " Status " << status << std::endl;
            std::cout << "Expected value: " << (float)expected << std::endl;
            std::cout << "Actual value: " << (float)value << std::endl;
            is_ok = false;
            break;
        }
    }

#if ENABLE_DEBUG_PRINT
    if (is_ok && rank_id == 0) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "[DEBUG] Rank 0: TPUT_SDMA Ring SUCCESSFUL!" << std::endl;
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
    status |= aclrtFree(input_ptr);
    status |= aclrtFree(output_ptr);
    ShmemFree(shmem_ptr);

    ShmemFinalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

template <typename T, size_t count>
bool RunPutSdmaRing(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    std::vector<pid_t> pids;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    for (int r = 0; r < n_ranks; ++r){
        pid_t pid = fork();
        if (pid == 0) { // child
            const bool ok = RunPutSdmaRingKernel<T, count>(first_rank_id + r, n_ranks, n_devices, first_device_id, local_mem_size);
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
template bool RunPutSdmaRing<float, 256>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunPutSdmaRing<int32_t, 4096>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunPutSdmaRing<uint8_t, 512>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// ============================================================================
// 2D Test Kernel - SDMA version
// Tests TPUT_SDMA with 2D shape GlobalTensor (rows x cols)
// 
// Data flow (Ring communication):
//   Rank i sends srcG directly to Rank (i-1)'s recv buffer
//   Rank i receives data from Rank (i+1) into its recv buffer
// ============================================================================
template <typename T, size_t rows, size_t cols>
__global__ AICORE void TPutSdmaKernel2DImpl(__gm__ T *dst, __gm__ T *src, __gm__ T *shmem, int nranks)
{
    constexpr size_t total_count = rows * cols;
    
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    KERNEL_DEBUG_PRINT("[TPutSdmaKernel2D] ======== Kernel Start ========\n");
    KERNEL_DEBUG_PRINT("[TPutSdmaKernel2D] dst=%p, src=%p, shmem=%p, nranks=%d\n", dst, src, shmem, nranks);
    KERNEL_DEBUG_PRINT("[TPutSdmaKernel2D] rows=%lu, cols=%lu, total_count=%lu\n", rows, cols, total_count);

    // 2D GlobalTensor shape: [1, 1, 1, rows, cols]
    ShapeDyn shape(1, 1, 1, rows, cols);
    StrideDyn stride(total_count, total_count, total_count, cols, 1);

    int my_rank = shmem_my_pe();
    int prev_rank = (my_rank + nranks - 1) % nranks;
    
    KERNEL_DEBUG_PRINT("[TPutSdmaKernel2D] my_rank=%d, prev_rank=%d\n", my_rank, prev_rank);

    // Shared memory layout: only need recv buffer (no send buffer needed)
    __gm__ T *shmem_data = (__gm__ T *)((__gm__ T *)shmem + 64 * sizeof(int32_t));
    __gm__ T *recv_shmem = shmem_data;
    
    KERNEL_DEBUG_PRINT("[TPutSdmaKernel2D] shmem_data=%p, recv_shmem=%p\n", shmem_data, recv_shmem);

    Global srcG(src, shape, stride);
    Global dstG(dst, shape, stride);
    Global recvG(recv_shmem, shape, stride);

    KERNEL_DEBUG_PRINT("[TPutSdmaKernel2D] GlobalTensors created, calling ShmemDeviceBarrierAll...\n");
    
    // Synchronize to ensure all ranks are ready
    ShmemDeviceBarrierAll();
    
    KERNEL_DEBUG_PRINT("[TPutSdmaKernel2D] First barrier done\n");

    // Get remote PE's recv buffer address
    __gm__ T *remote_recv_shmem = ShmemPtr(recv_shmem, prev_rank);
    Global remoteRecvG(remote_recv_shmem, shape, stride);
    
    KERNEL_DEBUG_PRINT("[TPutSdmaKernel2D] remote_recv_shmem=%p (target pe=%d)\n", remote_recv_shmem, prev_rank);
    
    // TPUT_SDMA: Direct transfer from local srcG to remote recvG
    // No intermediate local buffer needed
    KERNEL_DEBUG_PRINT("[TPutSdmaKernel2D] Calling TPUT_SDMA (local->remote)...\n");
    auto put_event = pto::comm::TPUT_SDMA(remoteRecvG, srcG);
    
    KERNEL_DEBUG_PRINT("[TPutSdmaKernel2D] TPUT_SDMA returned, waiting for completion...\n");
    
    // Wait for SDMA transfer completion
    pto::comm::sdma::SDMA::wait(put_event);
    
    KERNEL_DEBUG_PRINT("[TPutSdmaKernel2D] SDMA wait done, calling ShmemDeviceQuiet...\n");
    
    // Ensure all remote operations from this PE are complete
    ShmemDeviceQuiet();
    
    KERNEL_DEBUG_PRINT("[TPutSdmaKernel2D] Quiet done, calling second barrier...\n");
    
    // Then synchronize all PEs
    ShmemDeviceBarrierAll();
    
    KERNEL_DEBUG_PRINT("[TPutSdmaKernel2D] Second barrier done, copying result to output...\n");

    // Copy result from local recv buffer to output using SDMA
    auto result_event = pto::comm::TPUT_SDMA(dstG, recvG);
    pto::comm::sdma::SDMA::wait(result_event);
    
    KERNEL_DEBUG_PRINT("[TPutSdmaKernel2D] ======== Kernel Complete ========\n");
}

template <typename T, size_t rows, size_t cols>
bool RunPutSdmaRing2DKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, uint64_t local_mem_size)
{
    constexpr size_t total_count = rows * cols;
    
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

    void *input_ptr, *output_ptr;
    aclrtMalloc(&input_ptr, total_count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&output_ptr, total_count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);

    uint8_t *input_host, *output_host;
    aclrtMallocHost(reinterpret_cast<void**>(&input_host), total_count * sizeof(T));
    aclrtMallocHost(reinterpret_cast<void**>(&output_host), total_count * sizeof(T));

    // Initialize Input/Output Host - 2D data in row-major order
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            size_t idx = r * cols + c;
            reinterpret_cast<T*>(input_host)[idx] = static_cast<T>(idx + rank_id * 10000);
            reinterpret_cast<T*>(output_host)[idx] = static_cast<T>(-1);
        }
    }
    
    aclrtMemcpy(input_ptr, total_count * sizeof(T), input_host, total_count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    // Allocate symmetric heap memory for shared buffer (sync buffer + data buffer)
    void* shmem_ptr = ShmemMalloc(64 * sizeof(int32_t) + 4 * total_count * sizeof(T));
    if (shmem_ptr == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    // Barrier to ensure all ranks have initialized
    ShmemBarrierAll();

    TPutSdmaKernel2DImpl<T, rows, cols><<<1, nullptr, stream>>>((T*)output_ptr, (T*)input_ptr, (T*)shmem_ptr, n_ranks);
    status = aclrtSynchronizeStream(stream);

    // Barrier after kernel execution
    ShmemBarrierAll();

    aclrtMemcpy(output_host, total_count * sizeof(T), output_ptr, total_count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    // Verify: Each rank should receive data from next rank
    bool is_ok = true;
    for (size_t r = 0; r < rows && is_ok; ++r) {
        for (size_t c = 0; c < cols && is_ok; ++c) {
            size_t idx = r * cols + c;
            T value = reinterpret_cast<T*>(output_host)[idx];
            T expected = static_cast<T>(idx + (rank_id + 1) % n_ranks * 10000);
            if (value != expected) {
                std::cout << "Rank " << rank_id << " Device " << device_id << " Status " << status << std::endl;
                std::cout << "At [" << r << ", " << c << "] (idx=" << idx << "):" << std::endl;
                std::cout << "Expected value: " << (float)expected << std::endl;
                std::cout << "Actual value: " << (float)value << std::endl;
                is_ok = false;
            }
        }
    }

#if ENABLE_DEBUG_PRINT
    if (is_ok && rank_id == 0) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "[DEBUG] Rank 0: TPUT_SDMA 2D Ring SUCCESSFUL! (" << rows << "x" << cols << ")" << std::endl;
        std::cout << "Sample Result (First row): [ ";
        for (size_t c = 0; c < (cols > 5 ? 5 : cols); ++c) {
            std::cout << (float)reinterpret_cast<T*>(output_host)[c] << " ";
        }
        if (cols > 5) std::cout << "... ";
        std::cout << "]" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }
#endif

    status |= aclrtFreeHost(input_host);
    status |= aclrtFreeHost(output_host);
    status |= aclrtFree(input_ptr);
    status |= aclrtFree(output_ptr);
    ShmemFree(shmem_ptr);

    ShmemFinalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

template <typename T, size_t rows, size_t cols>
bool RunPutSdmaRing2D(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    std::vector<pid_t> pids;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) { // child
            const bool ok = RunPutSdmaRing2DKernel<T, rows, cols>(first_rank_id + r, n_ranks, n_devices, first_device_id, local_mem_size);
            _exit(ok ? 0 : 1);
        } else if (pid > 0) {
            pids.push_back(pid);
        } else {
            return false; // fork failed
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

// Explicit instantiations for 2D shape tests
template bool RunPutSdmaRing2D<float, 16, 16>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunPutSdmaRing2D<float, 8, 32>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunPutSdmaRing2D<int32_t, 4, 64>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
