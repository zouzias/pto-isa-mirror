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
// 1D Vector Tile Test Kernel
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TPutKernelImpl(__gm__ T *dst, __gm__ T *src, __gm__ T *shmem, int nranks)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    // UB Tile definition: 1D Vector Tile
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, count, pto::BLayout::RowMajor, -1, -1>;

    ShapeDyn shape(1, 1, 1, 1, count);
    StrideDyn stride(count, count, count, count, 1);

    int my_rank = shmem_my_pe();
    int next_rank = (my_rank + 1) % nranks;
    int prev_rank = (my_rank + nranks - 1) % nranks;

    __gm__ T *shmem_data = (__gm__ T *)((__gm__ T *)shmem + 64 * sizeof(int32_t));
    __gm__ T *send_shmem = (__gm__ T *)((__gm__ T *)shmem_data + 0);
    __gm__ T *recv_shmem = (__gm__ T *)((__gm__ T *)shmem_data + count);

    Global srcG(src, shape, stride);
    Global dstG(dst, shape, stride);

    Global sendG(send_shmem, shape, stride);
    Global recvG(recv_shmem, shape, stride);

    // Allocate UB tiles for data staging
    TileData srcBufTile(1, count);
    TileData dstBufTile(1, count);

    TASSIGN(srcBufTile, 0x0);
    TASSIGN(dstBufTile, 0x10000);

    // Load local data to UB, then store to local shared memory
    TLOAD(srcBufTile, srcG);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(sendG, srcBufTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

    // Get remote PE's recv buffer address
    // This converts local symmetric heap offset to actual remote address mapping
    __gm__ T *remote_recv_shmem = ShmemPtr(recv_shmem, prev_rank);
    Global remoteRecvG(remote_recv_shmem, shape, stride);
    
    // TPUT: write local sendG to remote recvG (previous rank's recv buffer)
    pto::comm::TPUT(remoteRecvG, sendG, srcBufTile);
    
    // Ensure all remote operations from this PE are complete
    ShmemDeviceQuiet();
    // Then synchronize all PEs
    ShmemDeviceBarrierAll();

    // Now recvG on my rank should have data from next rank
    // Load from local recvG to UB, then store to local dstG
    TLOAD(dstBufTile, recvG);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstG, dstBufTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

template <typename T, size_t count>
bool RunPutRingKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, uint64_t local_mem_size){
    
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
    const char *ip = "tcp://127.0.0.1:8769";
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

    TPutKernelImpl<T, count><<<1, nullptr, stream>>>((T*)output_ptr, (T*)input_ptr, (T*)shmem_ptr, n_ranks);
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
        std::cout << "[DEBUG] Rank 0: TPUT Ring SUCCESSFUL!" << std::endl;
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
bool RunPutRing(int n_ranks, int n_devices, int first_rank_id, int first_device_id){
    std::vector<pid_t> pids;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    for (int r = 0; r < n_ranks; ++r){
        pid_t pid = fork();
        if (pid == 0) { // child
            const bool ok = RunPutRingKernel<T, count>(first_rank_id + r, n_ranks, n_devices, first_device_id, local_mem_size);
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
template bool RunPutRing<float, 256>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunPutRing<int32_t, 4096>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunPutRing<uint8_t, 512>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// ============================================================================
// 2D Tile Test Kernel
// Tests TPUT with 2D Vec Tile (rows x cols) stored in UB
// Uses Vec Tile because Mat Tile uses L1 cache which may not be supported
// ============================================================================
template <typename T, size_t rows, size_t cols>
__global__ AICORE void TPutKernel2DImpl(__gm__ T *dst, __gm__ T *src, __gm__ T *shmem, int nranks)
{
    constexpr size_t total_count = rows * cols;
    
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    // UB Tile definition: 2D Vec Tile (rows x cols) stored in Unified Buffer
    // Note: TileType::Vec stores data in UB (__ubuf__), which is required for MTE transfer
    //       TileType::Mat stores data in L1 cache (__cbuf__), which may not be supported
    using TileData = pto::Tile<pto::TileType::Vec, T, rows, cols, pto::BLayout::RowMajor, -1, -1>;

    // 2D GlobalTensor shape: [1, 1, 1, rows, cols]
    ShapeDyn shape(1, 1, 1, rows, cols);
    StrideDyn stride(total_count, total_count, total_count, cols, 1);

    int my_rank = shmem_my_pe();
    int next_rank = (my_rank + 1) % nranks;
    int prev_rank = (my_rank + nranks - 1) % nranks;

    __gm__ T *shmem_data = (__gm__ T *)((__gm__ T *)shmem + 64 * sizeof(int32_t));
    __gm__ T *send_shmem = (__gm__ T *)((__gm__ T *)shmem_data + 0);
    __gm__ T *recv_shmem = (__gm__ T *)((__gm__ T *)shmem_data + total_count);

    Global srcG(src, shape, stride);
    Global dstG(dst, shape, stride);

    Global sendG(send_shmem, shape, stride);
    Global recvG(recv_shmem, shape, stride);

    // Allocate UB tiles for data staging - 2D Vec Tiles (rows x cols)
    TileData srcBufTile(rows, cols);
    TileData dstBufTile(rows, cols);

    TASSIGN(srcBufTile, 0x0);
    TASSIGN(dstBufTile, 0x10000);

    // Load local data to UB, then store to local shared memory
    TLOAD(srcBufTile, srcG);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(sendG, srcBufTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

    // Get remote PE's recv buffer address
    __gm__ T *remote_recv_shmem = ShmemPtr(recv_shmem, prev_rank);
    Global remoteRecvG(remote_recv_shmem, shape, stride);
    
    // TPUT: write local sendG to remote recvG (previous rank's recv buffer)
    pto::comm::TPUT(remoteRecvG, sendG, srcBufTile);
    
    // Ensure all remote operations from this PE are complete
    ShmemDeviceQuiet();
    // Then synchronize all PEs
    ShmemDeviceBarrierAll();

    // Now recvG on my rank should have data from next rank
    // Load from local recvG to UB, then store to local dstG
    TLOAD(dstBufTile, recvG);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstG, dstBufTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

template <typename T, size_t rows, size_t cols>
bool RunPutRing2DKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, uint64_t local_mem_size)
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
    const char *ip = "tcp://127.0.0.1:8769";
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

    TPutKernel2DImpl<T, rows, cols><<<1, nullptr, stream>>>((T*)output_ptr, (T*)input_ptr, (T*)shmem_ptr, n_ranks);
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
        std::cout << "[DEBUG] Rank 0: TPUT 2D Ring SUCCESSFUL! (" << rows << "x" << cols << ")" << std::endl;
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
bool RunPutRing2D(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    std::vector<pid_t> pids;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) { // child
            const bool ok = RunPutRing2DKernel<T, rows, cols>(first_rank_id + r, n_ranks, n_devices, first_device_id, local_mem_size);
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
template bool RunPutRing2D<float, 16, 16>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunPutRing2D<float, 8, 32>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunPutRing2D<int32_t, 4, 64>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
