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
#include <cstring>
#include <vector>
#include <string>
#include <iostream>

#include <acl/acl.h>
#include "pto/comm/kernels/Pto_putAsync.hpp"
#include "../common.hpp"

#define ENABLE_DEBUG_PRINT 1

// ============================================================================
// SDMA Path Test: Test PTO_PUT_ASYNC with aclrtMemcpyAsync (default)
// ============================================================================
template <typename T, size_t count>
bool RunPutAsyncSdmaTest(int srcDeviceId, int dstDeviceId)
{
    std::cout << "\n[TEST] RunPutAsyncSdmaTest<" << typeid(T).name() << ", " << count 
              << ">(src=" << srcDeviceId << ", dst=" << dstDeviceId << ")" << std::endl;

    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclInit failed: " << ret << std::endl;
        return false;
    }

    // Check and enable P2P access
    if (!pto::comm::detail::CanAccessPeer(srcDeviceId, dstDeviceId)) {
        std::cerr << "[ERROR] P2P not supported between Device " 
                  << srcDeviceId << " and " << dstDeviceId << std::endl;
        aclFinalize();
        return false;
    }
    
    ret = pto::comm::detail::EnablePeerAccessBidirectional(srcDeviceId, dstDeviceId);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] EnablePeerAccessBidirectional failed: " << ret << std::endl;
        aclFinalize();
        return false;
    }
    std::cout << "[INFO] P2P access enabled between Device " << srcDeviceId 
              << " and Device " << dstDeviceId << std::endl;

    // Allocate memory on source device
    ret = aclrtSetDevice(srcDeviceId);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtSetDevice(src) failed: " << ret << std::endl;
        aclFinalize();
        return false;
    }
    
    void* srcPtr = nullptr;
    ret = aclrtMalloc(&srcPtr, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST_P2P);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMalloc(src) failed: " << ret << std::endl;
        aclFinalize();
        return false;
    }

    // Initialize source data on host and copy to device (sync for initialization)
    T* srcHost = nullptr;
    ret = aclrtMallocHost(reinterpret_cast<void**>(&srcHost), count * sizeof(T));
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMallocHost(srcHost) failed: " << ret << std::endl;
        aclrtFree(srcPtr);
        aclFinalize();
        return false;
    }
    
    for (size_t i = 0; i < count; ++i) {
        srcHost[i] = static_cast<T>(i + 1);
    }
    ret = aclrtMemcpy(srcPtr, count * sizeof(T), srcHost, count * sizeof(T), 
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMemcpy(H2D src) failed: " << ret << std::endl;
        aclrtFreeHost(srcHost);
        aclrtFree(srcPtr);
        aclFinalize();
        return false;
    }

    // Allocate memory on destination device
    ret = aclrtSetDevice(dstDeviceId);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtSetDevice(dst) failed: " << ret << std::endl;
        aclrtFreeHost(srcHost);
        aclrtFree(srcPtr);
        aclFinalize();
        return false;
    }
    
    void* dstPtr = nullptr;
    ret = aclrtMalloc(&dstPtr, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST_P2P);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMalloc(dst) failed: " << ret << std::endl;
        aclrtFreeHost(srcHost);
        aclrtFree(srcPtr);
        aclFinalize();
        return false;
    }

    // Initialize destination to -1 (sync for initialization)
    T* initHost = nullptr;
    ret = aclrtMallocHost(reinterpret_cast<void**>(&initHost), count * sizeof(T));
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMallocHost(initHost) failed: " << ret << std::endl;
        aclrtFreeHost(srcHost);
        aclrtFree(srcPtr);
        aclrtFree(dstPtr);
        aclFinalize();
        return false;
    }
    
    for (size_t i = 0; i < count; ++i) {
        initHost[i] = static_cast<T>(-1);
    }
    ret = aclrtMemcpy(dstPtr, count * sizeof(T), initHost, count * sizeof(T),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMemcpy(H2D dst init) failed: " << ret << std::endl;
        aclrtFreeHost(srcHost);
        aclrtFreeHost(initHost);
        aclrtFree(srcPtr);
        aclrtFree(dstPtr);
        aclFinalize();
        return false;
    }

    // Create stream for async D2D transfer
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtCreateStream failed: " << ret << std::endl;
        aclrtFreeHost(srcHost);
        aclrtFreeHost(initHost);
        aclrtFree(srcPtr);
        aclrtFree(dstPtr);
        aclFinalize();
        return false;
    }

    // Call PTO_PUT_ASYNC (SDMA path, default) - THIS IS THE ASYNC OPERATION
    std::cout << "[INFO] Calling PTO_PUT_ASYNC (SDMA path)..." << std::endl;
    ret = pto::comm::PTO_PUT_ASYNC(
        dstPtr, count * sizeof(T),
        srcPtr, count * sizeof(T),
        stream
    );
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] PTO_PUT_ASYNC failed: " << ret << std::endl;
    }

    // Synchronize to wait for async transfer completion
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtSynchronizeStream failed: " << ret << std::endl;
    }

    // Verify result (sync for verification)
    T* dstHost = nullptr;
    aclrtMallocHost(reinterpret_cast<void**>(&dstHost), count * sizeof(T));
    aclrtMemcpy(dstHost, count * sizeof(T), dstPtr, count * sizeof(T),
                ACL_MEMCPY_DEVICE_TO_HOST);

    bool is_ok = true;
    for (size_t i = 0; i < count; ++i) {
        if (dstHost[i] != srcHost[i]) {
            std::cerr << "[ERROR] Mismatch at index " << i 
                      << ": expected " << static_cast<float>(srcHost[i])
                      << ", got " << static_cast<float>(dstHost[i]) << std::endl;
            is_ok = false;
            break;
        }
    }

#if ENABLE_DEBUG_PRINT
    if (is_ok) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "[OK] PTO_PUT_ASYNC SDMA test PASSED!" << std::endl;
        std::cout << "Transfer size: " << count << " elements (" << count * sizeof(T) << " bytes)" << std::endl;
        std::cout << "Sample data (first 5): [ ";
        for (size_t i = 0; i < (count > 5 ? 5 : count); ++i) {
            std::cout << static_cast<float>(dstHost[i]) << " ";
        }
        if (count > 5) std::cout << "... ";
        std::cout << "]" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }
#endif

    // Cleanup
    aclrtFreeHost(srcHost);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(initHost);
    aclrtFree(srcPtr);
    aclrtFree(dstPtr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(srcDeviceId);
    aclrtResetDevice(dstDeviceId);
    aclFinalize();

    return is_ok;
}

// ============================================================================
// AIV Path Test: Test PTO_PUT_ASYNC with AIV kernel
// Note: AIV path requires TPUT_ASYNC_SDMA_IMPL to be fully implemented
// ============================================================================
template <typename T, size_t count, int AivCores>
bool RunPutAsyncAivTest(int srcDeviceId, int dstDeviceId)
{
    std::cout << "\n[TEST] RunPutAsyncAivTest<" << typeid(T).name() << ", " << count 
              << ", AivCores=" << AivCores << ">(src=" << srcDeviceId << ", dst=" << dstDeviceId << ")" << std::endl;

    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclInit failed: " << ret << std::endl;
        return false;
    }

    // Initialize shmem TLS configuration
    int32_t shmem_ret = ShmemSetConfStoreTls(false, nullptr, 0);
    if (shmem_ret != 0) {
        std::cerr << "[ERROR] Failed to init shmem tls\n";
        aclFinalize();
        return false;
    }

    // Check and enable P2P access
    if (!pto::comm::detail::CanAccessPeer(srcDeviceId, dstDeviceId)) {
        std::cerr << "[ERROR] P2P not supported between Device " 
                  << srcDeviceId << " and " << dstDeviceId << std::endl;
        aclFinalize();
        return false;
    }
    
    ret = pto::comm::detail::EnablePeerAccessBidirectional(srcDeviceId, dstDeviceId);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] EnablePeerAccessBidirectional failed: " << ret << std::endl;
        aclFinalize();
        return false;
    }

    // Initialize shmem symmetric heap
    ret = aclrtSetDevice(srcDeviceId);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtSetDevice(src) failed: " << ret << std::endl;
        aclFinalize();
        return false;
    }
    ShmemEnv env;
    env.rank = 0;
    env.size = 1;
    env.ipPort = "tcp://127.0.0.1:8771";
    uint64_t required_bytes = 2ULL * count * sizeof(T);
    env.heapBytes = (required_bytes < (8ULL * 1024 * 1024)) ? (8ULL * 1024 * 1024) : required_bytes;
    if (ShmemInitForSdma(env) != 0) {
        std::cerr << "[ERROR] ShmemInitForSdma failed!" << std::endl;
        aclFinalize();
        return false;
    }

    // Initialize source data (sync)
    T* srcHost = nullptr;
    aclrtMallocHost(reinterpret_cast<void**>(&srcHost), count * sizeof(T));
    for (size_t i = 0; i < count; ++i) {
        srcHost[i] = static_cast<T>(i + 100);  // Different pattern for AIV test
    }

    // Initialize destination (sync)
    T* initHost = nullptr;
    aclrtMallocHost(reinterpret_cast<void**>(&initHost), count * sizeof(T));
    for (size_t i = 0; i < count; ++i) {
        initHost[i] = static_cast<T>(-1);
    }

    // Allocate symmetric heap memory for shared buffer (send + recv)
    void* shmem_ptr = ShmemMalloc(2 * count * sizeof(T));
    if (shmem_ptr == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        aclrtFreeHost(srcHost);
        aclrtFreeHost(initHost);
        ShmemFinalize();
        aclFinalize();
        return false;
    }

    T* shmem_data = reinterpret_cast<T*>(shmem_ptr);
    T* send_shmem = shmem_data;
    T* recv_shmem = shmem_data + count;

    aclrtMemcpy(send_shmem, count * sizeof(T), srcHost, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(recv_shmem, count * sizeof(T), initHost, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    // Barrier to ensure all ranks have initialized
    ShmemBarrierAll();

    // Create stream for async kernel launch
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // Call PTO_PUT_ASYNC with AIV path - THIS IS THE ASYNC OPERATION
    std::cout << "[INFO] Calling PTO_PUT_ASYNC<false, " << AivCores << "> (AIV path)..." << std::endl;
    ret = pto::comm::PTO_PUT_ASYNC<false, AivCores>(
        recv_shmem, count * sizeof(T),
        send_shmem, count * sizeof(T),
        stream
    );

    aclrtSynchronizeStream(stream);

    // Barrier after kernel execution
    ShmemBarrierAll();

    // Verify result (sync)
    T* dstHost = nullptr;
    aclrtMallocHost(reinterpret_cast<void**>(&dstHost), count * sizeof(T));
    aclrtMemcpy(dstHost, count * sizeof(T), recv_shmem, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    bool is_ok = true;
    for (size_t i = 0; i < count; ++i) {
        if (dstHost[i] != srcHost[i]) {
            std::cerr << "[ERROR] Mismatch at index " << i 
                      << ": expected " << static_cast<float>(srcHost[i])
                      << ", got " << static_cast<float>(dstHost[i]) << std::endl;
            is_ok = false;
            break;
        }
    }

#if ENABLE_DEBUG_PRINT
    if (is_ok) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "[OK] PTO_PUT_ASYNC AIV test PASSED!" << std::endl;
        std::cout << "Transfer size: " << count << " elements, AivCores: " << AivCores << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }
#endif

    // Cleanup
    aclrtFreeHost(srcHost);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(initHost);
    ShmemFree(shmem_ptr);
    ShmemFinalize();
    aclrtDestroyStream(stream);
    aclrtResetDevice(srcDeviceId);
    aclrtResetDevice(dstDeviceId);
    aclFinalize();

    return is_ok;
}

// ============================================================================
// Explicit template instantiations
// ============================================================================
// SDMA tests
template bool RunPutAsyncSdmaTest<float, 256>(int, int);
template bool RunPutAsyncSdmaTest<int32_t, 4096>(int, int);
template bool RunPutAsyncSdmaTest<uint8_t, 512>(int, int);

// AIV tests  
template bool RunPutAsyncAivTest<float, 256, 20>(int, int);
template bool RunPutAsyncAivTest<int32_t, 1024, 10>(int, int);
