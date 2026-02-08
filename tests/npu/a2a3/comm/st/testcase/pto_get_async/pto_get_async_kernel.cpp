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
#include "pto/comm/kernels/Pto_getAsync.hpp"

#define ENABLE_DEBUG_PRINT 1

// ============================================================================
// SDMA Path Test: Test PTO_GET_ASYNC with aclrtMemcpyAsync (default)
//
// GET operation: Read data from remote device to local device
// Data flow: remoteDevice (srcPtr) → localDevice (dstPtr)
// ============================================================================
template <typename T, size_t count>
bool RunGetAsyncSdmaTest(int localDeviceId, int remoteDeviceId)
{
    std::cout << "\n[TEST] RunGetAsyncSdmaTest<" << typeid(T).name() << ", " << count 
              << ">(local=" << localDeviceId << ", remote=" << remoteDeviceId << ")" << std::endl;

    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclInit failed: " << ret << std::endl;
        return false;
    }

    // Check and enable P2P access
    if (!pto::comm::detail::CanAccessPeer(localDeviceId, remoteDeviceId)) {
        std::cerr << "[ERROR] P2P not supported between Device " 
                  << localDeviceId << " and " << remoteDeviceId << std::endl;
        aclFinalize();
        return false;
    }
    
    ret = pto::comm::detail::EnablePeerAccessBidirectional(localDeviceId, remoteDeviceId);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] EnablePeerAccessBidirectional failed: " << ret << std::endl;
        aclFinalize();
        return false;
    }
    std::cout << "[INFO] P2P access enabled between Device " << localDeviceId 
              << " and Device " << remoteDeviceId << std::endl;

    // ========================================================================
    // Allocate and initialize source data on REMOTE device
    // ========================================================================
    ret = aclrtSetDevice(remoteDeviceId);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtSetDevice(remote) failed: " << ret << std::endl;
        aclFinalize();
        return false;
    }
    
    void* remoteSrcPtr = nullptr;
    ret = aclrtMalloc(&remoteSrcPtr, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST_P2P);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMalloc(remote) failed: " << ret << std::endl;
        aclFinalize();
        return false;
    }

    // Initialize remote source data on host and copy to device
    T* srcHost = nullptr;
    ret = aclrtMallocHost(reinterpret_cast<void**>(&srcHost), count * sizeof(T));
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMallocHost(srcHost) failed: " << ret << std::endl;
        aclrtFree(remoteSrcPtr);
        aclFinalize();
        return false;
    }
    
    for (size_t i = 0; i < count; ++i) {
        srcHost[i] = static_cast<T>(i + 1);  // Data pattern: 1, 2, 3, ...
    }
    ret = aclrtMemcpy(remoteSrcPtr, count * sizeof(T), srcHost, count * sizeof(T), 
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMemcpy(H2D remote) failed: " << ret << std::endl;
        aclrtFreeHost(srcHost);
        aclrtFree(remoteSrcPtr);
        aclFinalize();
        return false;
    }

    // ========================================================================
    // Allocate and initialize destination on LOCAL device
    // ========================================================================
    ret = aclrtSetDevice(localDeviceId);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtSetDevice(local) failed: " << ret << std::endl;
        aclrtFreeHost(srcHost);
        aclrtFree(remoteSrcPtr);
        aclFinalize();
        return false;
    }
    
    void* localDstPtr = nullptr;
    ret = aclrtMalloc(&localDstPtr, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST_P2P);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMalloc(local) failed: " << ret << std::endl;
        aclrtFreeHost(srcHost);
        aclrtFree(remoteSrcPtr);
        aclFinalize();
        return false;
    }

    // Initialize local destination to -1
    T* initHost = nullptr;
    ret = aclrtMallocHost(reinterpret_cast<void**>(&initHost), count * sizeof(T));
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMallocHost(initHost) failed: " << ret << std::endl;
        aclrtFreeHost(srcHost);
        aclrtFree(remoteSrcPtr);
        aclrtFree(localDstPtr);
        aclFinalize();
        return false;
    }
    
    for (size_t i = 0; i < count; ++i) {
        initHost[i] = static_cast<T>(-1);
    }
    ret = aclrtMemcpy(localDstPtr, count * sizeof(T), initHost, count * sizeof(T),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMemcpy(H2D local init) failed: " << ret << std::endl;
        aclrtFreeHost(srcHost);
        aclrtFreeHost(initHost);
        aclrtFree(remoteSrcPtr);
        aclrtFree(localDstPtr);
        aclFinalize();
        return false;
    }

    // ========================================================================
    // Execute async GET operation: remote → local
    // ========================================================================
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtCreateStream failed: " << ret << std::endl;
        aclrtFreeHost(srcHost);
        aclrtFreeHost(initHost);
        aclrtFree(remoteSrcPtr);
        aclrtFree(localDstPtr);
        aclFinalize();
        return false;
    }

    // Call PTO_GET_ASYNC (SDMA path, default) - THIS IS THE ASYNC OPERATION
    // GET: Read from remote (remoteSrcPtr) to local (localDstPtr)
    std::cout << "[INFO] Calling PTO_GET_ASYNC (SDMA path)..." << std::endl;
    ret = pto::comm::PTO_GET_ASYNC(
        localDstPtr, count * sizeof(T),   // local destination
        remoteSrcPtr, count * sizeof(T),  // remote source
        stream
    );
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] PTO_GET_ASYNC failed: " << ret << std::endl;
    }

    // Synchronize to wait for async transfer completion
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtSynchronizeStream failed: " << ret << std::endl;
    }

    // ========================================================================
    // Verify result
    // ========================================================================
    T* dstHost = nullptr;
    aclrtMallocHost(reinterpret_cast<void**>(&dstHost), count * sizeof(T));
    aclrtMemcpy(dstHost, count * sizeof(T), localDstPtr, count * sizeof(T),
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
        std::cout << "[OK] PTO_GET_ASYNC SDMA test PASSED!" << std::endl;
        std::cout << "Transfer size: " << count << " elements (" << count * sizeof(T) << " bytes)" << std::endl;
        std::cout << "Data flow: Device " << remoteDeviceId << " → Device " << localDeviceId << std::endl;
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
    aclrtFree(remoteSrcPtr);
    aclrtFree(localDstPtr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(localDeviceId);
    aclrtResetDevice(remoteDeviceId);
    aclFinalize();

    return is_ok;
}

// ============================================================================
// AIV Path Test: Test PTO_GET_ASYNC with AIV kernel
// Note: AIV path requires TGET_ASYNC_SDMA_IMPL to be fully implemented
// ============================================================================
template <typename T, size_t count, int AivCores>
bool RunGetAsyncAivTest(int localDeviceId, int remoteDeviceId)
{
    std::cout << "\n[TEST] RunGetAsyncAivTest<" << typeid(T).name() << ", " << count 
              << ", AivCores=" << AivCores << ">(local=" << localDeviceId << ", remote=" << remoteDeviceId << ")" << std::endl;

    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclInit failed: " << ret << std::endl;
        return false;
    }

    // Check and enable P2P access
    if (!pto::comm::detail::CanAccessPeer(localDeviceId, remoteDeviceId)) {
        std::cerr << "[ERROR] P2P not supported between Device " 
                  << localDeviceId << " and " << remoteDeviceId << std::endl;
        aclFinalize();
        return false;
    }
    
    ret = pto::comm::detail::EnablePeerAccessBidirectional(localDeviceId, remoteDeviceId);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] EnablePeerAccessBidirectional failed: " << ret << std::endl;
        aclFinalize();
        return false;
    }

    // Allocate and initialize source on remote device
    aclrtSetDevice(remoteDeviceId);
    void* remoteSrcPtr = nullptr;
    aclrtMalloc(&remoteSrcPtr, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST_P2P);

    T* srcHost = nullptr;
    aclrtMallocHost(reinterpret_cast<void**>(&srcHost), count * sizeof(T));
    for (size_t i = 0; i < count; ++i) {
        srcHost[i] = static_cast<T>(i + 100);  // Different pattern for AIV test
    }
    aclrtMemcpy(remoteSrcPtr, count * sizeof(T), srcHost, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    // Allocate and initialize destination on local device
    aclrtSetDevice(localDeviceId);
    void* localDstPtr = nullptr;
    aclrtMalloc(&localDstPtr, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST_P2P);

    T* initHost = nullptr;
    aclrtMallocHost(reinterpret_cast<void**>(&initHost), count * sizeof(T));
    for (size_t i = 0; i < count; ++i) {
        initHost[i] = static_cast<T>(-1);
    }
    aclrtMemcpy(localDstPtr, count * sizeof(T), initHost, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    // Create stream for async kernel launch
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // Call PTO_GET_ASYNC with AIV path - THIS IS THE ASYNC OPERATION
    std::cout << "[INFO] Calling PTO_GET_ASYNC<false, " << AivCores << "> (AIV path)..." << std::endl;
    ret = pto::comm::PTO_GET_ASYNC<false, AivCores>(
        localDstPtr, count * sizeof(T),
        remoteSrcPtr, count * sizeof(T),
        stream
    );

    aclrtSynchronizeStream(stream);

    // Verify result
    T* dstHost = nullptr;
    aclrtMallocHost(reinterpret_cast<void**>(&dstHost), count * sizeof(T));
    aclrtMemcpy(dstHost, count * sizeof(T), localDstPtr, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

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
        std::cout << "[OK] PTO_GET_ASYNC AIV test PASSED!" << std::endl;
        std::cout << "Transfer size: " << count << " elements, AivCores: " << AivCores << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }
#endif

    // Cleanup
    aclrtFreeHost(srcHost);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(initHost);
    aclrtFree(remoteSrcPtr);
    aclrtFree(localDstPtr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(localDeviceId);
    aclrtResetDevice(remoteDeviceId);
    aclFinalize();

    return is_ok;
}

// ============================================================================
// Explicit template instantiations
// ============================================================================
// SDMA tests
template bool RunGetAsyncSdmaTest<float, 256>(int, int);
template bool RunGetAsyncSdmaTest<int32_t, 4096>(int, int);
template bool RunGetAsyncSdmaTest<uint8_t, 512>(int, int);

// AIV tests  
template bool RunGetAsyncAivTest<float, 256, 20>(int, int);
template bool RunGetAsyncAivTest<int32_t, 1024, 10>(int, int);
