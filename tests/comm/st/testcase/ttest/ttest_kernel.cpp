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
// Kernel 1: TTEST True Condition Test
// Tests that TTEST returns true when signal matches expected value
// ============================================================================
__global__ AICORE void TTestTrueKernel(__gm__ int32_t *shmem_signal, __gm__ int32_t *result)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GSignal = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, 1);
    StrideDyn stride(1, 1, 1, 1, 1);

    int my_rank = shmem_my_pe();

    if (my_rank == 0) {
        // Rank 0: Set rank 1's signal value to 42 using ShmemPtr for remote address
        __gm__ int32_t *remote_signal = ShmemPtr(shmem_signal, 1);
        GSignal targetSignal(remote_signal, shape, stride);

        pto::comm::TNOTIFY(targetSignal, 42, pto::comm::NotifyOp::Set);
        ShmemDeviceQuiet();
    }

    ShmemDeviceBarrierAll();

    if (my_rank == 1) {
        // Rank 1: Test if local signal == 42 (should be true)
        GSignal localSignal(shmem_signal, shape, stride);

        bool testResult = pto::comm::TTEST(localSignal, 42, pto::comm::WaitCmp::EQ);
        *result = testResult ? 1 : 0;
    }

    ShmemDeviceBarrierAll();
}

// ============================================================================
// Kernel 2: TTEST False Condition Test
// Tests that TTEST returns false when signal does not match expected value
// ============================================================================
__global__ AICORE void TTestFalseKernel(__gm__ int32_t *shmem_signal, __gm__ int32_t *result)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GSignal = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, 1);
    StrideDyn stride(1, 1, 1, 1, 1);

    int my_rank = shmem_my_pe();

    if (my_rank == 0) {
        // Rank 0: Set rank 1's signal value to 42 using ShmemPtr
        __gm__ int32_t *remote_signal = ShmemPtr(shmem_signal, 1);
        GSignal targetSignal(remote_signal, shape, stride);

        pto::comm::TNOTIFY(targetSignal, 42, pto::comm::NotifyOp::Set);
        ShmemDeviceQuiet();
    }

    ShmemDeviceBarrierAll();

    if (my_rank == 1) {
        // Rank 1: Test if local signal == 100 (should be false, signal is 42)
        GSignal localSignal(shmem_signal, shape, stride);

        bool testResult = pto::comm::TTEST(localSignal, 100, pto::comm::WaitCmp::EQ);
        *result = testResult ? 1 : 0;
    }

    ShmemDeviceBarrierAll();
}

// ============================================================================
// Kernel 3: TTEST with different comparison operators
// Tests GE (>=), GT (>), LE (<=), LT (<), NE (!=) operators
// ============================================================================
template <pto::comm::WaitCmp cmp>
__global__ AICORE void TTestCompareKernel(__gm__ int32_t *shmem_signal, __gm__ int32_t *result, 
                                           int32_t signalValue, int32_t cmpValue)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GSignal = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, 1);
    StrideDyn stride(1, 1, 1, 1, 1);

    int my_rank = shmem_my_pe();

    if (my_rank == 0) {
        // Rank 0: Set rank 1's signal to specified value using ShmemPtr
        __gm__ int32_t *remote_signal = ShmemPtr(shmem_signal, 1);
        GSignal targetSignal(remote_signal, shape, stride);

        pto::comm::TNOTIFY(targetSignal, signalValue, pto::comm::NotifyOp::Set);
        ShmemDeviceQuiet();
    }

    ShmemDeviceBarrierAll();

    if (my_rank == 1) {
        // Rank 1: Test with specified comparison on local signal
        GSignal localSignal(shmem_signal, shape, stride);

        bool testResult = pto::comm::TTEST(localSignal, cmpValue, cmp);
        *result = testResult ? 1 : 0;
    }

    ShmemDeviceBarrierAll();
}

// ============================================================================
// Kernel 4: TTEST Polling with Timeout
// Demonstrates polling pattern: check, do work, check again
// ============================================================================
__global__ AICORE void TTestPollingTimeoutKernel(__gm__ int32_t *shmem_signal, __gm__ int32_t *poll_count,
                                                  __gm__ int32_t *final_result, int32_t delay_iters,
                                                  int32_t max_polls, bool send_signal)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GSignal = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, 1);
    StrideDyn stride(1, 1, 1, 1, 1);

    int my_rank = shmem_my_pe();

    // Sync start for polling
    ShmemDeviceBarrierAll();

    if (my_rank == 0 && send_signal) {
        // Rank 0: delay before sending signal
        for (int32_t i = 0; i < delay_iters; ++i) {
            __asm__ __volatile__("");
        }
        __gm__ int32_t *remote_signal = ShmemPtr(shmem_signal, 1);
        GSignal targetSignal(remote_signal, shape, stride);

        pto::comm::TNOTIFY(targetSignal, 999, pto::comm::NotifyOp::Set);
        ShmemDeviceQuiet();
    }

    if (my_rank == 1) {
        // Rank 1: Poll with TTEST until signal or timeout
        GSignal localSignal(shmem_signal, shape, stride);

        int32_t count = 0;
        bool found = false;
        while (count < max_polls) {
            if (pto::comm::TTEST(localSignal, 999, pto::comm::WaitCmp::EQ)) {
                found = true;
                break;
            }
            ++count;
        }

        *poll_count = count;
        *final_result = found ? 1 : 0;
    }

    ShmemDeviceBarrierAll();
}

// ============================================================================
// Kernel 5: TTEST Not Equal (NE) Test
// Tests that TTEST with NE returns true when values differ
// ============================================================================
__global__ AICORE void TTestNEKernel(__gm__ int32_t *shmem_signal, __gm__ int32_t *result)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GSignal = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, 1);
    StrideDyn stride(1, 1, 1, 1, 1);

    int my_rank = shmem_my_pe();

    if (my_rank == 0) {
        // Rank 0: Set rank 1's signal to 50 using ShmemPtr
        __gm__ int32_t *remote_signal = ShmemPtr(shmem_signal, 1);
        GSignal targetSignal(remote_signal, shape, stride);

        pto::comm::TNOTIFY(targetSignal, 50, pto::comm::NotifyOp::Set);
        ShmemDeviceQuiet();
    }

    ShmemDeviceBarrierAll();

    if (my_rank == 1) {
        // Rank 1: Test if local signal != 0 (should be true, signal is 50)
        GSignal localSignal(shmem_signal, shape, stride);

        bool testResult = pto::comm::TTEST(localSignal, 0, pto::comm::WaitCmp::NE);
        *result = testResult ? 1 : 0;
    }

    ShmemDeviceBarrierAll();
}

// ============================================================================
// Host-side Test Implementation
// ============================================================================

bool RunTTestTrueKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
{
    // Initialize shmem TLS configuration
    int32_t ret = ShmemSetConfStoreTls(false, nullptr, 0);
    if (ret != 0) {
        std::cerr << "[ERROR] Failed to init shmem tls\n";
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
    const char *ip = "tcp://127.0.0.1:8790";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;
    env.heapBytes = 8ULL * 1024 * 1024;

    if (!ShmemInitFromEnv(env)) {
        std::cerr << "[ERROR] ShmemInitFromEnv failed!" << std::endl;
        return false;
    }

    int32_t *shmem_signal = (int32_t *)ShmemMalloc(sizeof(int32_t));
    int32_t *result = (int32_t *)ShmemMalloc(sizeof(int32_t));

    if (shmem_signal == nullptr || result == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    int32_t zero = 0;
    aclrtMemcpy(shmem_signal, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(result, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    ShmemBarrierAll();

    TTestTrueKernel<<<1, nullptr, stream>>>(shmem_signal, result);
    status = aclrtSynchronizeStream(stream);

    ShmemBarrierAll();

    bool is_ok = true;

    if (rank_id == 1) {
        int32_t testResult = 0;
        aclrtMemcpy(&testResult, sizeof(int32_t), result, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

        if (testResult != 1) {
            std::cerr << "TTest True test failed! TTEST(EQ, 42) should return true (1), Got: " << testResult << std::endl;
            is_ok = false;
        } else {
            std::cout << "Rank 1: TTEST(EQ, 42) returned " << testResult << " (expected 1/true)" << std::endl;
        }
    }

    ShmemFree(shmem_signal);
    ShmemFree(result);
    ShmemFinalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

bool RunTTestFalseKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
{
    // Initialize shmem TLS configuration
    int32_t ret = ShmemSetConfStoreTls(false, nullptr, 0);
    if (ret != 0) {
        std::cerr << "[ERROR] Failed to init shmem tls\n";
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
    const char *ip = "tcp://127.0.0.1:8791";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;
    env.heapBytes = 8ULL * 1024 * 1024;

    if (!ShmemInitFromEnv(env)) {
        std::cerr << "[ERROR] ShmemInitFromEnv failed!" << std::endl;
        return false;
    }

    int32_t *shmem_signal = (int32_t *)ShmemMalloc(sizeof(int32_t));
    int32_t *result = (int32_t *)ShmemMalloc(sizeof(int32_t));

    if (shmem_signal == nullptr || result == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    int32_t zero = 0;
    aclrtMemcpy(shmem_signal, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    int32_t one = 1;  // Initialize to 1 so we can detect if TTEST correctly returns 0
    aclrtMemcpy(result, sizeof(int32_t), &one, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    ShmemBarrierAll();

    TTestFalseKernel<<<1, nullptr, stream>>>(shmem_signal, result);
    status = aclrtSynchronizeStream(stream);

    ShmemBarrierAll();

    bool is_ok = true;

    if (rank_id == 1) {
        int32_t testResult = 0;
        aclrtMemcpy(&testResult, sizeof(int32_t), result, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

        if (testResult != 0) {
            std::cerr << "TTest False test failed! TTEST(EQ, 100) when signal=42 should return false (0), Got: " << testResult << std::endl;
            is_ok = false;
        } else {
            std::cout << "Rank 1: TTEST(EQ, 100) when signal=42 returned " << testResult << " (expected 0/false)" << std::endl;
        }
    }

    ShmemFree(shmem_signal);
    ShmemFree(result);
    ShmemFinalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

template <pto::comm::WaitCmp cmp>
bool RunTTestCompareKernel(int rank_id, int n_ranks, int n_devices, int first_device_id,
                            int32_t signalValue, int32_t cmpValue, bool expectedResult)
{
    // Initialize shmem TLS configuration
    int32_t ret = ShmemSetConfStoreTls(false, nullptr, 0);
    if (ret != 0) {
        std::cerr << "[ERROR] Failed to init shmem tls\n";
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
    char ipPort[64];
    snprintf(ipPort, sizeof(ipPort), "tcp://127.0.0.1:%d", 8792 + static_cast<int>(cmp));
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ipPort;
    env.heapBytes = 8ULL * 1024 * 1024;

    if (!ShmemInitFromEnv(env)) {
        std::cerr << "[ERROR] ShmemInitFromEnv failed!" << std::endl;
        return false;
    }

    int32_t *shmem_signal = (int32_t *)ShmemMalloc(sizeof(int32_t));
    int32_t *result = (int32_t *)ShmemMalloc(sizeof(int32_t));

    if (shmem_signal == nullptr || result == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    int32_t zero = 0;
    aclrtMemcpy(shmem_signal, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(result, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    ShmemBarrierAll();

    TTestCompareKernel<cmp><<<1, nullptr, stream>>>(shmem_signal, result, signalValue, cmpValue);
    status = aclrtSynchronizeStream(stream);

    ShmemBarrierAll();

    bool is_ok = true;

    if (rank_id == 1) {
        int32_t testResult = 0;
        aclrtMemcpy(&testResult, sizeof(int32_t), result, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

        int32_t expectedInt = expectedResult ? 1 : 0;
        if (testResult != expectedInt) {
            std::cerr << "TTest Compare test failed! signal=" << signalValue 
                      << ", cmpValue=" << cmpValue << ", expected " << expectedInt 
                      << ", Got: " << testResult << std::endl;
            is_ok = false;
        } else {
            std::cout << "Rank 1: TTEST compare (signal=" << signalValue << ", cmpValue=" << cmpValue 
                      << ") returned " << testResult << " (expected " << expectedInt << ")" << std::endl;
        }
    }

    ShmemFree(shmem_signal);
    ShmemFree(result);
    ShmemFinalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

bool RunTTestPollingTimeoutKernel(int rank_id, int n_ranks, int n_devices, int first_device_id,
                                  int32_t delay_iters, int32_t max_polls, bool expected_found, bool send_signal)
{
    // Initialize shmem TLS configuration
    int32_t ret = ShmemSetConfStoreTls(false, nullptr, 0);
    if (ret != 0) {
        std::cerr << "[ERROR] Failed to init shmem tls\n";
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
    const char *ip = "tcp://127.0.0.1:8800";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;
    env.heapBytes = 8ULL * 1024 * 1024;

    if (!ShmemInitFromEnv(env)) {
        std::cerr << "[ERROR] ShmemInitFromEnv failed!" << std::endl;
        return false;
    }

    int32_t *shmem_signal = (int32_t *)ShmemMalloc(sizeof(int32_t));
    int32_t *poll_count = (int32_t *)ShmemMalloc(sizeof(int32_t));
    int32_t *final_result = (int32_t *)ShmemMalloc(sizeof(int32_t));

    if (shmem_signal == nullptr || poll_count == nullptr || final_result == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    int32_t zero = 0;
    aclrtMemcpy(shmem_signal, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(poll_count, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(final_result, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    ShmemBarrierAll();

    TTestPollingTimeoutKernel<<<1, nullptr, stream>>>(shmem_signal, poll_count, final_result,
                                                      delay_iters, max_polls, send_signal);
    status = aclrtSynchronizeStream(stream);

    ShmemBarrierAll();

    bool is_ok = true;

    if (rank_id == 1) {
        int32_t count = 0;
        int32_t found = 0;
        aclrtMemcpy(&count, sizeof(int32_t), poll_count, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(&found, sizeof(int32_t), final_result, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

        const int32_t expected = expected_found ? 1 : 0;
        if (found != expected) {
            std::cerr << "TTest Polling Timeout test failed! expected=" << expected
                      << ", found=" << found << std::endl;
            is_ok = false;
        } else if (count < 0 || count > max_polls) {
            std::cerr << "TTest Polling Timeout test failed! Poll count out of range: " << count << std::endl;
            is_ok = false;
        } else if (expected_found) {
            std::cout << "Rank 1: TTEST polling found signal after " << count << " iterations" << std::endl;
        } else {
            std::cout << "Rank 1: TTEST polling timed out after " << count << " iterations" << std::endl;
        }
    }

    ShmemFree(shmem_signal);
    ShmemFree(poll_count);
    ShmemFree(final_result);
    ShmemFinalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

bool RunTTestNEKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
{
    // Initialize shmem TLS configuration
    int32_t ret = ShmemSetConfStoreTls(false, nullptr, 0);
    if (ret != 0) {
        std::cerr << "[ERROR] Failed to init shmem tls\n";
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
    const char *ip = "tcp://127.0.0.1:8801";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;
    env.heapBytes = 8ULL * 1024 * 1024;

    if (!ShmemInitFromEnv(env)) {
        std::cerr << "[ERROR] ShmemInitFromEnv failed!" << std::endl;
        return false;
    }

    int32_t *shmem_signal = (int32_t *)ShmemMalloc(sizeof(int32_t));
    int32_t *result = (int32_t *)ShmemMalloc(sizeof(int32_t));

    if (shmem_signal == nullptr || result == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    int32_t zero = 0;
    aclrtMemcpy(shmem_signal, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(result, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    ShmemBarrierAll();

    TTestNEKernel<<<1, nullptr, stream>>>(shmem_signal, result);
    status = aclrtSynchronizeStream(stream);

    ShmemBarrierAll();

    bool is_ok = true;

    if (rank_id == 1) {
        int32_t testResult = 0;
        aclrtMemcpy(&testResult, sizeof(int32_t), result, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

        if (testResult != 1) {
            std::cerr << "TTest NE test failed! TTEST(NE, 0) when signal=50 should return true (1), Got: " << testResult << std::endl;
            is_ok = false;
        } else {
            std::cout << "Rank 1: TTEST(NE, 0) when signal=50 returned " << testResult << " (expected 1/true)" << std::endl;
        }
    }

    ShmemFree(shmem_signal);
    ShmemFree(result);
    ShmemFinalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

// ============================================================================
// Multi-process Launcher Functions
// ============================================================================

bool RunTTestTrue(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    std::vector<pid_t> pids;
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            const bool ok = RunTTestTrueKernel(first_rank_id + r, n_ranks, n_devices, first_device_id);
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

bool RunTTestFalse(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    std::vector<pid_t> pids;
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            const bool ok = RunTTestFalseKernel(first_rank_id + r, n_ranks, n_devices, first_device_id);
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

template <pto::comm::WaitCmp cmp>
bool RunTTestCompare(int n_ranks, int n_devices, int first_rank_id, int first_device_id,
                     int32_t signalValue, int32_t cmpValue, bool expectedResult)
{
    std::vector<pid_t> pids;
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            const bool ok = RunTTestCompareKernel<cmp>(first_rank_id + r, n_ranks, n_devices, first_device_id,
                                                        signalValue, cmpValue, expectedResult);
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

bool RunTTestPollingTimeout(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    std::vector<pid_t> pids;
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            const bool ok = RunTTestPollingTimeoutKernel(first_rank_id + r, n_ranks, n_devices, first_device_id,
                                                         50000, 200000, true, true);
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

bool RunTTestPollingTimeoutMiss(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    std::vector<pid_t> pids;
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            const bool ok = RunTTestPollingTimeoutKernel(first_rank_id + r, n_ranks, n_devices, first_device_id,
                                                         0, 50000, false, false);
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

bool RunTTestNE(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    std::vector<pid_t> pids;
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            const bool ok = RunTTestNEKernel(first_rank_id + r, n_ranks, n_devices, first_device_id);
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

// Non-template wrapper functions for host-side linkage (avoid including comm_types.hpp in main.cpp)
bool RunTTestCompare_GE(int n_ranks, int n_devices, int first_rank_id, int first_device_id,
                        int32_t signalValue, int32_t cmpValue, bool expectedResult)
{
    return RunTTestCompare<pto::comm::WaitCmp::GE>(n_ranks, n_devices, first_rank_id, first_device_id,
                                                    signalValue, cmpValue, expectedResult);
}

bool RunTTestCompare_GT(int n_ranks, int n_devices, int first_rank_id, int first_device_id,
                        int32_t signalValue, int32_t cmpValue, bool expectedResult)
{
    return RunTTestCompare<pto::comm::WaitCmp::GT>(n_ranks, n_devices, first_rank_id, first_device_id,
                                                    signalValue, cmpValue, expectedResult);
}

bool RunTTestCompare_LE(int n_ranks, int n_devices, int first_rank_id, int first_device_id,
                        int32_t signalValue, int32_t cmpValue, bool expectedResult)
{
    return RunTTestCompare<pto::comm::WaitCmp::LE>(n_ranks, n_devices, first_rank_id, first_device_id,
                                                    signalValue, cmpValue, expectedResult);
}

bool RunTTestCompare_LT(int n_ranks, int n_devices, int first_rank_id, int first_device_id,
                        int32_t signalValue, int32_t cmpValue, bool expectedResult)
{
    return RunTTestCompare<pto::comm::WaitCmp::LT>(n_ranks, n_devices, first_rank_id, first_device_id,
                                                    signalValue, cmpValue, expectedResult);
}
