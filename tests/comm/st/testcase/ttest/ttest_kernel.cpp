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

#include "pto/comm/pto_comm_inst.hpp"
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"

#include <pto/pto-inst.hpp>

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
        // Rank 0: Set signal value to 42
        GSignal targetSignal(shmem_signal, shape, stride);
        targetSignal.SetRank(1);

        pto::comm::TNOTIFY<pto::comm::NotifyOp::Set>(targetSignal, 42);
        pto::comm::TQUIET();
    }

    pto::comm::TBARRIER();

    if (my_rank == 1) {
        // Rank 1: Test if signal == 42 (should be true)
        GSignal localSignal(shmem_signal, shape, stride);
        localSignal.SetRank(my_rank);

        bool testResult = pto::comm::TTEST<pto::comm::WaitCmp::EQ>(localSignal, 42);
        *result = testResult ? 1 : 0;
    }

    pto::comm::TBARRIER();
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
        // Rank 0: Set signal value to 42
        GSignal targetSignal(shmem_signal, shape, stride);
        targetSignal.SetRank(1);

        pto::comm::TNOTIFY<pto::comm::NotifyOp::Set>(targetSignal, 42);
        pto::comm::TQUIET();
    }

    pto::comm::TBARRIER();

    if (my_rank == 1) {
        // Rank 1: Test if signal == 100 (should be false, signal is 42)
        GSignal localSignal(shmem_signal, shape, stride);
        localSignal.SetRank(my_rank);

        bool testResult = pto::comm::TTEST<pto::comm::WaitCmp::EQ>(localSignal, 100);
        *result = testResult ? 1 : 0;
    }

    pto::comm::TBARRIER();
}

// ============================================================================
// Kernel 3: TTEST with different comparison operators
// Tests GE (>=), GT (>), LE (<=), LT (<), NE (!=) operators
// ============================================================================
template <pto::comm::WaitCmp cmp>
__global__ AICORE void TTestCompareKernel(__gm__ int32_t *shmem_signal, __gm__ int32_t *result, int32_t signalValue, int32_t cmpValue)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GSignal = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, 1);
    StrideDyn stride(1, 1, 1, 1, 1);

    int my_rank = shmem_my_pe();

    if (my_rank == 0) {
        // Rank 0: Set signal to specified value
        GSignal targetSignal(shmem_signal, shape, stride);
        targetSignal.SetRank(1);

        pto::comm::TNOTIFY<pto::comm::NotifyOp::Set>(targetSignal, signalValue);
        pto::comm::TQUIET();
    }

    pto::comm::TBARRIER();

    if (my_rank == 1) {
        // Rank 1: Test with specified comparison
        GSignal localSignal(shmem_signal, shape, stride);
        localSignal.SetRank(my_rank);

        bool testResult = pto::comm::TTEST<cmp>(localSignal, cmpValue);
        *result = testResult ? 1 : 0;
    }

    pto::comm::TBARRIER();
}

// ============================================================================
// Kernel 4: TTEST Polling with Timeout
// Demonstrates polling pattern: check, do work, check again
// ============================================================================
__global__ AICORE void TTestPollingTimeoutKernel(__gm__ int32_t *shmem_signal, __gm__ int32_t *poll_count, __gm__ int32_t *final_result)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GSignal = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, 1);
    StrideDyn stride(1, 1, 1, 1, 1);

    int my_rank = shmem_my_pe();

    if (my_rank == 0) {
        // Rank 0: Send signal after some delay (simulated by barrier)
        pto::comm::TBARRIER();  // Wait for rank 1 to start polling

        GSignal targetSignal(shmem_signal, shape, stride);
        targetSignal.SetRank(1);

        pto::comm::TNOTIFY<pto::comm::NotifyOp::Set>(targetSignal, 999);
        pto::comm::TQUIET();
    } else if (my_rank == 1) {
        // Rank 1: Poll until signal becomes 999 or timeout
        GSignal localSignal(shmem_signal, shape, stride);
        localSignal.SetRank(my_rank);

        int32_t count = 0;
        const int32_t maxPolls = 100000;
        bool found = false;

        pto::comm::TBARRIER();  // Signal rank 0 to send

        while (count < maxPolls) {
            count++;
            if (pto::comm::TTEST<pto::comm::WaitCmp::EQ>(localSignal, 999)) {
                found = true;
                break;
            }
        }

        *poll_count = count;
        *final_result = found ? 1 : 0;
    }

    pto::comm::TBARRIER();
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
        // Rank 0: Set signal to 50
        GSignal targetSignal(shmem_signal, shape, stride);
        targetSignal.SetRank(1);

        pto::comm::TNOTIFY<pto::comm::NotifyOp::Set>(targetSignal, 50);
        pto::comm::TQUIET();
    }

    pto::comm::TBARRIER();

    if (my_rank == 1) {
        // Rank 1: Test if signal != 0 (should be true, signal is 50)
        GSignal localSignal(shmem_signal, shape, stride);
        localSignal.SetRank(my_rank);

        bool testResult = pto::comm::TTEST<pto::comm::WaitCmp::NE>(localSignal, 0);
        *result = testResult ? 1 : 0;
    }

    pto::comm::TBARRIER();
}

// ============================================================================
// Host-side Test Implementation
// ============================================================================

bool RunTTestTrueKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
{
    int32_t ret = shmem_set_conf_store_tls(false, nullptr, 0);
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

    ShmemEnv env;
    const char *ip = "tcp://127.0.0.1:8790";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;

    if (!ShmemInitFromEnv(env)) {
        std::cerr << "ShmemInitFromEnv failed!" << std::endl;
        return false;
    }

    int32_t *shmem_signal = (int32_t *)pto::comm::ContextManager::SymmetricAlloc(sizeof(int32_t));
    int32_t *result = (int32_t *)pto::comm::ContextManager::SymmetricAlloc(sizeof(int32_t));

    int32_t zero = 0;
    aclrtMemcpy(shmem_signal, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(result, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    #if defined(ASCEND_SHMEM)
        shmem_barrier_all();
    #elif defined(CANN_SHMEM)
        aclshmem_barrier_all();
    #endif

    TTestTrueKernel<<<1, nullptr, stream>>>(shmem_signal, result);
    status = aclrtSynchronizeStream(stream);

    #if defined(ASCEND_SHMEM)
        shmem_barrier_all();
    #elif defined(CANN_SHMEM)
        aclshmem_barrier_all();
    #endif

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

    pto::comm::ContextManager::SymmetricFree(shmem_signal);
    pto::comm::ContextManager::SymmetricFree(result);
    pto::comm::ContextManager::Finalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

bool RunTTestFalseKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
{
    int32_t ret = shmem_set_conf_store_tls(false, nullptr, 0);
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

    ShmemEnv env;
    const char *ip = "tcp://127.0.0.1:8791";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;

    if (!ShmemInitFromEnv(env)) {
        std::cerr << "ShmemInitFromEnv failed!" << std::endl;
        return false;
    }

    int32_t *shmem_signal = (int32_t *)pto::comm::ContextManager::SymmetricAlloc(sizeof(int32_t));
    int32_t *result = (int32_t *)pto::comm::ContextManager::SymmetricAlloc(sizeof(int32_t));

    int32_t zero = 0;
    aclrtMemcpy(shmem_signal, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    int32_t one = 1;  // Initialize to 1 so we can detect if TTEST correctly returns 0
    aclrtMemcpy(result, sizeof(int32_t), &one, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    #if defined(ASCEND_SHMEM)
        shmem_barrier_all();
    #elif defined(CANN_SHMEM)
        aclshmem_barrier_all();
    #endif

    TTestFalseKernel<<<1, nullptr, stream>>>(shmem_signal, result);
    status = aclrtSynchronizeStream(stream);

    #if defined(ASCEND_SHMEM)
        shmem_barrier_all();
    #elif defined(CANN_SHMEM)
        aclshmem_barrier_all();
    #endif

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

    pto::comm::ContextManager::SymmetricFree(shmem_signal);
    pto::comm::ContextManager::SymmetricFree(result);
    pto::comm::ContextManager::Finalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

template <pto::comm::WaitCmp cmp>
bool RunTTestCompareKernel(int rank_id, int n_ranks, int n_devices, int first_device_id,
                            int32_t signalValue, int32_t cmpValue, bool expectedResult)
{
    int32_t ret = shmem_set_conf_store_tls(false, nullptr, 0);
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

    ShmemEnv env;
    char ipPort[64];
    snprintf(ipPort, sizeof(ipPort), "tcp://127.0.0.1:%d", 8792 + static_cast<int>(cmp));
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ipPort;

    if (!ShmemInitFromEnv(env)) {
        std::cerr << "ShmemInitFromEnv failed!" << std::endl;
        return false;
    }

    int32_t *shmem_signal = (int32_t *)pto::comm::ContextManager::SymmetricAlloc(sizeof(int32_t));
    int32_t *result = (int32_t *)pto::comm::ContextManager::SymmetricAlloc(sizeof(int32_t));

    int32_t zero = 0;
    aclrtMemcpy(shmem_signal, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(result, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    #if defined(ASCEND_SHMEM)
        shmem_barrier_all();
    #elif defined(CANN_SHMEM)
        aclshmem_barrier_all();
    #endif

    TTestCompareKernel<cmp><<<1, nullptr, stream>>>(shmem_signal, result, signalValue, cmpValue);
    status = aclrtSynchronizeStream(stream);

    #if defined(ASCEND_SHMEM)
        shmem_barrier_all();
    #elif defined(CANN_SHMEM)
        aclshmem_barrier_all();
    #endif

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

    pto::comm::ContextManager::SymmetricFree(shmem_signal);
    pto::comm::ContextManager::SymmetricFree(result);
    pto::comm::ContextManager::Finalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

bool RunTTestPollingTimeoutKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
{
    int32_t ret = shmem_set_conf_store_tls(false, nullptr, 0);
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

    ShmemEnv env;
    const char *ip = "tcp://127.0.0.1:8800";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;

    if (!ShmemInitFromEnv(env)) {
        std::cerr << "ShmemInitFromEnv failed!" << std::endl;
        return false;
    }

    int32_t *shmem_signal = (int32_t *)pto::comm::ContextManager::SymmetricAlloc(sizeof(int32_t));
    int32_t *poll_count = (int32_t *)pto::comm::ContextManager::SymmetricAlloc(sizeof(int32_t));
    int32_t *final_result = (int32_t *)pto::comm::ContextManager::SymmetricAlloc(sizeof(int32_t));

    int32_t zero = 0;
    aclrtMemcpy(shmem_signal, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(poll_count, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(final_result, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    #if defined(ASCEND_SHMEM)
        shmem_barrier_all();
    #elif defined(CANN_SHMEM)
        aclshmem_barrier_all();
    #endif

    TTestPollingTimeoutKernel<<<1, nullptr, stream>>>(shmem_signal, poll_count, final_result);
    status = aclrtSynchronizeStream(stream);

    #if defined(ASCEND_SHMEM)
        shmem_barrier_all();
    #elif defined(CANN_SHMEM)
        aclshmem_barrier_all();
    #endif

    bool is_ok = true;

    if (rank_id == 1) {
        int32_t count = 0;
        int32_t found = 0;
        aclrtMemcpy(&count, sizeof(int32_t), poll_count, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(&found, sizeof(int32_t), final_result, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

        if (found != 1) {
            std::cerr << "TTest Polling Timeout test failed! Should find signal, found=" << found << std::endl;
            is_ok = false;
        } else if (count <= 0 || count > 100000) {
            std::cerr << "TTest Polling Timeout test failed! Poll count out of range: " << count << std::endl;
            is_ok = false;
        } else {
            std::cout << "Rank 1: TTEST polling found signal after " << count << " iterations" << std::endl;
        }
    }

    pto::comm::ContextManager::SymmetricFree(shmem_signal);
    pto::comm::ContextManager::SymmetricFree(poll_count);
    pto::comm::ContextManager::SymmetricFree(final_result);
    pto::comm::ContextManager::Finalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

bool RunTTestNEKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
{
    int32_t ret = shmem_set_conf_store_tls(false, nullptr, 0);
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

    ShmemEnv env;
    const char *ip = "tcp://127.0.0.1:8801";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;

    if (!ShmemInitFromEnv(env)) {
        std::cerr << "ShmemInitFromEnv failed!" << std::endl;
        return false;
    }

    int32_t *shmem_signal = (int32_t *)pto::comm::ContextManager::SymmetricAlloc(sizeof(int32_t));
    int32_t *result = (int32_t *)pto::comm::ContextManager::SymmetricAlloc(sizeof(int32_t));

    int32_t zero = 0;
    aclrtMemcpy(shmem_signal, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(result, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    #if defined(ASCEND_SHMEM)
        shmem_barrier_all();
    #elif defined(CANN_SHMEM)
        aclshmem_barrier_all();
    #endif

    TTestNEKernel<<<1, nullptr, stream>>>(shmem_signal, result);
    status = aclrtSynchronizeStream(stream);

    #if defined(ASCEND_SHMEM)
        shmem_barrier_all();
    #elif defined(CANN_SHMEM)
        aclshmem_barrier_all();
    #endif

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

    pto::comm::ContextManager::SymmetricFree(shmem_signal);
    pto::comm::ContextManager::SymmetricFree(result);
    pto::comm::ContextManager::Finalize();

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
            const bool ok = RunTTestPollingTimeoutKernel(first_rank_id + r, n_ranks, n_devices, first_device_id);
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

// Explicit template instantiations
template bool RunTTestCompare<pto::comm::WaitCmp::GE>(int, int, int, int, int32_t, int32_t, bool);
template bool RunTTestCompare<pto::comm::WaitCmp::GT>(int, int, int, int, int32_t, int32_t, bool);
template bool RunTTestCompare<pto::comm::WaitCmp::LE>(int, int, int, int, int32_t, int32_t, bool);
template bool RunTTestCompare<pto::comm::WaitCmp::LT>(int, int, int, int, int32_t, int32_t, bool);
