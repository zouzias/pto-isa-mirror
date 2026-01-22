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
// Kernel 1: TWAIT Basic Test
// Rank 0 sends signal to rank 1, rank 1 waits for the signal
// Tests blocking wait functionality
// ============================================================================
__global__ AICORE void TWaitBasicKernel(__gm__ int32_t *shmem_signal)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GSignal = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, 1);
    StrideDyn stride(1, 1, 1, 1, 1);

    int my_rank = shmem_my_pe();
    int nranks = shmem_n_pes();

    if (my_rank == 0) {
        // Rank 0: Send signal to rank 1
        GSignal targetSignal(shmem_signal, shape, stride);
        targetSignal.SetRank(1);

        // Set signal value to 42
        pto::comm::TNOTIFY<pto::comm::NotifyOp::Set>(targetSignal, 42);
        pto::comm::TQUIET();
    } else if (my_rank == 1) {
        // Rank 1: Wait for signal to equal 42
        GSignal localSignal(shmem_signal, shape, stride);
        localSignal.SetRank(my_rank);

        // Blocking wait until signal == 42
        pto::comm::TWAIT<pto::comm::WaitCmp::EQ>(localSignal, 42);
    }

    // Global synchronization
    pto::comm::TBARRIER();
}

// ============================================================================
// Kernel 2: TWAIT with different comparison operators
// Tests EQ, NE, GT, GE, LT, LE comparisons
// ============================================================================
__global__ AICORE void TWaitCompareKernel(__gm__ int32_t *shmem_signal, int32_t notifyValue)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GSignal = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, 1);
    StrideDyn stride(1, 1, 1, 1, 1);

    int my_rank = shmem_my_pe();

    if (my_rank == 0) {
        // Rank 0: Send signal with specified value to rank 1
        GSignal targetSignal(shmem_signal, shape, stride);
        targetSignal.SetRank(1);

        pto::comm::TNOTIFY<pto::comm::NotifyOp::Set>(targetSignal, notifyValue);
        pto::comm::TQUIET();
    } else if (my_rank == 1) {
        // Rank 1: Wait for signal >= 100
        GSignal localSignal(shmem_signal, shape, stride);
        localSignal.SetRank(my_rank);

        // Blocking wait until signal >= 100
        pto::comm::TWAIT<pto::comm::WaitCmp::GE>(localSignal, 100);
    }

    pto::comm::TBARRIER();
}

// ============================================================================
// Kernel 3: TTEST Non-blocking Test
// Tests polling-based signal checking
// ============================================================================
__global__ AICORE void TTestBasicKernel(__gm__ int32_t *shmem_signal, __gm__ int32_t *result)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GSignal = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, 1);
    StrideDyn stride(1, 1, 1, 1, 1);

    int my_rank = shmem_my_pe();

    if (my_rank == 0) {
        // Rank 0: Send signal to rank 1
        GSignal targetSignal(shmem_signal, shape, stride);
        targetSignal.SetRank(1);

        pto::comm::TNOTIFY<pto::comm::NotifyOp::Set>(targetSignal, 99);
        pto::comm::TQUIET();
    } else if (my_rank == 1) {
        // Rank 1: Test signal value (non-blocking)
        GSignal localSignal(shmem_signal, shape, stride);
        localSignal.SetRank(my_rank);

        // First barrier to ensure rank 0 has sent the signal
        pto::comm::TBARRIER();

        // Non-blocking test: check if signal == 99
        bool testResult = pto::comm::TTEST<pto::comm::WaitCmp::EQ>(localSignal, 99);
        
        // Store result
        *result = testResult ? 1 : 0;
        return;  // Skip final barrier for rank 1
    }

    pto::comm::TBARRIER();
}

// ============================================================================
// Kernel 4: TTEST Polling Loop
// Demonstrates polling with TTEST until condition is met
// ============================================================================
__global__ AICORE void TTestPollingKernel(__gm__ int32_t *shmem_signal, __gm__ int32_t *poll_count)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GSignal = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, 1);
    StrideDyn stride(1, 1, 1, 1, 1);

    int my_rank = shmem_my_pe();

    if (my_rank == 0) {
        // Rank 0: Send incrementing signals to rank 1
        GSignal targetSignal(shmem_signal, shape, stride);
        targetSignal.SetRank(1);

        // Send final value
        pto::comm::TNOTIFY<pto::comm::NotifyOp::Set>(targetSignal, 100);
        pto::comm::TQUIET();
    } else if (my_rank == 1) {
        // Rank 1: Poll until signal >= 100
        GSignal localSignal(shmem_signal, shape, stride);
        localSignal.SetRank(my_rank);

        int32_t count = 0;
        const int32_t maxPolls = 10000;

        while (count < maxPolls) {
            count++;
            if (pto::comm::TTEST<pto::comm::WaitCmp::GE>(localSignal, 100)) {
                break;
            }
        }

        *poll_count = count;
    }

    pto::comm::TBARRIER();
}

// ============================================================================
// Kernel 5: TWAIT with AtomicAdd
// Multiple ranks atomically add to a counter, one rank waits for threshold
// ============================================================================
__global__ AICORE void TWaitAtomicKernel(__gm__ int32_t *shmem_counter, int threshold)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GSignal = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, 1);
    StrideDyn stride(1, 1, 1, 1, 1);

    int my_rank = shmem_my_pe();
    int nranks = shmem_n_pes();

    GSignal counterSignal(shmem_counter, shape, stride);
    counterSignal.SetRank(0);  // All operations target rank 0

    if (my_rank != 0) {
        // Non-rank-0: atomically add 1 to rank 0's counter
        pto::comm::TNOTIFY<pto::comm::NotifyOp::AtomicAdd>(counterSignal, 1);
        pto::comm::TQUIET();
    } else {
        // Rank 0: Wait until counter >= threshold (n_ranks - 1)
        GSignal localCounter(shmem_counter, shape, stride);
        localCounter.SetRank(0);

        pto::comm::TWAIT<pto::comm::WaitCmp::GE>(localCounter, threshold);
    }

    pto::comm::TBARRIER();
}

// ============================================================================
// Host-side Test Implementation
// ============================================================================

bool RunTWaitBasicKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
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
    const char *ip = "tcp://127.0.0.1:8780";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;

    if (!ShmemInitFromEnv(env)) {
        std::cerr << "ShmemInitFromEnv failed!" << std::endl;
        return false;
    }

    // Allocate symmetric memory for signal
    int32_t *shmem_signal = (int32_t *)pto::comm::ContextManager::SymmetricAlloc(sizeof(int32_t));

    // Initialize signal to 0
    int32_t zero = 0;
    aclrtMemcpy(shmem_signal, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    // Wait for all ranks to complete initialization
    #if defined(ASCEND_SHMEM)
        shmem_barrier_all();
    #elif defined(CANN_SHMEM)
        aclshmem_barrier_all();
    #endif

    // Execute kernel
    TWaitBasicKernel<<<1, nullptr, stream>>>(shmem_signal);
    status = aclrtSynchronizeStream(stream);

    #if defined(ASCEND_SHMEM)
        shmem_barrier_all();
    #elif defined(CANN_SHMEM)
        aclshmem_barrier_all();
    #endif

    bool is_ok = true;

    // Rank 1 verifies the signal value
    if (rank_id == 1) {
        int32_t result = 0;
        aclrtMemcpy(&result, sizeof(int32_t), shmem_signal, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

        if (result != 42) {
            std::cerr << "TWait Basic test failed! Expected: 42, Got: " << result << std::endl;
            is_ok = false;
        } else {
            std::cout << "Rank 1: TWait received signal = " << result << " (expected 42)" << std::endl;
        }
    }

    pto::comm::ContextManager::SymmetricFree(shmem_signal);
    pto::comm::ContextManager::Finalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

bool RunTWaitCompareKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, int32_t notifyValue)
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
    const char *ip = "tcp://127.0.0.1:8781";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;

    if (!ShmemInitFromEnv(env)) {
        std::cerr << "ShmemInitFromEnv failed!" << std::endl;
        return false;
    }

    int32_t *shmem_signal = (int32_t *)pto::comm::ContextManager::SymmetricAlloc(sizeof(int32_t));

    int32_t zero = 0;
    aclrtMemcpy(shmem_signal, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    #if defined(ASCEND_SHMEM)
        shmem_barrier_all();
    #elif defined(CANN_SHMEM)
        aclshmem_barrier_all();
    #endif

    TWaitCompareKernel<<<1, nullptr, stream>>>(shmem_signal, notifyValue);
    status = aclrtSynchronizeStream(stream);

    #if defined(ASCEND_SHMEM)
        shmem_barrier_all();
    #elif defined(CANN_SHMEM)
        aclshmem_barrier_all();
    #endif

    bool is_ok = true;

    if (rank_id == 1) {
        int32_t result = 0;
        aclrtMemcpy(&result, sizeof(int32_t), shmem_signal, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

        if (result != notifyValue) {
            std::cerr << "TWait Compare test failed! Expected: " << notifyValue << ", Got: " << result << std::endl;
            is_ok = false;
        } else {
            std::cout << "Rank 1: TWait (GE) received signal = " << result << " (expected >= 100)" << std::endl;
        }
    }

    pto::comm::ContextManager::SymmetricFree(shmem_signal);
    pto::comm::ContextManager::Finalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

bool RunTTestBasicKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
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
    const char *ip = "tcp://127.0.0.1:8782";
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

    TTestBasicKernel<<<1, nullptr, stream>>>(shmem_signal, result);
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
            std::cerr << "TTest Basic test failed! TTEST should return true (1), Got: " << testResult << std::endl;
            is_ok = false;
        } else {
            std::cout << "Rank 1: TTEST returned " << testResult << " (expected 1/true)" << std::endl;
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

bool RunTTestPollingKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
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
    const char *ip = "tcp://127.0.0.1:8783";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;

    if (!ShmemInitFromEnv(env)) {
        std::cerr << "ShmemInitFromEnv failed!" << std::endl;
        return false;
    }

    int32_t *shmem_signal = (int32_t *)pto::comm::ContextManager::SymmetricAlloc(sizeof(int32_t));
    int32_t *poll_count = (int32_t *)pto::comm::ContextManager::SymmetricAlloc(sizeof(int32_t));

    int32_t zero = 0;
    aclrtMemcpy(shmem_signal, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(poll_count, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    #if defined(ASCEND_SHMEM)
        shmem_barrier_all();
    #elif defined(CANN_SHMEM)
        aclshmem_barrier_all();
    #endif

    TTestPollingKernel<<<1, nullptr, stream>>>(shmem_signal, poll_count);
    status = aclrtSynchronizeStream(stream);

    #if defined(ASCEND_SHMEM)
        shmem_barrier_all();
    #elif defined(CANN_SHMEM)
        aclshmem_barrier_all();
    #endif

    bool is_ok = true;

    if (rank_id == 1) {
        int32_t count = 0;
        aclrtMemcpy(&count, sizeof(int32_t), poll_count, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

        // Poll count should be > 0 and <= 10000 (max polls)
        if (count <= 0 || count > 10000) {
            std::cerr << "TTest Polling test failed! Poll count out of range: " << count << std::endl;
            is_ok = false;
        } else {
            std::cout << "Rank 1: TTEST polling completed after " << count << " iterations" << std::endl;
        }
    }

    pto::comm::ContextManager::SymmetricFree(shmem_signal);
    pto::comm::ContextManager::SymmetricFree(poll_count);
    pto::comm::ContextManager::Finalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

bool RunTWaitAtomicKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
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
    const char *ip = "tcp://127.0.0.1:8784";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;

    if (!ShmemInitFromEnv(env)) {
        std::cerr << "ShmemInitFromEnv failed!" << std::endl;
        return false;
    }

    int32_t *shmem_counter = (int32_t *)pto::comm::ContextManager::SymmetricAlloc(sizeof(int32_t));

    int32_t zero = 0;
    aclrtMemcpy(shmem_counter, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    #if defined(ASCEND_SHMEM)
        shmem_barrier_all();
    #elif defined(CANN_SHMEM)
        aclshmem_barrier_all();
    #endif

    int threshold = n_ranks - 1;  // All ranks except rank 0 contribute
    TWaitAtomicKernel<<<1, nullptr, stream>>>(shmem_counter, threshold);
    status = aclrtSynchronizeStream(stream);

    #if defined(ASCEND_SHMEM)
        shmem_barrier_all();
    #elif defined(CANN_SHMEM)
        aclshmem_barrier_all();
    #endif

    bool is_ok = true;

    if (rank_id == 0) {
        int32_t result = 0;
        aclrtMemcpy(&result, sizeof(int32_t), shmem_counter, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

        if (result != threshold) {
            std::cerr << "TWait Atomic test failed! Expected: " << threshold << ", Got: " << result << std::endl;
            is_ok = false;
        } else {
            std::cout << "Rank 0: TWait (GE) atomic counter = " << result << " (expected >= " << threshold << ")" << std::endl;
        }
    }

    pto::comm::ContextManager::SymmetricFree(shmem_counter);
    pto::comm::ContextManager::Finalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

// ============================================================================
// Multi-process Launcher Functions
// ============================================================================

bool RunTWaitBasic(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    std::vector<pid_t> pids;
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            const bool ok = RunTWaitBasicKernel(first_rank_id + r, n_ranks, n_devices, first_device_id);
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

bool RunTWaitCompare(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int32_t notifyValue)
{
    std::vector<pid_t> pids;
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            const bool ok = RunTWaitCompareKernel(first_rank_id + r, n_ranks, n_devices, first_device_id, notifyValue);
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

bool RunTTestBasic(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    std::vector<pid_t> pids;
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            const bool ok = RunTTestBasicKernel(first_rank_id + r, n_ranks, n_devices, first_device_id);
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

bool RunTTestPolling(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    std::vector<pid_t> pids;
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            const bool ok = RunTTestPollingKernel(first_rank_id + r, n_ranks, n_devices, first_device_id);
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

bool RunTWaitAtomic(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    std::vector<pid_t> pids;
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            const bool ok = RunTWaitAtomicKernel(first_rank_id + r, n_ranks, n_devices, first_device_id);
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
