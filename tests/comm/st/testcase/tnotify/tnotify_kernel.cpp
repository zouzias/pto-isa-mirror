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
// Kernel 1: AtomicAdd Test
// All ranks perform atomic add 1 to rank 0's counter
// The final counter value should equal n_ranks
// ============================================================================
__global__ AICORE void TNotifyAtomicAddKernel(__gm__ int32_t *shmem_counter)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GSignal = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, 1);
    StrideDyn stride(1, 1, 1, 1, 1);

    int my_rank = shmem_my_pe();
    int nranks = shmem_n_pes();
    int target_rank = 0;  // All ranks notify rank 0

    // Create GlobalTensor pointing to rank 0's counter
    GSignal counterSignal(shmem_counter, shape, stride);
    counterSignal.SetRank(target_rank);

    // Each rank performs atomic add 1 to rank 0's counter
    pto::comm::TNOTIFY<pto::comm::NotifyOp::AtomicAdd>(counterSignal, 1);
    pto::comm::TWAIT();

    // Global synchronization
    pto::comm::TBARRIER();
}

// ============================================================================
// Kernel 2: Set Test
// Each rank sets the next rank's signal to its own rank_id + 100
// Ring notification: rank i -> rank (i+1) % n_ranks
// ============================================================================
__global__ AICORE void TNotifySetKernel(__gm__ int32_t *shmem_signals)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GSignal = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, 1);
    StrideDyn stride(1, 1, 1, 1, 1);

    int my_rank = shmem_my_pe();
    int nranks = shmem_n_pes();
    int next_rank = (my_rank + 1) % nranks;

    // Create GlobalTensor pointing to next rank's signal
    GSignal nextSignal(shmem_signals, shape, stride);
    nextSignal.SetRank(next_rank);

    // Set next rank's signal to own rank_id + 100
    int32_t value = static_cast<int32_t>(my_rank + 100);
    pto::comm::TNOTIFY<pto::comm::NotifyOp::Set>(nextSignal, value);
    pto::comm::TWAIT();

    // Global synchronization
    pto::comm::TBARRIER();
}

// ============================================================================
// Kernel 3: Scoreboard Test
// Each rank notifies its corresponding slot in rank 0's scoreboard
// scoreboard[rank_id] = rank_id + 1000
// ============================================================================
template <size_t numSlots>
__global__ AICORE void TNotifyScoreboardKernel(__gm__ int32_t *shmem_scoreboard)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GSignal = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, 1);
    StrideDyn stride(1, 1, 1, 1, 1);

    int my_rank = shmem_my_pe();
    int target_rank = 0;

    // Calculate own slot offset in scoreboard
    __gm__ int32_t *my_slot = shmem_scoreboard + my_rank;

    // Create GlobalTensor pointing to specific slot in rank 0's scoreboard
    GSignal slotSignal(my_slot, shape, stride);
    slotSignal.SetRank(target_rank);

    // Set own slot value
    int32_t value = static_cast<int32_t>(my_rank + 1000);
    pto::comm::TNOTIFY<pto::comm::NotifyOp::Set>(slotSignal, value);
    pto::comm::TWAIT();

    // Global synchronization
    pto::comm::TBARRIER();
}

// ============================================================================
// Host-side Test Implementation
// ============================================================================

bool RunNotifyAtomicAddKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
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
    const char *ip = "tcp://127.0.0.1:8767";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;

    if (!ShmemInitFromEnv(env)) {
        std::cerr << "ShmemInitFromEnv failed!" << std::endl;
        return false;
    }

    // Allocate symmetric memory as counter
    int32_t *shmem_counter = (int32_t *)pto::comm::ContextManager::SymmetricAlloc(sizeof(int32_t));

    // Initialize counter to 0 (only on rank 0)
    if (rank_id == 0) {
        int32_t zero = 0;
        aclrtMemcpy(shmem_counter, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    }

    // Wait for all ranks to complete initialization
    #if defined(ASCEND_SHMEM)
        shmem_barrier_all();
    #elif defined(CANN_SHMEM)
        aclshmem_barrier_all();
    #endif

    // Execute kernel
    TNotifyAtomicAddKernel<<<1, nullptr, stream>>>(shmem_counter);
    status = aclrtSynchronizeStream(stream);

    // Host-side global synchronization to ensure all ranks' kernels have completed
    #if defined(ASCEND_SHMEM)
        shmem_barrier_all();
    #elif defined(CANN_SHMEM)
        aclshmem_barrier_all();
    #endif

    bool is_ok = true;

    // Only rank 0 verifies the result
    if (rank_id == 0) {
        int32_t result = 0;
        aclrtMemcpy(&result, sizeof(int32_t), shmem_counter, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

        if (result != static_cast<int32_t>(n_ranks)) {
            std::cerr << "AtomicAdd test failed! Expected: " << n_ranks << ", Got: " << result << std::endl;
            is_ok = false;
        } else {
            std::cout << "Rank 0: AtomicAdd counter = " << result << " (expected " << n_ranks << ")" << std::endl;
        }
    }

    pto::comm::ContextManager::SymmetricFree(shmem_counter);
    pto::comm::ContextManager::Finalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

bool RunNotifySetKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
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
    const char *ip = "tcp://127.0.0.1:8768";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;

    if (!ShmemInitFromEnv(env)) {
        std::cerr << "ShmemInitFromEnv failed!" << std::endl;
        return false;
    }

    // Allocate symmetric memory as signal
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
    TNotifySetKernel<<<1, nullptr, stream>>>(shmem_signal);
    status = aclrtSynchronizeStream(stream);

    // Host-side global synchronization to ensure all ranks' kernels have completed
    #if defined(ASCEND_SHMEM)
        shmem_barrier_all();
    #elif defined(CANN_SHMEM)
        aclshmem_barrier_all();
    #endif

    bool is_ok = true;

    // Verify: each rank's signal should equal previous rank's id + 100
    int prev_rank = (rank_id + n_ranks - 1) % n_ranks;
    int32_t expected = static_cast<int32_t>(prev_rank + 100);

    int32_t result = 0;
    aclrtMemcpy(&result, sizeof(int32_t), shmem_signal, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

    if (result != expected) {
        std::cerr << "Rank " << rank_id << ": Set test failed! Expected: " << expected << ", Got: " << result << std::endl;
        is_ok = false;
    } else {
        std::cout << "Rank " << rank_id << ": Set signal = " << result << " (expected " << expected << ")" << std::endl;
    }

    pto::comm::ContextManager::SymmetricFree(shmem_signal);
    pto::comm::ContextManager::Finalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

template <size_t numSlots>
bool RunNotifyScoreboardKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
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
    // Use different ports to avoid conflicts: 8769 + numSlots
    char ipPort[64];
    snprintf(ipPort, sizeof(ipPort), "tcp://127.0.0.1:%d", 8769 + static_cast<int>(numSlots));
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ipPort;

    if (!ShmemInitFromEnv(env)) {
        std::cerr << "ShmemInitFromEnv failed!" << std::endl;
        return false;
    }

    // Allocate symmetric memory as scoreboard
    int32_t *shmem_scoreboard = (int32_t *)pto::comm::ContextManager::SymmetricAlloc(numSlots * sizeof(int32_t));

    // Initialize scoreboard to 0
    std::vector<int32_t> zeros(numSlots, 0);
    aclrtMemcpy(shmem_scoreboard, numSlots * sizeof(int32_t), zeros.data(), numSlots * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    // Wait for all ranks to complete initialization
    #if defined(ASCEND_SHMEM)
        shmem_barrier_all();
    #elif defined(CANN_SHMEM)
        aclshmem_barrier_all();
    #endif

    // Execute kernel
    TNotifyScoreboardKernel<numSlots><<<1, nullptr, stream>>>(shmem_scoreboard);
    status = aclrtSynchronizeStream(stream);

    bool is_ok = true;

    // Only rank 0 verifies scoreboard, using busy-wait polling + timeout mechanism
    if (rank_id == 0) {
        constexpr int TIMEOUT_MS = 5000;  // Timeout 5 seconds
        constexpr int POLL_INTERVAL_US = 1000;  // Poll interval 1ms
        const int max_polls = (TIMEOUT_MS * 1000) / POLL_INTERVAL_US;

        std::vector<bool> slot_verified(n_ranks, false);
        int verified_count = 0;

        for (int poll = 0; poll < max_polls && verified_count < n_ranks; ++poll) {
            std::vector<int32_t> results(numSlots);
            aclrtMemcpy(results.data(), numSlots * sizeof(int32_t), shmem_scoreboard, numSlots * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

            for (int i = 0; i < n_ranks && i < static_cast<int>(numSlots); ++i) {
                if (slot_verified[i]) continue;  // Skip already verified slots

                int32_t expected = static_cast<int32_t>(i + 1000);
                if (results[i] == expected) {
                    slot_verified[i] = true;
                    verified_count++;
                    std::cout << "Scoreboard[" << i << "] = " << results[i] << " (expected " << expected << ")" << std::endl;
                }
            }

            if (verified_count < n_ranks) {
                usleep(POLL_INTERVAL_US);
            }
        }

        // Check if all slots are verified successfully
        for (int i = 0; i < n_ranks && i < static_cast<int>(numSlots); ++i) {
            if (!slot_verified[i]) {
                // Read one more time to output actual value for debugging
                std::vector<int32_t> final_results(numSlots);
                aclrtMemcpy(final_results.data(), numSlots * sizeof(int32_t), shmem_scoreboard, numSlots * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
                
                int32_t expected = static_cast<int32_t>(i + 1000);
                std::cerr << "Scoreboard slot " << i << " timeout after " << TIMEOUT_MS << "ms! Expected: " 
                          << expected << ", Got: " << final_results[i] << std::endl;
                is_ok = false;
            }
        }
    }

    // Non-rank-0 processes wait for a while to ensure rank 0 has enough time to verify
    // Avoid releasing resources too early which would cause rank 0 read failures
    if (rank_id != 0) {
        usleep(100000);  // Wait 100ms
    }

    pto::comm::ContextManager::SymmetricFree(shmem_scoreboard);
    pto::comm::ContextManager::Finalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

// ============================================================================
// Multi-process Launcher Functions
// ============================================================================

bool RunNotifyAtomicAdd(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    std::vector<pid_t> pids;
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            const bool ok = RunNotifyAtomicAddKernel(first_rank_id + r, n_ranks, n_devices, first_device_id);
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

bool RunNotifySet(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    std::vector<pid_t> pids;
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            const bool ok = RunNotifySetKernel(first_rank_id + r, n_ranks, n_devices, first_device_id);
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

template <size_t numSlots>
bool RunNotifyScoreboard(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    std::vector<pid_t> pids;
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            const bool ok = RunNotifyScoreboardKernel<numSlots>(first_rank_id + r, n_ranks, n_devices, first_device_id);
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

// Explicit instantiation
template bool RunNotifyScoreboard<4>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunNotifyScoreboard<8>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
