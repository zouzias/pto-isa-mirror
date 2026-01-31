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

    if (my_rank == 0) {
        // Rank 0: Send signal to rank 1
        __gm__ int32_t *remote_signal = ShmemPtr(shmem_signal, 1);
        GSignal targetSignal(remote_signal, shape, stride);

        // Set signal value to 42
        pto::comm::TNOTIFY(targetSignal, 42, pto::comm::NotifyOp::Set);
        ShmemDeviceQuiet();
        ShmemDeviceBarrierAll();
    } else if (my_rank == 1) {
        ShmemDeviceBarrierAll();
        // Rank 1: Wait for signal to equal 42
        GSignal localSignal(shmem_signal, shape, stride);

        // Blocking wait until signal == 42
        pto::comm::TWAIT(localSignal, 42, pto::comm::WaitCmp::EQ);
    }
    // Global synchronization
    ShmemDeviceQuiet();
    ShmemDeviceBarrierAll();
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
        __gm__ int32_t *remote_signal = ShmemPtr(shmem_signal, 1);
        GSignal targetSignal(remote_signal, shape, stride);

        pto::comm::TNOTIFY(targetSignal, notifyValue, pto::comm::NotifyOp::Set);
        ShmemDeviceQuiet();
    } else if (my_rank == 1) {
        // Rank 1: Wait for signal >= 100
        GSignal localSignal(shmem_signal, shape, stride);

        // Blocking wait until signal >= 100
        pto::comm::TWAIT(localSignal, 100, pto::comm::WaitCmp::GE);
    }

    ShmemDeviceQuiet();
    ShmemDeviceBarrierAll();
}

// ============================================================================
// Kernel 3: TWAIT with multi-rank atomic add
// Multiple ranks atomically add to rank 0's counter, rank 0 waits for threshold
// ============================================================================
__global__ AICORE void TWaitAtomicKernel(__gm__ int32_t *shmem_counter, int threshold, int iters)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GSignal = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, 1);
    StrideDyn stride(1, 1, 1, 1, 1);

    int my_rank = shmem_my_pe();

    __gm__ int32_t *remote_counter = ShmemPtr(shmem_counter, 0);
    GSignal counterSignal(remote_counter, shape, stride);

    ShmemDeviceBarrierAll();

    if (my_rank != 0) {
        // Non-rank-0: atomically add multiple times
        for (int i = 0; i < iters; ++i) {
            pto::comm::TNOTIFY(counterSignal, 1, pto::comm::NotifyOp::AtomicAdd);
        }
        ShmemDeviceQuiet();
        ShmemDeviceBarrierAll();
    } else {
        // Rank 0: Wait until counter >= threshold
        GSignal localCounter(shmem_counter, shape, stride);
        ShmemDeviceBarrierAll();
        pto::comm::TWAIT(localCounter, threshold, pto::comm::WaitCmp::GE);
    }

    ShmemDeviceQuiet();
    ShmemDeviceBarrierAll();
}

// ============================================================================
// Kernel 4: TWAIT 2D Signal Matrix
// Rank 0 sets a 2D signal matrix for rank 1, rank 1 waits on the matrix
// ============================================================================
template <int Rows, int Cols>
__global__ AICORE void TWaitMatrixKernel(__gm__ int32_t *shmem_matrix)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GSignal = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;

    ShapeDyn shape(1, 1, 1, Rows, Cols);
    StrideDyn stride(Rows * Cols, Rows * Cols, Rows * Cols, Cols, 1);
    ShapeDyn sigShape(1, 1, 1, 1, 1);
    StrideDyn sigStride(1, 1, 1, 1, 1);

    int my_rank = shmem_my_pe();

    if (my_rank == 0) {
        __gm__ int32_t *remote_matrix = ShmemPtr(shmem_matrix, 1);
        for (int r = 0; r < Rows; ++r) {
            for (int c = 0; c < Cols; ++c) {
                __gm__ int32_t *remote_elem = remote_matrix + r * Cols + c;
                GSignal targetElem(remote_elem, sigShape, sigStride);
                pto::comm::TNOTIFY(targetElem, 1, pto::comm::NotifyOp::Set);
            }
        }
    } else if (my_rank == 1) {
        GSignal localMatrix(shmem_matrix, shape, stride);
        pto::comm::TWAIT(localMatrix, 1, pto::comm::WaitCmp::EQ);
    }

    ShmemDeviceQuiet();
    ShmemDeviceBarrierAll();
}

// ============================================================================
// Kernel 5: TWAIT Multi-Phase
// Rank 0 updates signal in phases, rank 1 waits in phases
// ============================================================================
__global__ AICORE void TWaitMultiPhaseKernel(__gm__ int32_t *shmem_signal)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GSignal = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, 1);
    StrideDyn stride(1, 1, 1, 1, 1);

    int my_rank = shmem_my_pe();

    if (my_rank == 0) {
        __gm__ int32_t *remote_signal = ShmemPtr(shmem_signal, 1);
        GSignal targetSignal(remote_signal, shape, stride);

        pto::comm::TNOTIFY(targetSignal, 1, pto::comm::NotifyOp::Set);
        ShmemDeviceQuiet();
        ShmemDeviceBarrierAll();

        pto::comm::TNOTIFY(targetSignal, 3, pto::comm::NotifyOp::Set);
        ShmemDeviceQuiet();
        ShmemDeviceBarrierAll();

        pto::comm::TNOTIFY(targetSignal, 5, pto::comm::NotifyOp::Set);
        ShmemDeviceQuiet();
        ShmemDeviceBarrierAll();
    } else if (my_rank == 1) {
        GSignal localSignal(shmem_signal, shape, stride);

        ShmemDeviceBarrierAll();
        pto::comm::TWAIT(localSignal, 1, pto::comm::WaitCmp::EQ);

        ShmemDeviceBarrierAll();
        pto::comm::TWAIT(localSignal, 3, pto::comm::WaitCmp::GE);

        ShmemDeviceBarrierAll();
        pto::comm::TWAIT(localSignal, 5, pto::comm::WaitCmp::EQ);

        ShmemDeviceBarrierAll();
    }

    ShmemDeviceQuiet();
    ShmemDeviceBarrierAll();
}

// ============================================================================
// Host-side Test Implementation
// ============================================================================

bool RunTWaitBasicKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
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
    const char *ip = "tcp://127.0.0.1:8780";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;
    env.heapBytes = 8ULL * 1024 * 1024;

    if (!ShmemInitFromEnv(env)) {
        std::cerr << "[ERROR] ShmemInitFromEnv failed!" << std::endl;
        return false;
    }

    // Allocate symmetric memory for signal
    int32_t *shmem_signal = (int32_t *)ShmemMalloc(sizeof(int32_t));
    if (shmem_signal == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    // Initialize signal to 0
    int32_t zero = 0;
    aclrtMemcpy(shmem_signal, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    // Wait for all ranks to complete initialization
    ShmemBarrierAll();

    // Execute kernel
    TWaitBasicKernel<<<1, nullptr, stream>>>(shmem_signal);
    status = aclrtSynchronizeStream(stream);

    ShmemBarrierAll();

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

    ShmemFree(shmem_signal);
    ShmemFinalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

bool RunTWaitCompareKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, int32_t notifyValue)
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
    const char *ip = "tcp://127.0.0.1:8781";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;
    env.heapBytes = 8ULL * 1024 * 1024;

    if (!ShmemInitFromEnv(env)) {
        std::cerr << "[ERROR] ShmemInitFromEnv failed!" << std::endl;
        return false;
    }

    int32_t *shmem_signal = (int32_t *)ShmemMalloc(sizeof(int32_t));
    if (shmem_signal == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    int32_t zero = 0;
    aclrtMemcpy(shmem_signal, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    ShmemBarrierAll();

    TWaitCompareKernel<<<1, nullptr, stream>>>(shmem_signal, notifyValue);
    status = aclrtSynchronizeStream(stream);

    ShmemBarrierAll();

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

    ShmemFree(shmem_signal);
    ShmemFinalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

bool RunTWaitAtomicKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
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
    const char *ip = "tcp://127.0.0.1:8782";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;
    env.heapBytes = 8ULL * 1024 * 1024;

    if (!ShmemInitFromEnv(env)) {
        std::cerr << "[ERROR] ShmemInitFromEnv failed!" << std::endl;
        return false;
    }

    int32_t *shmem_counter = (int32_t *)ShmemMalloc(sizeof(int32_t));
    if (shmem_counter == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    int32_t zero = 0;
    aclrtMemcpy(shmem_counter, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    ShmemBarrierAll();

    constexpr int kAtomicIters = 50;
    const int threshold = (n_ranks - 1) * kAtomicIters;
    TWaitAtomicKernel<<<1, nullptr, stream>>>(shmem_counter, threshold, kAtomicIters);
    status = aclrtSynchronizeStream(stream);

    ShmemBarrierAll();

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

    ShmemFree(shmem_counter);
    ShmemFinalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

template <int Rows, int Cols>
bool RunTWaitMatrixKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
{
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

    ShmemEnv env;
    const char *ip = "tcp://127.0.0.1:8783";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;
    env.heapBytes = 8ULL * 1024 * 1024;

    if (!ShmemInitFromEnv(env)) {
        std::cerr << "[ERROR] ShmemInitFromEnv failed!" << std::endl;
        return false;
    }

    constexpr size_t total = Rows * Cols;
    int32_t *shmem_matrix = (int32_t *)ShmemMalloc(total * sizeof(int32_t));
    if (shmem_matrix == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    std::vector<int32_t> zeros(total, 0);
    aclrtMemcpy(shmem_matrix, total * sizeof(int32_t), zeros.data(), total * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    ShmemBarrierAll();

    TWaitMatrixKernel<Rows, Cols><<<1, nullptr, stream>>>(shmem_matrix);
    status = aclrtSynchronizeStream(stream);

    ShmemBarrierAll();

    bool is_ok = true;
    if (rank_id == 1) {
        std::vector<int32_t> result(total, 0);
        aclrtMemcpy(result.data(), total * sizeof(int32_t), shmem_matrix, total * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
        for (size_t i = 0; i < total; ++i) {
            if (result[i] != 1) {
                std::cerr << "TWait Matrix test failed at " << i << " got " << result[i] << std::endl;
                is_ok = false;
                break;
            }
        }
    }

    ShmemFree(shmem_matrix);
    ShmemFinalize();

    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();

    return (status == 0) && is_ok;
}

bool RunTWaitMultiPhaseKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
{
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

    ShmemEnv env;
    const char *ip = "tcp://127.0.0.1:8786";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;
    env.heapBytes = 8ULL * 1024 * 1024;

    if (!ShmemInitFromEnv(env)) {
        std::cerr << "[ERROR] ShmemInitFromEnv failed!" << std::endl;
        return false;
    }

    int32_t *shmem_signal = (int32_t *)ShmemMalloc(sizeof(int32_t));
    if (shmem_signal == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    int32_t zero = 0;
    aclrtMemcpy(shmem_signal, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    ShmemBarrierAll();

    TWaitMultiPhaseKernel<<<1, nullptr, stream>>>(shmem_signal);
    status = aclrtSynchronizeStream(stream);

    ShmemBarrierAll();

    bool is_ok = true;
    if (rank_id == 1) {
        int32_t result = 0;
        aclrtMemcpy(&result, sizeof(int32_t), shmem_signal, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
        if (result != 5) {
            std::cerr << "TWait MultiPhase failed! Expected: 5, Got: " << result << std::endl;
            is_ok = false;
        }
    }

    ShmemFree(shmem_signal);
    ShmemFinalize();

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

template <int Rows, int Cols>
bool RunTWaitMatrix(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    std::vector<pid_t> pids;
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            const bool ok = RunTWaitMatrixKernel<Rows, Cols>(first_rank_id + r, n_ranks, n_devices, first_device_id);
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

bool RunTWaitMultiPhase(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    std::vector<pid_t> pids;
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            const bool ok = RunTWaitMultiPhaseKernel(first_rank_id + r, n_ranks, n_devices, first_device_id);
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

template bool RunTWaitMatrix<4, 8>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunTWaitMatrix<7, 13>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
