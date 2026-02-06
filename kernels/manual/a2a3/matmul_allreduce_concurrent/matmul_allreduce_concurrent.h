/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once
#include <cstddef>
#include <cstdint>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>

// ============================================================================
// Demo Configuration: Concurrent GEMM computation with AllReduce communication
// ============================================================================

// Total number of AI cores to use
#define TOTAL_BLOCK_NUM 4

// Number of cores for GEMM computation (first half)
#define GEMM_BLOCK_NUM TOTAL_BLOCK_NUM

// Number of cores for AllReduce communication (second half)
#define ALLREDUCE_BLOCK_NUM TOTAL_BLOCK_NUM * 2

// ============================================================================
// Test Configuration
// ============================================================================
struct ConcurrentTestConfig {
    int warmup_iters = 10;
    int measure_iters = 10;
    bool verbose = true;
};

// ============================================================================
// Performance Statistics
// ============================================================================
struct ConcurrentPerfStats {
    double gemm_avg_us;
    double gemm_min_us;
    double gemm_max_us;
    double allreduce_avg_us;
    double allreduce_min_us;
    double allreduce_max_us;
    double total_avg_us;
    double concurrent_efficiency;  // How much time was saved by concurrent
};

inline ConcurrentPerfStats CalculateConcurrentStats(
    const std::vector<double> &gemm_latencies_us,
    const std::vector<double> &allreduce_latencies_us)
{
    ConcurrentPerfStats stats = {};
    
    if (gemm_latencies_us.empty() || allreduce_latencies_us.empty()) {
        return stats;
    }
    
    // GEMM stats
    std::vector<double> gemm_sorted = gemm_latencies_us;
    std::sort(gemm_sorted.begin(), gemm_sorted.end());
    stats.gemm_min_us = gemm_sorted.front();
    stats.gemm_max_us = gemm_sorted.back();
    stats.gemm_avg_us = std::accumulate(gemm_sorted.begin(), gemm_sorted.end(), 0.0) / gemm_sorted.size();
    
    // AllReduce stats
    std::vector<double> ar_sorted = allreduce_latencies_us;
    std::sort(ar_sorted.begin(), ar_sorted.end());
    stats.allreduce_min_us = ar_sorted.front();
    stats.allreduce_max_us = ar_sorted.back();
    stats.allreduce_avg_us = std::accumulate(ar_sorted.begin(), ar_sorted.end(), 0.0) / ar_sorted.size();
    
    // Total time is the max of the two (since they run in parallel)
    stats.total_avg_us = std::max(stats.gemm_avg_us, stats.allreduce_avg_us);
    
    // Concurrent efficiency: how much we saved compared to sequential
    double sequential_time = stats.gemm_avg_us + stats.allreduce_avg_us;
    stats.concurrent_efficiency = (sequential_time - stats.total_avg_us) / sequential_time * 100.0;
    
    return stats;
}

// ============================================================================
// Forward declarations
// ============================================================================

// Run the concurrent test with specified parameters
bool RunMatmulAllReduceConcurrent(
    int n_ranks, 
    int n_devices, 
    int first_rank_id, 
    int first_device_id,
    const ConcurrentTestConfig &config);
