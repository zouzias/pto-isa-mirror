
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
// Performance Test Configuration
// ============================================================================
struct PerfTestConfig {
    int warmup_iters = 20;       // Warmup iterations
    int measure_iters = 50;      // Measurement iterations
    bool verbose = true;         // Print per-iteration results
};

// ============================================================================
// Statistics Helper
// ============================================================================
struct PerfStats {
    double min_us;
    double max_us;
    double avg_us;
    double median_us;
    double std_dev_us;
    double bandwidth_gbps;   // GB/s
    double msg_rate_mops;    // Million ops/s
};

PerfStats CalculateStats(const std::vector<double> &latencies_us, size_t data_bytes) {
    PerfStats stats;
    
    if (latencies_us.empty()) {
        return stats;
    }
    
    std::vector<double> sorted = latencies_us;
    std::sort(sorted.begin(), sorted.end());
    
    stats.min_us = sorted.front();
    stats.max_us = sorted.back();
    stats.avg_us = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
    
    // Median
    size_t mid = sorted.size() / 2;
    stats.median_us = (sorted.size() % 2 == 0) 
        ? (sorted[mid - 1] + sorted[mid]) / 2.0 
        : sorted[mid];
    
    // Standard deviation
    double sq_sum = 0.0;
    for (auto v : sorted) {
        sq_sum += (v - stats.avg_us) * (v - stats.avg_us);
    }
    stats.std_dev_us = std::sqrt(sq_sum / sorted.size());
    
    // Effective bandwidth: data_bytes / latency
    stats.bandwidth_gbps = (data_bytes / (stats.avg_us * 1e-6)) / (1024.0 * 1024.0 * 1024.0);
    
    // Message rate
    stats.msg_rate_mops = 1.0 / (stats.avg_us * 1e-6) / 1e6;
    
    return stats;
}

// Forward declarations
template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
bool RunReducePerf(int n_ranks, int n_devices, int first_rank_id, int first_device_id,
                   const PerfTestConfig &config);
