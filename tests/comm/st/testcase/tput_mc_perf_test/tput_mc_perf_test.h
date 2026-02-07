
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

#define TPUT_BLOCK_NUM 32

// ============================================================================
// Performance Test Configuration
// ============================================================================
struct PerfTestConfig {
    int warmup_iters = 5;        // Warmup kernel launches (full kernel with loops)
    int repeat_iters = 20;       // TPUT repeat iterations inside kernel for steady-state
    int total_rows = 4096;       // Total rows in tensor
    int total_cols = 4096;       // Total cols in tensor
    int block_num = TPUT_BLOCK_NUM;
    bool verbose = true;
};

// ============================================================================
// Forward Declarations
// ============================================================================

template <typename T, int kTRows_, int kTCols_>
bool RunPutAutoChunkPerf(int n_ranks, int n_devices, int first_rank_id, int first_device_id,
                         const PerfTestConfig &config);

template <typename T, int kTRows_, int kTCols_>
bool RunPutPingPongPerf(int n_ranks, int n_devices, int first_rank_id, int first_device_id,
                        const PerfTestConfig &config);
