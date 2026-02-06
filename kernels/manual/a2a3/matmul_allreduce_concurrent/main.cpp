/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "matmul_allreduce_concurrent.h"

// ============================================================================
// Command Line Argument Parser
// ============================================================================
struct TestArgs {
    int n_ranks = 2;
    int n_devices = 2;
    int first_rank_id = 0;
    int first_device_id = 0;
    int warmup_iters = 10;
    int measure_iters = 10;
    bool verbose = true;
    bool help = false;
};

void PrintUsage(const char *prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "\n"
              << "MATMUL + ALLREDUCE Concurrent Demo\n"
              << "===============================\n"
              << "This demo demonstrates concurrent GEMM computation with AllReduce\n"
              << "communication using different AI cores.\n"
              << "\n"
              << "Configuration:\n"
              << "  - Total cores:      " << TOTAL_BLOCK_NUM << "\n"
              << "  - GEMM cores:       " << GEMM_BLOCK_NUM << " (cores 0-" << (GEMM_BLOCK_NUM-1) << ")\n"
              << "  - AllReduce cores:  " << ALLREDUCE_BLOCK_NUM << " (cores 0-" << (ALLREDUCE_BLOCK_NUM-1) << ")\n"
              << "\n"
              << "Options:\n"
              << "  -r, --ranks N       Number of ranks (default: 8)\n"
              << "  -d, --devices N     Number of devices (default: 8)\n"
              << "  -w, --warmup N      Warmup iterations (default: 5)\n"
              << "  -m, --measure N     Measurement iterations (default: 10)\n"
              << "  -q, --quiet         Disable verbose output\n"
              << "  -h, --help          Show this help message\n"
              << std::endl;
}

TestArgs ParseArgs(int argc, char **argv) {
    TestArgs args;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            args.help = true;
        } else if ((arg == "-r" || arg == "--ranks") && i + 1 < argc) {
            args.n_ranks = std::atoi(argv[++i]);
        } else if ((arg == "-d" || arg == "--devices") && i + 1 < argc) {
            args.n_devices = std::atoi(argv[++i]);
        } else if ((arg == "-w" || arg == "--warmup") && i + 1 < argc) {
            args.warmup_iters = std::atoi(argv[++i]);
        } else if ((arg == "-m" || arg == "--measure") && i + 1 < argc) {
            args.measure_iters = std::atoi(argv[++i]);
        } else if (arg == "-q" || arg == "--quiet") {
            args.verbose = false;
        }
    }
    
    return args;
}

// ============================================================================
// Main Entry Point
// ============================================================================
int main(int argc, char **argv) {
    TestArgs args = ParseArgs(argc, argv);
    
    if (args.help) {
        PrintUsage(argv[0]);
        return 0;
    }
    
    // Print test configuration
    std::cout << "============================================================" << std::endl;
    std::cout << "  MATMUL + ALLREDUCE Concurrent Demo" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "  This demo runs GEMM and AllReduce in parallel on different" << std::endl;
    std::cout << "  AI cores to demonstrate compute communication concurrent." << std::endl;
    std::cout << std::endl;
    std::cout << "  Configuration:" << std::endl;
    std::cout << "    Total cores:      " << TOTAL_BLOCK_NUM << std::endl;
    std::cout << "    GEMM cores:       " << GEMM_BLOCK_NUM << std::endl;
    std::cout << "    AllReduce cores:  " << ALLREDUCE_BLOCK_NUM << std::endl;
    std::cout << "    Ranks:            " << args.n_ranks << std::endl;
    std::cout << "    Devices:          " << args.n_devices << std::endl;
    std::cout << "    Warmup iters:     " << args.warmup_iters << std::endl;
    std::cout << "    Measurement iters:" << args.measure_iters << std::endl;
    std::cout << "    Verbose:          " << (args.verbose ? "yes" : "no") << std::endl;
    std::cout << "============================================================\n" << std::endl;
    
    ConcurrentTestConfig config;
    config.warmup_iters = args.warmup_iters;
    config.measure_iters = args.measure_iters;
    config.verbose = args.verbose;
    
    bool success = RunMatmulAllReduceConcurrent(
        args.n_ranks, 
        args.n_devices, 
        args.first_rank_id, 
        args.first_device_id,
        config);
    
    // Final summary
    std::cout << "\n============================================================" << std::endl;
    std::cout << "  Demo " << (success ? "PASSED" : "FAILED") << std::endl;
    std::cout << "============================================================\n" << std::endl;
    
    return success ? 0 : 1;
}
