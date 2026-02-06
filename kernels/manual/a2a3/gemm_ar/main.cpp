/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

#include "acl/acl.h"
#include "common.hpp"
#include "test_common.h"
#include "gemm_config.h"

// ============================================================================
// External Kernel Launch Function (defined in gemm_ar_kernel.cpp)
// ============================================================================
template <typename T>
void LaunchGEMME2E(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t* shmem, void *stream, bool is_overlap);


// ============================================================================
// Command Line Argument Parser
// ============================================================================
struct TestArgs {
    int n_ranks = 4;
    int n_devices = 4;
    int first_rank_id = 0;
    int first_device_id = 0;
    bool verbose = true;
    bool help = false;
};

void PrintUsage(const char *prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "\n"
              << "GEMM with AllReduce Demo\n"
              << "========================\n"
              << "\n"
              << "GEMM Configuration:\n"
              << "  - Global shape:        [" << GEMM_M << " x " << GEMM_K 
              << "] * [" << GEMM_K << " x " << GEMM_N << "]\n"
              << "  - Single core shape:   [" << SINGLE_CORE_M << ", " << SINGLE_CORE_K 
              << ", " << SINGLE_CORE_N << "]\n"
              << "  - Base tile:           [" << BASE_M << ", " << BASE_K 
              << ", " << BASE_N << "]\n"
              << "  - Block dim:           " << BLOCK_DIM << "\n"
              << "\n"
              << "Options:\n"
              << "  -r, --ranks N       Number of ranks (default: 2)\n"
              << "  -d, --devices N     Number of devices (default: 2)\n"
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
        } else if (arg == "-q" || arg == "--quiet") {
            args.verbose = false;
        }
    }
    
    return args;
}

// ============================================================================
// Single Rank GEMM Execution
// ============================================================================
bool RunGemmKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, bool verbose)
{
    int32_t ret = shmem_set_conf_store_tls(false, nullptr, 0);
    if (ret != 0) {
        std::cerr << "[Rank " << rank_id << "] Failed to init shmem tls\n";
        return false;
    }
    
    const int32_t device_id = rank_id % n_devices + first_device_id;
    int status = 0;
    aclrtStream stream = nullptr;
    
    status |= aclInit(nullptr);
    status |= aclrtSetDevice(device_id);
    status |= aclrtCreateStream(&stream);
    
    if (status != 0) {
        std::cerr << "[Rank " << rank_id << "] ACL initialization failed\n";
        return false;
    }
    
    // Initialize shmem
    ShmemEnv env;
    const char *ip = "tcp://127.0.0.1:8778";
    env.rank = rank_id;
    env.size = n_ranks;
    env.ipPort = ip;
    
    if (!ShmemInitFromEnv(env)) {
        std::cerr << "[Rank " << rank_id << "] Shmem initialization failed\n";
        return false;
    }
    
    if (verbose) {
        std::cout << "[Rank " << rank_id << "] Running on device " << device_id << std::endl;
    }
    
    // Calculate sizes
    size_t aFileSize = GEMM_M * GEMM_K * sizeof(uint16_t);
    size_t bFileSize = GEMM_K * GEMM_N * sizeof(uint16_t);
    size_t cFileSize = GEMM_M * GEMM_N * sizeof(float);
    
    // Allocate host memory
    uint8_t *dstHost = nullptr, *src0Host = nullptr, *src1Host = nullptr;
    uint8_t *dstDevice = nullptr, *src0Device = nullptr, *src1Device = nullptr;
    
    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);

    
    // Allocate device memory
    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);


    uint8_t *shmemDevice = reinterpret_cast<uint8_t*>(ShmemCalloc((1UL << 28), sizeof(uint32_t)));
    
    if (shmemDevice == nullptr) {
        std::cerr << "[Rank " << rank_id << "] ShmemMalloc failed\n";
        // Cleanup
        aclrtFree(dstDevice);
        aclrtFree(src0Device);
        aclrtFree(src1Device);
        
        aclrtFreeHost(dstHost);
        aclrtFreeHost(src0Host);
        aclrtFreeHost(src1Host);

        ShmemFinalize();

        return false;
    }

    
    // Initialize input data (simple initialization for demo)
    PtoTestCommon::ReadFile("../input/x1_gm.bin", aFileSize, src0Host, aFileSize);
    PtoTestCommon::ReadFile("../input/x2_gm.bin", bFileSize, src1Host, bFileSize);
    
    // Copy to device
    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    
    // Synchronize all ranks before kernel launch
    ShmemBarrierAll();
    
    // Launch GEMM kernel
    LaunchGEMME2E<uint16_t>(dstDevice, src0Device, src1Device, shmemDevice, stream, true);
    
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    if(rank_id == 1){
        PtoTestCommon::WriteFile("../output/output_z.bin", dstHost, cFileSize);
    }
    
    // Synchronize all ranks after kernel
    ShmemBarrierAll();
    
    if (verbose) {
        std::cout << "[Rank " << rank_id << "] GEMM completed" << std::endl;
    }
    
    // Cleanup
    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    
    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    
    ShmemFinalize();
    
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    
    std::vector<float> golden(cFileSize);
    std::vector<float> devFinal(cFileSize);
    PtoTestCommon::ReadFile("../output/golden.bin", cFileSize, golden.data(), cFileSize);
    PtoTestCommon::ReadFile("../output/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    for(int i = 0; i < cFileSize; ++i){
        devFinal[i] = devFinal[i] / n_ranks;
    }

    if(rank_id == 1){
        if (PtoTestCommon::ResultCmp(golden, devFinal, 0.001f)) {
            printf("test success\n");
            return true;
        } else {
            printf("test failed\n");
            return false;
        }
    }else{
        return true;
    }
}

// ============================================================================
// Multi-process Launcher
// ============================================================================
bool RunGemmMultiProcess(int n_ranks, int n_devices, int first_rank_id, int first_device_id, bool verbose)
{
    std::vector<pid_t> pids;
    
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child process
            const bool ok = RunGemmKernel(first_rank_id + r, n_ranks, n_devices, first_device_id, verbose);
            _exit(ok ? 0 : 1);
        } else if (pid > 0) {
            // Parent process
            pids.push_back(pid);
        } else {
            // Fork failed
            std::cerr << "Fork failed for rank " << r << std::endl;
            return false;
        }
    }
    
    // Wait for all child processes
    bool success = true;
    for (pid_t p : pids) {
        int status = 0;
        waitpid(p, &status, 0);
        if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0)) {
            success = false;
        }
    }
    return success;
}

// ============================================================================
// Main Entry Point
// ============================================================================
int main(int argc, char **argv)
{
    TestArgs args = ParseArgs(argc, argv);
    
    if (args.help) {
        PrintUsage(argv[0]);
        return 0;
    }
    
    // Print test configuration
    std::cout << "============================================================" << std::endl;
    std::cout << "  GEMM with AllReduce Demo" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "  GEMM Configuration:" << std::endl;
    std::cout << "    Global shape:     [" << GEMM_M << " x " << GEMM_K 
              << "] * [" << GEMM_K << " x " << GEMM_N << "]" << std::endl;
    std::cout << "    Block dim:        " << BLOCK_DIM << std::endl;
    std::cout << std::endl;
    std::cout << "  Test Parameters:" << std::endl;
    std::cout << "    Ranks:            " << args.n_ranks << std::endl;
    std::cout << "    Devices:          " << args.n_devices << std::endl;
    std::cout << "    Verbose:          " << (args.verbose ? "yes" : "no") << std::endl;
    std::cout << "============================================================\n" << std::endl;
    
    bool success = RunGemmMultiProcess(
        args.n_ranks,
        args.n_devices,
        args.first_rank_id,
        args.first_device_id,
        args.verbose);
    
    // Final summary
    std::cout << "\n============================================================" << std::endl;
    std::cout << "  Demo " << (success ? "PASSED" : "FAILED") << std::endl;
    std::cout << "============================================================\n" << std::endl;
    
    return success ? 0 : 1;
}
