/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Allgather Multi-core Async Demo — Host Entry Point
//
// Demonstrates AllGather using multi-core TPUT_ASYNC / TGET_ASYNC:
// each AICORE handles one rank's communication in parallel.
//
// Usage: mpirun -n <N> ./allgather_demo

#include <cstdlib>
#include <iostream>
#include "comm_mpi.h"
#include "../kernel/allgather_kernel.h"

int main(int argc, char **argv)
{
    if (!CommMpiInit(&argc, &argv)) {
        std::cerr << "[FATAL] MPI init failed. Launch with: mpirun -n <N> ./allgather_demo" << std::endl;
        return 1;
    }

    int rank = CommMpiRank();
    int size = CommMpiSize();

    if (size < 2) {
        if (rank == 0) {
            std::cerr << "[ERROR] Allgather requires at least 2 MPI ranks." << std::endl;
            std::cerr << "        Launch with: mpirun -n <N> ./allgather_demo" << std::endl;
        }
        CommMpiFinalize();
        return 1;
    }

    if (rank == 0) {
        std::cout << "========================================" << std::endl;
        std::cout << " PTO Allgather Multi-core Async Demo" << std::endl;
        std::cout << " Ranks: " << size << std::endl;
        std::cout << "========================================" << std::endl;
    }

    int failures = 0;

    if (rank == 0)
        std::cout << "\n--- Demo 1: Allgather via TPUT_ASYNC (Multi-core) ---" << std::endl;
    if (!RunAllgatherPutAsyncMC(size, 0, 0)) {
        if (rank == 0)
            std::cerr << "[TPUT_ASYNC_MC Allgather FAIL]" << std::endl;
        ++failures;
    }

    CommMpiBarrier();

    if (rank == 0)
        std::cout << "\n--- Demo 2: Allgather via TGET_ASYNC (Multi-core) ---" << std::endl;
    if (!RunAllgatherGetAsyncMC(size, 0, 0)) {
        if (rank == 0)
            std::cerr << "[TGET_ASYNC_MC Allgather FAIL]" << std::endl;
        ++failures;
    }

    if (rank == 0) {
        std::cout << "\n========================================" << std::endl;
        if (failures == 0)
            std::cout << " All demos PASSED" << std::endl;
        else
            std::cout << " " << failures << " demo(s) FAILED" << std::endl;
        std::cout << "========================================" << std::endl;
    }

    CommMpiBarrier();

    if (rank == 0) {
        std::cout << "\n[PERF] ======== Bandwidth Sweep (Rank 0) ========" << std::endl;
        std::cout << "[PERF] warmup=20 iters=100 nranks=" << size << std::endl;
    }

    if (!RunAllgatherMcAsyncSweep(size, 0, 0)) {
        if (rank == 0)
            std::cerr << "[PERF] MC Async sweep FAILED" << std::endl;
    }

    if (rank == 0) {
        std::cout << "[PERF] ==============================================" << std::endl;
    }

    CommMpiFinalize();
    return (failures == 0) ? 0 : 1;
}
