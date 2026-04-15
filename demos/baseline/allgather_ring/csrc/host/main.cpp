/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Ring Allgather Demo — Host Entry Point
//
// Demonstrates the Ring allgather collective using two transport paths:
//   1. TPUT_ASYNC (SDMA engine, asynchronous)
//   2. TPUT       (MTE2 pipeline, synchronous)
//
// Both use N-1 rounds for N ranks; each round pushes one chunk to the
// next rank in the ring. Comparing them reveals whether the L2 cache
// coherency issue is specific to SDMA or also affects MTE2 writes.
//
// Usage: mpirun -n <N> ./allgather_ring_demo

#include <cstdlib>
#include <iostream>
#include "comm_mpi.h"
#include "../kernel/allgather_ring_kernel.h"

int main(int argc, char **argv)
{
    if (!CommMpiInit(&argc, &argv)) {
        std::cerr << "[FATAL] MPI init failed. Launch with: mpirun -n <N> ./allgather_ring_demo" << std::endl;
        return 1;
    }

    int rank = CommMpiRank();
    int size = CommMpiSize();

    if (size < 2) {
        if (rank == 0) {
            std::cerr << "[ERROR] Allgather requires at least 2 MPI ranks." << std::endl;
            std::cerr << "        Launch with: mpirun -n <N> ./allgather_ring_demo" << std::endl;
        }
        CommMpiFinalize();
        return 1;
    }

    if (rank == 0) {
        std::cout << "========================================" << std::endl;
        std::cout << " PTO Ring Allgather — Deep Diagnostic" << std::endl;
        std::cout << " Ranks: " << size << ", Rounds: " << size - 1 << std::endl;
        std::cout << "========================================" << std::endl;
    }

    int failures = 0;
    int total = 4;

    if (rank == 0)
        std::cout << "\n--- [1/4] TPUT_ASYNC baseline ---" << std::endl;
    if (!RunAllgatherRing(size, 0, 0)) {
        if (rank == 0) std::cerr << "  => FAIL" << std::endl;
        ++failures;
    }

    if (rank == 0)
        std::cout << "\n--- [2/4] TPUT_ASYNC + dcci only ---" << std::endl;
    if (!RunAllgatherRingDcci(size, 0, 0)) {
        if (rank == 0) std::cerr << "  => FAIL" << std::endl;
        ++failures;
    }

    if (rank == 0)
        std::cout << "\n--- [3/4] TPUT_ASYNC + dcci + AICORE stage-through ---" << std::endl;
    if (!RunAllgatherRingStage(size, 0, 0)) {
        if (rank == 0) std::cerr << "  => FAIL" << std::endl;
        ++failures;
    }

    if (rank == 0)
        std::cout << "\n--- [4/4] TPUT sync (MTE2, reference) ---" << std::endl;
    if (!RunAllgatherRingSync(size, 0, 0)) {
        if (rank == 0) std::cerr << "  => FAIL" << std::endl;
        ++failures;
    }

    if (rank == 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << " Result: " << (total - failures) << "/" << total << " PASSED" << std::endl;
        std::cout << "========================================" << std::endl;
    }

    CommMpiFinalize();
    return (failures == 0) ? 0 : 1;
}
