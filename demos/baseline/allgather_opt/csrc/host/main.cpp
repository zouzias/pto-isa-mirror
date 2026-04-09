#include <cstdlib>
#include <iostream>
#include "comm_mpi.h"
#include "../kernel/allgather_opt_kernel.h"

int main(int argc, char **argv)
{
    if (!CommMpiInit(&argc, &argv)) {
        std::cerr << "[FATAL] MPI init failed. Launch with: mpirun -n <N> ./allgather_opt_demo" << std::endl;
        return 1;
    }

    int rank = CommMpiRank();
    int size = CommMpiSize();

    if (size < 2) {
        if (rank == 0)
            std::cerr << "[ERROR] Requires at least 2 MPI ranks." << std::endl;
        CommMpiFinalize();
        return 1;
    }

    if (rank == 0) {
        std::cout << "========================================" << std::endl;
        std::cout << " AllGather Optimization Sweep" << std::endl;
        std::cout << " Ranks: " << size << std::endl;
        std::cout << "========================================" << std::endl;
    }

    if (rank == 0) {
        std::cout << "\n[PERF] ======== Optimization Sweep (Rank 0) ========" << std::endl;
        std::cout << "[PERF] warmup=20 iters=100 nranks=" << size << std::endl;
    }

    if (!RunAllgatherOptSweep(size, 0, 0)) {
        if (rank == 0)
            std::cerr << "[PERF] Opt sweep FAILED" << std::endl;
    }

    if (rank == 0)
        std::cout << "[PERF] ==============================================" << std::endl;

    CommMpiFinalize();
    return 0;
}
