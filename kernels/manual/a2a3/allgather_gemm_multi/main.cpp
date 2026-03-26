/**
 * AllGather GEMM Demo - Main Entry Point (HCCL backend)
 *
 * Multi-card AllGather then GEMM: each card first gathers all portions of A
 * via AllGather, then computes the full C = A * B.
 *
 * Tile-level pipelining: comm kernel and compute kernel run on separate
 * streams; the compute kernel starts processing each row group as soon as
 * the comm kernel signals its availability via RowGroupSignals.
 *
 * This is the producer-consumer mirror of gemm_allgather:
 *   gemm_allgather:  GEMM (producer) → AllGather (consumer)
 *   allgather_gemm:  AllGather (producer) → GEMM (consumer)
 *
 * Launch with: mpirun -n <N_RANKS> ./allgather_gemm
 */

#include <cstdio>
#include <cstdlib>
#include "comm_mpi.h"

extern bool RunAllGatherGemm();

int main(int argc, char** argv)
{
    if (!CommMpiInit(&argc, &argv)) {
        fprintf(stderr, "[FATAL] CommMpiInit failed. Launch with: mpirun -n <N> ./allgather_gemm\n");
        return 1;
    }

    bool ok = RunAllGatherGemm();

    int rank = CommMpiRank();
    if (rank == 0) {
        if (ok) {
            printf("AllGather GEMM demo completed successfully.\n");
        } else {
            printf("AllGather GEMM demo FAILED.\n");
        }
    }

    CommMpiFinalize();
    return ok ? 0 : 1;
}
