/**
 * AllGather GEMM Demo - Main Entry Point
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
 */

#include <cstdio>
#include <cstdlib>

// Launcher function defined in allgather_gemm_comm_kernel.cpp
extern bool RunAllGatherGemm();

int main()
{
    bool ok = RunAllGatherGemm();
    if (ok) {
        printf("AllGather GEMM demo completed successfully.\n");
    } else {
        printf("AllGather GEMM demo FAILED.\n");
    }
    return ok ? 0 : 1;
}
