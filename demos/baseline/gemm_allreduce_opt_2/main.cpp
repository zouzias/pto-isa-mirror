/**
 * GEMM AllReduce Demo - Main Entry Point
 *
 * Multi-card GEMM with AllReduce: each card computes C = A * B,
 * then AllReduce aggregates results across all cards.
 *
 * Tile-level pipelining: compute kernel and comm kernel run on separate
 * streams; the comm kernel TPUT's each tile as soon as the compute kernel
 * signals its completion via a shared flag.
 *
 * Usage:
 *   ./gemm_allreduce [--nranks N] [--first-device ID]
 *   --nranks N: Number of ranks (NPUs) to use (default: 2)
 *   --first-device ID: First device ID to use (default: 0)
 *
 * Environment variables:
 *   GEMM_ALLREDUCE_NRANKS: Number of ranks (overrides --nranks)
 *   GEMM_ALLREDUCE_FIRST_DEVICE: First device ID (overrides --first-device)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// Launcher function defined in comm_kernel.cpp
extern bool RunGemmAllReduce(int n_ranks, int first_device_id);

static int parseIntArg(int argc, char* argv[], const char* argName, int defaultValue)
{
    // Check environment variable first
    const char* envName = (strcmp(argName, "--nranks") == 0) ? "GEMM_ALLREDUCE_NRANKS" 
                                                              : "GEMM_ALLREDUCE_FIRST_DEVICE";
    const char* envVal = getenv(envName);
    if (envVal != nullptr) {
        int val = atoi(envVal);
        if (val > 0) {
            return val;
        }
    }
    
    // Check command line arguments
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], argName) == 0) {
            int val = atoi(argv[i + 1]);
            if (val > 0) {
                return val;
            }
        }
    }
    
    return defaultValue;
}

static void printUsage(const char* progName)
{
    printf("Usage: %s [--nranks N] [--first-device ID]\n", progName);
    printf("  --nranks N:        Number of ranks (NPUs) to use (default: 2)\n");
    printf("  --first-device ID: First device ID to use (default: 0)\n");
    printf("\nEnvironment variables:\n");
    printf("  GEMM_ALLREDUCE_NRANKS:        Number of ranks (overrides --nranks)\n");
    printf("  GEMM_ALLREDUCE_FIRST_DEVICE:  First device ID (overrides --first-device)\n");
}

int main(int argc, char* argv[])
{
    // Parse help
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        }
    }
    
    int n_ranks = parseIntArg(argc, argv, "--nranks", 2);
    int first_device_id = parseIntArg(argc, argv, "--first-device", 0);
    
    printf("GEMM AllReduce Configuration:\n");
    printf("  Number of ranks: %d\n", n_ranks);
    printf("  First device ID: %d\n", first_device_id);
    printf("  Device range: [%d, %d)\n", first_device_id, first_device_id + n_ranks);
    printf("\n");
    
    bool ok = RunGemmAllReduce(n_ranks, first_device_id);
    if (ok) {
        printf("GEMM AllReduce demo completed successfully.\n");
    } else {
        printf("GEMM AllReduce demo FAILED.\n");
    }
    return ok ? 0 : 1;
}
