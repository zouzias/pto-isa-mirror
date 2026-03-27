/**
 * GEMM AllReduce Demo - Main Entry Point with Integrated Data Generation
 * HCCL backend — launched via mpirun
 *
 * Generates random input matrices (fp16), computes golden reference (fp32 CPU GEMM),
 * then runs multi-card GEMM with AllReduce.
 *
 * Usage:
 *   mpirun -n <NRANKS> ./gemm_allreduce [--first-device ID]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <thread>
#include <chrono>
#include <algorithm>

#include "comm_mpi.h"

#ifndef CONFIG_G_M
#define CONFIG_G_M 16384
#endif
#ifndef CONFIG_G_K
#define CONFIG_G_K 16384
#endif
#ifndef CONFIG_G_N
#define CONFIG_G_N 4096
#endif

static constexpr int G_M = CONFIG_G_M;
static constexpr int G_K = CONFIG_G_K;
static constexpr int G_N = CONFIG_G_N;

extern bool RunGemmAllReduce(int n_ranks, int first_device_id,
                             const uint16_t *a_parts,
                             const uint16_t *b_parts,
                             const float *golden);

static uint16_t floatToHalf(float f)
{
    union { float f; uint32_t u; } bits;
    bits.f = f;
    uint32_t x = bits.u;
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x007FFFFF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static void computeGolden(const float *A, const float *B, float *C, int M, int K, int N)
{
    std::memset(C, 0, (size_t)M * N * sizeof(float));

    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    unsigned n_threads = std::min(hw, 64u);

    int rows_per = (M + (int)n_threads - 1) / (int)n_threads;
    std::vector<std::thread> threads;

    for (unsigned t = 0; t < n_threads; t++) {
        int r0 = (int)t * rows_per;
        int r1 = std::min(r0 + rows_per, M);
        if (r0 >= M) break;

        threads.emplace_back([A, B, C, K, N, r0, r1]() {
            constexpr int BLK = 64;
            for (int i = r0; i < r1; i++)
                for (int kk = 0; kk < K; kk += BLK) {
                    int kEnd = std::min(kk + BLK, K);
                    for (int jj = 0; jj < N; jj += BLK) {
                        int jEnd = std::min(jj + BLK, N);
                        for (int k = kk; k < kEnd; k++) {
                            float aik = A[(size_t)i * K + k];
                            for (int j = jj; j < jEnd; j++)
                                C[(size_t)i * N + j] += aik * B[(size_t)k * N + j];
                        }
                    }
                }
        });
    }
    for (auto &th : threads) th.join();
}

static bool generateData(int nranks,
                          std::vector<uint16_t> &a_parts,
                          std::vector<uint16_t> &b_parts,
                          std::vector<float> &golden)
{
    int k_per_rank = G_K / nranks;
    if (G_K % nranks != 0) {
        fprintf(stderr, "[ERROR] K=%d not divisible by nranks=%d\n", G_K, nranks);
        return false;
    }

    printf("Data Parallel: K=%d split into %d ranks, %d per rank\n", G_K, nranks, k_per_rank);

    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(1, 4);

    size_t A_elems = (size_t)G_M * G_K;
    size_t B_elems = (size_t)G_K * G_N;
    std::vector<float> A_fp32(A_elems), B_fp32(B_elems);

    for (auto &v : A_fp32) v = (float)dist(gen);
    for (auto &v : B_fp32) v = (float)dist(gen);

    printf("  Computing golden reference (CPU GEMM %d×%d×%d)...\n", G_M, G_K, G_N);
    auto t0 = std::chrono::high_resolution_clock::now();
    golden.resize((size_t)G_M * G_N);
    computeGolden(A_fp32.data(), B_fp32.data(), golden.data(), G_M, G_K, G_N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    printf("  Golden computed in %.1f s\n", secs);

    size_t a_rank_elems = (size_t)G_M * k_per_rank;
    size_t b_rank_elems = (size_t)G_N * k_per_rank;
    a_parts.resize((size_t)nranks * a_rank_elems);
    b_parts.resize((size_t)nranks * b_rank_elems);

    for (int r = 0; r < nranks; r++) {
        int k_start = r * k_per_rank;
        uint16_t *a_dst = a_parts.data() + (size_t)r * a_rank_elems;
        uint16_t *b_dst = b_parts.data() + (size_t)r * b_rank_elems;

        for (int i = 0; i < G_M; i++)
            for (int j = 0; j < k_per_rank; j++)
                a_dst[(size_t)i * k_per_rank + j] =
                    floatToHalf(A_fp32[(size_t)i * G_K + k_start + j]);

        for (int i = 0; i < G_N; i++)
            for (int j = 0; j < k_per_rank; j++)
                b_dst[(size_t)i * k_per_rank + j] =
                    floatToHalf(B_fp32[(size_t)(k_start + j) * G_N + i]);

        printf("  Rank %d: A[%d×%d] cols[%d:%d], B[%d×%d] rows[%d:%d]\n",
               r, G_M, k_per_rank, k_start, k_start + k_per_rank,
               k_per_rank, G_N, k_start, k_start + k_per_rank);
    }

    double gsum = 0.0;
    for (auto v : golden) gsum += v;
    printf("  Golden: shape=(%d, %d), sum=%.2f\n", G_M, G_N, gsum);
    return true;
}

static int parseFirstDevice(int argc, char *argv[])
{
    const char *envVal = getenv("GEMM_ALLREDUCE_FIRST_DEVICE");
    if (envVal != nullptr) {
        int val = atoi(envVal);
        if (val >= 0) return val;
    }
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "--first-device") == 0) {
            int val = atoi(argv[i + 1]);
            if (val >= 0) return val;
        }
    }
    return 0;
}

int main(int argc, char *argv[])
{
    if (!CommMpiInit(&argc, &argv)) {
        fprintf(stderr, "[ERROR] MPI_Init failed. Launch with: mpirun -n <NRANKS> ./gemm_allreduce\n");
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: mpirun -n <NRANKS> %s [--first-device ID]\n", argv[0]);
            CommMpiFinalize();
            return 0;
        }
    }

    int n_ranks = CommMpiSize();
    int first_device_id = parseFirstDevice(argc, argv);

    if (CommMpiRank() == 0) {
        printf("GEMM AllReduce (HCCL): ranks=%d, devices=[%d, %d)\n\n",
               n_ranks, first_device_id, first_device_id + n_ranks);
    }

    int k_per_rank = G_K / n_ranks;
    size_t a_total = (size_t)n_ranks * G_M * k_per_rank;
    size_t b_total = (size_t)n_ranks * G_N * k_per_rank;
    size_t g_total = (size_t)G_M * G_N;

    std::vector<uint16_t> a_parts(a_total), b_parts(b_total);
    std::vector<float> golden(g_total);

    int data_ok = 0;
    if (CommMpiRank() == 0) {
        if (!generateData(n_ranks, a_parts, b_parts, golden)) {
            data_ok = 1;
        }
    }

    CommMpiBcast(&data_ok, 1, COMM_MPI_INT, 0);
    if (data_ok != 0) {
        CommMpiFinalize();
        return 1;
    }

    CommMpiBcast(a_parts.data(), (int)(a_total * sizeof(uint16_t)), COMM_MPI_CHAR, 0);
    CommMpiBcast(b_parts.data(), (int)(b_total * sizeof(uint16_t)), COMM_MPI_CHAR, 0);
    CommMpiBcast(golden.data(), (int)(g_total * sizeof(float)), COMM_MPI_CHAR, 0);

    if (CommMpiRank() == 0) {
        printf("  Broadcast input data to all %d ranks.\n", n_ranks);
    }

    bool ok = RunGemmAllReduce(n_ranks, first_device_id,
                               a_parts.data(), b_parts.data(), golden.data());

    if (CommMpiRank() == 0) {
        printf(ok ? "\nGEMM AllReduce demo completed successfully.\n"
                  : "\nGEMM AllReduce demo FAILED.\n");
    }

    CommMpiFinalize();
    return ok ? 0 : 1;
}
