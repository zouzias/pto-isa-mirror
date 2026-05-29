/**
 * main.cpp - host driver for Indexer.wq_b  (BF16 qr x BF16 Wq_b -> FP32 q).
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_qr.bin     M*K   bfloat16   (low-rank query)
 *   ./input/input_wq_b.bin   N*K   bfloat16   (Indexer.wq_b weight)
 *   ./output/golden_q.bin    M*N   float32    (numpy reference)
 *   ./output/output_q.bin    M*N   float32    (kernel output)
 *
 * M = B*S, K = Q_LORA_RANK, N = N_HEADS*HEAD_DIM (see ../README.md).
 */

#include "test_common.h"
#include "acl/acl.h"
#include "../../../../kernel_timing.h"
#include "generated_cases.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

extern "C" void launch_wq_b_gemm(uint8_t *out, uint8_t *qr, uint8_t *Wq_b,
                                 uint64_t M, uint64_t N, uint64_t K, void *stream);

int main()
{
    // Aliases produced by ../scripts/generate_cases.py.
    constexpr int kB        = kIdxB;
    constexpr int kS        = kIdxS;
    constexpr int kQLoraRank = kIdxQLoraRank;
    constexpr int kNHeads   = kIdxNHeads;
    constexpr int kHeadDim  = kIdxHeadDim;
    constexpr int kM = kB * kS;
    constexpr int kK = kQLoraRank;
    constexpr int kN = kNHeads * kHeadDim;

    constexpr size_t bf16Bytes  = 2;
    constexpr size_t floatBytes = 4;

    size_t qrBytes  = static_cast<size_t>(kM) * kK * bf16Bytes;
    size_t wBytes   = static_cast<size_t>(kN) * kK * bf16Bytes;
    size_t qBytes   = static_cast<size_t>(kM) * kN * floatBytes;

    printf("[wq_b_gemm] B=%d S=%d QLR=%d NHeads=%d HeadDim=%d -> M=%d N=%d K=%d\n",
           kB, kS, kQLoraRank, kNHeads, kHeadDim, kM, kN, kK);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *qrHost = nullptr, *wHost = nullptr, *qHost  = nullptr;
    uint8_t *qrDev  = nullptr, *wDev  = nullptr, *qDev   = nullptr;

    aclrtMallocHost((void **)&qrHost, qrBytes);
    aclrtMallocHost((void **)&wHost,  wBytes);
    aclrtMallocHost((void **)&qHost,  qBytes);

    aclrtMalloc((void **)&qrDev, qrBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&wDev,  wBytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&qDev,  qBytes,  ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("./input/input_qr.bin",   qrBytes, qrHost, qrBytes);
    ReadFile("./input/input_wq_b.bin", wBytes,  wHost,  wBytes);

    aclrtMemcpy(qrDev, qrBytes, qrHost, qrBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(wDev,  wBytes,  wHost,  wBytes,  ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("wq_b_gemm", stream, [&]() {
        launch_wq_b_gemm(qDev, qrDev, wDev,
                         static_cast<uint64_t>(kM),
                         static_cast<uint64_t>(kN),
                         static_cast<uint64_t>(kK),
                         stream);
    });

    aclrtMemcpy(qHost, qBytes, qDev, qBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("./output/output_q.bin", qHost, qBytes);

    aclrtFree(qDev);
    aclrtFree(wDev);
    aclrtFree(qrDev);
    aclrtFreeHost(qHost);
    aclrtFreeHost(wHost);
    aclrtFreeHost(qrHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(qBytes / sizeof(float));
    std::vector<float> devFinal(qBytes / sizeof(float));
    ReadFile("./output/golden_q.bin",  qBytes, golden.data(),   qBytes);
    ReadFile("./output/output_q.bin",  qBytes, devFinal.data(), qBytes);

    // BF16 inputs, FP32 accumulator. Small integer inputs are bit-exact in
    // BF16, so cumulative drift over K stays bounded.
    bool ok = ResultCmp(golden, devFinal, 1e-2f);
    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
