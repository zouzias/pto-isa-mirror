/**
 * main.cpp - host driver for Compressor.c_gemm (BF16 X x BF16 Wgate -> FP32 score).
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ./input/input_x.bin        M*K   bfloat16   (token features)
 *   ./input/input_wgate.bin    N*K   bfloat16   (Compressor.wgate weight)
 *   ./output/golden_score.bin  M*N   float32    (numpy reference)
 *   ./output/output_score.bin  M*N   float32    (kernel output)
 *
 * M = B*S, K = DIM, N = COFF*HEAD_DIM (see ../README.md).
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

extern "C" void launch_c_gemm(uint8_t *out, uint8_t *X, uint8_t *Wgate,
                              uint64_t M, uint64_t N, uint64_t K, void *stream);

int main()
{
    // Aliases produced by ../scripts/generate_cases.py — Known: emits
    // kCompB, kCompS, kCompDim, kCompHeadDim, kCompCoff (see
    // HCA/Compressor/scripts/generate_cases.py:_render_header).
    constexpr int kB        = kCompB;
    constexpr int kS        = kCompS;
    constexpr int kDim      = kCompDim;
    constexpr int kHeadDim  = kCompHeadDim;
    constexpr int kCoff     = kCompCoff;
    constexpr int kM = kB * kS;
    constexpr int kK = kDim;
    constexpr int kN = kCoff * kHeadDim;

    constexpr size_t bf16Bytes  = 2;
    constexpr size_t floatBytes = 4;

    size_t xBytes   = static_cast<size_t>(kM) * kK * bf16Bytes;
    size_t wBytes   = static_cast<size_t>(kN) * kK * bf16Bytes;
    size_t sBytes   = static_cast<size_t>(kM) * kN * floatBytes;

    printf("[c_gemm] B=%d S=%d Dim=%d HeadDim=%d Coff=%d  -> M=%d N=%d K=%d\n",
           kB, kS, kDim, kHeadDim, kCoff, kM, kN, kK);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *xHost  = nullptr, *wHost  = nullptr, *sHost  = nullptr;
    uint8_t *xDev   = nullptr, *wDev   = nullptr, *sDev   = nullptr;

    aclrtMallocHost((void **)&xHost, xBytes);
    aclrtMallocHost((void **)&wHost, wBytes);
    aclrtMallocHost((void **)&sHost, sBytes);

    aclrtMalloc((void **)&xDev,  xBytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&wDev,  wBytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&sDev,  sBytes,  ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("./input/input_x.bin",     xBytes, xHost, xBytes);
    ReadFile("./input/input_wgate.bin", wBytes, wHost, wBytes);

    aclrtMemcpy(xDev, xBytes, xHost, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(wDev, wBytes, wHost, wBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    (void)PtoTiming::TimeKernelCallUs("c_gemm", stream, [&]() {
        launch_c_gemm(sDev, xDev, wDev,
                      static_cast<uint64_t>(kM),
                      static_cast<uint64_t>(kN),
                      static_cast<uint64_t>(kK),
                      stream);
    });

    aclrtMemcpy(sHost, sBytes, sDev, sBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("./output/output_score.bin", sHost, sBytes);

    aclrtFree(sDev);
    aclrtFree(wDev);
    aclrtFree(xDev);
    aclrtFreeHost(sHost);
    aclrtFreeHost(wHost);
    aclrtFreeHost(xHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(sBytes / sizeof(float));
    std::vector<float> devFinal(sBytes / sizeof(float));
    ReadFile("./output/golden_score.bin", sBytes, golden.data(),   sBytes);
    ReadFile("./output/output_score.bin", sBytes, devFinal.data(), sBytes);

    // BF16 inputs, FP32 accumulator. Small integer inputs in numpy reference
    // are bit-exact in BF16 (<2^7), so cumulative drift over K=DIM=4096 stays
    // bounded.
    bool ok = ResultCmp(golden, devFinal, 1e-2f);
    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ok ? 0 : 1;
}
