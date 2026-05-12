/**
 * main.cpp - host driver for moe_segmented_ffn_top1.
 *
 * Structure mirrors moe_segmented_gemm_relu/main.cpp. The kernel needs an
 * extra FP16 weight buffer (w2) and an FP16 scratch buffer for the GEMM1 ->
 * GEMM2 hand-off; everything else is the same. The host allocates scratch on
 * device but does NOT copy any host-side bytes into it — the kernel writes
 * before it reads.
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ../output/t_padded.txt                (single int line; written by gen_data.py)
 *   ../input/input_packed_tokens.bin      (T_PADDED * kH float16)
 *   ../input/input_expert_count.bin       (kE         int32)
 *   ../input/input_expert_start.bin       (kE         int32)
 *   ../input/input_w1.bin                 (kE * kH * kF float16)
 *   ../input/input_w2.bin                 (kE * kF * kH float16)
 *   ../output/golden_packed_output.bin    (T_PADDED * kH float32; full FFN)
 *   ../output/output_packed_output.bin    (T_PADDED * kH float32) (this driver)
 *   ../output/golden_scratch.bin          (T_PADDED * kF float16; post-ReLU; debug)
 */

#include "test_common.h"
#include "acl/acl.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

// Non-template FP16 wrapper exposed by the kernel TU; host never names `half`.
// See compile_error_logbook.md §E13.
extern "C" void launchMoeSegmentedFfnTop1Fp16(uint8_t *packed_output,
                                              uint8_t *packed_tokens,
                                              int32_t *expert_count,
                                              int32_t *expert_start,
                                              uint8_t *w1,
                                              uint8_t *w2,
                                              uint8_t *scratch,
                                              void    *stream);

static int ReadTPadded()
{
    std::ifstream f("../output/t_padded.txt");
    if (!f.is_open()) {
        printf("[main] FATAL: cannot open ../output/t_padded.txt; did scripts/gen_data.py run?\n");
        std::exit(2);
    }
    int t_padded = 0;
    f >> t_padded;
    if (t_padded <= 0) {
        printf("[main] FATAL: invalid T_PADDED=%d from t_padded.txt\n", t_padded);
        std::exit(2);
    }
    return t_padded;
}

template <typename TOut>
inline bool ValidateDataResults(size_t outFileSize)
{
    std::vector<TOut> golden(outFileSize / sizeof(TOut));
    std::vector<TOut> devFinal(outFileSize / sizeof(TOut));

    ReadFile("../output/golden_packed_output.bin", outFileSize, golden.data(),   outFileSize);
    ReadFile("../output/output_packed_output.bin", outFileSize, devFinal.data(), outFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    if (ret) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ret;
}

int main()
{
    constexpr int kH = 64;
    constexpr int kF = 64;
    constexpr int kE = 4;
    constexpr size_t halfBytes  = 2;
    constexpr size_t floatBytes = 4;
    constexpr size_t int32Bytes = 4;

    const int T_padded = ReadTPadded();
    size_t packedTokensBytes = static_cast<size_t>(T_padded) * kH * halfBytes;   // FP16
    size_t packedOutputBytes = static_cast<size_t>(T_padded) * kH * floatBytes;  // FP32
    size_t w1Bytes           = static_cast<size_t>(kE) * kH * kF * halfBytes;
    size_t w2Bytes           = static_cast<size_t>(kE) * kF * kH * halfBytes;
    size_t scratchBytes      = static_cast<size_t>(T_padded) * kF * halfBytes;   // FP16
    size_t expertMetaBytes   = static_cast<size_t>(kE) * int32Bytes;

    printf("[main] T_padded=%d  kH=%d  kF=%d  kE=%d\n"
           "       packedTokensBytes=%zu  packedOutputBytes=%zu\n"
           "       w1Bytes=%zu  w2Bytes=%zu  scratchBytes=%zu\n"
           "       expertMetaBytes=%zu\n",
           T_padded, kH, kF, kE,
           packedTokensBytes, packedOutputBytes,
           w1Bytes, w2Bytes, scratchBytes,
           expertMetaBytes);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *tokensHost = nullptr, *outputHost = nullptr;
    uint8_t *w1Host = nullptr, *w2Host = nullptr;
    int32_t *countHost = nullptr, *startHost = nullptr;

    uint8_t *tokensDev = nullptr, *outputDev = nullptr;
    uint8_t *w1Dev = nullptr, *w2Dev = nullptr, *scratchDev = nullptr;
    int32_t *countDev = nullptr, *startDev = nullptr;

    aclrtMallocHost((void **)(&tokensHost), packedTokensBytes);
    aclrtMallocHost((void **)(&outputHost), packedOutputBytes);
    aclrtMallocHost((void **)(&w1Host),     w1Bytes);
    aclrtMallocHost((void **)(&w2Host),     w2Bytes);
    aclrtMallocHost((void **)(&countHost),  expertMetaBytes);
    aclrtMallocHost((void **)(&startHost),  expertMetaBytes);

    aclrtMalloc((void **)&tokensDev,  packedTokensBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&outputDev,  packedOutputBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&w1Dev,      w1Bytes,           ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&w2Dev,      w2Bytes,           ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&scratchDev, scratchBytes,      ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&countDev,   expertMetaBytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&startDev,   expertMetaBytes,   ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("../input/input_packed_tokens.bin", packedTokensBytes, tokensHost, packedTokensBytes);
    ReadFile("../input/input_expert_count.bin",  expertMetaBytes,   countHost,  expertMetaBytes);
    ReadFile("../input/input_expert_start.bin",  expertMetaBytes,   startHost,  expertMetaBytes);
    ReadFile("../input/input_w1.bin",            w1Bytes,           w1Host,     w1Bytes);
    ReadFile("../input/input_w2.bin",            w2Bytes,           w2Host,     w2Bytes);

    aclrtMemcpy(tokensDev, packedTokensBytes, tokensHost, packedTokensBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(w1Dev,     w1Bytes,           w1Host,     w1Bytes,           ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(w2Dev,     w2Bytes,           w2Host,     w2Bytes,           ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(countDev,  expertMetaBytes,   countHost,  expertMetaBytes,   ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(startDev,  expertMetaBytes,   startHost,  expertMetaBytes,   ACL_MEMCPY_HOST_TO_DEVICE);

    // scratchDev is kernel-managed; the kernel writes before it reads.

    launchMoeSegmentedFfnTop1Fp16(
        outputDev, tokensDev, countDev, startDev,
        w1Dev, w2Dev, scratchDev, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outputHost, packedOutputBytes, outputDev, packedOutputBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("../output/output_packed_output.bin", outputHost, packedOutputBytes);

    aclrtFree(startDev);
    aclrtFree(countDev);
    aclrtFree(scratchDev);
    aclrtFree(w2Dev);
    aclrtFree(w1Dev);
    aclrtFree(outputDev);
    aclrtFree(tokensDev);
    aclrtFreeHost(startHost);
    aclrtFreeHost(countHost);
    aclrtFreeHost(w2Host);
    aclrtFreeHost(w1Host);
    aclrtFreeHost(outputHost);
    aclrtFreeHost(tokensHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    bool dataSuccess = ValidateDataResults<float>(packedOutputBytes);
    if (dataSuccess) {
        printf("test success\n");
    } else {
        printf("test failed\n");
    }
    return 0;
}
