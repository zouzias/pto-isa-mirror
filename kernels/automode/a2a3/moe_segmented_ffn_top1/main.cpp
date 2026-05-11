/**
 * main.cpp - host driver for moe_segmented_ffn_top1.
 *
 * Pattern source: kernels/automode/a2a3/moe_segmented_gemm_relu/main.cpp.
 * Adds:
 *   - one extra GM input (w2),
 *   - one device-resident scratch GM buffer (hidden_scratch) of size
 *     kTileM * kF * sizeof(half), allocated once and reused by the kernel
 *     across all (expert, microtile) inner iters.
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ../output/t_padded.txt                (single int line; written by gen_data.py)
 *   ../input/input_packed_tokens.bin      (T_PADDED * H float16)
 *   ../input/input_expert_count.bin       (kE        int32)
 *   ../input/input_expert_start.bin       (kE        int32)
 *   ../input/input_w1.bin                 (kE * H * F float16)
 *   ../input/input_w2.bin                 (kE * F * O float16)
 *   ../output/golden_packed_output.bin    (T_PADDED * O float32; final FFN out)
 *   ../output/output_packed_output.bin    (T_PADDED * O float32)  (this driver)
 */

#include "test_common.h"
#include "acl/acl.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

extern "C" void launchMoeSegmentedFfnTop1Fp16(uint8_t *packed_output,
                                               uint8_t *hidden_scratch,
                                               uint8_t *packed_tokens,
                                               int32_t *expert_count,
                                               int32_t *expert_start,
                                               uint8_t *w1,
                                               uint8_t *w2,
                                               void *stream);

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

    // Wider tolerance than §A17 because the FP32->FP16 cast of the hidden
    // state can introduce one ULP of difference at the rounding boundary
    // (golden mimics it exactly; if device differs from golden's cast,
    // small residuals propagate through the second GEMM). For the
    // integer-valued [-3, 4] input distribution the result should still be
    // bit-exact, but 1e-2 is comfortably above any plausible drift.
    bool ret = ResultCmp(golden, devFinal, 0.01f);
    if (ret) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ret;
}

int main()
{
    constexpr int kH      = 64;
    constexpr int kF      = 64;
    constexpr int kO      = 64;
    constexpr int kE      = 4;
    constexpr int kTileM  = 128;
    constexpr size_t halfBytes  = 2;
    constexpr size_t floatBytes = 4;
    constexpr size_t int32Bytes = 4;

    const int T_padded = ReadTPadded();
    size_t packedTokensBytes = static_cast<size_t>(T_padded) * kH * halfBytes;
    size_t packedOutputBytes = static_cast<size_t>(T_padded) * kO * floatBytes;
    size_t w1Bytes           = static_cast<size_t>(kE) * kH * kF * halfBytes;
    size_t w2Bytes           = static_cast<size_t>(kE) * kF * kO * halfBytes;
    size_t expertMetaBytes   = static_cast<size_t>(kE) * int32Bytes;
    size_t hiddenScratchBytes = static_cast<size_t>(kTileM) * kF * halfBytes;

    printf("[main] T_padded=%d  H=%d  F=%d  O=%d  E=%d  TILE_M=%d\n"
           "       packedTokensBytes=%zu  packedOutputBytes=%zu\n"
           "       w1Bytes=%zu  w2Bytes=%zu  expertMetaBytes=%zu  hiddenScratchBytes=%zu\n",
           T_padded, kH, kF, kO, kE, kTileM,
           packedTokensBytes, packedOutputBytes,
           w1Bytes, w2Bytes, expertMetaBytes, hiddenScratchBytes);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *tokensHost = nullptr, *outputHost = nullptr;
    uint8_t *w1Host = nullptr, *w2Host = nullptr;
    int32_t *countHost = nullptr, *startHost = nullptr;

    uint8_t *tokensDev = nullptr, *outputDev = nullptr;
    uint8_t *w1Dev = nullptr, *w2Dev = nullptr;
    int32_t *countDev = nullptr, *startDev = nullptr;
    uint8_t *hiddenScratchDev = nullptr;

    aclrtMallocHost((void **)(&tokensHost), packedTokensBytes);
    aclrtMallocHost((void **)(&outputHost), packedOutputBytes);
    aclrtMallocHost((void **)(&w1Host),     w1Bytes);
    aclrtMallocHost((void **)(&w2Host),     w2Bytes);
    aclrtMallocHost((void **)(&countHost),  expertMetaBytes);
    aclrtMallocHost((void **)(&startHost),  expertMetaBytes);

    aclrtMalloc((void **)&tokensDev,        packedTokensBytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&outputDev,        packedOutputBytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&w1Dev,            w1Bytes,             ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&w2Dev,            w2Bytes,             ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&countDev,         expertMetaBytes,     ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&startDev,         expertMetaBytes,     ACL_MEM_MALLOC_HUGE_FIRST);
    // hidden_scratch: device-only; no host counterpart needed.
    aclrtMalloc((void **)&hiddenScratchDev, hiddenScratchBytes,  ACL_MEM_MALLOC_HUGE_FIRST);

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
    // hidden_scratch contents are don't-care on entry; the kernel always
    // writes a full TILE_M*F tile via TSTORE before reading it back via TLOAD.

    launchMoeSegmentedFfnTop1Fp16(outputDev, hiddenScratchDev, tokensDev,
                                   countDev, startDev, w1Dev, w2Dev, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outputHost, packedOutputBytes, outputDev, packedOutputBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("../output/output_packed_output.bin", outputHost, packedOutputBytes);

    aclrtFree(hiddenScratchDev);
    aclrtFree(startDev);
    aclrtFree(countDev);
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
