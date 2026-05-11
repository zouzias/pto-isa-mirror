/**
 * main.cpp - host driver for moe_segmented_gemm_relu.
 *
 * Identical structure to moe_segmented_gemm_one_layer/main.cpp; only the
 * launcher symbol name changes (`...Fp16` -> `...ReluFp16`). All buffer
 * sizes and dtypes are unchanged because ReLU is fused into the TSTORE in
 * the kernel and does not alter the output shape/dtype.
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ../output/t_padded.txt                (single int line; written by gen_data.py)
 *   ../input/input_packed_tokens.bin      (T_PADDED * H float16)
 *   ../input/input_expert_count.bin       (kE        int32)
 *   ../input/input_expert_start.bin       (kE        int32)
 *   ../input/input_expert_weight.bin      (kE * H * O float16)
 *   ../output/golden_packed_output.bin    (T_PADDED * O float32; POST-ReLU)
 *   ../output/output_packed_output.bin    (T_PADDED * O float32)  (this driver)
 *   ../output/golden_gemm_output.bin      (T_PADDED * O float32; PRE-ReLU; debug)
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
extern "C" void launchMoeSegmentedGemmReluFp16(uint8_t *packed_output,
                                                uint8_t *packed_tokens,
                                                int32_t *expert_count,
                                                int32_t *expert_start,
                                                uint8_t *expert_weight,
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
    constexpr int kO = 64;
    constexpr int kE = 4;
    constexpr size_t halfBytes  = 2;
    constexpr size_t floatBytes = 4;
    constexpr size_t int32Bytes = 4;

    const int T_padded = ReadTPadded();
    size_t packedTokensBytes = static_cast<size_t>(T_padded) * kH * halfBytes;
    size_t packedOutputBytes = static_cast<size_t>(T_padded) * kO * floatBytes;
    size_t expertWeightBytes = static_cast<size_t>(kE) * kH * kO * halfBytes;
    size_t expertMetaBytes   = static_cast<size_t>(kE) * int32Bytes;

    printf("[main] T_padded=%d  H=%d  O=%d  E=%d\n"
           "       packedTokensBytes=%zu  packedOutputBytes=%zu\n"
           "       expertWeightBytes=%zu  expertMetaBytes=%zu\n",
           T_padded, kH, kO, kE,
           packedTokensBytes, packedOutputBytes,
           expertWeightBytes, expertMetaBytes);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *tokensHost = nullptr, *outputHost = nullptr, *weightHost = nullptr;
    int32_t *countHost = nullptr, *startHost = nullptr;
    uint8_t *tokensDev  = nullptr, *outputDev  = nullptr, *weightDev  = nullptr;
    int32_t *countDev   = nullptr, *startDev   = nullptr;

    aclrtMallocHost((void **)(&tokensHost), packedTokensBytes);
    aclrtMallocHost((void **)(&outputHost), packedOutputBytes);
    aclrtMallocHost((void **)(&weightHost), expertWeightBytes);
    aclrtMallocHost((void **)(&countHost),  expertMetaBytes);
    aclrtMallocHost((void **)(&startHost),  expertMetaBytes);

    aclrtMalloc((void **)&tokensDev, packedTokensBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&outputDev, packedOutputBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&weightDev, expertWeightBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&countDev,  expertMetaBytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&startDev,  expertMetaBytes,   ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("../input/input_packed_tokens.bin", packedTokensBytes, tokensHost, packedTokensBytes);
    ReadFile("../input/input_expert_count.bin",  expertMetaBytes,   countHost,  expertMetaBytes);
    ReadFile("../input/input_expert_start.bin",  expertMetaBytes,   startHost,  expertMetaBytes);
    ReadFile("../input/input_expert_weight.bin", expertWeightBytes, weightHost, expertWeightBytes);

    aclrtMemcpy(tokensDev, packedTokensBytes, tokensHost, packedTokensBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(weightDev, expertWeightBytes, weightHost, expertWeightBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(countDev,  expertMetaBytes,   countHost,  expertMetaBytes,   ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(startDev,  expertMetaBytes,   startHost,  expertMetaBytes,   ACL_MEMCPY_HOST_TO_DEVICE);

    // FP16 x FP16 -> FP32 -> ReLU-in-TSTORE (the auto-mode-eligible A3 cube
    // combo plus the FIX-pipe ReluPreMode::NormalRelu fusion).
    launchMoeSegmentedGemmReluFp16(
        outputDev, tokensDev, countDev, startDev, weightDev, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outputHost, packedOutputBytes, outputDev, packedOutputBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("../output/output_packed_output.bin", outputHost, packedOutputBytes);

    aclrtFree(startDev);
    aclrtFree(countDev);
    aclrtFree(weightDev);
    aclrtFree(outputDev);
    aclrtFree(tokensDev);
    aclrtFreeHost(startHost);
    aclrtFreeHost(countHost);
    aclrtFreeHost(weightHost);
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
