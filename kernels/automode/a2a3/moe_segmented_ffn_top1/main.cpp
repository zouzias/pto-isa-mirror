/**
 * main.cpp - host driver for moe_segmented_ffn_top1.
 *
 * Stage-isolation build (debug):
 *   - poisons scratchDev (0x7B) and outputDev (0x5A) BEFORE the launch so we
 *     can distinguish "kernel wrote zeros" from "kernel did not write";
 *   - copies BOTH packed_output AND scratch back to host after sync;
 *   - writes both to ../output/ for the compare script;
 *   - prints expert_count / expert_start (after reading from disk) so we can
 *     confirm the inner loop bound;
 *   - checks the return codes of every ACL call (silent ACL failures look
 *     exactly like zero / unchanged outputs).
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
 *   ../output/output_scratch.bin          (T_PADDED * kF float16) (this driver, NEW)
 *   ../output/golden_scratch.bin          (T_PADDED * kF float16; post-ReLU; debug)
 */

#include "test_common.h"
#include "acl/acl.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
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

// Mirrors kernels/manual/a2a3/tget_bandwidth/tget_bandwidth_kernel.cpp:91-98.
// Prints and returns false on any non-success aclError; lets us bail before
// downstream operations that would mask the original failure.
static bool CheckAcl(aclError ret, const char *op)
{
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] " << op << " failed: " << static_cast<int>(ret) << std::endl;
        return false;
    }
    return true;
}

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

    // Poison patterns used by Patch 2 / Patch 3. The Python compare script
    // checks for these exact bytes when reporting "kernel never wrote".
    constexpr uint8_t kPoisonOutput  = 0x5A;  // packed_output (FP32)
    constexpr uint8_t kPoisonScratch = 0x7B;  // scratch       (FP16)

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

    if (!CheckAcl(aclInit(nullptr), "aclInit")) std::exit(3);
    if (!CheckAcl(aclrtSetDevice(0), "aclrtSetDevice")) std::exit(3);
    aclrtStream stream;
    if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream")) std::exit(3);

    uint8_t *tokensHost  = nullptr, *outputHost  = nullptr;
    uint8_t *scratchHost = nullptr;
    uint8_t *w1Host = nullptr, *w2Host = nullptr;
    int32_t *countHost = nullptr, *startHost = nullptr;

    uint8_t *tokensDev = nullptr, *outputDev = nullptr;
    uint8_t *w1Dev = nullptr, *w2Dev = nullptr, *scratchDev = nullptr;
    int32_t *countDev = nullptr, *startDev = nullptr;

    CheckAcl(aclrtMallocHost((void **)(&tokensHost),  packedTokensBytes), "aclrtMallocHost(tokensHost)");
    CheckAcl(aclrtMallocHost((void **)(&outputHost),  packedOutputBytes), "aclrtMallocHost(outputHost)");
    CheckAcl(aclrtMallocHost((void **)(&scratchHost), scratchBytes),      "aclrtMallocHost(scratchHost)");
    CheckAcl(aclrtMallocHost((void **)(&w1Host),      w1Bytes),           "aclrtMallocHost(w1Host)");
    CheckAcl(aclrtMallocHost((void **)(&w2Host),      w2Bytes),           "aclrtMallocHost(w2Host)");
    CheckAcl(aclrtMallocHost((void **)(&countHost),   expertMetaBytes),   "aclrtMallocHost(countHost)");
    CheckAcl(aclrtMallocHost((void **)(&startHost),   expertMetaBytes),   "aclrtMallocHost(startHost)");

    CheckAcl(aclrtMalloc((void **)&tokensDev,  packedTokensBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(tokensDev)");
    CheckAcl(aclrtMalloc((void **)&outputDev,  packedOutputBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(outputDev)");
    CheckAcl(aclrtMalloc((void **)&w1Dev,      w1Bytes,           ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(w1Dev)");
    CheckAcl(aclrtMalloc((void **)&w2Dev,      w2Bytes,           ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(w2Dev)");
    CheckAcl(aclrtMalloc((void **)&scratchDev, scratchBytes,      ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(scratchDev)");
    CheckAcl(aclrtMalloc((void **)&countDev,   expertMetaBytes,   ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(countDev)");
    CheckAcl(aclrtMalloc((void **)&startDev,   expertMetaBytes,   ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(startDev)");

    ReadFile("../input/input_packed_tokens.bin", packedTokensBytes, tokensHost, packedTokensBytes);
    ReadFile("../input/input_expert_count.bin",  expertMetaBytes,   countHost,  expertMetaBytes);
    ReadFile("../input/input_expert_start.bin",  expertMetaBytes,   startHost,  expertMetaBytes);
    ReadFile("../input/input_w1.bin",            w1Bytes,           w1Host,     w1Bytes);
    ReadFile("../input/input_w2.bin",            w2Bytes,           w2Host,     w2Bytes);

    // Visibility into the per-expert iteration bounds. If these look wrong
    // (zero counts, starts not monotonic, sum != T_padded) the kernel never
    // had a chance to write anything regardless of body correctness.
    int32_t sumCount = 0;
    printf("[main] expert metadata read from disk:\n");
    for (int e = 0; e < kE; ++e) {
        printf("       expert[%d]: count=%d  start=%d\n", e, countHost[e], startHost[e]);
        sumCount += countHost[e];
    }
    printf("       sum(count)=%d   T_padded=%d   match=%s\n",
           sumCount, T_padded, (sumCount == T_padded ? "YES" : "NO (BUG?)"));

    CheckAcl(aclrtMemcpy(tokensDev, packedTokensBytes, tokensHost, packedTokensBytes, ACL_MEMCPY_HOST_TO_DEVICE),
             "aclrtMemcpy(tokensDev)");
    CheckAcl(aclrtMemcpy(w1Dev,     w1Bytes,           w1Host,     w1Bytes,           ACL_MEMCPY_HOST_TO_DEVICE),
             "aclrtMemcpy(w1Dev)");
    CheckAcl(aclrtMemcpy(w2Dev,     w2Bytes,           w2Host,     w2Bytes,           ACL_MEMCPY_HOST_TO_DEVICE),
             "aclrtMemcpy(w2Dev)");
    CheckAcl(aclrtMemcpy(countDev,  expertMetaBytes,   countHost,  expertMetaBytes,   ACL_MEMCPY_HOST_TO_DEVICE),
             "aclrtMemcpy(countDev)");
    CheckAcl(aclrtMemcpy(startDev,  expertMetaBytes,   startHost,  expertMetaBytes,   ACL_MEMCPY_HOST_TO_DEVICE),
             "aclrtMemcpy(startDev)");

    // PATCH 2 — poison scratch + output before the launch so the compare
    // script can tell "kernel didn't write" from "kernel wrote zeros".
    // aclrtMemset(devPtr, max_count, value_byte, count) replicates a single
    // byte across the buffer (see kernels/manual/a2a3/gemm_ar/main.cpp:835-839).
    CheckAcl(aclrtMemset(scratchDev, scratchBytes,      kPoisonScratch, scratchBytes),
             "aclrtMemset(scratchDev=0x7B)");
    CheckAcl(aclrtMemset(outputDev,  packedOutputBytes, kPoisonOutput,  packedOutputBytes),
             "aclrtMemset(outputDev=0x5A)");

    printf("[main] poisoned scratchDev=0x%02X (%zu B), outputDev=0x%02X (%zu B)\n",
           kPoisonScratch, scratchBytes, kPoisonOutput, packedOutputBytes);

    launchMoeSegmentedFfnTop1Fp16(
        outputDev, tokensDev, countDev, startDev,
        w1Dev, w2Dev, scratchDev, stream);

    if (!CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream")) {
        std::cerr << "[main] stream sync failed — kernel likely crashed or never ran.\n";
    }

    // PATCH 1 — copy BOTH packed_output AND scratch back to host.
    CheckAcl(aclrtMemcpy(outputHost,  packedOutputBytes, outputDev,  packedOutputBytes, ACL_MEMCPY_DEVICE_TO_HOST),
             "aclrtMemcpy(outputHost <- outputDev)");
    CheckAcl(aclrtMemcpy(scratchHost, scratchBytes,      scratchDev, scratchBytes,      ACL_MEMCPY_DEVICE_TO_HOST),
             "aclrtMemcpy(scratchHost <- scratchDev)");

    WriteFile("../output/output_packed_output.bin", outputHost,  packedOutputBytes);
    WriteFile("../output/output_scratch.bin",       scratchHost, scratchBytes);

    // Sanity peek: is the first byte still the poison pattern?
    printf("[main] after launch: outputHost[0]=0x%02X (poison=0x%02X), "
           "scratchHost[0]=0x%02X (poison=0x%02X)\n",
           outputHost[0],  kPoisonOutput,
           scratchHost[0], kPoisonScratch);

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
    aclrtFreeHost(scratchHost);
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
