/**
 * main_debug.cpp - host driver for moe_segmented_ffn_top1 DEBUG mode.
 *
 * Launches the GEMM1-only debug kernel
 * `launchMoeSegmentedFfnTop1DebugHiddenFp16` and writes the full-size
 * post-ReLU FP16 hidden state to ../output/debug_hidden_fp16.bin so the
 * host can compare it against the Python golden
 * ../output/golden_hidden_fp16.bin.
 *
 * Purpose: isolate Assumption A.combined (the combined
 *   TSTORE<AccTile<float>, GlobalTensor<half, ...>, AtomicNone, NormalRelu>
 * form) from Assumption A.reuse (tile reuse across GEMM1 / GEMM2 in the
 * main FFN kernel).
 *
 * Reads the same inputs as the main FFN driver (packed_tokens, expert_*,
 * w1). w2 is not needed.
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ../output/t_padded.txt                (single int line; written by gen_data.py)
 *   ../input/input_packed_tokens.bin      (T_PADDED * H float16)
 *   ../input/input_expert_count.bin       (kE        int32)
 *   ../input/input_expert_start.bin       (kE        int32)
 *   ../input/input_w1.bin                 (kE * H * F float16)
 *   ../output/golden_hidden_fp16.bin      (T_PADDED * F float16; written by gen_data.py)
 *   ../output/debug_hidden_fp16.bin       (T_PADDED * F float16; written by this driver)
 *
 * Compare with `python ../scripts/compare_hidden_debug.py` (from build/).
 *
 * Pattern source: main.cpp (this project) — same harness, fewer GM
 * buffers (no w2, no output FP32 buffer, no hidden_scratch — the debug
 * kernel writes directly into a full-size FP16 buffer).
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

extern "C" void launchMoeSegmentedFfnTop1DebugHiddenFp16(uint8_t *hidden_debug_out,
                                                          uint8_t *packed_tokens,
                                                          int32_t *expert_count,
                                                          int32_t *expert_start,
                                                          uint8_t *w1,
                                                          void *stream);

static int ReadTPadded()
{
    std::ifstream f("../output/t_padded.txt");
    if (!f.is_open()) {
        printf("[main_debug] FATAL: cannot open ../output/t_padded.txt; did scripts/gen_data.py run?\n");
        std::exit(2);
    }
    int t_padded = 0;
    f >> t_padded;
    if (t_padded <= 0) {
        printf("[main_debug] FATAL: invalid T_PADDED=%d from t_padded.txt\n", t_padded);
        std::exit(2);
    }
    return t_padded;
}

int main()
{
    constexpr int kH = 64;
    constexpr int kF = 64;
    constexpr int kE = 4;
    constexpr size_t halfBytes  = 2;
    constexpr size_t int32Bytes = 4;

    const int T_padded = ReadTPadded();
    size_t packedTokensBytes = static_cast<size_t>(T_padded) * kH * halfBytes;
    size_t hiddenDebugBytes  = static_cast<size_t>(T_padded) * kF * halfBytes;
    size_t w1Bytes           = static_cast<size_t>(kE) * kH * kF * halfBytes;
    size_t expertMetaBytes   = static_cast<size_t>(kE) * int32Bytes;

    printf("[main_debug] DEBUG mode (GEMM1 only; isolates A.combined)\n");
    printf("[main_debug] T_padded=%d  H=%d  F=%d  E=%d\n"
           "             packedTokensBytes=%zu  hiddenDebugBytes=%zu\n"
           "             w1Bytes=%zu  expertMetaBytes=%zu\n",
           T_padded, kH, kF, kE,
           packedTokensBytes, hiddenDebugBytes, w1Bytes, expertMetaBytes);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *tokensHost = nullptr, *hiddenHost = nullptr, *w1Host = nullptr;
    int32_t *countHost = nullptr, *startHost = nullptr;
    uint8_t *tokensDev = nullptr, *hiddenDev = nullptr, *w1Dev = nullptr;
    int32_t *countDev = nullptr, *startDev = nullptr;

    aclrtMallocHost((void **)(&tokensHost), packedTokensBytes);
    aclrtMallocHost((void **)(&hiddenHost), hiddenDebugBytes);
    aclrtMallocHost((void **)(&w1Host),     w1Bytes);
    aclrtMallocHost((void **)(&countHost),  expertMetaBytes);
    aclrtMallocHost((void **)(&startHost),  expertMetaBytes);

    aclrtMalloc((void **)&tokensDev, packedTokensBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&hiddenDev, hiddenDebugBytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&w1Dev,     w1Bytes,           ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&countDev,  expertMetaBytes,   ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&startDev,  expertMetaBytes,   ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("../input/input_packed_tokens.bin", packedTokensBytes, tokensHost, packedTokensBytes);
    ReadFile("../input/input_expert_count.bin",  expertMetaBytes,   countHost,  expertMetaBytes);
    ReadFile("../input/input_expert_start.bin",  expertMetaBytes,   startHost,  expertMetaBytes);
    ReadFile("../input/input_w1.bin",            w1Bytes,           w1Host,     w1Bytes);

    aclrtMemcpy(tokensDev, packedTokensBytes, tokensHost, packedTokensBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(w1Dev,     w1Bytes,           w1Host,     w1Bytes,           ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(countDev,  expertMetaBytes,   countHost,  expertMetaBytes,   ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(startDev,  expertMetaBytes,   startHost,  expertMetaBytes,   ACL_MEMCPY_HOST_TO_DEVICE);

    launchMoeSegmentedFfnTop1DebugHiddenFp16(hiddenDev, tokensDev,
                                              countDev, startDev, w1Dev, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(hiddenHost, hiddenDebugBytes, hiddenDev, hiddenDebugBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("../output/debug_hidden_fp16.bin", hiddenHost, hiddenDebugBytes);

    aclrtFree(startDev);
    aclrtFree(countDev);
    aclrtFree(w1Dev);
    aclrtFree(hiddenDev);
    aclrtFree(tokensDev);
    aclrtFreeHost(startHost);
    aclrtFreeHost(countHost);
    aclrtFreeHost(w1Host);
    aclrtFreeHost(hiddenHost);
    aclrtFreeHost(tokensHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    printf("[main_debug] wrote ../output/debug_hidden_fp16.bin (%zu bytes)\n", hiddenDebugBytes);
    printf("[main_debug] run `python ../scripts/compare_hidden_debug.py` to compare against golden_hidden_fp16.bin\n");
    return 0;
}
