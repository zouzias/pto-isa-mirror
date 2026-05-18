/**
 * main.cpp - host driver for full_moe_combined.
 *
 * End-to-end MoE forward pass driven by a host wrapper (`launchFullMoeCombined`)
 * that fires six `__global__ AICORE` kernels on the same stream.
 *
 *   launchFullMoeCombined(...)
 *     -> launchRouterMatmulFp16     (cube)
 *     -> launchMoeTopkPadded<float> (vec)
 *     -> launchOutValPad<float>     (vec)  // device-side pad bridge
 *     -> launchScatterFp16          (vec)
 *     -> launchExpertFfnFp16        (cube + cube)
 *     -> launchGather<float>        (vec)
 *
 * Compared with full_moe_separate, this folder:
 *   - Uses a device-side `outval_pad` kernel instead of a host-side memcpy
 *     bridge to convert outVal (kT, kTopK) -> (kT, kPadded) with -1e30 in
 *     padding columns.
 *   - Times each kernel launch with an immediate aclrtSynchronizeStream so
 *     per-stage latency is visible from host logs.
 *
 * I/O contract — identical to full_moe_separate (same gen_data.py).
 */

#include "test_common.h"
#include "acl/acl.h"
#include "../../kernel_timing.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

// ----------------------------------------------------------------------------
// Kernel launcher declarations (one per kernel `.so` library).
// ----------------------------------------------------------------------------

extern "C" void launchRouterMatmulFp16(uint8_t *logits, uint8_t *x, uint8_t *w_router, void *stream);

template <typename T>
void launchMoeTopkPadded(uint8_t *outVal, uint8_t *outIdx,
                         uint8_t *src,    uint8_t *idx,
                         void *stream);

template <typename T>
void launchOutValPad(T *outVal_padded, T *outVal_compact, void *stream);

extern "C" void launchScatterFp16(uint8_t *A, int32_t *A_id, int32_t *rank_id,
                                  int32_t *expert_count, int32_t *expert_start,
                                  uint8_t *X, int32_t *expert_id,
                                  void *stream);

extern "C" void launchExpertFfnFp16(uint8_t *B, uint8_t *A,
                                    int32_t *expert_count, int32_t *expert_start,
                                    uint8_t *W1, uint8_t *W2,
                                    uint8_t *Y_scratch, void *stream);

template <typename T>
void launchGather(T *C, T *B,
                  int32_t *A_id, int32_t *rank_id,
                  T *outVal, T *weights_scratch,
                  void *stream);

// ----------------------------------------------------------------------------
// Host wrapper. Fires six kernels on `stream` and times each stage.
// ----------------------------------------------------------------------------
static void launchFullMoeCombined(
    float    *C,
    uint8_t  *X,         uint8_t  *W_router, uint8_t  *W1, uint8_t  *W2,
    uint8_t  *idx_init,
    uint8_t  *logits,
    uint8_t  *expert_id_u8, int32_t *expert_id_i32,
    uint8_t  *outVal_compact_u8, float *outVal_compact_f,
    float    *outVal_padded,
    uint8_t  *A,         int32_t  *A_id,     int32_t  *rank_id,
    int32_t  *expert_count, int32_t *expert_start,
    uint8_t  *Y_scratch, uint8_t  *B,
    float    *weights_scratch,
    void     *stream)
{
    (void)PtoTiming::TimeKernelCallUs("full_moe_combined/router_matmul", stream, [&]() {
        launchRouterMatmulFp16(logits, X, W_router, stream);
    });
    (void)PtoTiming::TimeKernelCallUs("full_moe_combined/moe_topk_padded", stream, [&]() {
        launchMoeTopkPadded<float>(outVal_compact_u8, expert_id_u8, logits, idx_init, stream);
    });
    (void)PtoTiming::TimeKernelCallUs("full_moe_combined/outval_pad", stream, [&]() {
        launchOutValPad<float>(outVal_padded, outVal_compact_f, stream);
    });
    (void)PtoTiming::TimeKernelCallUs("full_moe_combined/scatter", stream, [&]() {
        launchScatterFp16(A, A_id, rank_id, expert_count, expert_start, X, expert_id_i32, stream);
    });
    (void)PtoTiming::TimeKernelCallUs("full_moe_combined/expert_ffn", stream, [&]() {
        launchExpertFfnFp16(B, A, expert_count, expert_start, W1, W2, Y_scratch, stream);
    });
    (void)PtoTiming::TimeKernelCallUs("full_moe_combined/gather", stream, [&]() {
        launchGather<float>(C, reinterpret_cast<float *>(B), A_id, rank_id,
                            outVal_padded, weights_scratch, stream);
    });
}

// ----------------------------------------------------------------------------
namespace {

// v1 shape — patched by sweep.sh.
constexpr int kT     = 256;
constexpr int kH     = 64;
constexpr int kF     = 64;
constexpr int kE     = 32;
constexpr int kTopK  = 1;

constexpr int kPackedRows   = kT * kTopK;
constexpr int kOverspillPad = 16;
constexpr int kAlloc        = kPackedRows + kOverspillPad;

constexpr int kPadded = (kTopK < 8) ? 8 : kTopK;

constexpr size_t kHalfBytes  = 2;
constexpr size_t kFloatBytes = 4;

constexpr float kOutValPad = -1e30f;

}  // namespace

int main()
{
    // Buffer sizes.
    size_t xBytes              = static_cast<size_t>(kT)     * kH         * kHalfBytes;
    size_t wRouterBytes        = static_cast<size_t>(kH)     * kE         * kHalfBytes;
    size_t w1Bytes             = static_cast<size_t>(kE)     * kH * kF    * kHalfBytes;
    size_t w2Bytes             = static_cast<size_t>(kE)     * kF * kH    * kHalfBytes;
    size_t idxInitBytes        = static_cast<size_t>(kE)                  * sizeof(uint32_t);

    size_t logitsBytes         = static_cast<size_t>(kT)     * kE         * kFloatBytes;
    size_t expertIdBytes       = static_cast<size_t>(kT)     * kTopK      * sizeof(uint32_t);
    size_t outValCompactBytes  = static_cast<size_t>(kT)     * kTopK      * kFloatBytes;
    size_t outValPaddedBytes   = static_cast<size_t>(kT)     * kPadded    * kFloatBytes;

    size_t aBytes              = static_cast<size_t>(kAlloc) * kH         * kHalfBytes;
    size_t aIdBytes            = static_cast<size_t>(kAlloc)              * sizeof(int32_t);
    size_t rankIdBytes         = aIdBytes;
    size_t countBytes          = static_cast<size_t>(kE)                  * sizeof(int32_t);
    size_t startBytes          = countBytes;

    size_t yScratchBytes       = static_cast<size_t>(kAlloc) * kF         * kHalfBytes;
    size_t bBytes              = static_cast<size_t>(kAlloc) * kH         * kFloatBytes;
    size_t weightsScratchBytes = static_cast<size_t>(kT)     * kPadded    * kFloatBytes;

    size_t cBytes              = static_cast<size_t>(kT)     * kH         * kFloatBytes;

    printf("[main] kT=%d  kH=%d  kF=%d  kE=%d  kTopK=%d  kPadded=%d  kAlloc=%d\n",
           kT, kH, kF, kE, kTopK, kPadded, kAlloc);

    // ACL init.
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    // Host allocations: inputs + final output + pre-padded outVal seed.
    uint8_t  *xHost = nullptr, *wRouterHost = nullptr, *w1Host = nullptr, *w2Host = nullptr;
    uint32_t *idxInitHost = nullptr;
    float    *outValPaddedSeedHost = nullptr;
    float    *cHost = nullptr;

    aclrtMallocHost((void **)&xHost,                xBytes);
    aclrtMallocHost((void **)&wRouterHost,          wRouterBytes);
    aclrtMallocHost((void **)&w1Host,               w1Bytes);
    aclrtMallocHost((void **)&w2Host,               w2Bytes);
    aclrtMallocHost((void **)&idxInitHost,          idxInitBytes);
    aclrtMallocHost((void **)&outValPaddedSeedHost, outValPaddedBytes);
    aclrtMallocHost((void **)&cHost,                cBytes);

    // Device allocations.
    uint8_t  *xDev = nullptr, *wRouterDev = nullptr, *w1Dev = nullptr, *w2Dev = nullptr;
    uint32_t *idxInitDev = nullptr;
    uint8_t  *logitsDev = nullptr;
    uint32_t *expertIdDev = nullptr;
    float    *outValCompactDev = nullptr, *outValPaddedDev = nullptr;
    uint8_t  *aDev = nullptr, *yScratchDev = nullptr, *bDev = nullptr;
    int32_t  *aIdDev = nullptr, *rankIdDev = nullptr, *countDev = nullptr, *startDev = nullptr;
    float    *weightsScratchDev = nullptr;
    float    *cDev = nullptr;

    aclrtMalloc((void **)&xDev,               xBytes,              ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&wRouterDev,         wRouterBytes,        ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&w1Dev,              w1Bytes,             ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&w2Dev,              w2Bytes,             ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&idxInitDev,         idxInitBytes,        ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMalloc((void **)&logitsDev,          logitsBytes,         ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&expertIdDev,        expertIdBytes,       ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&outValCompactDev,   outValCompactBytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&outValPaddedDev,    outValPaddedBytes,   ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMalloc((void **)&aDev,               aBytes,              ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&aIdDev,             aIdBytes,            ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&rankIdDev,          rankIdBytes,         ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&countDev,           countBytes,          ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&startDev,           startBytes,          ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMalloc((void **)&yScratchDev,        yScratchBytes,       ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&bDev,               bBytes,              ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&weightsScratchDev,  weightsScratchBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMalloc((void **)&cDev,               cBytes,              ACL_MEM_MALLOC_HUGE_FIRST);

    // Load inputs.
    ReadFile("../input/input_X.bin",        xBytes,       xHost,       xBytes);
    ReadFile("../input/input_W_router.bin", wRouterBytes, wRouterHost, wRouterBytes);
    ReadFile("../input/input_W1.bin",       w1Bytes,      w1Host,      w1Bytes);
    ReadFile("../input/input_W2.bin",       w2Bytes,      w2Host,      w2Bytes);
    ReadFile("../input/input_idx_init.bin", idxInitBytes, idxInitHost, idxInitBytes);

    // Pre-pad the outVal_padded GM buffer with -1e30 so the device-side
    // outval_pad kernel only needs to overwrite the first kTopK columns of
    // each row; padding columns kTopK..kPadded-1 stay -1e30.
    for (size_t i = 0; i < outValPaddedBytes / sizeof(float); ++i) {
        outValPaddedSeedHost[i] = kOutValPad;
    }

    // H2D copy: inputs + the pre-padded outVal_padded seed.
    aclrtMemcpy(xDev,            xBytes,            xHost,                xBytes,            ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(wRouterDev,      wRouterBytes,      wRouterHost,          wRouterBytes,      ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(w1Dev,           w1Bytes,           w1Host,               w1Bytes,           ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(w2Dev,           w2Bytes,           w2Host,               w2Bytes,           ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(idxInitDev,      idxInitBytes,      idxInitHost,          idxInitBytes,      ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(outValPaddedDev, outValPaddedBytes, outValPaddedSeedHost, outValPaddedBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    // Zero-init C. Poison the scratches.
    aclrtMemset(cDev,              cBytes,             0x00, cBytes);
    aclrtMemset(aDev,              aBytes,             0x5A, aBytes);
    aclrtMemset(aIdDev,            aIdBytes,           0xFF, aIdBytes);
    aclrtMemset(rankIdDev,         rankIdBytes,        0xFF, rankIdBytes);
    aclrtMemset(yScratchDev,       yScratchBytes,      0x6C, yScratchBytes);
    aclrtMemset(bDev,              bBytes,             0x4D, bBytes);
    aclrtMemset(weightsScratchDev, weightsScratchBytes, 0x77, weightsScratchBytes);

    // ========================================================================
    // One host wrapper, six timed kernel launches on the stream.
    // ========================================================================
    printf("[combined] launching full MoE pipeline...\n");
    launchFullMoeCombined(
        cDev,
        xDev, wRouterDev, w1Dev, w2Dev,
        reinterpret_cast<uint8_t *>(idxInitDev),
        logitsDev,
        reinterpret_cast<uint8_t *>(expertIdDev),
        reinterpret_cast<int32_t *>(expertIdDev),
        reinterpret_cast<uint8_t *>(outValCompactDev),
        outValCompactDev,
        outValPaddedDev,
        aDev, aIdDev, rankIdDev, countDev, startDev,
        yScratchDev, bDev,
        weightsScratchDev,
        stream);

    // D2H final output.
    aclrtMemcpy(cHost, cBytes, cDev, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("../output/output_C.bin", cHost, cBytes);

    // Free.
    aclrtFree(cDev);
    aclrtFree(weightsScratchDev);
    aclrtFree(bDev);
    aclrtFree(yScratchDev);
    aclrtFree(startDev);
    aclrtFree(countDev);
    aclrtFree(rankIdDev);
    aclrtFree(aIdDev);
    aclrtFree(aDev);
    aclrtFree(outValPaddedDev);
    aclrtFree(outValCompactDev);
    aclrtFree(expertIdDev);
    aclrtFree(logitsDev);
    aclrtFree(idxInitDev);
    aclrtFree(w2Dev);
    aclrtFree(w1Dev);
    aclrtFree(wRouterDev);
    aclrtFree(xDev);
    aclrtFreeHost(cHost);
    aclrtFreeHost(outValPaddedSeedHost);
    aclrtFreeHost(idxInitHost);
    aclrtFreeHost(w2Host);
    aclrtFreeHost(w1Host);
    aclrtFreeHost(wRouterHost);
    aclrtFreeHost(xHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    // Validate final C against golden.
    std::vector<float> gold(cBytes / sizeof(float));
    std::vector<float> out (cBytes / sizeof(float));
    ReadFile("../output/golden_C.bin", cBytes, gold.data(), cBytes);
    ReadFile("../output/output_C.bin", cBytes, out.data(),  cBytes);

    // Same 5e-2 abs tol as full_moe_separate — full pipeline error budget.
    bool ok = ResultCmp(gold, out, 5e-2f);
    printf("C                : %s\n", ok ? "success" : "FAILED");
    if (ok) {
        printf("test data success\n");
        printf("test success\n");
    } else {
        printf("test data failed\n");
        printf("test failed\n");
    }
    return ok ? 0 : 1;
}
