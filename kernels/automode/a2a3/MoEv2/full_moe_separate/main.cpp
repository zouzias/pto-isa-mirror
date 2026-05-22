/**
 * main.cpp - host driver for full_moe_separate.
 *
 * End-to-end MoE forward pass driven by FIVE separate kernel launches with
 * explicit aclrtSynchronizeStream between each stage:
 *
 *   1. router_matmul    : logits = X @ W_router
 *   2. moe_topk_padded  : expert_id, outVal_compact = topK(logits)
 *   <host bridge>       : pad outVal_compact (kT, kTopK) -> outVal_padded (kT, kPadded)
 *                          with -1e30 in cols kTopK..kPadded-1 (needed by gather softmax).
 *   3. scatter          : A, A_id, rank_id, count, start = pack(X, expert_id)
 *   4. expert_ffn       : B = FFN(A, W1, W2, count, start)   (2 stages, cube)
 *   5. gather           : C = softmax_weighted_gather(B, A_id, rank_id, outVal_padded)
 *
 * The "separate" model: each launcher is fired individually with
 * aclrtSynchronizeStream between each so that we can intercept intermediates
 * (currently for the outVal padding bridge). The companion full_moe_combined
 * folder runs the same pipeline with one host wrapper and a device-side
 * outval_pad kernel instead of the host bridge here.
 *
 * I/O contract (all little-endian, contiguous, no header):
 *   ../input/input_X.bin           kT * kH            half (fp16)
 *   ../input/input_W_router.bin    kH * kE            half
 *   ../input/input_W1.bin          kE * kH * kF       half
 *   ../input/input_W2.bin          kE * kF * kH       half
 *   ../input/input_idx_init.bin    kE                 uint32   (identity 0..kE-1 for topk)
 *   ../output/golden_C.bin         kT * kH            float32
 *   ../output/output_C.bin         kT * kH            float32  (kernel-emitted, this run)
 */

#include "test_common.h"
#include "acl/acl.h"
#include "../../kernel_timing.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

// ----------------------------------------------------------------------------
// Kernel launcher declarations (one per stage's .so library).
// ----------------------------------------------------------------------------

extern "C" void launchRouterMatmulFp16(uint8_t *logits, uint8_t *x, uint8_t *w_router, void *stream);

template <typename T>
void launchMoeTopkPadded(uint8_t *outVal, uint8_t *outIdx,
                         uint8_t *src,    uint8_t *idx,
                         void *stream);

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
namespace {

// v1 shape — patched by sweep.sh.
constexpr int kT     = 256;
constexpr int kH     = 64;
constexpr int kF     = 64;
constexpr int kE     = 32;
constexpr int kTopK  = 1;

constexpr int kPackedRows   = kT * kTopK;
constexpr int kOverspillPad = 64;
constexpr int kAlloc        = kPackedRows + kOverspillPad;

// Softmax tile column padding for 32-byte UB alignment in gather.
// fp32 -> Cols * 4 % 32 == 0 -> Cols % 8 == 0.
constexpr int kPadded = (kTopK < 8) ? 8 : kTopK;

// router_matmul tile height (must match router_matmul_cfg::kTileM).
// The loop runs while m0 < kT with stride kTileM, so the last tile reads
// kTileM rows from X and writes kTileM rows to logits even when kT is not
// a multiple of kTileM.  Allocate both buffers with the padded row count so
// the tail TLOAD/TSTORE stays within the device allocation.  The padding
// rows of X are zero-filled (zero @ W_router = 0) and the padding rows of
// logits are never read by the topk kernel (it loops over exactly kT rows).
constexpr int kTileM_router = 128;
constexpr int kTRouterAlloc = ((kT + kTileM_router - 1) / kTileM_router) * kTileM_router;

// Cube blockAlign for fp16 = 16. W1, W2, the ABI Y-scratch, and A are zero-padded to
// K/N-aligned dimensions so GEMM MatTile padding columns read zeros.
// NOTE: when kH or kF are not multiples of 16, the scatter kernel still writes
// A with stride kH (unaligned). For the full pipeline with non-aligned kH,
// aDev must be zero-initialized and scatter's dstGlobal stride must use
// kH_aligned — tracked as a known limitation for non-aligned kH.
constexpr int kH_aligned = ((kH + 15) / 16) * 16;
constexpr int kF_aligned = ((kF + 15) / 16) * 16;

constexpr size_t kHalfBytes  = 2;
constexpr size_t kFloatBytes = 4;

constexpr float  kOutValPad  = -1e30f;

}  // namespace

int main()
{
    // ------------------------------------------------------------------------
    // Sizes.
    // ------------------------------------------------------------------------
    size_t xBytes              = static_cast<size_t>(kT)           * kH      * kHalfBytes;
    size_t xDevBytes           = static_cast<size_t>(kTRouterAlloc) * kH      * kHalfBytes;  // padded
    size_t wRouterBytes        = static_cast<size_t>(kH)     * kE         * kHalfBytes;
    size_t w1Bytes             = static_cast<size_t>(kE) * kH_aligned * kF_aligned * kHalfBytes;
    size_t w2Bytes             = static_cast<size_t>(kE) * kF_aligned * kH_aligned * kHalfBytes;
    size_t idxInitBytes        = static_cast<size_t>(kE)                  * sizeof(uint32_t);

    size_t logitsBytes         = static_cast<size_t>(kT)           * kE    * kFloatBytes;
    size_t logitsDevBytes      = static_cast<size_t>(kTRouterAlloc) * kE    * kFloatBytes;   // padded
    size_t expertIdBytes       = static_cast<size_t>(kT)     * kTopK      * sizeof(uint32_t);
    size_t outValCompactBytes  = static_cast<size_t>(kT)     * kTopK      * kFloatBytes;
    size_t outValPaddedBytes   = static_cast<size_t>(kT)     * kPadded    * kFloatBytes;

    size_t aBytes              = static_cast<size_t>(kAlloc) * kH         * kHalfBytes;
    size_t aIdBytes            = static_cast<size_t>(kAlloc)              * sizeof(int32_t);
    size_t rankIdBytes         = aIdBytes;
    size_t countBytes          = static_cast<size_t>(kE)                  * sizeof(int32_t);
    size_t startBytes          = countBytes;

    size_t yScratchBytes       = static_cast<size_t>(kAlloc) * kF_aligned  * kHalfBytes;
    size_t bBytes              = static_cast<size_t>(kAlloc) * kH          * kFloatBytes;
    size_t weightsScratchBytes = static_cast<size_t>(kT)     * kPadded    * kFloatBytes;

    size_t cBytes              = static_cast<size_t>(kT)     * kH         * kFloatBytes;

    printf("[main] kT=%d  kH=%d  kF=%d  kE=%d  kTopK=%d  kPadded=%d  kAlloc=%d\n",
           kT, kH, kF, kE, kTopK, kPadded, kAlloc);

    // ------------------------------------------------------------------------
    // ACL init.
    // ------------------------------------------------------------------------
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    // ------------------------------------------------------------------------
    // Host allocations (inputs + final output + outVal bridge).
    // ------------------------------------------------------------------------
    uint8_t  *xHost = nullptr, *wRouterHost = nullptr, *w1Host = nullptr, *w2Host = nullptr;
    uint32_t *idxInitHost = nullptr;
    float    *outValCompactHost = nullptr, *outValPaddedHost = nullptr;
    float    *cHost = nullptr;

    aclrtMallocHost((void **)&xHost,             xBytes);
    aclrtMallocHost((void **)&wRouterHost,       wRouterBytes);
    aclrtMallocHost((void **)&w1Host,            w1Bytes);
    aclrtMallocHost((void **)&w2Host,            w2Bytes);
    aclrtMallocHost((void **)&idxInitHost,       idxInitBytes);
    aclrtMallocHost((void **)&outValCompactHost, outValCompactBytes);
    aclrtMallocHost((void **)&outValPaddedHost,  outValPaddedBytes);
    aclrtMallocHost((void **)&cHost,             cBytes);

    // ------------------------------------------------------------------------
    // Device allocations (inputs + intermediates + outputs + scratches).
    // ------------------------------------------------------------------------
    uint8_t  *xDev = nullptr, *wRouterDev = nullptr, *w1Dev = nullptr, *w2Dev = nullptr;
    uint32_t *idxInitDev = nullptr;
    uint8_t  *logitsDev = nullptr;
    uint32_t *expertIdDev = nullptr;
    float    *outValCompactDev = nullptr, *outValPaddedDev = nullptr;
    uint8_t  *aDev = nullptr, *yScratchDev = nullptr, *bDev = nullptr;
    int32_t  *aIdDev = nullptr, *rankIdDev = nullptr, *countDev = nullptr, *startDev = nullptr;
    float    *weightsScratchDev = nullptr;
    float    *cDev = nullptr;

    aclrtMalloc((void **)&xDev,               xDevBytes,           ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&wRouterDev,         wRouterBytes,        ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&w1Dev,              w1Bytes,             ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&w2Dev,              w2Bytes,             ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&idxInitDev,         idxInitBytes,        ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMalloc((void **)&logitsDev,          logitsDevBytes,      ACL_MEM_MALLOC_HUGE_FIRST);
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

    // ------------------------------------------------------------------------
    // Load inputs from disk.
    // ------------------------------------------------------------------------
    ReadFile("../input/input_X.bin",        xBytes,       xHost,       xBytes);
    ReadFile("../input/input_W_router.bin", wRouterBytes, wRouterHost, wRouterBytes);
    ReadFile("../input/input_W1.bin",       w1Bytes,      w1Host,      w1Bytes);
    ReadFile("../input/input_W2.bin",       w2Bytes,      w2Host,      w2Bytes);
    ReadFile("../input/input_idx_init.bin", idxInitBytes, idxInitHost, idxInitBytes);

    // Pre-fill the host-side outVal_padded buffer with -1e30 so the kTopK..
    // kPadded-1 padding columns are already in place; we only copy the
    // first kTopK fp32 floats of each row from the compact buffer.
    for (size_t i = 0; i < outValPaddedBytes / sizeof(float); ++i) {
        outValPaddedHost[i] = kOutValPad;
    }

    // ------------------------------------------------------------------------
    // H2D copy: inputs + the pre-padded outVal_padded.
    // C must be zero on entry (gather pass-2 TLOAD-TADD-TSTORE accumulates).
    // ------------------------------------------------------------------------
    // Zero-fill the full padded xDev first so kT..kTRouterAlloc-1 tail rows
    // that the router_matmul last tile reads are 0 (not garbage device memory).
    aclrtMemset(xDev, xDevBytes, 0x00, xDevBytes);
    aclrtMemcpy(xDev,         xDevBytes,    xHost,       xBytes,       ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(wRouterDev,   wRouterBytes, wRouterHost, wRouterBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(w1Dev,        w1Bytes,      w1Host,      w1Bytes,      ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(w2Dev,        w2Bytes,      w2Host,      w2Bytes,      ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(idxInitDev,   idxInitBytes, idxInitHost, idxInitBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(outValPaddedDev, outValPaddedBytes, outValPaddedHost,
                outValPaddedBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    aclrtMemset(cDev, cBytes, 0x00, cBytes);
    // Initialize scratches with poison so we can detect stages that skip writes.
    aclrtMemset(aDev,              aBytes,             0x5A, aBytes);
    aclrtMemset(aIdDev,            aIdBytes,           0xFF, aIdBytes);
    aclrtMemset(rankIdDev,         rankIdBytes,        0xFF, rankIdBytes);
    aclrtMemset(yScratchDev,       yScratchBytes,      0x6C, yScratchBytes);
    aclrtMemset(bDev,              bBytes,             0x4D, bBytes);
    aclrtMemset(weightsScratchDev, weightsScratchBytes, 0x77, weightsScratchBytes);

    // ========================================================================
    // Pipeline — five kernel launches with sync between each (the "separate"
    // model). After the host bridge, six total launches.
    // ========================================================================

    // --- Stage 1: router matmul ---------------------------------------------
    printf("[stage 1] router_matmul\n");
    (void)PtoTiming::TimeKernelCallUs("full_moe_separate/router_matmul", stream, [&]() {
        launchRouterMatmulFp16(logitsDev, xDev, wRouterDev, stream);
    });

    // --- Stage 2: top-K -----------------------------------------------------
    printf("[stage 2] moe_topk_padded\n");
    (void)PtoTiming::TimeKernelCallUs("full_moe_separate/moe_topk_padded", stream, [&]() {
        launchMoeTopkPadded<float>(reinterpret_cast<uint8_t *>(outValCompactDev),
                                   reinterpret_cast<uint8_t *>(expertIdDev),
                                   logitsDev,
                                   reinterpret_cast<uint8_t *>(idxInitDev),
                                   stream);
    });

    // --- Host bridge: pad outVal_compact (kT, kTopK) -> outVal_padded
    //                  (kT, kPadded) with -1e30 in the trailing kPadded - kTopK
    //                  columns. We pre-filled outValPaddedHost with -1e30 above,
    //                  so we only memcpy the first kTopK floats per row in.
    printf("[bridge ] host pad outVal_compact -> outVal_padded\n");
    aclrtMemcpy(outValCompactHost, outValCompactBytes, outValCompactDev,
                outValCompactBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    for (int t = 0; t < kT; ++t) {
        std::memcpy(&outValPaddedHost[t * kPadded],
                    &outValCompactHost[t * kTopK],
                    static_cast<size_t>(kTopK) * sizeof(float));
    }
    aclrtMemcpy(outValPaddedDev, outValPaddedBytes, outValPaddedHost,
                outValPaddedBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    // --- Stage 3: scatter ---------------------------------------------------
    printf("[stage 3] scatter\n");
    (void)PtoTiming::TimeKernelCallUs("full_moe_separate/scatter", stream, [&]() {
        launchScatterFp16(aDev, aIdDev, rankIdDev, countDev, startDev, xDev,
                          reinterpret_cast<int32_t *>(expertIdDev), stream);
    });

    // --- Stage 4: expert FFN (fused cube) ----------------------------------
    printf("[stage 4] expert_ffn\n");
    (void)PtoTiming::TimeKernelCallUs("full_moe_separate/expert_ffn", stream, [&]() {
        launchExpertFfnFp16(bDev, aDev, countDev, startDev, w1Dev, w2Dev,
                            yScratchDev, stream);
    });

    // --- Stage 5: gather (softmax-weighted) ---------------------------------
    printf("[stage 5] gather\n");
    (void)PtoTiming::TimeKernelCallUs("full_moe_separate/gather", stream, [&]() {
        launchGather<float>(cDev, reinterpret_cast<float *>(bDev),
                            aIdDev, rankIdDev,
                            outValPaddedDev, weightsScratchDev, stream);
    });

    // ------------------------------------------------------------------------
    // D2H + validate.
    // ------------------------------------------------------------------------
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
    aclrtFreeHost(outValPaddedHost);
    aclrtFreeHost(outValCompactHost);
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

    // 5e-2 abs tol: the full pipeline stacks fp16 matmul rounding, fp32 acc,
    // fp32->fp16 cast between GEMMs, softmax, weighted sum. We're forgiving.
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
