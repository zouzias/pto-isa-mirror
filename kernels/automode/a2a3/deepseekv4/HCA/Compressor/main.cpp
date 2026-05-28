/**
 * main.cpp - host driver for HCA Compressor (auto-mode A3 prototype).
 *
 * Heavily Compressed Attention Compressor for DeepSeek-V4-Flash.
 * Drives three kernel stages on a single ACL stream; stream-order semantics
 * (§A18 in docs_for_ai/known_good_kernel_examples.md) guarantee sequential
 * execution. An explicit aclrtSynchronizeStream barrier is inserted after each
 * launch so that device writes are visible to the host and to the next stage.
 *
 * Pipeline:
 *   1. GEMM         (cube)  H @ W_kv^T -> C,  H @ W_z^T -> Z
 *   2. Biased Softmax (vec) softmax(Z.view(B,nb,r,c) + B, dim=2) -> scores
 *   3. Weighted Sum  (vec)  (C.view(B,nb,r,c) * scores).sum(dim=2) -> out
 *
 * I/O contract (little-endian, contiguous, no header):
 *   ../input/input_H.bin        [B*S*d     float16]
 *   ../input/input_W_kv.bin     [c*d       float16]   weight [out=c, in=d]
 *   ../input/input_W_z.bin      [c*d       float16]
 *   ../input/input_B.bin        [r*c       float32]   positional bias (fp32)
 *   ../output/golden_output.bin [B*nb*c    float16]   PyTorch reference
 *   ../output/output_npu.bin    [B*nb*c    float16]   written by this driver
 *
 * Poison bytes (detect "kernel never wrote" in compare_outputs.py):
 *   C       : 0xA1   Z       : 0xA2   scores  : 0xA3   output  : 0xA4
 */

#include "test_common.h"
#include "acl/acl.h"
#include "../kernel_timing.h"
#include "generated_cases.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

// ---------------------------------------------------------------------------
// Kernel launchers — implemented in the three kernel translation units.
// ---------------------------------------------------------------------------

extern "C" void launch_gemm(
    uint8_t       *C,
    uint8_t       *Z,
    const uint8_t *H,
    const uint8_t *W_kv,
    const uint8_t *W_z,
    void          *stream);

extern "C" void launch_biased_softmax(
    uint8_t       *scores,
    const uint8_t *Z,
    const uint8_t *B,
    void          *stream);

extern "C" void launch_weighted_sum(
    uint8_t       *output,
    const uint8_t *C,
    const uint8_t *scores,
    void          *stream);

// ---------------------------------------------------------------------------
// Shape constants from generated_cases.h
// ---------------------------------------------------------------------------

static constexpr int kB  = kHcaBatch;
static constexpr int kS  = kHcaSeqLen;
static constexpr int kD  = kHcaHiddenDim;
static constexpr int kC  = kHcaCompressDim;
static constexpr int kR  = kHcaCompressRatio;
static constexpr int kNb = kHcaNumBlocks;    // S / R

// Element counts
static constexpr size_t kHNumel      = (size_t)kB * kS * kD;   // [B, S, d]
static constexpr size_t kWNumel      = (size_t)kC * kD;         // [c, d]
static constexpr size_t kBiasNumel   = (size_t)kR * kC;         // [r, c]
static constexpr size_t kCZNumel     = (size_t)kB * kS * kC;    // [B, S, c]
static constexpr size_t kScoreNumel  = (size_t)kB * kNb * kR * kC; // [B,nb,r,c]
static constexpr size_t kOutNumel    = (size_t)kB * kNb * kC;   // [B, nb, c]

static constexpr size_t kHalfBytes = 2;   // sizeof(float16)
static constexpr size_t kFloatBytes = 4;  // sizeof(float32)

// Byte sizes
static constexpr size_t kHBytes      = kHNumel    * kHalfBytes;
static constexpr size_t kWBytes      = kWNumel    * kHalfBytes;
static constexpr size_t kBiasBytes   = kBiasNumel * kFloatBytes;
static constexpr size_t kCZBytes     = kCZNumel   * kHalfBytes;
static constexpr size_t kScoreBytes  = kScoreNumel * kHalfBytes;
static constexpr size_t kOutBytes    = kOutNumel  * kHalfBytes;

// ---------------------------------------------------------------------------
// Helper: validate one output buffer against a golden file
// ---------------------------------------------------------------------------

static bool ValidateStage(
    const char   *name,
    const char   *golden_path,
    const char   *out_path,
    size_t        byte_count,
    uint8_t       poison_byte)
{
    vector<uint8_t> golden(byte_count);
    vector<uint8_t> actual(byte_count);

    ReadFile(golden_path, byte_count, golden.data(), byte_count);
    ReadFile(out_path,    byte_count, actual.data(), byte_count);

    // Check if output is still all-poisoned (kernel never wrote)
    bool poisoned = true;
    for (size_t i = 0; i < min(byte_count, (size_t)256); ++i) {
        if (actual[i] != poison_byte) { poisoned = false; break; }
    }
    if (poisoned) {
        printf("[%s] POISONED — kernel never wrote (poison=0x%02X)\n", name, poison_byte);
        return false;
    }

    // Use ResultCmp from test_common.h (element-wise, fp16 reinterpreted as uint16 pairs)
    vector<uint16_t> g16(byte_count / kHalfBytes);
    vector<uint16_t> a16(byte_count / kHalfBytes);
    memcpy(g16.data(), golden.data(), byte_count);
    memcpy(a16.data(), actual.data(), byte_count);

    // ResultCmp on raw uint16 would compare bit patterns; use float tolerance via direct loop.
    const uint16_t *gp = reinterpret_cast<const uint16_t *>(golden.data());
    const uint16_t *ap = reinterpret_cast<const uint16_t *>(actual.data());
    size_t n = byte_count / kHalfBytes;
    size_t mismatches = 0;
    float  max_err    = 0.f;

    for (size_t i = 0; i < n; ++i) {
        // Reinterpret fp16 bits as float for comparison
        float gf, af;
        uint16_t gv = gp[i], av = ap[i];
        uint32_t gw = (uint32_t)gv << 16, aw = (uint32_t)av << 16;
        memcpy(&gf, &gw, 4);
        memcpy(&af, &aw, 4);
        float err = gf - af;
        if (err < 0.f) err = -err;
        if (err > max_err) max_err = err;
        // tolerance: atol=0.5, rtol=0.05
        float tol = 0.5f + 0.05f * (gf < 0.f ? -gf : gf);
        if (err > tol) ++mismatches;
    }

    bool ok = (mismatches == 0);
    printf("[%s] %s  n=%zu  mismatches=%zu  max_err=%.5f\n",
           name, ok ? "PASS" : "FAIL", n, mismatches, max_err);
    return ok;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main()
{
    printf("HCA Compressor — case: %s\n", kHcaCaseName);
    printf("  B=%d  S=%d  d=%d  c=%d  r=%d  nb=%d\n",
           kB, kS, kD, kC, kR, kNb);

    // ACL init
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    // --- Host buffers --------------------------------------------------
    uint8_t *hH     = nullptr, *hW_kv  = nullptr, *hW_z   = nullptr;
    uint8_t *hBias  = nullptr;
    uint8_t *hOut   = nullptr;
    uint8_t *hGolden = nullptr;

    aclrtMallocHost((void **)&hH,      kHBytes);
    aclrtMallocHost((void **)&hW_kv,   kWBytes);
    aclrtMallocHost((void **)&hW_z,    kWBytes);
    aclrtMallocHost((void **)&hBias,   kBiasBytes);
    aclrtMallocHost((void **)&hOut,    kOutBytes);
    aclrtMallocHost((void **)&hGolden, kOutBytes);

    // --- Device buffers ------------------------------------------------
    uint8_t *dH    = nullptr, *dW_kv = nullptr, *dW_z  = nullptr;
    uint8_t *dBias = nullptr;
    uint8_t *dC    = nullptr, *dZ    = nullptr;
    uint8_t *dScores = nullptr;
    uint8_t *dOut  = nullptr;

    aclrtMalloc((void **)&dH,      kHBytes,     ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dW_kv,   kWBytes,     ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dW_z,    kWBytes,     ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dBias,   kBiasBytes,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dC,      kCZBytes,    ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dZ,      kCZBytes,    ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dScores, kScoreBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dOut,    kOutBytes,   ACL_MEM_MALLOC_HUGE_FIRST);

    // --- Read inputs ---------------------------------------------------
    ReadFile("../input/input_H.bin",     kHBytes,    hH,    kHBytes);
    ReadFile("../input/input_W_kv.bin",  kWBytes,    hW_kv, kWBytes);
    ReadFile("../input/input_W_z.bin",   kWBytes,    hW_z,  kWBytes);
    ReadFile("../input/input_B.bin",     kBiasBytes, hBias, kBiasBytes);

    // --- Upload inputs -------------------------------------------------
    aclrtMemcpy(dH,    kHBytes,    hH,    kHBytes,    ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dW_kv, kWBytes,    hW_kv, kWBytes,    ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dW_z,  kWBytes,    hW_z,  kWBytes,    ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dBias, kBiasBytes, hBias, kBiasBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    // --- Poison output buffers (detect "never wrote" vs "wrote zeros") -
    aclrtMemset(dC,      kCZBytes,    0xA1, kCZBytes);
    aclrtMemset(dZ,      kCZBytes,    0xA2, kCZBytes);
    aclrtMemset(dScores, kScoreBytes, 0xA3, kScoreBytes);
    aclrtMemset(dOut,    kOutBytes,   0xA4, kOutBytes);

    // === Stage 1: GEMM =================================================
    // Projects H -> C (KV entries) and H -> Z (compression weights) via
    // two independent GEMMs: C = H @ W_kv^T,  Z = H @ W_z^T.
    // Stream-order guarantees: the next launch sees dC and dZ fully written.
    (void)PtoTiming::TimeKernelCallUs("hca_gemm", stream, [&]() {
        launch_gemm(dC, dZ, dH, dW_kv, dW_z, stream);
    });
    // Explicit barrier: ensures GEMM results are visible to Biased Softmax.
    aclrtSynchronizeStream(stream);

    // === Stage 2: Biased Softmax =======================================
    // Adds positional bias B to Z (reshaped to [B,nb,r,c]) and applies
    // row-wise softmax along the ratio dimension (dim=2).
    (void)PtoTiming::TimeKernelCallUs("hca_biased_softmax", stream, [&]() {
        launch_biased_softmax(dScores, dZ, dBias, stream);
    });
    // Explicit barrier: ensures softmax scores are visible to Weighted Sum.
    aclrtSynchronizeStream(stream);

    // === Stage 3: Weighted Sum =========================================
    // Element-wise multiply C by scores (both [B,nb,r,c]) and reduce
    // along the ratio dimension: out[b,nb_i,k] = sum_j C[b,nb_i,j,k]*scores[b,nb_i,j,k].
    (void)PtoTiming::TimeKernelCallUs("hca_weighted_sum", stream, [&]() {
        launch_weighted_sum(dOut, dC, dScores, stream);
    });
    aclrtSynchronizeStream(stream);

    // --- Download output -----------------------------------------------
    aclrtMemcpy(hOut, kOutBytes, dOut, kOutBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("../output/output_npu.bin", hOut, kOutBytes);

    // --- Validate final output -----------------------------------------
    bool ok = ValidateStage(
        "output",
        "../output/golden_output.bin",
        "../output/output_npu.bin",
        kOutBytes,
        /*poison=*/0xA4);

    if (ok) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }

    // --- Cleanup -------------------------------------------------------
    aclrtFree(dOut);
    aclrtFree(dScores);
    aclrtFree(dZ);
    aclrtFree(dC);
    aclrtFree(dBias);
    aclrtFree(dW_z);
    aclrtFree(dW_kv);
    aclrtFree(dH);

    aclrtFreeHost(hGolden);
    aclrtFreeHost(hOut);
    aclrtFreeHost(hBias);
    aclrtFreeHost(hW_z);
    aclrtFreeHost(hW_kv);
    aclrtFreeHost(hH);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    printf("%s\n", ok ? "test success" : "test failed");
    return ok ? 0 : 1;
}
