/**
 * main.cpp - host driver for mla_basic.
 *
 * Multi-Head Latent Attention (DeepSeek V2/V3-style) v1 driver. Launches the
 * seven `__global__ AICORE` kernels (5 cube + 2 vec) back-to-back on a single
 * ACL stream; the ACL stream-order semantics provide cross-kernel sync (see
 * §A18 in docs_for_ai/known_good_kernel_examples.md).
 *
 * Pipeline:
 *   1.  Q_proj           (cube)   X    @ W_q   -> Q       [S, Nh, Hd] FP16
 *   2.  KV_compression   (cube)   X    @ W_dkv -> C_kv    [S, L]      FP16
 *   3.  KV_cache_store   (vec)    C_kv          -> C_cache [S, L]     FP16
 *   4.  KV_reconstruction(cube)   C_cache @ W_uk -> K, C_cache @ W_uv -> V
 *                                                 each [S, Nh, Hd] FP16
 *   5a. Attn_QK          (cube)   Q  @ K^T     -> scores  [Nh, S, S]  FP16
 *   5b. Attn_softmax     (vec)    softmax(scores * 1/sqrt(Hd)) -> probs
 *                                                          [Nh, S, S] FP16
 *   5c. Attn_PV          (cube)   probs @ V    -> out     [S, Nh, Hd] FP16
 *
 * I/O contract (all raw little-endian, contiguous, no headers):
 *   ../input/input_x.bin       (S * H            FP16)
 *   ../input/input_w_q.bin     (H * Nh*Hd        FP16)
 *   ../input/input_w_dkv.bin   (H * L            FP16)
 *   ../input/input_w_uk.bin    (L * Nh*Hd        FP16)
 *   ../input/input_w_uv.bin    (L * Nh*Hd        FP16)
 *   ../output/golden_out.bin   (S * Nh*Hd        FP16)
 *
 *   ../output/output_q.bin       (S * Nh*Hd      FP16)   debug
 *   ../output/output_c_kv.bin    (S * L          FP16)   debug
 *   ../output/output_c_cache.bin (S * L          FP16)   debug
 *   ../output/output_k.bin       (S * Nh*Hd      FP16)   debug
 *   ../output/output_v.bin       (S * Nh*Hd      FP16)   debug
 *   ../output/output_scores.bin  (Nh * S * S     FP16)   debug
 *   ../output/output_probs.bin   (Nh * S * S     FP16)   debug
 *   ../output/output_out.bin     (S * Nh*Hd      FP16)   compared to golden
 *
 * Poisoning: every device output buffer is `aclrtMemset`-poisoned with a
 * unique byte BEFORE the launch so compare_outputs.py can distinguish
 * "kernel never wrote" from "kernel wrote zeros".
 */

#include "test_common.h"
#include "acl/acl.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

// ----- Kernel launchers exposed by the two kernel TUs ----------------------

extern "C" void launchMlaQProjectionFp16    (uint8_t *q, uint8_t *x, uint8_t *w_q, void *stream);
extern "C" void launchMlaKVCompressionFp16  (uint8_t *c_kv, uint8_t *x, uint8_t *w_dkv, void *stream);
extern "C" void launchMlaKVCacheStoreFp16   (uint8_t *c_cache, uint8_t *c_kv, void *stream);
extern "C" void launchMlaKVReconstructionFp16(uint8_t *k, uint8_t *v, uint8_t *c_cache,
                                              uint8_t *w_uk, uint8_t *w_uv, void *stream);
extern "C" void launchMlaAttnQKFp16         (uint8_t *scores, uint8_t *q, uint8_t *k, void *stream);
extern "C" void launchMlaAttnSoftmaxFp16    (uint8_t *probs, uint8_t *scores, float scale, void *stream);
extern "C" void launchMlaAttnPVFp16         (uint8_t *out, uint8_t *probs, uint8_t *v, void *stream);

// ----- MLA shapes; must match scripts/gen_data.py and the kernel namespaces.
static constexpr int kBatch    = 1;
static constexpr int kSeqLen   = 128;
static constexpr int kHidden   = 4096;
static constexpr int kNumHeads = 32;
static constexpr int kHeadDim  = 128;
static constexpr int kLatent   = 64;
static constexpr int kQKVHidden = kNumHeads * kHeadDim;   // 4096

static constexpr size_t kHalfBytes = 2;

// Per-buffer element counts.
static constexpr size_t kXNumel        = static_cast<size_t>(kSeqLen) * kHidden;          // 128 * 4096
static constexpr size_t kWqNumel       = static_cast<size_t>(kHidden) * kQKVHidden;       // 4096 * 4096
static constexpr size_t kWdkvNumel     = static_cast<size_t>(kHidden) * kLatent;          // 4096 * 64
static constexpr size_t kWukNumel      = static_cast<size_t>(kLatent) * kQKVHidden;       // 64 * 4096
static constexpr size_t kWuvNumel      = static_cast<size_t>(kLatent) * kQKVHidden;       // 64 * 4096
static constexpr size_t kQNumel        = static_cast<size_t>(kSeqLen) * kQKVHidden;       // 128 * 4096
static constexpr size_t kCKvNumel      = static_cast<size_t>(kSeqLen) * kLatent;          // 128 * 64
static constexpr size_t kKNumel        = kQNumel;
static constexpr size_t kVNumel        = kQNumel;
static constexpr size_t kScoresNumel   = static_cast<size_t>(kNumHeads) * kSeqLen * kSeqLen;  // 32 * 128 * 128
static constexpr size_t kProbsNumel    = kScoresNumel;
static constexpr size_t kOutNumel      = kQNumel;

// Byte sizes.
static constexpr size_t kXBytes      = kXNumel      * kHalfBytes;
static constexpr size_t kWqBytes     = kWqNumel     * kHalfBytes;
static constexpr size_t kWdkvBytes   = kWdkvNumel   * kHalfBytes;
static constexpr size_t kWukBytes    = kWukNumel    * kHalfBytes;
static constexpr size_t kWuvBytes    = kWuvNumel    * kHalfBytes;
static constexpr size_t kQBytes      = kQNumel      * kHalfBytes;
static constexpr size_t kCKvBytes    = kCKvNumel    * kHalfBytes;
static constexpr size_t kKBytes      = kKNumel      * kHalfBytes;
static constexpr size_t kVBytes      = kVNumel      * kHalfBytes;
static constexpr size_t kScoresBytes = kScoresNumel * kHalfBytes;
static constexpr size_t kProbsBytes  = kProbsNumel  * kHalfBytes;
static constexpr size_t kOutBytes    = kOutNumel    * kHalfBytes;

// Poison bytes (distinct per buffer so compare_outputs.py can name the
// stage if a buffer is still poisoned post-launch).
static constexpr uint8_t kPoisonQ        = 0xA1;
static constexpr uint8_t kPoisonCKv      = 0xA2;
static constexpr uint8_t kPoisonCCache   = 0xA3;
static constexpr uint8_t kPoisonK        = 0xA4;
static constexpr uint8_t kPoisonV        = 0xA5;
static constexpr uint8_t kPoisonScores   = 0xA6;
static constexpr uint8_t kPoisonProbs    = 0xA7;
static constexpr uint8_t kPoisonOut      = 0xA8;

// Mirrors kernels/manual/a2a3/tget_bandwidth/tget_bandwidth_kernel.cpp:91-98.
static bool CheckAcl(aclError ret, const char *op)
{
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] " << op << " failed: " << static_cast<int>(ret) << std::endl;
        return false;
    }
    return true;
}

static bool ValidateOutput(size_t outBytes)
{
    std::vector<uint8_t> goldenBytes(outBytes);
    std::vector<uint8_t> deviceBytes(outBytes);
    ReadFile("../output/golden_out.bin",  outBytes, goldenBytes.data(), outBytes);
    ReadFile("../output/output_out.bin",  outBytes, deviceBytes.data(), outBytes);

    // Compare as half-precision floats by raw 16-bit pattern, with a tolerance
    // applied through ResultCmp. ResultCmp dispatches on the element type; we
    // pre-cast the byte buffers into half via the vector<uint16_t> reinterpret
    // trick is awkward here, so instead we compare element-wise in float32
    // with absolute tolerance.
    size_t numel = outBytes / kHalfBytes;
    auto *golden = reinterpret_cast<uint16_t *>(goldenBytes.data());
    auto *device = reinterpret_cast<uint16_t *>(deviceBytes.data());

    // FP16 -> FP32 conversion via reinterpret bitmask (IEEE 754 half).
    auto half_to_float = [](uint16_t h) -> float {
        uint32_t sign     = (h & 0x8000u) << 16;
        uint32_t exponent = (h & 0x7C00u) >> 10;
        uint32_t mantissa = (h & 0x03FFu);
        uint32_t f;
        if (exponent == 0) {
            if (mantissa == 0) { f = sign; }
            else {
                // subnormal
                exponent = 1;
                while ((mantissa & 0x0400u) == 0) {
                    mantissa <<= 1;
                    exponent  -= 1;
                }
                mantissa &= 0x03FFu;
                f = sign | ((exponent + 112) << 23) | (mantissa << 13);
            }
        } else if (exponent == 0x1F) {
            f = sign | 0x7F800000u | (mantissa << 13);  // inf / nan
        } else {
            f = sign | ((exponent + 112) << 23) | (mantissa << 13);
        }
        float out;
        __builtin_memcpy(&out, &f, sizeof(out));
        return out;
    };

    // Tolerance: FP16 attention values for these shapes can range up to ~50.
    // Absolute tolerance 0.5, relative tolerance 0.05. This is loose but
    // matches a "very basic" correctness check; tighten in a follow-up.
    const float absTol = 0.5f;
    const float relTol = 0.05f;
    size_t   nMismatch = 0;
    float    maxAbsErr = 0.0f;
    size_t   firstBad  = static_cast<size_t>(-1);

    for (size_t i = 0; i < numel; ++i) {
        float g = half_to_float(golden[i]);
        float d = half_to_float(device[i]);
        float absErr = std::fabs(g - d);
        if (absErr > maxAbsErr) maxAbsErr = absErr;
        float tol = absTol + relTol * std::fabs(g);
        if (absErr > tol) {
            if (firstBad == static_cast<size_t>(-1)) firstBad = i;
            ++nMismatch;
        }
    }

    printf("[validate] elements=%zu  mismatches=%zu  max_abs_err=%.4f  "
           "(tol = %g abs + %g rel)\n",
           numel, nMismatch, maxAbsErr, absTol, relTol);
    if (nMismatch != 0 && firstBad != static_cast<size_t>(-1)) {
        float g = half_to_float(golden[firstBad]);
        float d = half_to_float(device[firstBad]);
        printf("[validate] first mismatch at index %zu: golden=%.4f  device=%.4f\n",
               firstBad, g, d);
    }
    return nMismatch == 0;
}

int main()
{
    printf("[main] MLA basic: B=%d S=%d H=%d Nh=%d Hd=%d L=%d\n",
           kBatch, kSeqLen, kHidden, kNumHeads, kHeadDim, kLatent);
    printf("[main] sizes (bytes): x=%zu  w_q=%zu  w_dkv=%zu  w_uk=%zu  w_uv=%zu\n",
           kXBytes, kWqBytes, kWdkvBytes, kWukBytes, kWuvBytes);
    printf("[main]                 q=%zu  c_kv=%zu  k=%zu  v=%zu\n",
           kQBytes, kCKvBytes, kKBytes, kVBytes);
    printf("[main]                 scores=%zu  probs=%zu  out=%zu\n",
           kScoresBytes, kProbsBytes, kOutBytes);

    if (!CheckAcl(aclInit(nullptr),         "aclInit"))         std::exit(3);
    if (!CheckAcl(aclrtSetDevice(0),        "aclrtSetDevice"))  std::exit(3);
    aclrtStream stream;
    if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream")) std::exit(3);

    // ---- Host allocations ------------------------------------------------
    uint8_t *xHost = nullptr, *wqHost = nullptr, *wdkvHost = nullptr;
    uint8_t *wukHost = nullptr, *wuvHost = nullptr;
    uint8_t *outHost = nullptr;
    // (Debug stage outputs)
    uint8_t *qHost      = nullptr;
    uint8_t *ckvHost    = nullptr;
    uint8_t *ccacheHost = nullptr;
    uint8_t *kHost      = nullptr;
    uint8_t *vHost      = nullptr;
    uint8_t *scoresHost = nullptr;
    uint8_t *probsHost  = nullptr;

    CheckAcl(aclrtMallocHost((void **)&xHost,      kXBytes),     "MallocHost(x)");
    CheckAcl(aclrtMallocHost((void **)&wqHost,     kWqBytes),    "MallocHost(w_q)");
    CheckAcl(aclrtMallocHost((void **)&wdkvHost,   kWdkvBytes),  "MallocHost(w_dkv)");
    CheckAcl(aclrtMallocHost((void **)&wukHost,    kWukBytes),   "MallocHost(w_uk)");
    CheckAcl(aclrtMallocHost((void **)&wuvHost,    kWuvBytes),   "MallocHost(w_uv)");
    CheckAcl(aclrtMallocHost((void **)&outHost,    kOutBytes),   "MallocHost(out)");
    CheckAcl(aclrtMallocHost((void **)&qHost,      kQBytes),     "MallocHost(q)");
    CheckAcl(aclrtMallocHost((void **)&ckvHost,    kCKvBytes),   "MallocHost(c_kv)");
    CheckAcl(aclrtMallocHost((void **)&ccacheHost, kCKvBytes),   "MallocHost(c_cache)");
    CheckAcl(aclrtMallocHost((void **)&kHost,      kKBytes),     "MallocHost(k)");
    CheckAcl(aclrtMallocHost((void **)&vHost,      kVBytes),     "MallocHost(v)");
    CheckAcl(aclrtMallocHost((void **)&scoresHost, kScoresBytes),"MallocHost(scores)");
    CheckAcl(aclrtMallocHost((void **)&probsHost,  kProbsBytes), "MallocHost(probs)");

    // ---- Device allocations ----------------------------------------------
    uint8_t *xDev = nullptr, *wqDev = nullptr, *wdkvDev = nullptr;
    uint8_t *wukDev = nullptr, *wuvDev = nullptr;
    uint8_t *qDev = nullptr, *ckvDev = nullptr, *ccacheDev = nullptr;
    uint8_t *kDev = nullptr, *vDev = nullptr;
    uint8_t *scoresDev = nullptr, *probsDev = nullptr;
    uint8_t *outDev = nullptr;

    CheckAcl(aclrtMalloc((void **)&xDev,      kXBytes,      ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(x)");
    CheckAcl(aclrtMalloc((void **)&wqDev,     kWqBytes,     ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(w_q)");
    CheckAcl(aclrtMalloc((void **)&wdkvDev,   kWdkvBytes,   ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(w_dkv)");
    CheckAcl(aclrtMalloc((void **)&wukDev,    kWukBytes,    ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(w_uk)");
    CheckAcl(aclrtMalloc((void **)&wuvDev,    kWuvBytes,    ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(w_uv)");
    CheckAcl(aclrtMalloc((void **)&qDev,      kQBytes,      ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(q)");
    CheckAcl(aclrtMalloc((void **)&ckvDev,    kCKvBytes,    ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(c_kv)");
    CheckAcl(aclrtMalloc((void **)&ccacheDev, kCKvBytes,    ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(c_cache)");
    CheckAcl(aclrtMalloc((void **)&kDev,      kKBytes,      ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(k)");
    CheckAcl(aclrtMalloc((void **)&vDev,      kVBytes,      ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(v)");
    CheckAcl(aclrtMalloc((void **)&scoresDev, kScoresBytes, ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(scores)");
    CheckAcl(aclrtMalloc((void **)&probsDev,  kProbsBytes,  ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(probs)");
    CheckAcl(aclrtMalloc((void **)&outDev,    kOutBytes,    ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(out)");

    // ---- Read inputs from disk -------------------------------------------
    ReadFile("../input/input_x.bin",     kXBytes,    xHost,    kXBytes);
    ReadFile("../input/input_w_q.bin",   kWqBytes,   wqHost,   kWqBytes);
    ReadFile("../input/input_w_dkv.bin", kWdkvBytes, wdkvHost, kWdkvBytes);
    ReadFile("../input/input_w_uk.bin",  kWukBytes,  wukHost,  kWukBytes);
    ReadFile("../input/input_w_uv.bin",  kWuvBytes,  wuvHost,  kWuvBytes);

    // ---- Upload inputs ----------------------------------------------------
    CheckAcl(aclrtMemcpy(xDev,    kXBytes,    xHost,    kXBytes,    ACL_MEMCPY_HOST_TO_DEVICE), "Memcpy(x)");
    CheckAcl(aclrtMemcpy(wqDev,   kWqBytes,   wqHost,   kWqBytes,   ACL_MEMCPY_HOST_TO_DEVICE), "Memcpy(w_q)");
    CheckAcl(aclrtMemcpy(wdkvDev, kWdkvBytes, wdkvHost, kWdkvBytes, ACL_MEMCPY_HOST_TO_DEVICE), "Memcpy(w_dkv)");
    CheckAcl(aclrtMemcpy(wukDev,  kWukBytes,  wukHost,  kWukBytes,  ACL_MEMCPY_HOST_TO_DEVICE), "Memcpy(w_uk)");
    CheckAcl(aclrtMemcpy(wuvDev,  kWuvBytes,  wuvHost,  kWuvBytes,  ACL_MEMCPY_HOST_TO_DEVICE), "Memcpy(w_uv)");

    // ---- Poison every device output buffer -------------------------------
    CheckAcl(aclrtMemset(qDev,      kQBytes,      kPoisonQ,      kQBytes),      "Memset(q)");
    CheckAcl(aclrtMemset(ckvDev,    kCKvBytes,    kPoisonCKv,    kCKvBytes),    "Memset(c_kv)");
    CheckAcl(aclrtMemset(ccacheDev, kCKvBytes,    kPoisonCCache, kCKvBytes),    "Memset(c_cache)");
    CheckAcl(aclrtMemset(kDev,      kKBytes,      kPoisonK,      kKBytes),      "Memset(k)");
    CheckAcl(aclrtMemset(vDev,      kVBytes,      kPoisonV,      kVBytes),      "Memset(v)");
    CheckAcl(aclrtMemset(scoresDev, kScoresBytes, kPoisonScores, kScoresBytes), "Memset(scores)");
    CheckAcl(aclrtMemset(probsDev,  kProbsBytes,  kPoisonProbs,  kProbsBytes),  "Memset(probs)");
    CheckAcl(aclrtMemset(outDev,    kOutBytes,    kPoisonOut,    kOutBytes),    "Memset(out)");

    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    printf("[main] softmax scale = 1/sqrt(%d) = %.6f\n", kHeadDim, scale);

    // ---- Pipeline launches (all on the same stream; ACL stream-order) ----
    launchMlaQProjectionFp16    (qDev, xDev, wqDev, stream);
    launchMlaKVCompressionFp16  (ckvDev, xDev, wdkvDev, stream);
    launchMlaKVCacheStoreFp16   (ccacheDev, ckvDev, stream);
    launchMlaKVReconstructionFp16(kDev, vDev, ccacheDev, wukDev, wuvDev, stream);
    launchMlaAttnQKFp16         (scoresDev, qDev, kDev, stream);
    launchMlaAttnSoftmaxFp16    (probsDev, scoresDev, scale, stream);
    launchMlaAttnPVFp16         (outDev, probsDev, vDev, stream);

    if (!CheckAcl(aclrtSynchronizeStream(stream), "SynchronizeStream")) {
        std::cerr << "[main] stream sync failed — kernel likely crashed or never ran.\n";
    }

    // ---- Copy every output back for debug + main validation ---------------
    CheckAcl(aclrtMemcpy(qHost,      kQBytes,      qDev,      kQBytes,      ACL_MEMCPY_DEVICE_TO_HOST), "DtoH(q)");
    CheckAcl(aclrtMemcpy(ckvHost,    kCKvBytes,    ckvDev,    kCKvBytes,    ACL_MEMCPY_DEVICE_TO_HOST), "DtoH(c_kv)");
    CheckAcl(aclrtMemcpy(ccacheHost, kCKvBytes,    ccacheDev, kCKvBytes,    ACL_MEMCPY_DEVICE_TO_HOST), "DtoH(c_cache)");
    CheckAcl(aclrtMemcpy(kHost,      kKBytes,      kDev,      kKBytes,      ACL_MEMCPY_DEVICE_TO_HOST), "DtoH(k)");
    CheckAcl(aclrtMemcpy(vHost,      kVBytes,      vDev,      kVBytes,      ACL_MEMCPY_DEVICE_TO_HOST), "DtoH(v)");
    CheckAcl(aclrtMemcpy(scoresHost, kScoresBytes, scoresDev, kScoresBytes, ACL_MEMCPY_DEVICE_TO_HOST), "DtoH(scores)");
    CheckAcl(aclrtMemcpy(probsHost,  kProbsBytes,  probsDev,  kProbsBytes,  ACL_MEMCPY_DEVICE_TO_HOST), "DtoH(probs)");
    CheckAcl(aclrtMemcpy(outHost,    kOutBytes,    outDev,    kOutBytes,    ACL_MEMCPY_DEVICE_TO_HOST), "DtoH(out)");

    WriteFile("../output/output_q.bin",       qHost,      kQBytes);
    WriteFile("../output/output_c_kv.bin",    ckvHost,    kCKvBytes);
    WriteFile("../output/output_c_cache.bin", ccacheHost, kCKvBytes);
    WriteFile("../output/output_k.bin",       kHost,      kKBytes);
    WriteFile("../output/output_v.bin",       vHost,      kVBytes);
    WriteFile("../output/output_scores.bin",  scoresHost, kScoresBytes);
    WriteFile("../output/output_probs.bin",   probsHost,  kProbsBytes);
    WriteFile("../output/output_out.bin",     outHost,    kOutBytes);

    printf("[main] first poison check: q[0]=0x%02X (poison=0x%02X), "
           "out[0]=0x%02X (poison=0x%02X)\n",
           qHost[0], kPoisonQ, outHost[0], kPoisonOut);

    // ---- Cleanup ----------------------------------------------------------
    aclrtFree(outDev);
    aclrtFree(probsDev);
    aclrtFree(scoresDev);
    aclrtFree(vDev);
    aclrtFree(kDev);
    aclrtFree(ccacheDev);
    aclrtFree(ckvDev);
    aclrtFree(qDev);
    aclrtFree(wuvDev);
    aclrtFree(wukDev);
    aclrtFree(wdkvDev);
    aclrtFree(wqDev);
    aclrtFree(xDev);
    aclrtFreeHost(probsHost);
    aclrtFreeHost(scoresHost);
    aclrtFreeHost(vHost);
    aclrtFreeHost(kHost);
    aclrtFreeHost(ccacheHost);
    aclrtFreeHost(ckvHost);
    aclrtFreeHost(qHost);
    aclrtFreeHost(outHost);
    aclrtFreeHost(wuvHost);
    aclrtFreeHost(wukHost);
    aclrtFreeHost(wdkvHost);
    aclrtFreeHost(wqHost);
    aclrtFreeHost(xHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    bool ok = ValidateOutput(kOutBytes);
    if (ok) {
        printf("test data success\n");
        printf("test success\n");
    } else {
        printf("test data failed\n");
        printf("test failed\n");
    }
    return 0;
}
