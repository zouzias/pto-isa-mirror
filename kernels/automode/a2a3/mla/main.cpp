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
 *   ../input/input_w_dq.bin    (H * qL           FP16)
 *   ../input/input_w_qk.bin    (qL * Nh*L        FP16)  absorbed W_uq @ W_uk^T
 *   ../input/input_w_dkv.bin   (H * L            FP16)
 *   ../input/input_w_uv.bin    (L * Nh*Hd        FP16)
 *   ../output/golden_out.bin   (S * Nh*Hd        FP16)
 *
 *   ../output/output_q.bin       (S * Nh*L       FP16)   debug — now Q_absorbed
 *   ../output/output_c_kv.bin    (S * L          FP16)   debug
 *   ../output/output_c_cache.bin (S * L          FP16)   debug
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
#include "generated_cases.h"   // emitted by scripts/generate_cases.py

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

// ----- Kernel launchers exposed by the two kernel TUs ----------------------

extern "C" void launchMlaQCompressionFp16   (uint8_t *c_q, uint8_t *x, uint8_t *w_dq, void *stream);
extern "C" void launchMlaQAbsorbFp16        (uint8_t *q_absorbed, uint8_t *c_q, uint8_t *w_qk, void *stream);
extern "C" void launchMlaKVCompressionFp16  (uint8_t *c_kv, uint8_t *x, uint8_t *w_dkv, void *stream);
extern "C" void launchMlaKVCacheStoreFp16   (uint8_t *c_cache, uint8_t *c_kv, void *stream);
extern "C" void launchMlaVReconstructionFp16(uint8_t *v, uint8_t *c_cache,
                                             uint8_t *w_uv, void *stream);
// DeepSeek-V2 absorbed-weight attention: A=Q_absorbed (per-head), B=C_cache (shared).
extern "C" void launchMlaAttnQKFp16         (uint8_t *scores, uint8_t *q_absorbed, uint8_t *c_cache, void *stream);
extern "C" void launchMlaAttnSoftmaxFp16    (uint8_t *probs, uint8_t *scores_nope,
                                             uint8_t *scores_rope, float scale, void *stream);
extern "C" void launchMlaAttnPVFp16         (uint8_t *out, uint8_t *probs, uint8_t *v, void *stream);
// DeepSeek-V2 decoupled RoPE additions.
extern "C" void launchMlaQRopeProjectionFp16(uint8_t *q_rope, uint8_t *x,
                                             uint8_t *w_q_rope, void *stream);
extern "C" void launchMlaKRopeProjectionFp16(uint8_t *k_rope, uint8_t *x,
                                             uint8_t *w_k_rope, void *stream);
extern "C" void launchMlaRoPEFp16           (uint8_t *y, uint8_t *x,
                                             uint8_t *cos, uint8_t *sin,
                                             unsigned numBlocks, void *stream);
extern "C" void launchMlaAttnQKRopeFp16     (uint8_t *scores_rope, uint8_t *q_rope_rot,
                                             uint8_t *k_rope_rot, void *stream);

// ----- MLA shapes; sourced from the generated case header so main.cpp and
//       both kernel TUs stay in lock-step. Re-run scripts/generate_cases.py
//       to switch the active case.
static constexpr int kBatch    = 1;
static constexpr int kSeqLen   = static_cast<int>(kMlaSeqLen);
static constexpr int kHidden   = static_cast<int>(kMlaHidden);
static constexpr int kNumHeads = static_cast<int>(kMlaNumHeads);
static constexpr int kHeadDim  = static_cast<int>(kMlaHeadDim);
static constexpr int kLatent   = static_cast<int>(kMlaLatent);
static constexpr int kNopeDim  = static_cast<int>(kMlaNopeDim);
static constexpr int kRopeDim  = static_cast<int>(kMlaRopeDim);
static constexpr int kRopeHalf = kRopeDim / 2;
static constexpr int kQLatent  = static_cast<int>(kMlaQLatent);
static constexpr int kQKVHidden  = kNumHeads * kHeadDim;
static constexpr int kQNopeWidth = kNumHeads * kNopeDim;
static constexpr int kQRopeWidth = kNumHeads * kRopeDim;
static_assert(kNopeDim + kRopeDim == kHeadDim,
              "DeepSeek-V2: head_dim must split into nope_dim + rope_dim");

static constexpr size_t kHalfBytes = 2;

// Per-buffer element counts.
static constexpr size_t kXNumel        = static_cast<size_t>(kSeqLen) * kHidden;          // S * H
static constexpr size_t kWdqNumel      = static_cast<size_t>(kHidden) * kQLatent;         // H * qL
// DeepSeek-V2 absorbed weight: W_qk[h] = W_uq[h] @ W_uk[h]^T, flattened to [qL, Nh*L].
// Replaces W_uq + W_uk for the kernel. Shape equals what W_uq used to be.
static constexpr size_t kWqkNumel      = static_cast<size_t>(kQLatent) * kQNopeWidth;     // qL * Nh*L
static constexpr size_t kWdkvNumel     = static_cast<size_t>(kHidden) * kLatent;          // H * L
static constexpr size_t kWuvNumel      = static_cast<size_t>(kLatent) * kQKVHidden;       // L * Nh*Hd
static constexpr size_t kCQNumel       = static_cast<size_t>(kSeqLen) * kQLatent;         // S * qL
// Q_absorbed replaces Q_nope; same shape [S, Nh*L] (== [S, 2048] in current cfg).
static constexpr size_t kQNumel        = static_cast<size_t>(kSeqLen) * kQNopeWidth;
static constexpr size_t kCKvNumel      = static_cast<size_t>(kSeqLen) * kLatent;          // S * L  (cache; also "K" input to attention)
// K_nope is GONE — its role is absorbed into W_qk; attention reads C_cache directly.
static constexpr size_t kVNumel        = static_cast<size_t>(kSeqLen) * kQKVHidden;       // S * Nh*Hd  (V unchanged)
static constexpr size_t kScoresNumel   = static_cast<size_t>(kNumHeads) * kSeqLen * kSeqLen;
static constexpr size_t kProbsNumel    = kScoresNumel;
static constexpr size_t kOutNumel      = kVNumel;                                         // Out has V shape
// RoPE buffers.
static constexpr size_t kWqRopeNumel   = static_cast<size_t>(kHidden) * kQRopeWidth;  // 4096 * 2048
static constexpr size_t kWkRopeNumel   = static_cast<size_t>(kHidden) * kRopeDim;     // 4096 * 64
static constexpr size_t kCosNumel      = static_cast<size_t>(kSeqLen) * kRopeHalf;    // 128 * 32
static constexpr size_t kSinNumel      = kCosNumel;
static constexpr size_t kQRopeNumel    = static_cast<size_t>(kNumHeads) * kSeqLen * kRopeDim; // [Nh, S, Rd]
static constexpr size_t kKRopeNumel    = static_cast<size_t>(kSeqLen) * kRopeDim;             // [S,    Rd]
static constexpr size_t kScoresRopeNumel = kScoresNumel;

// Byte sizes.
size_t kXBytes      = kXNumel      * kHalfBytes;
size_t kWdqBytes    = kWdqNumel    * kHalfBytes;
size_t kWqkBytes    = kWqkNumel    * kHalfBytes;
size_t kWdkvBytes   = kWdkvNumel   * kHalfBytes;
size_t kWuvBytes    = kWuvNumel    * kHalfBytes;
static constexpr size_t kCQBytes     = kCQNumel     * kHalfBytes;
static constexpr size_t kQBytes      = kQNumel      * kHalfBytes;
static constexpr size_t kCKvBytes    = kCKvNumel    * kHalfBytes;
static constexpr size_t kVBytes      = kVNumel      * kHalfBytes;
static constexpr size_t kScoresBytes    = kScoresNumel    * kHalfBytes;
static constexpr size_t kProbsBytes     = kProbsNumel     * kHalfBytes;
static constexpr size_t kOutBytes       = kOutNumel       * kHalfBytes;
static constexpr size_t kWqRopeBytes    = kWqRopeNumel    * kHalfBytes;
static constexpr size_t kWkRopeBytes    = kWkRopeNumel    * kHalfBytes;
static constexpr size_t kCosBytes       = kCosNumel       * kHalfBytes;
static constexpr size_t kSinBytes       = kSinNumel       * kHalfBytes;
static constexpr size_t kQRopeBytes     = kQRopeNumel     * kHalfBytes;
static constexpr size_t kKRopeBytes     = kKRopeNumel     * kHalfBytes;
static constexpr size_t kScoresRopeBytes = kScoresRopeNumel * kHalfBytes;

// Poison bytes (distinct per buffer so compare_outputs.py can name the
// stage if a buffer is still poisoned post-launch).
static constexpr uint8_t kPoisonQ        = 0xA1;  // Q_absorbed
static constexpr uint8_t kPoisonCKv      = 0xA2;
static constexpr uint8_t kPoisonCCache   = 0xA3;
// (no kPoisonK — K_nope is no longer reconstructed)
static constexpr uint8_t kPoisonV        = 0xA5;
static constexpr uint8_t kPoisonScores    = 0xA6;
static constexpr uint8_t kPoisonProbs     = 0xA7;
static constexpr uint8_t kPoisonOut       = 0xA8;
// RoPE poison bytes.
static constexpr uint8_t kPoisonQRope     = 0xB1;
static constexpr uint8_t kPoisonKRope     = 0xB2;
static constexpr uint8_t kPoisonQRopeRot  = 0xB3;
static constexpr uint8_t kPoisonKRopeRot  = 0xB4;
static constexpr uint8_t kPoisonScoresRope= 0xB5;
static constexpr uint8_t kPoisonCQ        = 0xC1;

// Mirrors kernels/manual/a2a3/tget_bandwidth/tget_bandwidth_kernel.cpp:91-98.
static bool CheckAcl(aclError ret, const char *op)
{
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] " << op << " failed: " << static_cast<int>(ret) << std::endl;
        return false;
    }
    return true;
}

// IEEE 754 FP16 -> FP32 via bit-pattern extraction.
static float HalfToFloat(uint16_t h)
{
    uint32_t sign     = (h & 0x8000u) << 16;
    uint32_t exponent = (h & 0x7C00u) >> 10;
    uint32_t mantissa = (h & 0x03FFu);
    uint32_t f;
    if (exponent == 0) {
        if (mantissa == 0) { f = sign; }
        else {
            exponent = 1;
            while ((mantissa & 0x0400u) == 0) { mantissa <<= 1; --exponent; }
            mantissa &= 0x03FFu;
            f = sign | ((exponent + 112) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1F) {
        f = sign | 0x7F800000u | (mantissa << 13);
    } else {
        f = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    float out;
    __builtin_memcpy(&out, &f, sizeof(out));
    return out;
}

// Simple file reader; returns bytes read (0 if file not found).
static size_t ReadFilePlain(const char *path, void *buf, size_t size)
{
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    size_t n = fread(buf, 1, size, f);
    fclose(f);
    return n;
}

// Per-stage result checker. Reads golden and device output files from disk,
// checks for poison (kernel never wrote), then compares FP16 element-by-element.
// Returns true = pass; false = fail / poisoned / missing output.
// goldenPath may be nullptr to skip the golden comparison (copy-stage check).
static bool ValidateStage(const char *name,
                          const char *goldenPath,
                          const char *devicePath,
                          size_t bytes,
                          float absTol, float relTol,
                          uint8_t poisonByte)
{
    std::vector<uint8_t> deviceBuf(bytes, 0);
    size_t deviceRead = ReadFilePlain(devicePath, deviceBuf.data(), bytes);
    if (deviceRead != bytes) {
        printf("[stage %-14s] ERROR  — output file missing or wrong size"
               " (read %zu / %zu): %s\n", name, deviceRead, bytes, devicePath);
        return false;
    }

    // Poison check: sample first 256 bytes.
    size_t checkN = std::min(bytes, (size_t)256);
    bool allPoison = true;
    for (size_t i = 0; i < checkN; ++i) {
        if (deviceBuf[i] != poisonByte) { allPoison = false; break; }
    }
    if (allPoison) {
        printf("[stage %-14s] POISONED — kernel never wrote this buffer"
               " (poison=0x%02X, checked %zu bytes)\n", name, poisonByte, checkN);
        return false;
    }

    if (goldenPath == nullptr) {
        printf("[stage %-14s] WRITTEN  (no golden; buffer not poisoned)\n", name);
        return true;
    }

    std::vector<uint8_t> goldenBuf(bytes, 0);
    size_t goldenRead = ReadFilePlain(goldenPath, goldenBuf.data(), bytes);
    if (goldenRead != bytes) {
        printf("[stage %-14s] SKIP   — golden file missing or wrong size"
               " (read %zu / %zu): %s\n", name, goldenRead, bytes, goldenPath);
        return true; // skip is not a kernel failure
    }

    size_t numel     = bytes / kHalfBytes;
    auto *golden16   = reinterpret_cast<const uint16_t *>(goldenBuf.data());
    auto *device16   = reinterpret_cast<const uint16_t *>(deviceBuf.data());
    size_t nMismatch = 0;
    float  maxAbsErr = 0.0f;
    size_t firstBad  = static_cast<size_t>(-1);

    for (size_t i = 0; i < numel; ++i) {
        float g = HalfToFloat(golden16[i]);
        float d = HalfToFloat(device16[i]);
        float absErr = std::fabs(g - d);
        if (absErr > maxAbsErr) maxAbsErr = absErr;
        if (absErr > absTol + relTol * std::fabs(g)) {
            if (firstBad == static_cast<size_t>(-1)) firstBad = i;
            ++nMismatch;
        }
    }

    bool ok = (nMismatch == 0);
    printf("[stage %-14s] %-5s  elements=%-8zu  mismatches=%-8zu  max_abs_err=%.4f"
           "  (tol=%g+%g*|g|)\n",
           name, ok ? "PASS" : "FAIL", numel, nMismatch, maxAbsErr, absTol, relTol);
    if (!ok && firstBad != static_cast<size_t>(-1)) {
        size_t shown = 0;
        for (size_t i = firstBad; i < numel && shown < 5; ++i) {
            float g = HalfToFloat(golden16[i]);
            float d = HalfToFloat(device16[i]);
            if (std::fabs(g - d) > absTol + relTol * std::fabs(g)) {
                printf("              mismatch[%zu]:  golden=%.5f  device=%.5f"
                       "  err=%.5f\n", i, g, d, std::fabs(g - d));
                ++shown;
            }
        }
    }
    return ok;
}

int main(int argc, char **argv)
{
    // CLI per kernel_test_guidance.md §3-§4. The active case is baked in at
    // build time (see scripts/generate_cases.py), so --case / --cases are
    // accepted only as a sanity check: if the user passes a tuple here we
    // verify it matches the compile-time constants and fail fast otherwise.
    int        npuId       = 0;
    bool       intermediate = false;
    std::string filterCase;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--intermediate") {
            intermediate = true;
        } else if (arg.rfind("--npu=", 0) == 0) {
            npuId = std::atoi(arg.c_str() + 6);
        } else if (arg == "--npu" && (i + 1) < argc) {
            npuId = std::atoi(argv[++i]);
        } else if (arg.rfind("--case=", 0) == 0) {
            filterCase = arg.substr(7);
        } else if (arg.rfind("--cases=", 0) == 0) {
            filterCase = arg.substr(8);
        } else if ((arg == "--case" || arg == "--cases") && (i + 1) < argc) {
            filterCase = argv[++i];
        }
        // Unknown args are silently ignored so run.sh can forward common
        // flags (--sys_cnt_multiple etc.) without us tracking them all.
    }

    // Sanity-check: if the user passed a numeric tuple via --case/--cases,
    // confirm it matches the case the binary was actually compiled against.
    // Compare *value* form (e.g. "128,4096,32,128,64,64,64") rather than the
    // case-name slug, since run.sh forwards the raw tuple.
    if (!filterCase.empty() && filterCase.find(',') != std::string::npos) {
        char expected[128];
        std::snprintf(expected, sizeof(expected), "%u,%u,%u,%u,%u,%u,%u",
                      kMlaSeqLen, kMlaHidden, kMlaNumHeads, kMlaHeadDim,
                      kMlaLatent, kMlaQLatent, kMlaRopeDim);
        // Tolerate whitespace in user input.
        std::string normalized;
        normalized.reserve(filterCase.size());
        for (char c : filterCase) {
            if (c != ' ' && c != '\t') normalized.push_back(c);
        }
        if (normalized != expected) {
            std::cerr << "[main] requested case='" << filterCase
                      << "' but this build is '" << expected
                      << "' (active=" << kMlaCaseName
                      << "). Re-run scripts/generate_cases.py + rebuild "
                      << "to switch cases.\n";
            return 2;
        }
    }
    (void)intermediate; // ValidateStage already prints per-stage info

    printf("[main] active case: %s\n", kMlaCaseName);
    printf("[main] MLA DeepSeek-V2: B=%d S=%d H=%d Nh=%d Hd=%d (nope=%d rope=%d) L=%d qL=%d\n",
           kBatch, kSeqLen, kHidden, kNumHeads, kHeadDim, kNopeDim, kRopeDim, kLatent, kQLatent);
    printf("[main] sizes (bytes): x=%zu  w_dq=%zu  w_qk=%zu  w_dkv=%zu  w_uv=%zu\n",
           kXBytes, kWdqBytes, kWqkBytes, kWdkvBytes, kWuvBytes);
    printf("[main]                 c_q=%zu  q_absorbed=%zu  c_kv=%zu  v=%zu\n",
           kCQBytes, kQBytes, kCKvBytes, kVBytes);
    printf("[main]                 scores=%zu  probs=%zu  out=%zu\n",
           kScoresBytes, kProbsBytes, kOutBytes);

    if (!CheckAcl(aclInit(nullptr),          "aclInit"))          std::exit(3);
    if (!CheckAcl(aclrtSetDevice(npuId),     "aclrtSetDevice"))   std::exit(3);
    aclrtStream stream;
    if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream")) std::exit(3);

    // ---- Host allocations ------------------------------------------------
    uint8_t *xHost = nullptr, *wdqHost = nullptr, *wqkHost = nullptr, *wdkvHost = nullptr;
    uint8_t *wuvHost = nullptr;
    uint8_t *cQHost   = nullptr;
    uint8_t *outHost = nullptr;
    // (Debug stage outputs)
    uint8_t *qHost      = nullptr;   // Q_absorbed
    uint8_t *ckvHost    = nullptr;
    uint8_t *ccacheHost = nullptr;
    uint8_t *vHost      = nullptr;
    uint8_t *scoresHost = nullptr;
    uint8_t *probsHost  = nullptr;
    // RoPE host buffers.
    uint8_t *wqRopeHost     = nullptr;
    uint8_t *wkRopeHost     = nullptr;
    uint8_t *cosHost        = nullptr;
    uint8_t *sinHost        = nullptr;
    uint8_t *qRopeHost      = nullptr;
    uint8_t *kRopeHost      = nullptr;
    uint8_t *qRopeRotHost   = nullptr;
    uint8_t *kRopeRotHost   = nullptr;
    uint8_t *scoresRopeHost = nullptr;

    CheckAcl(aclrtMallocHost((void **)&xHost,      kXBytes),     "MallocHost(x)");
    CheckAcl(aclrtMallocHost((void **)&wdqHost,    kWdqBytes),   "MallocHost(w_dq)");
    CheckAcl(aclrtMallocHost((void **)&wqkHost,    kWqkBytes),   "MallocHost(w_qk)");
    CheckAcl(aclrtMallocHost((void **)&wdkvHost,   kWdkvBytes),  "MallocHost(w_dkv)");
    CheckAcl(aclrtMallocHost((void **)&wuvHost,    kWuvBytes),   "MallocHost(w_uv)");
    CheckAcl(aclrtMallocHost((void **)&cQHost,     kCQBytes),    "MallocHost(c_q)");
    CheckAcl(aclrtMallocHost((void **)&outHost,    kOutBytes),   "MallocHost(out)");
    CheckAcl(aclrtMallocHost((void **)&qHost,      kQBytes),     "MallocHost(q)");
    CheckAcl(aclrtMallocHost((void **)&ckvHost,    kCKvBytes),   "MallocHost(c_kv)");
    CheckAcl(aclrtMallocHost((void **)&ccacheHost, kCKvBytes),   "MallocHost(c_cache)");
    CheckAcl(aclrtMallocHost((void **)&vHost,      kVBytes),     "MallocHost(v)");
    CheckAcl(aclrtMallocHost((void **)&scoresHost, kScoresBytes),"MallocHost(scores)");
    CheckAcl(aclrtMallocHost((void **)&probsHost,  kProbsBytes), "MallocHost(probs)");
    CheckAcl(aclrtMallocHost((void **)&wqRopeHost,    kWqRopeBytes),     "MallocHost(w_q_rope)");
    CheckAcl(aclrtMallocHost((void **)&wkRopeHost,    kWkRopeBytes),     "MallocHost(w_k_rope)");
    CheckAcl(aclrtMallocHost((void **)&cosHost,       kCosBytes),        "MallocHost(cos)");
    CheckAcl(aclrtMallocHost((void **)&sinHost,       kSinBytes),        "MallocHost(sin)");
    CheckAcl(aclrtMallocHost((void **)&qRopeHost,     kQRopeBytes),      "MallocHost(q_rope)");
    CheckAcl(aclrtMallocHost((void **)&kRopeHost,     kKRopeBytes),      "MallocHost(k_rope)");
    CheckAcl(aclrtMallocHost((void **)&qRopeRotHost,  kQRopeBytes),      "MallocHost(q_rope_rot)");
    CheckAcl(aclrtMallocHost((void **)&kRopeRotHost,  kKRopeBytes),      "MallocHost(k_rope_rot)");
    CheckAcl(aclrtMallocHost((void **)&scoresRopeHost,kScoresRopeBytes), "MallocHost(scores_rope)");

    // ---- Device allocations ----------------------------------------------
    uint8_t *xDev = nullptr, *wdqDev = nullptr, *wqkDev = nullptr, *wdkvDev = nullptr;
    uint8_t *wuvDev = nullptr;
    uint8_t *cQDev = nullptr;
    uint8_t *qDev = nullptr, *ckvDev = nullptr, *ccacheDev = nullptr;
    uint8_t *vDev = nullptr;
    uint8_t *scoresDev = nullptr, *probsDev = nullptr;
    uint8_t *outDev = nullptr;
    // RoPE device buffers.
    uint8_t *wqRopeDev     = nullptr;
    uint8_t *wkRopeDev     = nullptr;
    uint8_t *cosDev        = nullptr;
    uint8_t *sinDev        = nullptr;
    uint8_t *qRopeDev      = nullptr;
    uint8_t *kRopeDev      = nullptr;
    uint8_t *qRopeRotDev   = nullptr;
    uint8_t *kRopeRotDev   = nullptr;
    uint8_t *scoresRopeDev = nullptr;

    CheckAcl(aclrtMalloc((void **)&xDev,      kXBytes,      ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(x)");
    CheckAcl(aclrtMalloc((void **)&wdqDev,    kWdqBytes,    ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(w_dq)");
    CheckAcl(aclrtMalloc((void **)&wqkDev,    kWqkBytes,    ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(w_qk)");
    CheckAcl(aclrtMalloc((void **)&wdkvDev,   kWdkvBytes,   ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(w_dkv)");
    CheckAcl(aclrtMalloc((void **)&wuvDev,    kWuvBytes,    ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(w_uv)");
    CheckAcl(aclrtMalloc((void **)&cQDev,     kCQBytes,     ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(c_q)");
    CheckAcl(aclrtMalloc((void **)&qDev,      kQBytes,      ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(q)");
    CheckAcl(aclrtMalloc((void **)&ckvDev,    kCKvBytes,    ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(c_kv)");
    CheckAcl(aclrtMalloc((void **)&ccacheDev, kCKvBytes,    ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(c_cache)");
    CheckAcl(aclrtMalloc((void **)&vDev,      kVBytes,      ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(v)");
    CheckAcl(aclrtMalloc((void **)&scoresDev, kScoresBytes, ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(scores)");
    CheckAcl(aclrtMalloc((void **)&probsDev,  kProbsBytes,  ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(probs)");
    CheckAcl(aclrtMalloc((void **)&outDev,    kOutBytes,    ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(out)");
    CheckAcl(aclrtMalloc((void **)&wqRopeDev,    kWqRopeBytes,     ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(w_q_rope)");
    CheckAcl(aclrtMalloc((void **)&wkRopeDev,    kWkRopeBytes,     ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(w_k_rope)");
    CheckAcl(aclrtMalloc((void **)&cosDev,       kCosBytes,        ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(cos)");
    CheckAcl(aclrtMalloc((void **)&sinDev,       kSinBytes,        ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(sin)");
    CheckAcl(aclrtMalloc((void **)&qRopeDev,     kQRopeBytes,      ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(q_rope)");
    CheckAcl(aclrtMalloc((void **)&kRopeDev,     kKRopeBytes,      ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(k_rope)");
    CheckAcl(aclrtMalloc((void **)&qRopeRotDev,  kQRopeBytes,      ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(q_rope_rot)");
    CheckAcl(aclrtMalloc((void **)&kRopeRotDev,  kKRopeBytes,      ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(k_rope_rot)");
    CheckAcl(aclrtMalloc((void **)&scoresRopeDev,kScoresRopeBytes, ACL_MEM_MALLOC_HUGE_FIRST), "Malloc(scores_rope)");

    // ---- Read inputs from disk -------------------------------------------
    // ReadFile's second arg is `size_t&` (out-param for actual file size),
    // so it cannot bind to constexpr byte-size constants. Use a local.
    size_t actualSize = 0;
    ReadFile("../input/input_x.bin",        actualSize, xHost,      kXBytes);
    ReadFile("../input/input_w_dq.bin",     actualSize, wdqHost,    kWdqBytes);
    ReadFile("../input/input_w_qk.bin",     actualSize, wqkHost,    kWqkBytes);
    ReadFile("../input/input_w_dkv.bin",    actualSize, wdkvHost,   kWdkvBytes);
    ReadFile("../input/input_w_uv.bin",     actualSize, wuvHost,    kWuvBytes);
    ReadFile("../input/input_w_q_rope.bin", actualSize, wqRopeHost, kWqRopeBytes);
    ReadFile("../input/input_w_k_rope.bin", actualSize, wkRopeHost, kWkRopeBytes);
    ReadFile("../input/input_cos.bin",      actualSize, cosHost,    kCosBytes);
    ReadFile("../input/input_sin.bin",      actualSize, sinHost,    kSinBytes);

    // ---- Upload inputs ----------------------------------------------------
    CheckAcl(aclrtMemcpy(xDev,        kXBytes,      xHost,      kXBytes,      ACL_MEMCPY_HOST_TO_DEVICE), "Memcpy(x)");
    CheckAcl(aclrtMemcpy(wdqDev,      kWdqBytes,    wdqHost,    kWdqBytes,    ACL_MEMCPY_HOST_TO_DEVICE), "Memcpy(w_dq)");
    CheckAcl(aclrtMemcpy(wqkDev,      kWqkBytes,    wqkHost,    kWqkBytes,    ACL_MEMCPY_HOST_TO_DEVICE), "Memcpy(w_qk)");
    CheckAcl(aclrtMemcpy(wdkvDev,     kWdkvBytes,   wdkvHost,   kWdkvBytes,   ACL_MEMCPY_HOST_TO_DEVICE), "Memcpy(w_dkv)");
    CheckAcl(aclrtMemcpy(wuvDev,      kWuvBytes,    wuvHost,    kWuvBytes,    ACL_MEMCPY_HOST_TO_DEVICE), "Memcpy(w_uv)");
    CheckAcl(aclrtMemcpy(wqRopeDev,   kWqRopeBytes, wqRopeHost, kWqRopeBytes, ACL_MEMCPY_HOST_TO_DEVICE), "Memcpy(w_q_rope)");
    CheckAcl(aclrtMemcpy(wkRopeDev,   kWkRopeBytes, wkRopeHost, kWkRopeBytes, ACL_MEMCPY_HOST_TO_DEVICE), "Memcpy(w_k_rope)");
    CheckAcl(aclrtMemcpy(cosDev,      kCosBytes,    cosHost,    kCosBytes,    ACL_MEMCPY_HOST_TO_DEVICE), "Memcpy(cos)");
    CheckAcl(aclrtMemcpy(sinDev,      kSinBytes,    sinHost,    kSinBytes,    ACL_MEMCPY_HOST_TO_DEVICE), "Memcpy(sin)");

    // ---- Poison every device output buffer -------------------------------
    CheckAcl(aclrtMemset(cQDev,         kCQBytes,        kPoisonCQ,        kCQBytes),        "Memset(c_q)");
    CheckAcl(aclrtMemset(qDev,          kQBytes,         kPoisonQ,         kQBytes),         "Memset(q)");
    CheckAcl(aclrtMemset(ckvDev,        kCKvBytes,       kPoisonCKv,       kCKvBytes),       "Memset(c_kv)");
    CheckAcl(aclrtMemset(ccacheDev,     kCKvBytes,       kPoisonCCache,    kCKvBytes),       "Memset(c_cache)");
    CheckAcl(aclrtMemset(vDev,          kVBytes,         kPoisonV,         kVBytes),         "Memset(v)");
    CheckAcl(aclrtMemset(scoresDev,     kScoresBytes,    kPoisonScores,    kScoresBytes),    "Memset(scores)");
    CheckAcl(aclrtMemset(probsDev,      kProbsBytes,     kPoisonProbs,     kProbsBytes),     "Memset(probs)");
    CheckAcl(aclrtMemset(outDev,        kOutBytes,       kPoisonOut,       kOutBytes),       "Memset(out)");
    CheckAcl(aclrtMemset(qRopeDev,      kQRopeBytes,     kPoisonQRope,     kQRopeBytes),     "Memset(q_rope)");
    CheckAcl(aclrtMemset(kRopeDev,      kKRopeBytes,     kPoisonKRope,     kKRopeBytes),     "Memset(k_rope)");
    CheckAcl(aclrtMemset(qRopeRotDev,   kQRopeBytes,     kPoisonQRopeRot,  kQRopeBytes),     "Memset(q_rope_rot)");
    CheckAcl(aclrtMemset(kRopeRotDev,   kKRopeBytes,     kPoisonKRopeRot,  kKRopeBytes),     "Memset(k_rope_rot)");
    CheckAcl(aclrtMemset(scoresRopeDev, kScoresRopeBytes,kPoisonScoresRope,kScoresRopeBytes),"Memset(scores_rope)");

    // DeepSeek-V2 softmax scale = 1/sqrt(head_dim) = 1/sqrt(128).
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    printf("[main] softmax scale = 1/sqrt(%d) = %.6f\n", kHeadDim, scale);

    // ---- Pipeline launches -----------------------------------------------
    // nope path
    launchMlaQCompressionFp16   (cQDev, xDev, wdqDev, stream);
    launchMlaQAbsorbFp16        (qDev, cQDev, wqkDev, stream);
    launchMlaKVCompressionFp16  (ckvDev, xDev, wdkvDev, stream);
    launchMlaKVCacheStoreFp16   (ccacheDev, ckvDev, stream);
    launchMlaVReconstructionFp16(vDev, ccacheDev, wuvDev, stream);
    // Absorbed attention: A=Q_absorbed (per-head), B=C_cache (shared across heads).
    launchMlaAttnQKFp16         (scoresDev, qDev, ccacheDev, stream);
    // rope path
    launchMlaQRopeProjectionFp16(qRopeDev, xDev, wqRopeDev, stream);
    launchMlaKRopeProjectionFp16(kRopeDev, xDev, wkRopeDev, stream);
    launchMlaRoPEFp16           (qRopeRotDev, qRopeDev, cosDev, sinDev, kNumHeads, stream);
    launchMlaRoPEFp16           (kRopeRotDev, kRopeDev, cosDev, sinDev, 1,         stream);
    launchMlaAttnQKRopeFp16     (scoresRopeDev, qRopeRotDev, kRopeRotDev, stream);
    // combine + softmax + PV
    launchMlaAttnSoftmaxFp16    (probsDev, scoresDev, scoresRopeDev, scale, stream);
    launchMlaAttnPVFp16         (outDev, probsDev, vDev, stream);
    aclrtSynchronizeStream(stream);

    // ---- Copy every output back for debug + main validation ---------------
    CheckAcl(aclrtMemcpy(cQHost,     kCQBytes,     cQDev,     kCQBytes,     ACL_MEMCPY_DEVICE_TO_HOST), "DtoH(c_q)");
    CheckAcl(aclrtMemcpy(qHost,      kQBytes,      qDev,      kQBytes,      ACL_MEMCPY_DEVICE_TO_HOST), "DtoH(q_nope)");
    CheckAcl(aclrtMemcpy(ckvHost,    kCKvBytes,    ckvDev,    kCKvBytes,    ACL_MEMCPY_DEVICE_TO_HOST), "DtoH(c_kv)");
    CheckAcl(aclrtMemcpy(ccacheHost, kCKvBytes,    ccacheDev, kCKvBytes,    ACL_MEMCPY_DEVICE_TO_HOST), "DtoH(c_cache)");
    CheckAcl(aclrtMemcpy(vHost,      kVBytes,      vDev,      kVBytes,      ACL_MEMCPY_DEVICE_TO_HOST), "DtoH(v)");
    CheckAcl(aclrtMemcpy(scoresHost, kScoresBytes, scoresDev, kScoresBytes, ACL_MEMCPY_DEVICE_TO_HOST), "DtoH(scores)");
    CheckAcl(aclrtMemcpy(probsHost,  kProbsBytes,  probsDev,  kProbsBytes,  ACL_MEMCPY_DEVICE_TO_HOST), "DtoH(probs)");
    CheckAcl(aclrtMemcpy(outHost,    kOutBytes,    outDev,    kOutBytes,    ACL_MEMCPY_DEVICE_TO_HOST), "DtoH(out)");
    CheckAcl(aclrtMemcpy(qRopeHost,      kQRopeBytes,     qRopeDev,      kQRopeBytes,     ACL_MEMCPY_DEVICE_TO_HOST), "DtoH(q_rope)");
    CheckAcl(aclrtMemcpy(kRopeHost,      kKRopeBytes,     kRopeDev,      kKRopeBytes,     ACL_MEMCPY_DEVICE_TO_HOST), "DtoH(k_rope)");
    CheckAcl(aclrtMemcpy(qRopeRotHost,   kQRopeBytes,     qRopeRotDev,   kQRopeBytes,     ACL_MEMCPY_DEVICE_TO_HOST), "DtoH(q_rope_rot)");
    CheckAcl(aclrtMemcpy(kRopeRotHost,   kKRopeBytes,     kRopeRotDev,   kKRopeBytes,     ACL_MEMCPY_DEVICE_TO_HOST), "DtoH(k_rope_rot)");
    CheckAcl(aclrtMemcpy(scoresRopeHost, kScoresRopeBytes,scoresRopeDev, kScoresRopeBytes,ACL_MEMCPY_DEVICE_TO_HOST), "DtoH(scores_rope)");

    WriteFile("../output/output_c_q.bin",        cQHost,         kCQBytes);
    WriteFile("../output/output_q.bin",          qHost,          kQBytes);
    WriteFile("../output/output_c_kv.bin",       ckvHost,        kCKvBytes);
    WriteFile("../output/output_c_cache.bin",    ccacheHost,     kCKvBytes);
    WriteFile("../output/output_q_rope.bin",     qRopeHost,      kQRopeBytes);
    WriteFile("../output/output_k_rope.bin",     kRopeHost,      kKRopeBytes);
    WriteFile("../output/output_q_rope_rot.bin", qRopeRotHost,   kQRopeBytes);
    WriteFile("../output/output_k_rope_rot.bin", kRopeRotHost,   kKRopeBytes);
    WriteFile("../output/output_scores_rope.bin",scoresRopeHost, kScoresRopeBytes);
    WriteFile("../output/output_v.bin",       vHost,      kVBytes);
    WriteFile("../output/output_scores.bin",  scoresHost, kScoresBytes);
    WriteFile("../output/output_probs.bin",   probsHost,  kProbsBytes);
    WriteFile("../output/output_out.bin",     outHost,    kOutBytes);

    printf("[main] first byte check: q[0]=0x%02X (poison=0x%02X)  "
           "out[0]=0x%02X (poison=0x%02X)\n",
           qHost[0], kPoisonQ, outHost[0], kPoisonOut);

    // ---- Per-stage intermediate result checking ----------------------------
    // All stage outputs have been written to disk above. ValidateStage reads
    // from disk so it is independent of device/host buffer lifetime.
    // C_cache has no separate golden — it is a copy of C_kv, so we compare
    // output_c_cache.bin against golden_c_kv.bin.
    printf("\n[main] ===== per-stage validation =====\n");
    bool allOk = true;
    allOk &= ValidateStage("C_q",
        "../output/golden_c_q.bin",    "../output/output_c_q.bin",
        kCQBytes,     0.05f, 0.02f, kPoisonCQ);
    allOk &= ValidateStage("Q_absorbed",
        "../output/golden_q.bin",      "../output/output_q.bin",
        kQBytes,      0.1f, 0.02f, kPoisonQ);
    allOk &= ValidateStage("C_kv",
        "../output/golden_c_kv.bin",   "../output/output_c_kv.bin",
        kCKvBytes,    0.1f, 0.02f, kPoisonCKv);
    allOk &= ValidateStage("C_cache",
        "../output/golden_c_kv.bin",   "../output/output_c_cache.bin",
        kCKvBytes,    0.1f, 0.02f, kPoisonCCache);
    // K_nope is no longer reconstructed (absorbed into W_qk).
    allOk &= ValidateStage("V",
        "../output/golden_v.bin",      "../output/output_v.bin",
        kVBytes,      0.1f, 0.02f, kPoisonV);
    // runAttnQK writes the nope-only term; combined scores live only inside
    // the softmax kernel. Compare device scores against golden_scores_nope.
    allOk &= ValidateStage("scores_nope",
        "../output/golden_scores_nope.bin", "../output/output_scores.bin",
        kScoresBytes, 0.2f, 0.05f, kPoisonScores);
    allOk &= ValidateStage("Q_rope",
        "../output/golden_q_rope.bin", "../output/output_q_rope.bin",
        kQRopeBytes, 0.05f, 0.02f, kPoisonQRope);
    allOk &= ValidateStage("K_rope",
        "../output/golden_k_rope.bin", "../output/output_k_rope.bin",
        kKRopeBytes, 0.05f, 0.02f, kPoisonKRope);
    allOk &= ValidateStage("Q_rope_rot",
        "../output/golden_q_rope_rot.bin", "../output/output_q_rope_rot.bin",
        kQRopeBytes, 0.05f, 0.02f, kPoisonQRopeRot);
    allOk &= ValidateStage("K_rope_rot",
        "../output/golden_k_rope_rot.bin", "../output/output_k_rope_rot.bin",
        kKRopeBytes, 0.05f, 0.02f, kPoisonKRopeRot);
    allOk &= ValidateStage("scores_rope",
        "../output/golden_scores_rope.bin", "../output/output_scores_rope.bin",
        kScoresRopeBytes, 0.2f, 0.05f, kPoisonScoresRope);
    allOk &= ValidateStage("probs",
        "../output/golden_probs.bin",  "../output/output_probs.bin",
        kProbsBytes,  0.1f, 0.02f, kPoisonProbs);
    allOk &= ValidateStage("out",
        "../output/golden_out.bin",    "../output/output_out.bin",
        kOutBytes,    0.5f, 0.05f, kPoisonOut);
    printf("[main] ===== end validation =====\n\n");

    // ---- Cleanup ----------------------------------------------------------
    aclrtFree(scoresRopeDev);
    aclrtFree(kRopeRotDev);
    aclrtFree(qRopeRotDev);
    aclrtFree(kRopeDev);
    aclrtFree(qRopeDev);
    aclrtFree(sinDev);
    aclrtFree(cosDev);
    aclrtFree(wkRopeDev);
    aclrtFree(wqRopeDev);
    aclrtFree(outDev);
    aclrtFree(probsDev);
    aclrtFree(scoresDev);
    aclrtFree(vDev);
    aclrtFree(ccacheDev);
    aclrtFree(ckvDev);
    aclrtFree(qDev);
    aclrtFree(cQDev);
    aclrtFree(wuvDev);
    aclrtFree(wdkvDev);
    aclrtFree(wqkDev);
    aclrtFree(wdqDev);
    aclrtFree(xDev);
    aclrtFreeHost(scoresRopeHost);
    aclrtFreeHost(kRopeRotHost);
    aclrtFreeHost(qRopeRotHost);
    aclrtFreeHost(kRopeHost);
    aclrtFreeHost(qRopeHost);
    aclrtFreeHost(sinHost);
    aclrtFreeHost(cosHost);
    aclrtFreeHost(wkRopeHost);
    aclrtFreeHost(wqRopeHost);
    aclrtFreeHost(probsHost);
    aclrtFreeHost(scoresHost);
    aclrtFreeHost(vHost);
    aclrtFreeHost(ccacheHost);
    aclrtFreeHost(ckvHost);
    aclrtFreeHost(qHost);
    aclrtFreeHost(outHost);
    aclrtFreeHost(cQHost);
    aclrtFreeHost(wuvHost);
    aclrtFreeHost(wdkvHost);
    aclrtFreeHost(wqkHost);
    aclrtFreeHost(wdqHost);
    aclrtFreeHost(xHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(npuId);
    aclFinalize();

    if (allOk) {
        printf("test data success\n");
        printf("test success\n");
        return 0;
    } else {
        printf("test data failed\n");
        printf("test failed\n");
        return 1;   // §7 of kernel_test_guidance.md — non-zero on validation failure.
    }
}
