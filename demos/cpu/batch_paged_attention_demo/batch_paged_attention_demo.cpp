/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include <pto/pto-inst.hpp>

using namespace pto;

namespace {

constexpr int kExitSuccess = 0;
constexpr int kExitFailure = 1;

constexpr int kBatch = 2;
constexpr int kNumHeads = 16;
constexpr int kKvHeadNum = 1;
constexpr int kHeadDim = 16;
constexpr int kBlockSize = 16;
constexpr int kMaxNumBlocks = 4;
constexpr int kTotalBlocks = 8;

constexpr std::uint32_t kRngSeed = 20251220U;

constexpr float kAbsTol = 1e-2f;
constexpr float kRelTol = 1e-2f;
constexpr float kRelDenomEps = 1e-6f;
constexpr int kWarmupIters = 2;
constexpr int kIters = 20;

static const int kContextLens[kBatch] = {33, 17};

struct VerifyStats {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    float mean_abs = 0.0f;
    float rmse = 0.0f;
    std::size_t max_abs_idx = 0;
    std::size_t max_rel_idx = 0;
    std::size_t bad_count = 0;
    bool has_nan_or_inf = false;
};

VerifyStats verify_allclose(const std::vector<float> &actual, const std::vector<float> &ref, float abs_tol,
                            float rel_tol, float denom_eps)
{
    VerifyStats stats;
    if (actual.size() != ref.size() || actual.empty()) {
        stats.bad_count = std::max(actual.size(), ref.size());
        return stats;
    }

    double sum_abs = 0.0;
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float a = actual[i];
        const float r = ref[i];
        if (!std::isfinite(a) || !std::isfinite(r)) {
            stats.has_nan_or_inf = true;
            ++stats.bad_count;
            continue;
        }

        const float abs_diff = std::abs(a - r);
        const float denom = std::max(std::abs(r), denom_eps);
        const float rel_diff = abs_diff / denom;

        sum_abs += static_cast<double>(abs_diff);
        sum_sq += static_cast<double>(abs_diff) * static_cast<double>(abs_diff);

        if (abs_diff > stats.max_abs) {
            stats.max_abs = abs_diff;
            stats.max_abs_idx = i;
        }
        if (rel_diff > stats.max_rel) {
            stats.max_rel = rel_diff;
            stats.max_rel_idx = i;
        }

        const float allowed = abs_tol + rel_tol * std::abs(r);
        if (!(abs_diff <= allowed)) {
            ++stats.bad_count;
        }
    }

    const double n = static_cast<double>(actual.size());
    stats.mean_abs = static_cast<float>(sum_abs / n);
    stats.rmse = static_cast<float>(std::sqrt(sum_sq / n));
    return stats;
}

// Reference implementation matching simpler/examples golden.py:
// - half-typed Q/K/V inputs cast to float32 for computation
// - Global max_bn with valid_len==0 handling (mij=-1e30)
// - Mask before scale
// - pij fp16 truncation (simulating device precision)
// - mij clamped to -1e30 (avoids NaN from exp(-inf - (-inf)))
void batch_paged_attention_reference(const std::vector<half> &query, const std::vector<half> &key_cache,
                                     const std::vector<half> &value_cache, const int *block_table,
                                     const int *context_lens, std::vector<float> &out)
{
    const float scale = 1.0f;

    // Compute global max_bn across all batches (matches simpler orchestration)
    int global_max_bn = 0;
    for (int b = 0; b < kBatch; ++b) {
        const int bn_b = (context_lens[b] + kBlockSize - 1) / kBlockSize;
        if (bn_b > global_max_bn) global_max_bn = bn_b;
    }

    for (int b = 0; b < kBatch; ++b) {
        const int ctx_len = context_lens[b];

        for (int h = 0; h < kNumHeads; ++h) {
            std::vector<float> oi(kHeadDim, 0.0f);
            float mi = -std::numeric_limits<float>::infinity();
            float li = 0.0f;

            for (int bn = 0; bn < global_max_bn; ++bn) {
                const int start = bn * kBlockSize;
                const int valid_len = (start < ctx_len) ? std::min(kBlockSize, ctx_len - start) : 0;

                if (valid_len == 0) {
                    // Block beyond sequence: mij=-1e30, lij=0, oi_new=0
                    // Use -1e30 instead of -inf to avoid NaN in online update
                    const float mij_val = -1e30f;
                    const float mi_new = std::max(mi, mij_val);
                    const float alpha = std::exp(mi - mi_new);
                    const float beta = std::exp(mij_val - mi_new);
                    li = alpha * li;
                    for (int d = 0; d < kHeadDim; ++d) {
                        oi[d] = alpha * oi[d];
                    }
                    mi = mi_new;
                    continue;
                }

                const int phys_block = block_table[b * kMaxNumBlocks + bn];

                // sij = q @ k^T (float32 computation, matching golden.py)
                std::vector<float> sij(kBlockSize, 0.0f);
                for (int j = 0; j < kBlockSize; ++j) {
                    float dot = 0.0f;
                    for (int d = 0; d < kHeadDim; ++d) {
                        dot += static_cast<float>(query[(b * kNumHeads + h) * kHeadDim + d]) *
                               static_cast<float>(key_cache[(phys_block * kBlockSize + j) * kHeadDim + d]);
                    }
                    sij[j] = dot;
                }

                // Mask invalid positions first (matching simpler order: mask then scale)
                for (int j = valid_len; j < kBlockSize; ++j) {
                    sij[j] = -std::numeric_limits<float>::infinity();
                }

                // Then apply scale
                for (int j = 0; j < kBlockSize; ++j) {
                    sij[j] *= scale;
                }

                // Row max (clamp to -1e30, matching golden.py)
                float mij = -std::numeric_limits<float>::infinity();
                for (int j = 0; j < kBlockSize; ++j) {
                    mij = std::max(mij, sij[j]);
                }
                mij = std::max(mij, -1e30f);

                // exp
                std::vector<float> pij(kBlockSize, 0.0f);
                for (int j = 0; j < kBlockSize; ++j) {
                    pij[j] = std::exp(sij[j] - mij);
                }

                // fp16 truncation (matching golden.py: pij.to(fp16).to(fp32))
                for (int j = 0; j < kBlockSize; ++j) {
                    pij[j] = static_cast<float>(static_cast<half>(pij[j]));
                }

                // sum
                float lij = 0.0f;
                for (int j = 0; j < kBlockSize; ++j) {
                    lij += pij[j];
                }

                // P @ V (float32 computation with half-precision V values)
                std::vector<float> oi_new(kHeadDim, 0.0f);
                for (int d = 0; d < kHeadDim; ++d) {
                    for (int j = 0; j < kBlockSize; ++j) {
                        oi_new[d] +=
                            pij[j] * static_cast<float>(value_cache[(phys_block * kBlockSize + j) * kHeadDim + d]);
                    }
                }

                // Online softmax update
                const float mi_new = std::max(mi, mij);
                const float alpha = std::exp(mi - mi_new);
                const float beta = std::exp(mij - mi_new);
                li = alpha * li + beta * lij;
                for (int d = 0; d < kHeadDim; ++d) {
                    oi[d] = alpha * oi[d] + beta * oi_new[d];
                }
                mi = mi_new;
            }

            // Final normalization
            const float inv_li = (li > 0.0f) ? (1.0f / li) : 0.0f;
            float *o_row = &out[(b * kNumHeads + h) * kHeadDim];
            for (int d = 0; d < kHeadDim; ++d) {
                o_row[d] = oi[d] * inv_li;
            }
        }
    }
}

// PTO implementation matching simpler/examples kernel code:
// - Half-typed Q/K/V tiles and matmul operands (TileLeft<half>, TileRight<half>)
// - Float32 accumulators (TileAcc<float>)
// - Mask order: TFILLPAD_INPLACE -> SetValue workaround -> TMULS(scale)
// - pij fp16 truncation via TCVT
// - Half-typed PV matmul (LeftP<half> x RightV<half> -> AccO<float>)
// - Global max_bn with valid_len==0 handling
void batch_paged_attention_pto(const std::vector<half> &query, const std::vector<half> &key_cache,
                               const std::vector<half> &value_cache, const int *block_table,
                               const int *context_lens, std::vector<float> &out)
{
    constexpr int kH = kNumHeads;
    constexpr int kD = kHeadDim;
    constexpr int kBS = kBlockSize;

    // Half-typed GlobalTensors for Q/K/V (matching simpler aic_qk_matmul / aic_pv_matmul)
    using GlobalQ = GlobalTensor<half, Shape<1, 1, 1, kH, kD>, Stride<kH * kD, kH * kD, kH * kD, kD, 1>>;
    using GlobalKBlock =
        GlobalTensor<half, Shape<1, 1, 1, kBS, kD>, Stride<kBS * kD, kBS * kD, kBS * kD, kD, 1>>;
    using GlobalVBlock = GlobalKBlock;
    // Output remains float32
    using GlobalO = GlobalTensor<float, Shape<1, 1, 1, kH, kD>, Stride<kH * kD, kH * kD, kH * kD, kD, 1>>;

    // Half-typed Vec tiles for Q/K/V loading
    using QPlainHalf = Tile<TileType::Vec, half, kH, kD, BLayout::RowMajor, kH, kD, SLayout::NoneBox>;
    using KPlainHalf = Tile<TileType::Vec, half, kBS, kD, BLayout::RowMajor, kBS, kD, SLayout::NoneBox>;
    using KTPlainHalf = Tile<TileType::Vec, half, kD, kBS, BLayout::RowMajor, kD, kBS, SLayout::NoneBox>;
    using VPlainHalf = Tile<TileType::Vec, half, kBS, kD, BLayout::RowMajor, kBS, kD, SLayout::NoneBox>;

    // Float32 tiles for scores and intermediates
    using ScoresPlain = Tile<TileType::Vec, float, kH, kBS, BLayout::RowMajor, kH, kBS, SLayout::NoneBox>;
    // Dynamic-column tile: ValidCol set at runtime to indicate valid region
    using ScoresDyn = Tile<TileType::Vec, float, kH, kBS, BLayout::RowMajor, kH, -1>;
    // Padded tile: TFILLPAD_INPLACE fills columns beyond ValidCol with PadValue::Min (-inf)
    using ScoresPad =
        Tile<TileType::Vec, float, kH, kBS, BLayout::RowMajor, kH, kBS, SLayout::NoneBox, 512, PadValue::Min>;
    using RowRedPlain = Tile<TileType::Vec, float, kH, kBS, BLayout::ColMajor, kH, kBS, SLayout::NoneBox>;
    using OPlain = Tile<TileType::Vec, float, kH, kD, BLayout::RowMajor, kH, kD, SLayout::NoneBox>;

    // Half-typed tile for pij fp16 truncation (matching aiv_softmax_prepare TCVT)
    using PijHalf = Tile<TileType::Vec, half, kH, kBS, BLayout::RowMajor, kH, kBS, SLayout::NoneBox>;

    // Half-typed matmul operands with float32 accumulators
    // (matching simpler aic_qk_matmul: TileMatA<half>, TileMatB<half> -> TileAccC<float>)
    using LeftQ = TileLeft<half, kH, kD, kH, kD>;
    using RightKT = TileRight<half, kD, kBS, kD, kBS>;
    using AccS = TileAcc<float, kH, kBS, kH, kBS>;

    // (matching simpler aic_pv_matmul: pij<half> x V<half> -> oi<float>)
    using LeftP = TileLeft<half, kH, kBS, kH, kBS>;
    using RightV = TileRight<half, kBS, kD, kBS, kD>;
    using AccO = TileAcc<float, kH, kD, kH, kD>;

    const float scale = 1.0f;
    const float neg_inf = -std::numeric_limits<float>::infinity();

    // Compute global max_bn across all batches (matches simpler orchestration)
    int global_max_bn = 0;
    for (int b = 0; b < kBatch; ++b) {
        const int bn_b = (context_lens[b] + kBS - 1) / kBS;
        if (bn_b > global_max_bn) global_max_bn = bn_b;
    }

    for (int b = 0; b < kBatch; ++b) {
        const int ctx_len = context_lens[b];

        // Load query for this batch (half-typed)
        GlobalQ qGlobal(const_cast<half *>(&query[b * kH * kD]));
        QPlainHalf qTileHalf;
        TLOAD(qTileHalf, qGlobal);
        LeftQ qLeft;
        TMOV(qLeft, qTileHalf);

        // Initialize accumulators
        RowRedPlain mi, li, mij, lij, miNew, alphaVec, betaVec, tempLij;
        OPlain oi, oiNew;
        TEXPANDS(mi, neg_inf);
        TEXPANDS(li, 0.0f);
        TEXPANDS(oi, 0.0f);

        for (int bn = 0; bn < global_max_bn; ++bn) {
            const int start = bn * kBS;
            const int valid_len = (start < ctx_len) ? std::min(kBS, ctx_len - start) : 0;

            if (valid_len == 0) {
                // Block entirely beyond sequence: mij=-1e30, lij=0, oi_new=0
                // Use -1e30 instead of -inf to avoid NaN in online update (exp(-inf - (-inf)) = NaN)
                // Matches simpler aiv_softmax_prepare valid_len==0 handling
                constexpr float NEG_LARGE = -1e30f;
                TEXPANDS(mij, NEG_LARGE);
                TEXPANDS(lij, 0.0f);
                TEXPANDS(oiNew, 0.0f);

                // Online softmax update (same formula as normal path)
                TMAX(miNew, mi, mij);
                TSUB(alphaVec, mi, miNew);
                TEXP(alphaVec, alphaVec);
                TSUB(betaVec, mij, miNew);
                TEXP(betaVec, betaVec);

                TMUL(li, li, alphaVec);
                TMUL(tempLij, lij, betaVec);
                TADD(li, li, tempLij);

                TROWEXPANDMUL(oi, oi, alphaVec);
                TROWEXPANDMUL(oiNew, oiNew, betaVec);
                TADD(oi, oi, oiNew);

                TMOV(mi, miNew);
                continue;
            }

            const int phys_block = block_table[b * kMaxNumBlocks + bn];

            // Load K block (half) and transpose
            GlobalKBlock kGlobal(const_cast<half *>(&key_cache[phys_block * kBS * kD]));
            KPlainHalf kTileHalf;
            TLOAD(kTileHalf, kGlobal);
            KTPlainHalf ktTileHalf;
            TTRANS(ktTileHalf, kTileHalf, kTileHalf);
            RightKT kRight;
            TMOV(kRight, ktTileHalf);

            // TASSIGN scores to shared UB offset BEFORE TMOV
            // This ensures TMOV writes directly into the shared buffer
            constexpr std::size_t SCORES_OFF = 0x0;
            ScoresPlain scores;
            TASSIGN(scores, SCORES_OFF);
            
            // S = Q @ K^T (half x half -> float32 accumulator)
            AccS scoresAcc;
            TMATMUL(scoresAcc, qLeft, kRight);
            TMOV(scores, scoresAcc);

            // Mask invalid positions: TFILLPAD_INPLACE using TASSIGN aliasing
            // All tiles share the same UB offset, so TFILLPAD_INPLACE operates in-place.
            ScoresDyn scoresDynTile(static_cast<std::size_t>(valid_len));
            TASSIGN(scoresDynTile, SCORES_OFF);
            ScoresPad scoresPadTile;
            TASSIGN(scoresPadTile, SCORES_OFF);
            TFILLPAD_INPLACE(scoresPadTile, scoresDynTile);

            // Apply scale after masking (matching simpler order: mask -> scale)
            TMULS(scores, scores, scale);

            // Softmax: rowmax, center, exp
            TROWMAX(mij, scores, scores);
            ScoresPlain centered;
            TROWEXPANDSUB(centered, scores, mij);
            ScoresPlain pij;
            TEXP(pij, centered);

            // Truncate pij to fp16 and back to fp32 (matching simpler aiv_softmax_prepare TCVT)
            PijHalf pijF16;
            TCVT(pijF16, pij, RoundMode::CAST_ROUND);
            TCVT(pij, pijF16, RoundMode::CAST_ROUND);
            TROWSUM(lij, pij, pij);

            // P @ V (using half-typed operands, matching simpler aic_pv_matmul)
            GlobalVBlock vGlobal(const_cast<half *>(&value_cache[phys_block * kBS * kD]));
            VPlainHalf vTileHalf;
            TLOAD(vTileHalf, vGlobal);
            LeftP pLeft;
            TMOV(pLeft, pijF16);
            RightV vRight;
            TMOV(vRight, vTileHalf);
            AccO oNewAcc;
            TMATMUL(oNewAcc, pLeft, vRight);
            TMOV(oiNew, oNewAcc);

            // Online softmax update
            TMAX(miNew, mi, mij);
            TSUB(alphaVec, mi, miNew);
            TEXP(alphaVec, alphaVec);
            TSUB(betaVec, mij, miNew);
            TEXP(betaVec, betaVec);

            // li = alpha * li + beta * lij
            TMUL(li, li, alphaVec);
            TMUL(tempLij, lij, betaVec);
            TADD(li, li, tempLij);

            // oi = alpha * oi + beta * oi_new
            TROWEXPANDMUL(oi, oi, alphaVec);
            TROWEXPANDMUL(oiNew, oiNew, betaVec);
            TADD(oi, oi, oiNew);

            // mi = mi_new
            TMOV(mi, miNew);
        }

        // Final normalization: out = oi / li
        TROWEXPANDDIV(oi, oi, li);

        // Store output
        GlobalO oGlobal(&out[b * kH * kD]);
        TSTORE(oGlobal, oi);
    }
}

} // namespace

int main()
{
    std::cout << "batch_paged_attention_demo: B=" << kBatch << " H=" << kNumHeads << " KVH=" << kKvHeadNum
              << " D=" << kHeadDim << " BlockSize=" << kBlockSize << " MaxBlocks=" << kMaxNumBlocks << "\n";
    std::cout << "context_lens: [";
    for (int b = 0; b < kBatch; ++b) {
        if (b > 0) {
            std::cout << ", ";
        }
        std::cout << kContextLens[b];
    }
    std::cout << "]\n";

    const std::size_t q_total = static_cast<std::size_t>(kBatch) * kNumHeads * kHeadDim;
    const std::size_t kv_total = static_cast<std::size_t>(kTotalBlocks) * kBlockSize * kHeadDim;
    const std::size_t out_total = static_cast<std::size_t>(kBatch) * kNumHeads * kHeadDim;

    std::mt19937 rng(kRngSeed);

    // Generate random block table (matching golden.py's torch.randint)
    std::vector<int> block_table(kBatch * kMaxNumBlocks);
    std::uniform_int_distribution<int> bt_dist(0, kTotalBlocks - 1);
    for (int i = 0; i < kBatch * kMaxNumBlocks; ++i) {
        block_table[i] = bt_dist(rng);
    }

    // Generate data with uniform distribution and fp16 precision (matching golden.py)
    std::vector<half> query(q_total);
    std::vector<half> key_cache(kv_total);
    std::vector<half> value_cache(kv_total);

    std::uniform_real_distribution<float> dist_qk(-0.5f, 0.5f);
    std::uniform_real_distribution<float> dist_v(-1.0f, 1.0f);
    for (std::size_t i = 0; i < q_total; ++i) {
        query[i] = static_cast<half>(dist_qk(rng));
    }
    for (std::size_t i = 0; i < kv_total; ++i) {
        key_cache[i] = static_cast<half>(dist_qk(rng));
    }
    for (std::size_t i = 0; i < kv_total; ++i) {
        value_cache[i] = static_cast<half>(dist_v(rng));
    }

    std::vector<float> out_pto(out_total, 0.0f);
    std::vector<float> out_ref(out_total, 0.0f);

    for (int i = 0; i < kWarmupIters; ++i) {
        batch_paged_attention_pto(query, key_cache, value_cache, block_table.data(), kContextLens, out_pto);
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) {
        batch_paged_attention_pto(query, key_cache, value_cache, block_table.data(), kContextLens, out_pto);
    }
    const auto t1 = std::chrono::steady_clock::now();
    batch_paged_attention_reference(query, key_cache, value_cache, block_table.data(), kContextLens, out_ref);

    const double elapsed_s =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count() / static_cast<double>(kIters);

    int total_ctx = 0;
    for (int b = 0; b < kBatch; ++b) {
        total_ctx += kContextLens[b];
    }
    const double matmul_flops = 4.0 * static_cast<double>(kBatch) * static_cast<double>(kNumHeads) *
                                static_cast<double>(total_ctx) * static_cast<double>(kHeadDim);
    const double gflops = (elapsed_s > 0.0) ? (matmul_flops / elapsed_s / 1e9) : 0.0;

    const auto stats = verify_allclose(out_pto, out_ref, kAbsTol, kRelTol, kRelDenomEps);
    std::cout << "verify_allclose: abs_tol=" << kAbsTol << " rel_tol=" << kRelTol << " (bad=" << stats.bad_count
              << ", max_abs=" << stats.max_abs << " @idx=" << stats.max_abs_idx << ", max_rel=" << stats.max_rel
              << " @idx=" << stats.max_rel_idx << ", mean_abs=" << stats.mean_abs << ", rmse=" << stats.rmse << ")\n";

    if (stats.has_nan_or_inf) {
        std::cerr << "[FAIL] NaN/Inf detected\n";
        return kExitFailure;
    }
    if (stats.bad_count != 0) {
        std::cerr << "[FAIL] verification failed\n";
        return kExitFailure;
    }

    float checksum = 0.0f;
    for (std::size_t i = 0; i < out_pto.size(); ++i) {
        checksum += out_pto[i];
    }
    std::cout << "checksum(out) = " << checksum << "\n";
    std::cout << "perf: avg_ms=" << (elapsed_s * 1e3) << " approx_matmul_flops=" << matmul_flops
              << " gflops=" << gflops;
    if (const char *peak_env = std::getenv("PTO_CPU_PEAK_GFLOPS")) {
        char *end = nullptr;
        const double peak = std::strtod(peak_env, &end);
        if (end != peak_env && peak > 0.0) {
            std::cout << " peak_gflops=" << peak << " mfu=" << (gflops / peak);
        }
    }
    std::cout << "\n";
    std::cout << "[PASS] batch_paged_attention_demo\n";
    return kExitSuccess;
}
