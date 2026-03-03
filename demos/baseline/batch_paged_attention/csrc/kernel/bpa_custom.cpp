/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Batch Paged Attention NPU kernel
//
// Cross-core pipeline (CUBE + VEC) for paged attention with online softmax:
//   CUBE: QK matmul (Q @ K^T) and PV matmul (P @ V)
//   VEC:  Softmax preparation (mask, scale, rowmax, exp, fp16 truncation, rowsum)
//         and online softmax accumulation with final normalization
//
// Data flows through GM workspace between cores, synchronized via TSync_Custom.

#include "kernel_operator.h"
#include "bpa_custom.h"

#include <pto/pto-inst.hpp>
#if defined(__DAV_C220_CUBE__) || defined(__DAV_C220_VEC__)
#include <pto/npu/a2a3/custom/TSyncCVID.hpp>
#include <pto/npu/a2a3/custom/TSync_Custom.hpp>
#elif defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#include <pto/npu/a5/custom/TSyncCVID.hpp>
#include <pto/npu/a5/custom/TSync_Custom.hpp>
#endif

using namespace pto;

// Architecture detection
#if defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__)
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

// Cross-core sync flag IDs (each TSync_Custom uses 2 consecutive flags)
// Flag  0,1: QK scores ready  (CUBE -> VEC)
// Flag  2,3: oi_new ready     (CUBE -> VEC)
// Flag  4,5: pij ready        (VEC  -> CUBE)
// Flag  6,7: update done      (VEC  -> CUBE)
constexpr uint16_t QK_FLAG_ID = 0;
constexpr uint16_t OINEW_FLAG_ID = 2;
constexpr uint16_t PIJ_FLAG_ID = 4;
constexpr uint16_t DONE_FLAG_ID = 6;

template <int kH, int kD, int kBS>
AICORE void runBPA(__gm__ half *query, __gm__ half *key_cache, __gm__ half *value_cache,
                   __gm__ int32_t *block_table, __gm__ int32_t *context_lens, __gm__ float *output,
                   uint32_t batch, uint32_t max_bn, __gm__ uint8_t *workspace)
{
    constexpr int kMaxNumBlocks = kBpaMaxNumBlocks;
    constexpr float NEG_LARGE = -1e30f;
    const float scale = 1.0f;

    // Workspace pointers (shared between CUBE and VEC via GM)
    __gm__ float *ws_sij = reinterpret_cast<__gm__ float *>(workspace + kBpaSijOffset);
    __gm__ half *ws_pij = reinterpret_cast<__gm__ half *>(workspace + kBpaPijOffset);
    __gm__ float *ws_oinew = reinterpret_cast<__gm__ float *>(workspace + kBpaOiNewOffset);
    __gm__ uint8_t *cv_comm_buf = workspace + kBpaPerBlockWorkspace;

    // Cross-core ID setup (pairs CUBE and VEC cores on the same AICore)
    int comm_slot = TSYNC_CVID(get_block_idx(), cv_comm_buf);

    // Cross-core sync channels
    constexpr TSync_Custom<SyncOpType::TSTORE_C2GM, SyncOpType::TLOAD> qk_sync = {QK_FLAG_ID};
    constexpr TSync_Custom<SyncOpType::TSTORE_C2GM, SyncOpType::TLOAD> oinew_sync = {OINEW_FLAG_ID};
    constexpr TSync_Custom<SyncOpType::TSTORE_V2GM, SyncOpType::TLOAD> pij_sync = {PIJ_FLAG_ID};
    constexpr TSync_Custom<SyncOpType::TSTORE_V2GM, SyncOpType::TLOAD> done_sync = {DONE_FLAG_ID};

    // =====================================================================
    // CUBE path: QK matmul and PV matmul
    // =====================================================================
    if constexpr (DAV_CUBE) {
        // --- Mat tile types (L1 storage for TLOAD from GM) ---
        // Left operand: ColMajor BLayout, RowMajor SLayout (ND fractal)
        using TileMatQ = Tile<TileType::Mat, half, kH, kD, BLayout::ColMajor, kH, kD, SLayout::RowMajor>;
        using TileMatP = Tile<TileType::Mat, half, kH, kBS, BLayout::ColMajor, kH, kBS, SLayout::RowMajor>;
        // Right operand: RowMajor BLayout, ColMajor SLayout (DN fractal)
        using TileMatKT = Tile<TileType::Mat, half, kD, kBS, BLayout::RowMajor, kD, kBS, SLayout::ColMajor>;
        using TileMatV = Tile<TileType::Mat, half, kBS, kD, BLayout::ColMajor, kBS, kD, SLayout::RowMajor>;

        // --- Matmul operand tiles ---
        using LeftQ = TileLeft<half, kH, kD, kH, kD>;
        using RightKT = TileRight<half, kD, kBS, kD, kBS>;
        using AccS = TileAcc<float, kH, kBS, kH, kBS>;

        using LeftP = TileLeft<half, kH, kBS, kH, kBS>;
        using RightV = TileRight<half, kBS, kD, kBS, kD>;
        using AccO = TileAcc<float, kH, kD, kH, kD>;

        // --- GlobalTensor types ---
        // Q: (kH, kD) row-major
        using GlobalQ = GlobalTensor<half, Shape<1, 1, 1, kH, kD>, Stride<1, 1, 1, kD, 1>>;
        // K^T: logical (kD, kBS) from K stored as (kBS, kD) row-major -> DN layout
        using GlobalKT = GlobalTensor<half, Shape<1, 1, 1, kD, kBS>, Stride<1, 1, 1, 1, kD>, Layout::DN>;
        // V: (kBS, kD) row-major
        using GlobalV = GlobalTensor<half, Shape<1, 1, 1, kBS, kD>, Stride<1, 1, 1, kD, 1>>;
        // Sij workspace: (kH, kBS) float row-major
        using GlobalSij = GlobalTensor<float, Shape<1, 1, 1, kH, kBS>, Stride<1, 1, 1, kBS, 1>>;
        // Pij workspace: (kH, kBS) half row-major
        using GlobalPij = GlobalTensor<half, Shape<1, 1, 1, kH, kBS>, Stride<1, 1, 1, kBS, 1>>;
        // OiNew workspace: (kH, kD) float row-major
        using GlobalOiNew = GlobalTensor<float, Shape<1, 1, 1, kH, kD>, Stride<1, 1, 1, kD, 1>>;

        // --- Allocate L1 tiles ---
        TileMatQ qMatTile;
        TileMatKT ktMatTile;
        TileMatP pMatTile;
        TileMatV vMatTile;
        LeftQ qLeft;
        RightKT kRight;
        AccS accS;
        LeftP pLeft;
        RightV vRight;
        AccO accO;

        // L1 address assignment: Mat tiles and matmul operand tiles share addresses
        // Left operands: qMatTile/qLeft at offset 0, pMatTile/pLeft at offset 2
        // Right operands: ktMatTile/kRight at offset 1, vMatTile/vRight at offset 3
        constexpr uint32_t tileHalfBytes = sizeof(half);
        uint32_t l1_off = 0;
        TASSIGN(qMatTile, l1_off);
        TASSIGN(qLeft, l1_off);
        l1_off += kH * kD * tileHalfBytes;

        TASSIGN(ktMatTile, l1_off);
        TASSIGN(kRight, l1_off);
        l1_off += kD * kBS * tileHalfBytes;

        TASSIGN(pMatTile, l1_off);
        TASSIGN(pLeft, l1_off);
        l1_off += kH * kBS * tileHalfBytes;

        TASSIGN(vMatTile, l1_off);
        TASSIGN(vRight, l1_off);

        TASSIGN(accS, 0x0);
        TASSIGN(accO, 0x0);

        // --- Pipeline init ---
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);

        for (uint32_t b = 0; b < batch; ++b) {
            // Load Q for this batch (loaded once, reused across all blocks)
            GlobalQ qGlobal(query + b * kH * kD);

            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
            TLOAD(qMatTile, qGlobal);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            TMOV(qLeft, qMatTile);
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);  // Pre-signal for first KT TMOV (Q at offset 0, KT at offset 512)

            for (uint32_t bn = 0; bn < max_bn; ++bn) {
                // Wait for VEC to finish previous block's update
                if (bn > 0) {
                    done_sync.wait();
                }

                int32_t ctx_len = context_lens[b];
                int32_t start = static_cast<int32_t>(bn * kBS);
                int32_t valid_len = (start < ctx_len) ? min(kBS, ctx_len - start) : 0;

                if (valid_len == 0) {
                    // No data for this block - signal VEC and skip matmuls
                    qk_sync.record();
                    pij_sync.wait();
                    oinew_sync.record();
                    continue;
                }

                int32_t phys_block = block_table[b * kMaxNumBlocks + bn];

                // === QK Matmul: S = Q @ K^T ===
                GlobalKT ktGlobal(key_cache + phys_block * kBS * kD);

                wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
                TLOAD(ktMatTile, ktGlobal);
                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

                wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
                TMOV(kRight, ktMatTile);
                set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

                // Wait for previous TSTORE to finish reading accumulator
                if (bn > 0 || b > 0) {
                    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
                }

                TMATMUL(accS, qLeft, kRight);

                // M done with L1 data (qLeft, kRight), MTE1 can load pij/V
                set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);

                // Store sij to GM workspace
                set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                GlobalSij sijGlobal(ws_sij);
                TSTORE(sijGlobal, accS);
                set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

                // Signal VEC: sij ready
                qk_sync.record();

                // Wait VEC: pij ready
                pij_sync.wait();

                // === PV Matmul: O_new = P @ V ===
                GlobalPij pijGlobal(ws_pij);

                wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
                TLOAD(pMatTile, pijGlobal);
                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

                wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
                TMOV(pLeft, pMatTile);

                GlobalV vGlobal(value_cache + phys_block * kBS * kD);
                set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);

                wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
                TLOAD(vMatTile, vGlobal);
                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

                TMOV(vRight, vMatTile);
                set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

                // Wait for TSTORE(sij) to finish reading accumulator
                wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

                TMATMUL(accO, pLeft, vRight);

                // M done with L1 data (pLeft, vRight), MTE1 can load next KT
                set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);

                // Store oi_new to GM workspace
                set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                GlobalOiNew oiNewGlobal(ws_oinew);
                TSTORE(oiNewGlobal, accO);
                set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

                // Signal VEC: oi_new ready
                oinew_sync.record();
            }

            // Wait for VEC to finish final block's update
            done_sync.wait();
        }

        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        // Drain the last FIX→M from final TSTORE(oinew)
        if (batch > 0 && max_bn > 0) {
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        }
    }

    // =====================================================================
    // VEC path: Softmax preparation and online update
    // =====================================================================
    if constexpr (DAV_VEC) {
        // --- Vec tile types (UB storage) ---
        using ScoresVec =
            Tile<TileType::Vec, float, kH, kBS, BLayout::RowMajor, kH, kBS, SLayout::NoneBox>;
        using ScoresDynVec =
            Tile<TileType::Vec, float, kH, kBS, BLayout::RowMajor, kH, -1>;
        using ScoresPadVec =
            Tile<TileType::Vec, float, kH, kBS, BLayout::RowMajor, kH, kBS, SLayout::NoneBox, 512, PadValue::Min>;
        using OVec =
            Tile<TileType::Vec, float, kH, kD, BLayout::RowMajor, kH, kD, SLayout::NoneBox>;
        using PijHalfVec =
            Tile<TileType::Vec, half, kH, kBS, BLayout::RowMajor, kH, kBS, SLayout::NoneBox>;

        // Scalar tile types: ND for element-wise ops, DN for row-broadcast ops.
        // TROWEXPAND* with a RowMajor scalar tile takes the 32B path which does NOT
        // broadcast one scalar per row -- it uses all 8 elements per row as separate
        // column multipliers, producing wrong results.  ColMajor (DN) tiles take the
        // vbrcb path which correctly broadcasts one scalar per row.
        // Conversion between layouts uses a GM round-trip: TSTORE(DN) -> TLOAD(ND)
        // or TSTORE(ND) -> TLOAD(DN).  (Matches simpler/examples aiv_online_update.)
        constexpr uint32_t kRedCols = BLOCK_BYTE_SIZE / sizeof(float);  // 8
        constexpr uint32_t kScalarRows = kH / kRedCols;                 // 2
        constexpr uint32_t kAlignedRows =
            ((kH * static_cast<uint32_t>(sizeof(float)) + 31) / 32) * (32 / static_cast<uint32_t>(sizeof(float)));  // 16

        // ND layout (kScalarRows x kRedCols RowMajor) for element-wise ops
        using ScalarND =
            Tile<TileType::Vec, float, kScalarRows, kRedCols, BLayout::RowMajor, kScalarRows, kRedCols, SLayout::NoneBox>;
        // DN layout (kAlignedRows x 1 ColMajor) for TROWMAX/TROWSUM output & TROWEXPAND* ops
        using ScalarDN =
            Tile<TileType::Vec, float, kAlignedRows, 1, BLayout::ColMajor, kH, 1>;

        // --- GlobalTensor types for workspace ---
        using GlobalSij =
            GlobalTensor<float, Shape<1, 1, 1, kH, kBS>, Stride<1, 1, 1, kBS, 1>>;
        using GlobalPij =
            GlobalTensor<half, Shape<1, 1, 1, kH, kBS>, Stride<1, 1, 1, kBS, 1>>;
        using GlobalOiNew =
            GlobalTensor<float, Shape<1, 1, 1, kH, kD>, Stride<1, 1, 1, kD, 1>>;
        using GlobalO =
            GlobalTensor<float, Shape<1, 1, 1, kH, kD>, Stride<1, 1, 1, kD, 1>>;

        // GlobalTensor types for ND<->DN scalar conversion via GM round-trip
        using GlobalScalarND =
            GlobalTensor<float, Shape<1, 1, 1, kScalarRows, kRedCols>, Stride<1, 1, 1, kRedCols, 1>>;
        using GlobalScalarDN =
            GlobalTensor<float, Shape<1, 1, 1, kAlignedRows, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;

        // --- Allocate UB tiles ---
        constexpr uint32_t SCORES_OFF = 0x0;
        constexpr uint32_t CENTERED_OFF = SCORES_OFF + kH * kBS * sizeof(float);
        constexpr uint32_t PIJ_F16_OFF = CENTERED_OFF + kH * kBS * sizeof(float);
        // ND scalar tiles (kScalarRows * kRedCols * sizeof(float) = 64 bytes each)
        constexpr uint32_t kScalarNDSize = kScalarRows * kRedCols * sizeof(float);
        constexpr uint32_t MIJ_ND_OFF = PIJ_F16_OFF + kH * kBS * sizeof(half);
        constexpr uint32_t LIJ_ND_OFF = MIJ_ND_OFF + kScalarNDSize;
        constexpr uint32_t MI_OFF = LIJ_ND_OFF + kScalarNDSize;
        constexpr uint32_t LI_OFF = MI_OFF + kScalarNDSize;
        constexpr uint32_t MI_NEW_OFF = LI_OFF + kScalarNDSize;
        constexpr uint32_t ALPHA_OFF = MI_NEW_OFF + kScalarNDSize;
        constexpr uint32_t BETA_OFF = ALPHA_OFF + kScalarNDSize;
        constexpr uint32_t TEMP_OFF = BETA_OFF + kScalarNDSize;
        // Data tiles (kH * kD * sizeof(float) = 1024 bytes each)
        constexpr uint32_t OI_OFF = TEMP_OFF + kScalarNDSize;
        constexpr uint32_t OI_NEW_OFF = OI_OFF + kH * kD * sizeof(float);
        // DN scalar tiles (kAlignedRows * sizeof(float) = 64 bytes each)
        // Used for TROWMAX/TROWSUM output and ND->DN conversion results.
        // Placed at end; sequential usage makes potential physical overlap safe.
        constexpr uint32_t kScalarDNSize = kAlignedRows * sizeof(float);
        constexpr uint32_t MIJ_DN_OFF = OI_NEW_OFF + kH * kD * sizeof(float);
        constexpr uint32_t LIJ_DN_OFF = MIJ_DN_OFF + kScalarDNSize;
        // Reusable DN slots for alpha/beta/li (never live simultaneously with mij/lij DN)
        constexpr uint32_t ALPHA_DN_OFF = MIJ_DN_OFF;
        constexpr uint32_t BETA_DN_OFF = LIJ_DN_OFF;
        constexpr uint32_t LI_DN_OFF = MIJ_DN_OFF;

        ScoresVec scores;
        ScoresVec centered;
        PijHalfVec pij_f16;
        ScalarND mijND, lijND, mi, li, miNew, alphaVec, betaVec, tempLij;
        ScalarDN mijDN, lijDN, alphaDN, betaDN, liDN;
        OVec oi, oiNew;

        TASSIGN(scores, SCORES_OFF);
        TASSIGN(centered, CENTERED_OFF);
        TASSIGN(pij_f16, PIJ_F16_OFF);
        TASSIGN(mijND, MIJ_ND_OFF);
        TASSIGN(lijND, LIJ_ND_OFF);
        TASSIGN(mi, MI_OFF);
        TASSIGN(li, LI_OFF);
        TASSIGN(miNew, MI_NEW_OFF);
        TASSIGN(alphaVec, ALPHA_OFF);
        TASSIGN(betaVec, BETA_OFF);
        TASSIGN(tempLij, TEMP_OFF);
        TASSIGN(oi, OI_OFF);
        TASSIGN(oiNew, OI_NEW_OFF);
        TASSIGN(mijDN, MIJ_DN_OFF);
        TASSIGN(lijDN, LIJ_DN_OFF);
        TASSIGN(alphaDN, ALPHA_DN_OFF);
        TASSIGN(betaDN, BETA_DN_OFF);
        TASSIGN(liDN, LI_DN_OFF);

        // GM scratch addresses for ND<->DN scalar conversion (reuse ws_sij area,
        // which is free after VEC loads sij into UB and during online update)
        __gm__ float *gm_scratch_0 = ws_sij;
        __gm__ float *gm_scratch_1 = ws_sij + kH;  // +64 bytes

        set_mask_norm();
        set_vector_mask(-1, -1);

        // Pipe init
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

        for (uint32_t b = 0; b < batch; ++b) {
            // Ensure previous batch's output TSTORE has completed reading oi
            // from UB before overwriting it with TEXPANDS below.
            if (b > 0) {
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            }

            // Initialize accumulators for this batch
            TEXPANDS(mi, -__builtin_huge_valf());
            TEXPANDS(li, 0.0f);
            TEXPANDS(oi, 0.0f);

            for (uint32_t bn = 0; bn < max_bn; ++bn) {
                int32_t ctx_len = context_lens[b];
                int32_t start = static_cast<int32_t>(bn * kBS);
                int32_t valid_len = (start < ctx_len) ? min(kBS, ctx_len - start) : 0;

                // Wait for CUBE: sij ready
                qk_sync.wait();

                if (valid_len == 0) {
                    // Block beyond sequence: mij=-1e30, lij=0, oi_new=0
                    TEXPANDS(mijND, NEG_LARGE);
                    TEXPANDS(lijND, 0.0f);
                    TEXPANDS(oiNew, 0.0f);

                    // Produce dummy pij for CUBE (zeros)
                    TEXPANDS(pij_f16, static_cast<half>(0));
                    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    GlobalPij pijGlobal(ws_pij);
                    TSTORE(pijGlobal, pij_f16);
                    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

                    // Signal CUBE: pij ready
                    pij_sync.record();

                    // Wait for CUBE: oi_new ready (CUBE also skips in this case)
                    oinew_sync.wait();
                } else {
                    // --- Softmax preparation ---

                    // Load sij from workspace
                    GlobalSij sijGlobal(ws_sij);
                    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                    TLOAD(scores, sijGlobal);
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

                    // Mask invalid positions with -inf (TFILLPAD_INPLACE)
                    if (valid_len < kBS) {
                        ScoresDynVec scoresDyn(static_cast<std::size_t>(valid_len));
                        TASSIGN(scoresDyn, SCORES_OFF);
                        ScoresPadVec scoresPad;
                        TASSIGN(scoresPad, SCORES_OFF);
                        TFILLPAD_INPLACE(scoresPad, scoresDyn);
                        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                    }

                    // Apply scale
                    TMULS(scores, scores, scale);
                    pipe_barrier(PIPE_V);

                    // Row max -> DN tile (used directly for TROWEXPANDSUB)
                    TROWMAX(mijDN, scores, centered);
                    pipe_barrier(PIPE_V);

                    // Center and exp: pij = exp(scores - mij)
                    TROWEXPANDSUB(centered, scores, mijDN);
                    pipe_barrier(PIPE_V);
                    ScoresVec pij_f32;
                    TASSIGN(pij_f32, CENTERED_OFF);  // Reuse centered buffer
                    TEXP(pij_f32, centered);

                    // FP16 truncation (matching golden.py precision)
                    TCVT(pij_f16, pij_f32, RoundMode::CAST_ROUND);
                    TCVT(pij_f32, pij_f16, RoundMode::CAST_ROUND);
                    pipe_barrier(PIPE_V);

                    // Row sum -> DN tile
                    TROWSUM(lijDN, pij_f32, centered);
                    pipe_barrier(PIPE_V);

                    // Convert mij and lij from DN to ND for element-wise ops in online update.
                    // Store DN tiles to GM, reload as ND (ws_sij is free after TLOAD above).
                    GlobalScalarDN gmMijDN(gm_scratch_0);
                    GlobalScalarND gmMijND(gm_scratch_0);
                    GlobalScalarDN gmLijDN(gm_scratch_1);
                    GlobalScalarND gmLijND(gm_scratch_1);
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                    TSTORE(gmMijDN, mijDN);
                    TSTORE(gmLijDN, lijDN);
                    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                    TLOAD(mijND, gmMijND);
                    TLOAD(lijND, gmLijND);
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

                    // Store pij (fp16) to workspace for CUBE PV matmul
                    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    GlobalPij pijGlobal(ws_pij);
                    TSTORE(pijGlobal, pij_f16);
                    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

                    // Signal CUBE: pij ready
                    pij_sync.record();

                    // Wait for CUBE: oi_new ready
                    oinew_sync.wait();

                    // Load oi_new from workspace
                    GlobalOiNew oiNewGlobal(ws_oinew);
                    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                    TLOAD(oiNew, oiNewGlobal);
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                }

                // --- Online softmax update (all element-wise ops use ND tiles) ---
                // mi_new = max(mi, mij)
                TMAX(miNew, mi, mijND);
                pipe_barrier(PIPE_V);

                // alpha = exp(mi - mi_new), beta = exp(mij - mi_new)
                TSUB(alphaVec, mi, miNew);
                pipe_barrier(PIPE_V);
                TEXP(alphaVec, alphaVec);
                pipe_barrier(PIPE_V);
                TSUB(betaVec, mijND, miNew);
                pipe_barrier(PIPE_V);
                TEXP(betaVec, betaVec);
                pipe_barrier(PIPE_V);

                // li = alpha * li + beta * lij
                TMUL(li, li, alphaVec);
                pipe_barrier(PIPE_V);
                TMUL(tempLij, lijND, betaVec);
                pipe_barrier(PIPE_V);
                TADD(li, li, tempLij);
                pipe_barrier(PIPE_V);

                // Convert alpha/beta from ND to DN for TROWEXPANDMUL
                GlobalScalarND gmAlphaND(gm_scratch_0);
                GlobalScalarDN gmAlphaDN(gm_scratch_0);
                GlobalScalarND gmBetaND(gm_scratch_1);
                GlobalScalarDN gmBetaDN(gm_scratch_1);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                TSTORE(gmAlphaND, alphaVec);
                TSTORE(gmBetaND, betaVec);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                TLOAD(alphaDN, gmAlphaDN);
                TLOAD(betaDN, gmBetaDN);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

                // oi = alpha * oi + beta * oi_new (DN tiles for correct broadcast)
                TROWEXPANDMUL(oi, oi, alphaDN);
                pipe_barrier(PIPE_V);
                TROWEXPANDMUL(oiNew, oiNew, betaDN);
                pipe_barrier(PIPE_V);
                TADD(oi, oi, oiNew);

                // mi = mi_new
                TMOV(mi, miNew);

                if (valid_len > 0) {
                    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                }

                // Signal CUBE: update done
                done_sync.record();
            }

            // Final normalization: convert li to DN for TROWEXPANDDIV
            // Use ws_oinew (not gm_scratch_0 = ws_sij) as GM scratch for the
            // ND->DN round-trip.  After done_sync.record() above, CUBE is free
            // to start the next batch and will overwrite ws_sij with new sij
            // data.  ws_oinew is safe: CUBE only writes to it after completing
            // QK + softmax for the next batch, by which time this conversion
            // has long finished.
            pipe_barrier(PIPE_V);
            GlobalScalarND gmLiND(reinterpret_cast<__gm__ float *>(workspace + kBpaOiNewOffset));
            GlobalScalarDN gmLiDN(reinterpret_cast<__gm__ float *>(workspace + kBpaOiNewOffset));
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            TSTORE(gmLiND, li);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
            TLOAD(liDN, gmLiDN);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

            TROWEXPANDDIV(oi, oi, liDN);

            // Store output
            GlobalO oGlobal(output + b * kH * kD);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(oGlobal, oi);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
}

extern "C" __global__ AICORE void bpa_custom(GM_ADDR query, GM_ADDR key_cache, GM_ADDR value_cache,
                                              GM_ADDR block_table, GM_ADDR context_lens, GM_ADDR output,
                                              uint32_t batch, uint32_t max_bn, GM_ADDR workspace)
{
    runBPA<kBpaNumHeads, kBpaHeadDim, kBpaBlockSize>(
        reinterpret_cast<__gm__ half *>(query), reinterpret_cast<__gm__ half *>(key_cache),
        reinterpret_cast<__gm__ half *>(value_cache), reinterpret_cast<__gm__ int32_t *>(block_table),
        reinterpret_cast<__gm__ int32_t *>(context_lens), reinterpret_cast<__gm__ float *>(output), batch, max_bn,
        reinterpret_cast<__gm__ uint8_t *>(workspace));
}
