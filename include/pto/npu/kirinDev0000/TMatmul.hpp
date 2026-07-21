/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMATMUL_HPP
#define TMATMUL_HPP

#include <cstdint>
#include <type_traits>

namespace pto {

inline namespace TMatmulInternal {
constexpr const int MMAD_MAX_SUPPORT_LENGTH = 4095;
constexpr const int TF32_MODE_BIT = 46;
constexpr const int TF32_TRANS_MODE_BIT = 47;
} // namespace TMatmulInternal

// ============================================================================
// PTO-level enums for matmul_to_cbuf features (mapped to CCE matmul_t enums)
// ============================================================================

enum class MatmulPreQuant : uint8_t {
    NoQuant = 0,
    ReqS8Vector = 2,
    ReqS8Scalar = 3,
    DeqF16Vector = 4,
    DeqF16Scalar = 5,
    DeqS16Vector = 6,
    DeqS16Scalar = 7,
    DeqS32Vector = 8,
    DeqS32Scalar = 9,
    ReqS4Vector = 10,
    ReqS4Scalar = 11,
    Qs322Fp8E4m3IntVector = 12,
    Qs322Fp8E4m3IntScalar = 13,
};

enum class MatmulPostQuant : uint8_t {
    NoQuant = 0,
    Shift2S16Scalar = 1,
    Shift2S16Vector = 2,
    Shift2S8Scalar = 3,
    Shift2S8Vector = 4,
    Shift2S4Scalar = 5,
    Shift2S4Vector = 6,
    Shift2S32Scalar = 7,
    Shift2S32Vector = 8,
};

enum class MatmulPreRelu : uint8_t {
    NoRelu = 0,
    NormalRelu = 1,
    ScalarRelu = 2,
    VectorRelu = 3,
    LutActivation = 4,
};

// MatmulPostRelu: kirinDev0000 hardware does not support post-stage relu.
// (post_fp_t.act_post field exists in the header but is not functional on L510.)

enum class MatmulEltwiseOp : uint8_t {
    NoEltwise = 0,
    Add = 1,
    Sub = 2,
    Mul = 3,
    Max = 4,
};

enum class MatmulLsbMask : uint8_t {
    Disable = 0,
    Mask1Lsb = 1,
    Mask2Lsbs = 2,
    Mask3Lsbs = 3,
    Mask4Lsbs = 4,
};

enum class MatmulInstrId : uint8_t {
    Id0 = 0,
    Id1 = 1,
    Id2 = 2,
    Id3 = 3,
};

// ============================================================================
// Config builder: assembles uint64_t config from PTO enums
// L510 (non-5161) config layout:
//   [0]      init_ctrl
//   [1]      broadcast_en
//   [2:6]    pre_quant (5 bits)
//   [7:9]    pre_relu (3 bits)
//   [10:14]  post_quant (5 bits)
//   [15:17]  reserved (post_fp_t.act_post, not functional on L510)
//   [18]     clip_relu_en
//   [20:22]  eltwise_op (3 bits)
//   [23]     antiq_en
//   [24]     m_d_broadca_en
//   [28:30]  lsb_mask (3 bits)
//   [31]     gemv_ctrl
//   [36]     depend_en
//   [37:38]  instr_id (2 bits)
//   [39]     bandwidth_ctrl
//   [40]     nbrc_bias_ctrl
//   [48:63]  size_n (16 bits)
// ============================================================================

struct MatmulConfig {
    bool initCtrl = true;
    bool broadcastEn = false;
    MatmulPreQuant preQuant = MatmulPreQuant::NoQuant;
    MatmulPreRelu preRelu = MatmulPreRelu::NoRelu;
    MatmulPostQuant postQuant = MatmulPostQuant::NoQuant;
    bool clipReluEn = false;
    MatmulEltwiseOp eltwiseOp = MatmulEltwiseOp::NoEltwise;
    bool antiqEn = false;
    bool mDBroadcaEn = false;
    MatmulLsbMask lsbMask = MatmulLsbMask::Disable;
    bool gemvCtrl = false;
    bool dependEn = false;
    MatmulInstrId instrId = MatmulInstrId::Id0;
    bool bandwidthCtrl = false;
    bool nbrcBiasCtrl = false;
    uint16_t sizeN = 0;

    constexpr uint64_t Build() const
    {
        return (static_cast<uint64_t>(initCtrl) & 0x1) | ((static_cast<uint64_t>(broadcastEn) & 0x1) << 1) |
               ((static_cast<uint64_t>(preQuant) & 0x1f) << 2) | ((static_cast<uint64_t>(preRelu) & 0x7) << 7) |
               ((static_cast<uint64_t>(postQuant) & 0x1f) << 10) | ((static_cast<uint64_t>(clipReluEn) & 0x1) << 18) |
               ((static_cast<uint64_t>(eltwiseOp) & 0x7) << 20) | ((static_cast<uint64_t>(antiqEn) & 0x1) << 23) |
               ((static_cast<uint64_t>(mDBroadcaEn) & 0x1) << 24) | ((static_cast<uint64_t>(lsbMask) & 0x7) << 28) |
               ((static_cast<uint64_t>(gemvCtrl) & 0x1) << 31) | ((static_cast<uint64_t>(dependEn) & 0x1) << 36) |
               ((static_cast<uint64_t>(instrId) & 0x3) << 37) | ((static_cast<uint64_t>(bandwidthCtrl) & 0x1) << 39) |
               ((static_cast<uint64_t>(nbrcBiasCtrl) & 0x1) << 40) | ((static_cast<uint64_t>(sizeN) & 0xffff) << 48);
    }
};

// Helper: set M_FPC for FP16 fixed-point shift (FIXED_SHIFT_VAL=42).
// Non-MACRO __tf__ functions call this before TMatmulCore.
template <typename T>
__tf__ PTO_INTERNAL void SetMFpcForFp16()
{
    if constexpr (std::is_same_v<T, half>) {
        uint64_t mFpc = 0;
        mFpc |= ((uint64_t)42) << 40;
        set_m_fpc(mFpc);
    }
}

// Helper: set M_RELU_ALPHA for Scalar ReLU.
// M_RELU_ALPHA layout:
//   [2:0]    LUT mode (not supported, 0)
//   [31:13]  M2 in (1,8,10) format: sign(1) + exponent(8) + mantissa(10)
//   [63:45]  reserved
// Converts float32 (1,8,23) to (1,8,10) by truncating mantissa.
__tf__ PTO_INTERNAL void SetMReluAlpha(float reluScalar)
{
    uint32_t floatBits = 0;
    *reinterpret_cast<float*>(reinterpret_cast<char*>(&floatBits)) = reluScalar;
    uint32_t sign = (floatBits >> 31) & 0x1;
    uint32_t exp = (floatBits >> 23) & 0xFF;
    uint32_t mantissa = (floatBits >> 13) & 0x3FF;
    uint32_t m2 = (sign << 18) | (exp << 10) | mantissa;
    uint64_t mReluAlpha = static_cast<uint64_t>(m2) << 13;
    set_m_relu_alpha(mReluAlpha);
}

// Helper: set M_QUANT_PRE for FP16 fixed-point shift and clip relu.
// M_QUANT_PRE layout:
//   [31:13]  M1 in (1,8,10) format (pre-stage quant scale / FP16 fixed shift)
//   [46]     reserved
//   [63:48]  PRE_CLIP_VAL (clip relu max, format depends on output type)
// For FP16: M1 = (42-58+127) packed into (1,8,10).
// For clip relu: PRE_CLIP_VAL converted from float to output-type format.
template <typename OutType>
__tf__ PTO_INTERNAL void SetMQuantPre(float clipReluVal)
{
    uint64_t mQuantPre = 0;
    if constexpr (std::is_same_v<OutType, half> || std::is_same_v<OutType, int16_t>) {
        mQuantPre |= (((uint64_t)(42 - 58 + 127) & 0xff) << 10) << 13;
    }
    if (clipReluVal != 0.0f) {
        uint64_t clipVal = 0;
        if constexpr (std::is_same_v<OutType, half>) {
            half h = static_cast<half>(clipReluVal);
            clipVal = *reinterpret_cast<uint16_t*>(&h);
        } else if constexpr (std::is_same_v<OutType, int8_t>) {
            int16_t v = static_cast<int16_t>(clipReluVal);
            clipVal = static_cast<uint16_t>(v) & 0xFFFF;
        } else if constexpr (std::is_same_v<OutType, int16_t>) {
            int16_t v = static_cast<int16_t>(clipReluVal);
            clipVal = static_cast<uint16_t>(v) & 0xFFFF;
        }
        mQuantPre |= (clipVal & 0xFFFF) << 48;
    }
    set_m_quant_pre(mQuantPre);
}

// ============================================================================
// Layer 1: Core __tf__ function — wraps matmul_to_cbuf with full config
// ============================================================================

template <typename TileRes, typename TileLeft, typename TileRight>
__tf__ PTO_INTERNAL void TMatmulCore(
    typename TileRes::TileDType __out__ cData, typename TileLeft::TileDType __in__ aData,
    typename TileRight::TileDType __in__ bData, __cbuf__ int32_t* biasAddr, uint64_t btAddr, uint16_t m, uint16_t k,
    uint16_t n, uint64_t config)
{
    constexpr uint32_t BLOCK_BYTE = 32;
    constexpr uint32_t FRACTAL_ROW = 16;

    // set_matrix_para: [15:0]=M, [31:16]=K
    uint64_t matrixPara = (static_cast<uint64_t>(m) & 0xFFFF) | ((static_cast<uint64_t>(k) & 0xFFFF) << 16);
    set_matrix_para(matrixPara);

    // set_cube_stride_para: [15:0]=LOOP4_SRC_STRIDE, [31:16]=LOOP4_DST_STRIDE (in 32B units)
    //
    //
    //
    //
    //
    //
    //
    //
    constexpr uint32_t srcStride = TileLeft::Rows * TileLeft::InnerCols * sizeof(typename TileLeft::DType) / BLOCK_BYTE;
    constexpr uint32_t dstStride = TileRes::Rows * TileRes::InnerCols * sizeof(typename TileRes::DType) / BLOCK_BYTE;
    uint64_t cubeStridePara =
        (static_cast<uint64_t>(srcStride) & 0xFFFF) | ((static_cast<uint64_t>(dstStride) & 0xFFFF) << 16);
    set_cube_stride_para(cubeStridePara);

    using T = typename TileRes::DType;

    // Zero-initialize C tile to avoid 5A residue in padding area.
    // initCtrl=1 only zeros the computed M×N region; padding stays uninitialized.
    __cbuf__ void* c = (__cbuf__ void*)__cce_get_tile_ptr(cData);
    if (config & 0x1) {
        constexpr uint32_t totalBlocks = (TileRes::Rows * TileRes::Cols * sizeof(typename TileRes::DType)) / BLOCK_BYTE;
        // Xm: [14:0]=repeat times, [30:16]=block number per repeat (32B units), [46:32]=repeat gap (32B units)
        constexpr int64_t repeatConfig = ((static_cast<int64_t>(0) & 0x7FFF) << 32) |
                                         ((static_cast<int64_t>(totalBlocks) & 0x7FFF) << 16) |
                                         (static_cast<int64_t>(1) & 0x7FFF);
        if constexpr (std::is_same_v<T, half>) {
            pto_create_cbuf_matrix((__cbuf__ half*)c, repeatConfig, (half)0);
        } else {
            pto_create_cbuf_matrix((__cbuf__ uint16_t*)c, repeatConfig, (uint16_t)0);
        }
    }

    __cbuf__ typename TileLeft::DType* a = (__cbuf__ typename TileLeft::DType*)__cce_get_tile_ptr(aData);
    __cb__ typename TileRight::DType* b = (__cb__ typename TileRight::DType*)__cce_get_tile_ptr(bData);

    matmul_to_cbuf(c, a, b, biasAddr, btAddr, config);
}

// ============================================================================
// Layer 2: Feature-specific __tf__ functions
// ============================================================================

// --- 1. Normal: C = A*B or C += A*B ---
template <typename TileRes, typename TileLeft, typename TileRight>
__tf__ PTO_INTERNAL void TMatmul(
    typename TileRes::TileDType __out__ cData, typename TileLeft::TileDType __in__ aData,
    typename TileRight::TileDType __in__ bData, uint16_t m, uint16_t k, uint16_t n, bool initCtrl)
{
    using T = typename TileRes::DType;
    MatmulConfig cfg;
    cfg.initCtrl = initCtrl;
    cfg.sizeN = n;
    if constexpr (std::is_same_v<T, half>) {
        cfg.preQuant = MatmulPreQuant::DeqF16Scalar;
    }
    SetMFpcForFp16<T>();
    SetMQuantPre<T>(0.0f);
    TMatmulCore<TileRes, TileLeft, TileRight>(cData, aData, bData, nullptr, 0, m, k, n, cfg.Build());
}

// --- 2a. Non-broadcast bias (bias in L1): C = A*B + bias or C += A*B + bias ---
template <typename TileRes, typename TileLeft, typename TileRight, typename TileBias>
__tf__ PTO_INTERNAL void TMatmulBiasNonBroadcast(
    typename TileRes::TileDType __out__ cData, typename TileLeft::TileDType __in__ aData,
    typename TileRight::TileDType __in__ bData, typename TileBias::TileDType __in__ biasData, uint16_t m, uint16_t k,
    uint16_t n, bool initCtrl)
{
    __cbuf__ typename TileBias::DType* biasPtr = (__cbuf__ typename TileBias::DType*)__cce_get_tile_ptr(biasData);
    MatmulConfig cfg;
    cfg.initCtrl = initCtrl;
    cfg.broadcastEn = false;
    cfg.sizeN = n;
    SetMFpcForFp16<typename TileRes::DType>();
    SetMQuantPre<typename TileRes::DType>(0.0f);
    TMatmulCore<TileRes, TileLeft, TileRight>(cData, aData, bData, (__cbuf__ int32_t*)biasPtr, 0, m, k, n, cfg.Build());
}

// --- 2b. Broadcast bias (bias in BiasTable): C = A*B + bias_vec or C += A*B + bias_vec ---
template <typename TileRes, typename TileLeft, typename TileRight>
__tf__ PTO_INTERNAL void TMatmulBiasBroadcast(
    typename TileRes::TileDType __out__ cData, typename TileLeft::TileDType __in__ aData,
    typename TileRight::TileDType __in__ bData, uint64_t btAddr, uint16_t m, uint16_t k, uint16_t n, bool initCtrl)
{
    using T = typename TileRes::DType;
    constexpr uint32_t BLOCK_BYTE = 32;
    __cbuf__ void* c = (__cbuf__ void*)__cce_get_tile_ptr(cData);
    constexpr uint32_t totalBlocks = (TileRes::Rows * TileRes::Cols * sizeof(T)) / BLOCK_BYTE;
    constexpr int64_t repeatConfig = ((static_cast<int64_t>(0) & 0x7FFF) << 32) |
                                     ((static_cast<int64_t>(totalBlocks) & 0x7FFF) << 16) |
                                     (static_cast<int64_t>(1) & 0x7FFF);
    if constexpr (std::is_same_v<T, half>) {
        pto_create_cbuf_matrix((__cbuf__ half*)c, repeatConfig, (half)0);
    } else {
        pto_create_cbuf_matrix((__cbuf__ uint16_t*)c, repeatConfig, (uint16_t)0);
    }
    MatmulConfig cfg;
    cfg.initCtrl = initCtrl;
    cfg.broadcastEn = true;
    cfg.sizeN = n;
    if constexpr (std::is_same_v<T, half>) {
        cfg.preQuant = MatmulPreQuant::DeqF16Scalar;
    }
    SetMFpcForFp16<T>();
    SetMQuantPre<T>(0.0f);
    TMatmulCore<TileRes, TileLeft, TileRight>(cData, aData, bData, nullptr, btAddr, m, k, n, cfg.Build());
}

// ============================================================================
// MatmulMacroConfig: user-supplied quant parameter struct for TMATMUL_MACRO
// Quant types are NOT here — they are inferred from input/output data types.
// ============================================================================

enum class MatmulReluMode : uint8_t {
    NoRelu = 0,
    NormalRelu = 1,
    ScalarRelu = 2,
    VectorRelu = 3,
};

enum class MatmulQuantMode : uint8_t {
    ScalarQuant = 0,
    VectorQuant = 1,
};

struct MatmulMacroConfig {
    uint64_t preQuantTileAddr = 0;
    uint64_t vectorReluTileAddr = 0;
    float reluScalar = 0.0f;
    float clipReluVal = 0.0f;
    bool gemvCtrl = false;
    MatmulReluMode reluMode = MatmulReluMode::NoRelu;
    MatmulQuantMode quantMode = MatmulQuantMode::VectorQuant;
};

// ============================================================================
// Type inference: infer pre/post quant types from L0C src type and output type
// ============================================================================

template <typename L0cType, typename OutType>
AICORE constexpr MatmulPreQuant InferPreQuant()
{
    if constexpr (std::is_same_v<OutType, int32_t>) {
        return MatmulPreQuant::NoQuant;
    } else if constexpr (std::is_same_v<OutType, half>) {
        return MatmulPreQuant::DeqF16Vector;
    } else if constexpr (std::is_same_v<OutType, int8_t>) {
        return MatmulPreQuant::ReqS8Vector;
    } else if constexpr (std::is_same_v<OutType, int16_t>) {
        return MatmulPreQuant::DeqS16Vector;
    } else {
        return MatmulPreQuant::NoQuant;
    }
}

// ============================================================================
// Layer 2: __tf__ functions for TMATMUL_MACRO and TMATMUL_MACRO_ACC
// ============================================================================

// --- TMATMUL_MACRO_ACC: partial sum for split-K ---
//   Three modes:
//     Bias (hasBias):       init_ctrl=0, broadcast_en=isBroadcast, manual zero C (k=0 with bias)
//     Clear (!hasBias, isClear): init_ctrl=1, broadcast_en=0, hw zero C (k=0 no bias)
//     Acc  (!hasBias, !isClear): init_ctrl=0, broadcast_en=1, btAddr=C_tile_L1 (accumulate via bias-broadcast)
template <
    typename TileRes, typename TileLeft, typename TileRight, bool hasBias, bool isBroadcast = false,
    bool isClear = false>
__tf__ PTO_INTERNAL void TMatmulMacroAcc(
    typename TileRes::TileDType __out__ cData, typename TileLeft::TileDType __in__ aData,
    typename TileRight::TileDType __in__ bData, __cbuf__ int32_t* biasAddr, uint64_t btAddr, uint16_t m, uint16_t k,
    uint16_t n, const MatmulMacroConfig& cfg)
{
    using OutType = typename TileRes::DType;

    constexpr auto preQuant = MatmulPreQuant::NoQuant;
    constexpr auto preRelu = MatmulPreRelu::NoRelu;

    // ACC mode uses non-broadcast bias from L1: biasPtr points to C tile in L1
    constexpr bool broadcastEn = hasBias && isBroadcast;
    constexpr bool initCtrl = isClear;

    if constexpr (hasBias) {
        constexpr uint32_t BLOCK_BYTE = 32;
        __cbuf__ void* c = (__cbuf__ void*)__cce_get_tile_ptr(cData);
        constexpr uint32_t totalBlocks = (TileRes::Rows * TileRes::Cols * sizeof(OutType)) / BLOCK_BYTE;
        constexpr int64_t repeatConfig = ((static_cast<int64_t>(0) & 0x7FFF) << 32) |
                                         ((static_cast<int64_t>(totalBlocks) & 0x7FFF) << 16) |
                                         (static_cast<int64_t>(1) & 0x7FFF);
        if constexpr (std::is_same_v<OutType, half>) {
            pto_create_cbuf_matrix((__cbuf__ half*)c, repeatConfig, (half)0);
        } else {
            pto_create_cbuf_matrix((__cbuf__ uint16_t*)c, repeatConfig, (uint16_t)0);
        }
    }

    using InputType = typename TileLeft::DType;
    uint64_t mFpc = 0;
    if constexpr (std::is_same_v<InputType, half>) {
        mFpc |= ((uint64_t)42) << 40;
    }
    set_m_fpc(mFpc);

    SetMQuantPre<InputType>(0.0f);

    MatmulConfig matmulCfg;
    matmulCfg.initCtrl = initCtrl;
    matmulCfg.broadcastEn = broadcastEn;
    matmulCfg.preQuant = preQuant;
    matmulCfg.preRelu = preRelu;
    matmulCfg.sizeN = n;
    TMatmulCore<TileRes, TileLeft, TileRight>(cData, aData, bData, biasAddr, btAddr, m, k, n, matmulCfg.Build());
}

// --- TMATMUL_MACRO: final iteration or no split-K ---
//   Output: F16/S8/S16 (with pre-stage quant + optional relu)
//   Three mutually exclusive modes (hasBias and isAcc are mutually exclusive):
//     Clear (!hasBias && !isAcc): init_ctrl=1 (hw zero C), broadcast_en=0
//     Bias  (hasBias):            init_ctrl=0, broadcast_en=isBroadcast, manual zero C
//     Acc   (isAcc):              init_ctrl=0, broadcast_en=1, btAddr=C_tile_L1 (accumulate via bias-broadcast)
template <
    typename TileRes, typename TileLeft, typename TileRight, bool hasBias, bool isBroadcast = false, bool isAcc = false>
__tf__ PTO_INTERNAL void TMatmulMacro(
    typename TileRes::TileDType __out__ cData, typename TileLeft::TileDType __in__ aData,
    typename TileRight::TileDType __in__ bData, __cbuf__ int32_t* biasAddr, uint64_t btAddr, uint16_t m, uint16_t k,
    uint16_t n, const MatmulMacroConfig& cfg)
{
    static_assert(!(hasBias && isAcc), "bias and acc are mutually exclusive");
    using OutType = typename TileRes::DType;
    constexpr auto preQuant = InferPreQuant<int32_t, OutType>();
    if (cfg.quantMode == MatmulQuantMode::ScalarQuant) {
        PTO_ASSERT(false, "kirinDev0000 does not support scalar pre-quant.");
    }
    const auto preRelu = static_cast<MatmulPreRelu>(cfg.reluMode);

    // Acc mode uses non-broadcast bias from L1: biasPtr points to C tile in L1
    constexpr bool broadcastEn = hasBias && isBroadcast;
    bool initCtrl;
    if constexpr (hasBias) {
        initCtrl = false;
        constexpr uint32_t BLOCK_BYTE = 32;
        __cbuf__ void* c = (__cbuf__ void*)__cce_get_tile_ptr(cData);
        constexpr uint32_t totalBlocks = (TileRes::Rows * TileRes::Cols * sizeof(OutType)) / BLOCK_BYTE;
        constexpr int64_t repeatConfig = ((static_cast<int64_t>(0) & 0x7FFF) << 32) |
                                         ((static_cast<int64_t>(totalBlocks) & 0x7FFF) << 16) |
                                         (static_cast<int64_t>(1) & 0x7FFF);
        if constexpr (std::is_same_v<OutType, half>) {
            pto_create_cbuf_matrix((__cbuf__ half*)c, repeatConfig, (half)0);
        } else {
            pto_create_cbuf_matrix((__cbuf__ uint16_t*)c, repeatConfig, (uint16_t)0);
        }
    } else if constexpr (isAcc) {
        initCtrl = false;
    } else {
        initCtrl = true;
    }

    // M_FPC register:
    //   [7:0]    = vector relu addr (64B units, relative to mem_block1)
    //   [15:8]   = pre quant vector addr (128B units, relative to mem_block0)
    //   [45:40]  = FIXED_SHIFT_VAL = 42 for f16f16
    // fbuf mem_block0: 2KB (0x800), mem_block1: 1KB (0x800 offset)
    uint64_t mFpc = 0;
    if constexpr (preQuant != MatmulPreQuant::NoQuant) {
        mFpc |= ((cfg.preQuantTileAddr >> 7) & 0xFF) << 8;
    }
    if (preRelu == MatmulPreRelu::VectorRelu) {
        constexpr uint64_t memBlock1Base = 0x800;
        mFpc |= (((cfg.vectorReluTileAddr - memBlock1Base) >> 6) & 0xFF);
    }
    using InputType = typename TileLeft::DType;
    if constexpr (std::is_same_v<InputType, half>) {
        mFpc |= ((uint64_t)42) << 40;
    }
    set_m_fpc(mFpc);

    if (preRelu == MatmulPreRelu::ScalarRelu) {
        SetMReluAlpha(cfg.reluScalar);
    }

    const bool clipReluEn = (cfg.clipReluVal != 0.0f);
    SetMQuantPre<OutType>(cfg.clipReluVal);

    MatmulConfig matmulCfg;
    matmulCfg.initCtrl = initCtrl;
    matmulCfg.broadcastEn = broadcastEn;
    matmulCfg.preQuant = preQuant;
    matmulCfg.preRelu = preRelu;
    matmulCfg.clipReluEn = clipReluEn;
    matmulCfg.gemvCtrl = cfg.gemvCtrl;
    matmulCfg.sizeN = n;
    TMatmulCore<TileRes, TileLeft, TileRight>(cData, aData, bData, biasAddr, btAddr, m, k, n, matmulCfg.Build());
}

// --- 6. GEMV mode: M=1, gemv_ctrl=1 ---
template <typename TileRes, typename TileLeft, typename TileRight>
__tf__ PTO_INTERNAL void TMatmulGemv(
    typename TileRes::TileDType __out__ cData, typename TileLeft::TileDType __in__ aData,
    typename TileRight::TileDType __in__ bData, uint16_t k, uint16_t n, bool initCtrl)
{
    MatmulConfig cfg;
    cfg.initCtrl = initCtrl;
    cfg.gemvCtrl = true;
    cfg.sizeN = n;
    SetMFpcForFp16<typename TileRes::DType>();
    SetMQuantPre<typename TileRes::DType>(0.0f);
    TMatmulCore<TileRes, TileLeft, TileRight>(cData, aData, bData, nullptr, 0, 1, k, n, cfg.Build());
}

// --- Full-featured: bias + quant + activation + eltwise + lsb_mask + gemv + data_depend ---
template <typename TileRes, typename TileLeft, typename TileRight, typename TileBias>
__tf__ PTO_INTERNAL void TMatmulFull(
    typename TileRes::TileDType __out__ cData, typename TileLeft::TileDType __in__ aData,
    typename TileRight::TileDType __in__ bData, typename TileBias::TileDType __in__ biasData, uint64_t btAddr,
    uint16_t m, uint16_t k, uint16_t n, const MatmulConfig& cfg)
{
    __cbuf__ int32_t* biasPtr = nullptr;
    if constexpr (!std::is_same_v<TileBias, void>) {
        biasPtr = (__cbuf__ int32_t*)__cce_get_tile_ptr(biasData);
    }
    MatmulConfig fullCfg = cfg;
    fullCfg.sizeN = n;
    SetMFpcForFp16<typename TileRes::DType>();
    SetMQuantPre<typename TileRes::DType>(0.0f);
    TMatmulCore<TileRes, TileLeft, TileRight>(cData, aData, bData, biasPtr, btAddr, m, k, n, fullCfg.Build());
}

// ============================================================================
// Layer 3: Validation
// ============================================================================

PTO_INTERNAL void CheckDynamicMmad(uint16_t aMatrixRow, uint16_t aMatrixCol, uint16_t bMatrixCol)
{
    PTO_ASSERT(
        aMatrixRow >= 1 && aMatrixRow <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixRow is [1, 4095].");
    PTO_ASSERT(
        aMatrixCol >= 1 && aMatrixCol <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(
        bMatrixCol >= 1 && bMatrixCol <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");
}

template <typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void CheckMadValid()
{
    using AType = typename TileLeft::DType;
    using BType = typename TileRight::DType;
    using CType = typename TileRes::DType;
    if constexpr (std::is_same_v<CType, half>) {
        static_assert(
            std::is_same_v<AType, half> && std::is_same_v<BType, half>,
            "TMATMUL: Left Type and Right Type must be half when Acc Type is half.");
    } else if constexpr (std::is_same_v<CType, int32_t>) {
        static_assert(
            (std::is_same_v<AType, int8_t> && std::is_same_v<BType, int8_t>) ||
                (std::is_same_v<AType, int16_t> && std::is_same_v<BType, int16_t>) ||
                (std::is_same_v<AType, int16_t> && std::is_same_v<BType, int8_t>) ||
                (std::is_same_v<AType, half> && std::is_same_v<BType, half>),
            "TMATMUL: For int32 accumulation, supported input pairs are s8s8/s16s16/s16s8/f16f16.");
    } else if constexpr (std::is_same_v<CType, int16_t> || std::is_same_v<CType, int8_t>) {
        static_assert(
            std::is_same_v<AType, half> && std::is_same_v<BType, half>,
            "TMATMUL: Left Type and Right Type must be half when Acc Type is int16_t or int8_t.");
    } else {
        static_assert(sizeof(CType) == 0, "TMATMUL: Acc Type only supports int32_t/half/int16_t/int8_t.");
    }

    static_assert(
        TileLeft::Loc == TileType::Left || TileLeft::Loc == TileType::Mat,
        "TMATMUL: TileLeft TileType must be TileType::Left or TileType::Mat (L1 simulating L0A).");
    static_assert(TileRight::Loc == TileType::Right, "TMATMUL: TileRight TileType must be TileType::Right.");
    static_assert(
        TileRes::Loc == TileType::Acc || TileRes::Loc == TileType::Mat,
        "TMATMUL: TileRes TileType must be TileType::Acc or TileType::Mat (L1 simulating L0C).");
    static_assert(
        (!TileLeft::isRowMajor) && (TileRight::isRowMajor) && (!TileRes::isRowMajor) &&
            (TileLeft::SFractal == SLayout::RowMajor) && (TileRight::SFractal == SLayout::ColMajor) &&
            (TileRes::SFractal == SLayout::RowMajor),
        "TMATMUL: Non-conforming matrix fractal.");
}

// ============================================================================
// Layer 4: PTO IMPL functions (called by PTO_INST macros)
// ============================================================================

// --- 1. TMATMUL_IMPL: C = A*B ---
template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_IMPL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();

    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMmad(m, k, n);

    TMatmul<TileRes, TileLeft, TileRight>(cMatrix.data(), aMatrix.data(), bMatrix.data(), m, k, n, true);
}

// --- 1b. TMATMUL_ACC_IMPL: C += A*B ---
template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_ACC_IMPL(TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();

    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMmad(m, k, n);

    TMatmul<TileRes, TileLeft, TileRight>(cOutMatrix.data(), aMatrix.data(), bMatrix.data(), m, k, n, false);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_ACC_IMPL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    TMATMUL_ACC_IMPL<Phase>(cMatrix, cMatrix, aMatrix, bMatrix);
}

// --- 2a. TMATMUL_BIAS_IMPL: C = A*B + bias
//   If TileBias::Loc == TileType::Bias  → broadcast (bias vector in BiasTable)
//   If TileBias::Loc == TileType::Mat   → non-broadcast (bias matrix in L1, element-wise) ---
template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight, typename TileBias>
PTO_INTERNAL void TMATMUL_BIAS_IMPL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileBias& biasData)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();
    static_assert(std::is_same_v<typename TileRes::DType, typename TileBias::DType>, "No supported bias data type.");

    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMmad(m, k, n);

    if constexpr (TileBias::Loc == TileType::Bias) {
        static_assert(TileBias::Rows == 1, "Broadcast bias must be single row.");
        TMatmulBiasBroadcast<TileRes, TileLeft, TileRight>(
            cMatrix.data(), aMatrix.data(), bMatrix.data(), (uint64_t)biasData.data(), m, k, n, false);
    } else if constexpr (TileBias::Loc == TileType::Mat) {
        static_assert(
            (TileBias::Rows == 1) && (TileBias::isRowMajor),
            "Non-broadcast bias must be Mat tile (L1) with single row, row major.");
        TMatmulBiasNonBroadcast<TileRes, TileLeft, TileRight, TileBias>(
            cMatrix.data(), aMatrix.data(), bMatrix.data(), biasData.data(), m, k, n, false);
    } else {
        static_assert(sizeof(typename TileBias::DType) == 0, "Bias tile must be TileType::Bias or TileType::Mat.");
    }
}

// --- 6. TGEMV_IMPL: C = A*B (M=1, gemv_ctrl=1) ---
template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TGEMV_IMPL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();
    uint16_t k = bMatrix.GetValidRow();
    uint16_t n = bMatrix.GetValidCol();
    PTO_ASSERT(k >= 1 && k <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(n >= 1 && n <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");
    TMatmulGemv<TileRes, TileLeft, TileRight>(cMatrix.data(), aMatrix.data(), bMatrix.data(), k, n, true);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TGEMV_ACC_IMPL(TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();
    uint16_t k = bMatrix.GetValidRow();
    uint16_t n = bMatrix.GetValidCol();
    PTO_ASSERT(k >= 1 && k <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(n >= 1 && n <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");
    TMatmulGemv<TileRes, TileLeft, TileRight>(cOutMatrix.data(), aMatrix.data(), bMatrix.data(), k, n, false);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight, typename TileBias>
PTO_INTERNAL void TGEMV_BIAS_IMPL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileBias& biasData)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();
    static_assert(std::is_same_v<typename TileRes::DType, typename TileBias::DType>, "No supported bias data type.");
    static_assert(
        (TileBias::Loc == TileType::Mat) && (TileBias::Rows == 1), "TileBias must be Mat tile with single row.");

    uint16_t k = bMatrix.GetValidRow();
    uint16_t n = bMatrix.GetValidCol();
    PTO_ASSERT(k >= 1 && k <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(n >= 1 && n <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");

    MatmulConfig cfg;
    cfg.initCtrl = false;
    cfg.broadcastEn = false;
    cfg.gemvCtrl = true;
    cfg.sizeN = n;
    TMatmulFull<TileRes, TileLeft, TileRight, TileBias>(
        cMatrix.data(), aMatrix.data(), bMatrix.data(), biasData.data(), 0, 1, k, n, cfg);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight,
    typename TileBias = void, bool isClear = false>
PTO_INTERNAL void TMATMUL_MACRO_ACC_IMPL(
    TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileBias* biasData, const MatmulMacroConfig& cfg)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();

    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMmad(m, k, n);

    constexpr bool hasBias = !std::is_same_v<TileBias, void>;

    __cbuf__ int32_t* biasPtr = nullptr;
    uint64_t btAddr = 0;
    if constexpr (hasBias) {
        if constexpr (TileBias::Loc == TileType::Bias) {
            static_assert(TileBias::Rows == 1, "Broadcast bias must be single row.");
            btAddr = (uint64_t)biasData->data();
        } else if constexpr (TileBias::Loc == TileType::Mat) {
            static_assert(
                (TileBias::Rows == TileRes::Rows) && (TileBias::Cols == TileRes::Cols),
                "Non-broadcast bias must have same shape as TileRes.");
            biasPtr = (__cbuf__ int32_t*)__cce_get_tile_ptr(biasData->data());
        } else {
            static_assert(sizeof(typename TileBias::DType) == 0, "Bias tile must be TileType::Bias or TileType::Mat.");
        }
    } else if constexpr (!isClear) {
        // ACC mode: use C tile in L1 as bias source for accumulation (non-broadcast)
        biasPtr = (__cbuf__ int32_t*)__cce_get_tile_ptr(cMatrix.data());
    }

    if constexpr (hasBias) {
        if constexpr (TileBias::Loc == TileType::Bias) {
            TMatmulMacroAcc<TileRes, TileLeft, TileRight, true, true, false>(
                cMatrix.data(), aMatrix.data(), bMatrix.data(), biasPtr, btAddr, m, k, n, cfg);
        } else {
            TMatmulMacroAcc<TileRes, TileLeft, TileRight, true, false, false>(
                cMatrix.data(), aMatrix.data(), bMatrix.data(), biasPtr, btAddr, m, k, n, cfg);
        }
    } else {
        TMatmulMacroAcc<TileRes, TileLeft, TileRight, false, false, isClear>(
            cMatrix.data(), aMatrix.data(), bMatrix.data(), biasPtr, btAddr, m, k, n, cfg);
    }
}

// --- TMATMUL_MACRO_IMPL: final iteration or no split-K, with pre-stage quant + bias ---
//   Output: F16/S8/S16 (with pre-stage quant + optional relu)
//   initCtrl=false (bias requires init_ctrl=0)
//   isAcc: false = zero C before matmul; true = keep existing C (split-K final)
template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight, typename TileBias,
    bool isAcc = false>
PTO_INTERNAL void TMATMUL_MACRO_IMPL(
    TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileBias& biasData, const MatmulMacroConfig& cfg)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();

    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMmad(m, k, n);

    constexpr bool isBroadcast = (TileBias::Loc == TileType::Bias);

    __cbuf__ int32_t* biasPtr = nullptr;
    uint64_t btAddr = 0;
    if constexpr (TileBias::Loc == TileType::Bias) {
        static_assert(TileBias::Rows == 1, "Broadcast bias must be single row.");
        btAddr = (uint64_t)biasData.data();
    } else if constexpr (TileBias::Loc == TileType::Mat) {
        static_assert(
            (TileBias::Rows == TileRes::Rows) && (TileBias::Cols == TileRes::Cols),
            "Non-broadcast bias must have same shape as TileRes.");
        biasPtr = (__cbuf__ int32_t*)__cce_get_tile_ptr(biasData.data());
    } else {
        static_assert(sizeof(typename TileBias::DType) == 0, "Bias tile must be TileType::Bias or TileType::Mat.");
    }

    TMatmulMacro<TileRes, TileLeft, TileRight, true, isBroadcast, isAcc>(
        cMatrix.data(), aMatrix.data(), bMatrix.data(), biasPtr, btAddr, m, k, n, cfg);
}

// --- TMATMUL_MACRO_IMPL: no bias ---
//   isAcc=false: initCtrl=true (hardware zero C), no bias
//   isAcc=true: init_ctrl=0, broadcast_en=1, btAddr=C_tile_L1 (accumulate via bias-broadcast)
template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight, bool isAcc = false>
PTO_INTERNAL void TMATMUL_MACRO_IMPL(
    TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, const MatmulMacroConfig& cfg)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();

    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMmad(m, k, n);

    __cbuf__ int32_t* biasPtr = nullptr;
    if constexpr (isAcc) {
        // ACC mode: use C tile in L1 as bias source for accumulation (non-broadcast)
        biasPtr = (__cbuf__ int32_t*)__cce_get_tile_ptr(cMatrix.data());
    }

    TMatmulMacro<TileRes, TileLeft, TileRight, false, false, isAcc>(
        cMatrix.data(), aMatrix.data(), bMatrix.data(), biasPtr, 0, m, k, n, cfg);
}

// --- MX not supported ---
template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale>
PTO_INTERNAL void TMATMUL_MX_IMPL(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix)
{
    static_assert(sizeof(TileRes::DType) == 0, "no support instruction.");
}

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale>
PTO_INTERNAL void TMATMUL_MX_IMPL(
    TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix,
    TileRightScale& bScaleMatrix)
{
    static_assert(sizeof(TileRes::DType) == 0, "no support instruction.");
}

template <
    typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
    typename TileBias>
PTO_INTERNAL void TMATMUL_MX_IMPL(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix,
    TileBias& biasData)
{
    static_assert(sizeof(TileRes::DType) == 0, "no support instruction.");
}

// --- TF32 not supported ---
template <bool isEnable, RoundMode tf32TransMode = RoundMode::CAST_ROUND>
PTO_INTERNAL void TSETTF32MODE_IMPL()
{
    static_assert(!isEnable, "Fix: KirinDev0000 does not support setting the TF32 mode to enabled.");
    set_ctrl(sbitset0(get_ctrl(), TF32_MODE_BIT));
}
} // namespace pto
#endif
