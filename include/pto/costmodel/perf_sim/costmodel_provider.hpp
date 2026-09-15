/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_PERF_SIM_COSTMODEL_PROVIDER_HPP
#define PTO_PERF_SIM_COSTMODEL_PROVIDER_HPP

#include "config.hpp"
#include "latency.hpp"
#include "recorder.hpp"

#include <pto/costmodel/lightweight_costmodel.hpp>

#include <string>

namespace pto::perf_sim {

// ── Magic number constants ──
static constexpr uint64_t kPercentMultiplier = 100;
static constexpr uint64_t kMatrixLatencyBase = 4;
static constexpr uint64_t kElemsPerMatrixUnit = 16;
static constexpr uint64_t kGMLatencyBase = 3;
static constexpr uint64_t kGMCycleMultiplier = 2;
static constexpr uint64_t kGMElemsPerUnit = 64;
static constexpr uint64_t kMTE1LatencyBase = 1;
static constexpr uint64_t kMTE1ElemsPerUnit = 64;
static constexpr uint64_t kDefaultLatencyBase = 2;
static constexpr uint64_t kDefaultElemsPerUnit = 32;

// ── Runtime context passed to costmodel ──

struct CostModelRuntimeCtx {
    uint32_t block_dim = 1;
    uint32_t npu_arch = 2201;
};

inline CostModelRuntimeCtx& GetCostModelCtx()
{
    static thread_local CostModelRuntimeCtx ctx;
    return ctx;
}

// ── String → lightweight costmodel mappings ──

// clang-format off
#define PTO_PERF_SIM_OPCODE_LIST                                                                                      \
    X(TADD)                                                                                                           \
    X(TSUB)                                                                                                           \
    X(TMUL)                                                                                                           \
    X(TDIV)                                                                                                           \
    X(TRECIP)                                                                                                         \
    X(TADDS)                                                                                                          \
    X(TSUBS)                                                                                                          \
    X(TMULS)                                                                                                          \
    X(TDIVS) X(TMINS) X(TMAXS) X(TABS) X(TNEG) X(TEXP) X(TSQRT) X(TRSQRT) X(TLOG) X(TRELU) X(TLRELU) X(TNOT)          \
        X(TROWSUM) X(TROWMAX) X(TROWMIN) X(TROWPROD) X(TCOLSUM) X(TCOLMAX) X(TCOLMIN) X(TCOLPROD) X(TMATMUL) X(TGEMV) \
            X(TCVT) X(TMOV) X(TLOAD) X(TSTORE) X(TTRANS) X(TPREFETCH) X(TSORT32) X(TMRGSORT) X(TSEL) X(TSCATTER)      \
                X(TEXTRACT) X(TINSERT) X(TROWEXPAND) X(TCOLEXPAND) X(TLOADCONV)
// clang-format on

inline bool TryMapOpcode(const std::string& opcode, ::pto::mocker::lightweight::PtoOpcode& out)
{
#define X(name)                                            \
    if (opcode == #name) {                                 \
        out = ::pto::mocker::lightweight::PtoOpcode::name; \
        return true;                                       \
    }
    PTO_PERF_SIM_OPCODE_LIST
#undef X
    return false;
}

// ── String → lightweight::DType mapping (X-macro, paired str↔enum) ──

#define PTO_PERF_SIM_DTYPE_LIST \
    X("fp32", Float)            \
    X("fp16", Half)             \
    X("int8", Int8)             \
    X("int16", Int16)           \
    X("int32", Int32)           \
    X("uint8", Uint8)           \
    X("uint16", Uint16)         \
    X("uint32", Uint32)         \
    X("bf16", BFloat16)         \
    X("fp8_e4m3", Float8E4M3)   \
    X("fp8_e5m2", Float8E5M2)   \
    X("hif8", HFloat8)          \
    X("fp4_e1m2", Float4E1M2)   \
    X("fp4_e2m1", Float4E2M1)

inline bool TryMapDType(const std::string& dtype, ::pto::mocker::lightweight::DType& out)
{
#define X(str, enum_val)                                     \
    if (dtype == str) {                                     \
        out = ::pto::mocker::lightweight::DType::enum_val;  \
        return true;                                        \
    }
    PTO_PERF_SIM_DTYPE_LIST
#undef X
    return false;
}

#if !defined(__NPU_ARCH__) || (__NPU_ARCH__ == 2201)
// ── LightweightFormula backend ──

inline uint64_t EstimateLightweightCycles(const std::string& opcode, int rows, int cols, const std::string& dtype)
{
    using namespace ::pto::mocker::lightweight;

    PtoOpcode pto_op;
    if (!TryMapOpcode(opcode, pto_op))
        return 0;

    CostModelInput input{};
    input.op = pto_op;
    if (!TryMapDType(dtype, input.dtype))
        return 0;
    input.rows = rows;
    input.cols = cols;
    input.data_size = static_cast<int64_t>(rows) * cols;

    if (pto_op == PtoOpcode::TMATMUL || pto_op == PtoOpcode::TGEMV) {
        input.k = cols;
    }

    CostModelResult result = EstimateCycles(input);
    if (result.cycles > 0.0)
        return static_cast<uint64_t>(result.cycles);
    return 0;
}

#endif

struct A5VfTileOpInput {
    std::string opcode;
    std::string src_dtype;
    std::string dst_dtype;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t valid_rows = 0;
    int64_t valid_cols = 0;
    ::pto::VFImplKind vf_impl_kind = ::pto::VFImplKind::VFIMPL_DEFAULT;
    ::pto::RoundMode round_mode = ::pto::RoundMode::CAST_NONE;
    ::pto::SaturationMode saturation_mode = ::pto::SaturationMode::ON;
    std::string_view op_params{};
    ::pto::mocker::lightweight::A5VfShapePathHint shape_path_hint =
        ::pto::mocker::lightweight::A5VfShapePathHint::Infer;
};

inline bool TryEstimateA5VfTileOpCycles(const A5VfTileOpInput& tile_op, uint64_t& cycles)
{
    using namespace ::pto::mocker::lightweight;

    PtoOpcode op;
    DType src_dtype;
    DType dst_dtype;
    if (!TryMapOpcode(tile_op.opcode, op) || !TryMapDType(tile_op.src_dtype, src_dtype)) {
        return false;
    }
    if (!tile_op.dst_dtype.empty() && !TryMapDType(tile_op.dst_dtype, dst_dtype)) {
        return false;
    }

    CostModelInput input{};
    input.op = op;
    input.dtype = src_dtype;
    input.rows = tile_op.rows;
    input.cols = tile_op.cols;
    input.arch = CostModelArch::A5;
    input.valid_rows = tile_op.valid_rows;
    input.valid_cols = tile_op.valid_cols;
    input.vf_impl_kind = tile_op.vf_impl_kind;
    input.round_mode = tile_op.round_mode;
    input.saturation_mode = tile_op.saturation_mode;
    input.a5_op_params = tile_op.op_params;
    input.a5_shape_path_hint = tile_op.shape_path_hint;
    if (!tile_op.dst_dtype.empty()) {
        input.dst_dtype = dst_dtype;
    }
    return a5::TryEstimateA5VfCycles(input, cycles);
}

// ── Fallback for unsupported instructions ──

inline uint64_t FallbackCycles(const std::string& opcode, int rows, int cols)
{
    uint64_t elems = static_cast<uint64_t>(rows) * cols;
    PipeStage stage = StaticPipeStageLookup(opcode);
    if (stage == PipeStage::Scalar)
        return 1;
    if (stage == PipeStage::Matrix)
        return kMatrixLatencyBase + elems / kElemsPerMatrixUnit;
    if (IsGMAccessPipe(stage))
        return kGMLatencyBase + elems * kGMCycleMultiplier / kGMElemsPerUnit;
    if (stage == PipeStage::MTE1)
        return kMTE1LatencyBase + elems / kMTE1ElemsPerUnit;
    return kDefaultLatencyBase + elems / kDefaultElemsPerUnit;
}

// ── Unified estimation entry (used when CCE mock trace is unavailable) ──

inline uint64_t EstimateInstrCycles(const std::string& opcode, int rows, int cols, const std::string& dtype)
{
#if !defined(__NPU_ARCH__) || (__NPU_ARCH__ == 2201)
    if (opcode == "TDIVS" && (dtype == "int16" || dtype == "int32")) {
        return 4;
    }
    uint64_t cycles = EstimateLightweightCycles(opcode, rows, cols, dtype);
    return cycles > 0 ? cycles : FallbackCycles(opcode, rows, cols);
#else
    (void)dtype;
    return FallbackCycles(opcode, rows, cols);
#endif
}

} // namespace pto::perf_sim

#endif
