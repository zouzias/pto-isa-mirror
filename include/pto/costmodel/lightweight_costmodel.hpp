/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_MOCKER_LIGHTWEIGHT_COSTMODEL_HPP
#define PTO_MOCKER_LIGHTWEIGHT_COSTMODEL_HPP

#include <array>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <type_traits>

#include <pto/common/type.hpp>
#include <pto/costmodel/a2a3/formula_costmodel/formula_backend_transfer.hpp>
#if !defined(__NPU_ARCH__) || (__NPU_ARCH__ == 2201)
#include <pto/costmodel/a2a3/formula_costmodel/formula_backend_compute.hpp>
#endif
#include <pto/costmodel/arch_config.hpp>
#include <pto/costmodel/a5/vf_costmodel.hpp>

namespace pto::mocker::lightweight {

enum class PtoOpcode {
    TADD,
    TSUB,
    TMUL,
    TDIV,
    TRECIP,
    TADDS,
    TSUBS,
    TMULS,
    TDIVS,
    TMINS,
    TMAXS,
    TABS,
    TNEG,
    TEXP,
    TSQRT,
    TRSQRT,
    TLOG,
    TRELU,
    TLRELU,
    TNOT,
    TROWSUM,
    TROWMAX,
    TROWMIN,
    TROWPROD,
    TCOLSUM,
    TCOLMAX,
    TCOLMIN,
    TCOLPROD,
    TMATMUL,
    TGEMV,
    TCVT,
    TMOV,
    TLOAD,
    TSTORE,
    TTRANS,
    TSORT32,
    TMRGSORT,
    TSEL,
    TSCATTER,
    TEXTRACT,
    TINSERT,
    TROWEXPAND,
    TCOLEXPAND,
    TLOADCONV,
    TPREFETCH,
};

enum class DType : uint8_t {
    Float,
    Half,
    Int8,
    Int16,
    Int32,
    Uint8,
    Uint16,
    Uint32,
    BFloat16,
    Float8E4M3,
    Float8E5M2,
    HFloat8,
    Float4E1M2,
    Float4E2M1,
};

using MemLayout = ::pto::Layout;
using RoundMode = ::pto::RoundMode;
using SaturationMode = ::pto::SaturationMode;
using TransferTileType = fit::TransferTileType;
using VFImplKind = ::pto::VFImplKind;

enum class CostModelArch : uint8_t {
    A2A3,
    A5,
};

// The A5 wrapper can determine this from the Tile types without executing the
// NPU implementation.  Direct users of CostModelInput may leave it as Infer.
enum class A5VfShapePathHint : uint8_t {
    Infer,
    Path1D,
    Path2D,
};

struct CostModelInput {
    PtoOpcode op;
    DType dtype;
    int64_t rows;
    int64_t cols;
    CostModelArch arch = CostModelArch::A2A3;

    DType dtype2 = DType::Float;
    int64_t k = 0;

    DType dst_dtype = DType::Float;
    RoundMode round_mode = RoundMode::CAST_NONE;

    MemLayout layout = MemLayout::ND;

    int64_t dst_rows = 0;
    int64_t dst_cols = 0;

    int64_t block_len = 0;
    int64_t src_count = 1;

    int64_t channels = 0;
    int64_t height = 0;
    int64_t width = 0;

    TransferTileType tile_type = TransferTileType::Unknown;
    int64_t data_size = 0;

    int64_t valid_rows = 0;
    int64_t valid_cols = 0;
    VFImplKind vf_impl_kind = VFImplKind::VFIMPL_DEFAULT;
    SaturationMode saturation_mode = SaturationMode::ON;
    std::string_view a5_op_params{};
    A5VfShapePathHint a5_shape_path_hint = A5VfShapePathHint::Infer;
};

struct CostModelResult {
    double cycles = 0.0;
    long double latency_us = 0.0L;
};

struct PredictRuntimeConfig {
    long double frequency_mhz = 0.0L;
    evaluator::BandwidthTable bandwidth_bytes_per_us{};
};

namespace a5 {
inline bool TryEstimateA5VfCycles(const CostModelInput& input, uint64_t& cycles);
}

inline constexpr const char* DTypeToString(DType dtype)
{
    switch (dtype) {
        case DType::Float:
            return "Float";
        case DType::Half:
            return "Half";
        case DType::Float8E4M3:
            return "Float8E4M3";
        case DType::Float8E5M2:
            return "Float8E5M2";
        case DType::HFloat8:
            return "HFloat8";
        case DType::Float4E1M2:
            return "Float4E1M2";
        case DType::Float4E2M1:
            return "Float4E2M1";
        case DType::Int8:
            return "Int8";
        case DType::Int16:
            return "Int16";
        case DType::Int32:
            return "Int32";
        case DType::Uint8:
            return "Uint8";
        case DType::Uint16:
            return "Uint16";
        case DType::Uint32:
            return "Uint32";
        case DType::BFloat16:
            return "BFloat16";
        default:
            return "Unknown";
    }
}

inline constexpr const char* PtoOpcodeToString(PtoOpcode op)
{
    constexpr std::array<const char*, 45> names = {
        "TADD",     "TSUB",    "TMUL",     "TDIV",     "TRECIP",  "TADDS",      "TSUBS",      "TMULS",     "TDIVS",
        "TMINS",    "TMAXS",   "TABS",     "TNEG",     "TEXP",    "TSQRT",      "TRSQRT",     "TLOG",      "TRELU",
        "TLRELU",   "TNOT",    "TROWSUM",  "TROWMAX",  "TROWMIN", "TROWPROD",   "TCOLSUM",    "TCOLMAX",   "TCOLMIN",
        "TCOLPROD", "TMATMUL", "TGEMV",    "TCVT",     "TMOV",    "TLOAD",      "TSTORE",     "TTRANS",    "TSORT32",
        "TMRGSORT", "TSEL",    "TSCATTER", "TEXTRACT", "TINSERT", "TROWEXPAND", "TCOLEXPAND", "TLOADCONV", "TPREFETCH",
    };
    auto idx = static_cast<std::underlying_type_t<PtoOpcode>>(op);
    if (idx >= 0 && static_cast<size_t>(idx) < names.size()) {
        return names[static_cast<size_t>(idx)];
    }
    return "Unknown";
}

inline constexpr const char* TransferTileTypeToString(TransferTileType tile_type)
{
    switch (tile_type) {
        case TransferTileType::Unknown:
            return "Unknown";
        case TransferTileType::VecTile:
            return "VecTile";
        case TransferTileType::MatTile:
            return "MatTile";
        case TransferTileType::AccTile:
            return "AccTile";
        case TransferTileType::LeftTile:
            return "LeftTile";
        case TransferTileType::RightTile:
            return "RightTile";
        case TransferTileType::BiasTile:
            return "BiasTile";
        case TransferTileType::ScalingTile:
            return "ScalingTile";
        default:
            return "Unknown";
    }
}

inline bool WarnAndFallbackToZero(const CostModelInput& input, CostModelResult& result, std::string_view reason)
{
    result.cycles = 0.0;
    result.latency_us = 0.0L;
    std::cerr << "[WARN] lightweight::EstimateCycles fallback to 0 cycles: " << reason
              << ", op=" << PtoOpcodeToString(input.op) << ", dtype=" << DTypeToString(input.dtype)
              << ", rows=" << input.rows << ", cols=" << input.cols
              << ", tile_type=" << TransferTileTypeToString(input.tile_type) << ", data_size=" << input.data_size
              << '\n';
    return false;
}

inline PredictRuntimeConfig GetDefaultPredictRuntimeConfig()
{
    const auto& default_arch = evaluator::GetDefaultArchConfig();
    return {
        default_arch.frequency_hz / evaluator::kMicrosPerSecond,
        default_arch.bandwidth,
    };
}

#if !defined(__NPU_ARCH__) || (__NPU_ARCH__ == 2201)
template <typename FpType>
inline bool TryEstimateSupportedCycles(PtoOpcode op, uint64_t rows, uint64_t cols, uint64_t& cycles)
{
    switch (op) {
        case PtoOpcode::TSUB:
            return fit::TryEstimateFormulaCycles<FpType>("TSUB", rows, cols, cycles);
        case PtoOpcode::TMUL:
            return fit::TryEstimateFormulaCycles<FpType>("TMUL", rows, cols, cycles);
        case PtoOpcode::TADDS:
            return fit::TryEstimateFormulaCycles<FpType>("TADDS", rows, cols, cycles);
        case PtoOpcode::TDIVS:
            return fit::TryEstimateFormulaCycles<FpType>("TDIVS", rows, cols, cycles);
        case PtoOpcode::TMULS:
            return fit::TryEstimateFormulaCycles<FpType>("TMULS", rows, cols, cycles);
        case PtoOpcode::TMINS:
            return fit::TryEstimateFormulaCycles<FpType>("TMINS", rows, cols, cycles);
        case PtoOpcode::TROWSUM:
            return fit::TryEstimateFormulaCycles<FpType>("TROWSUM", rows, cols, cycles);
        case PtoOpcode::TROWMAX:
            return fit::TryEstimateFormulaCycles<FpType>("TROWMAX", rows, cols, cycles);
        case PtoOpcode::TCOLSUM:
            return fit::TryEstimateFormulaCycles<FpType>("TCOLSUM", rows, cols, cycles);
        case PtoOpcode::TCOLMAX:
            return fit::TryEstimateFormulaCycles<FpType>("TCOLMAX", rows, cols, cycles);
        case PtoOpcode::TROWEXPAND:
            return fit::TryEstimateFormulaCycles<FpType>("TROWEXPAND", rows, cols, cycles);
        case PtoOpcode::TEXP:
            return fit::TryEstimateFormulaCyclesAnyDType<FpType>("TEXP", rows, cols, cycles);
        case PtoOpcode::TSQRT:
            return fit::TryEstimateFormulaCyclesAnyDType<FpType>("TSQRT", rows, cols, cycles);
        default:
            return false;
    }
}

inline bool TryEstimateMatmulCycles(const CostModelInput& input, uint64_t& cycles)
{
    if (input.rows <= 0 || input.k <= 0 || input.cols <= 0) {
        return false;
    }
    const uint64_t m = static_cast<uint64_t>(input.rows);
    const uint64_t k = static_cast<uint64_t>(input.k);
    const uint64_t n = static_cast<uint64_t>(input.cols);

    switch (input.dtype) {
        case DType::Float:
            return fit::TryEstimateMadCycles<float, float, float>(m, k, n, cycles);
        case DType::Half:
            return fit::TryEstimateMadCycles<half, half, half>(m, k, n, cycles);
        default:
            return false;
    }
}
#endif

inline bool TryGetDTypeSizeBytes(DType dtype, uint64_t& bytes)
{
    constexpr uint64_t kByteOne = 1;
    constexpr uint64_t kByteTwo = 2;
    constexpr uint64_t kByteFour = 4;
    switch (dtype) {
        case DType::Int8:
        case DType::Uint8:
            bytes = kByteOne;
            return true;
        case DType::Half:
        case DType::Int16:
        case DType::Uint16:
        case DType::BFloat16:
            bytes = kByteTwo;
            return true;
        case DType::Float:
        case DType::Int32:
        case DType::Uint32:
            bytes = kByteFour;
            return true;
        default:
            return false;
    }
}

inline bool TryEstimateTransferLatency(
    const CostModelInput& input, const PredictRuntimeConfig& predict_config, CostModelResult& result)
{
    if (input.data_size <= 0) {
        return false;
    }
    uint64_t dtype_bytes = 0;
    if (!TryGetDTypeSizeBytes(input.dtype, dtype_bytes)) {
        return false;
    }
    const uint64_t elements = static_cast<uint64_t>(input.data_size);
    const uint64_t bytes = elements * dtype_bytes;

    fit::TransferOp transfer_op = fit::TransferOp::TLoad;
    switch (input.op) {
        case PtoOpcode::TLOAD:
            transfer_op = fit::TransferOp::TLoad;
            break;
        case PtoOpcode::TSTORE:
            transfer_op = fit::TransferOp::TStore;
            break;
        case PtoOpcode::TMOV:
            transfer_op = fit::TransferOp::TMov;
            break;
        default:
            return false;
    }

    long double latency_us = 0.0L;
    if (!fit::TryEstimateTransferLatencyUs(
            transfer_op, input.tile_type, bytes, predict_config.bandwidth_bytes_per_us, latency_us)) {
        return false;
    }
    result.latency_us = latency_us;
    result.cycles = static_cast<double>(latency_us * predict_config.frequency_mhz);
    return true;
}

inline bool TryEstimateA5Cycles(
    const CostModelInput& input, const PredictRuntimeConfig& predict_config, CostModelResult& result)
{
    uint64_t cycles = 0;
    if (!a5::TryEstimateA5VfCycles(input, cycles)) {
        result = {};
        std::cerr << "[costmodel][A5] Unsupported formula key: op=" << PtoOpcodeToString(input.op)
                  << ", dtype=" << DTypeToString(input.dtype) << ", rows=" << input.rows
                  << ", cols=" << input.cols << '\n';
        return false;
    }
    result.cycles = static_cast<double>(cycles);
    result.latency_us = evaluator::CyclesToUs(cycles, predict_config.frequency_mhz);
    return true;
}

inline bool TryEstimateTransferCycles(
    const CostModelInput& input, const PredictRuntimeConfig& predict_config, CostModelResult& result)
{
    if (TryEstimateTransferLatency(input, predict_config, result)) {
        return true;
    }
    return WarnAndFallbackToZero(input, result, "unsupported transfer op/tile_type/data_size/dtype");
}

#if !defined(__NPU_ARCH__) || (__NPU_ARCH__ == 2201)
inline bool TryEstimateMatmulCyclesWithResult(
    const CostModelInput& input, const PredictRuntimeConfig& predict_config, CostModelResult& result)
{
    uint64_t cycles = 0;
    if (!TryEstimateMatmulCycles(input, cycles)) {
        return WarnAndFallbackToZero(input, result, "unsupported matmul shape/dtype combination");
    }
    result.cycles = static_cast<double>(cycles);
    result.latency_us = evaluator::CyclesToUs(cycles, predict_config.frequency_mhz);
    return true;
}

inline bool TryEstimateElementwiseCycles(
    const CostModelInput& input, const PredictRuntimeConfig& predict_config, CostModelResult& result)
{
    if (input.rows <= 0 || input.cols <= 0) {
        return WarnAndFallbackToZero(input, result, "unsupported rows/cols");
    }

    const uint64_t rows = static_cast<uint64_t>(input.rows);
    const uint64_t cols = static_cast<uint64_t>(input.cols);
    uint64_t cycles = 0;
    bool estimated = false;
    switch (input.dtype) {
        case DType::Float:
            estimated = TryEstimateSupportedCycles<float>(input.op, rows, cols, cycles);
            break;
        case DType::Half:
            estimated = TryEstimateSupportedCycles<half>(input.op, rows, cols, cycles);
            break;
        default:
            return WarnAndFallbackToZero(input, result, "unsupported dtype");
    }
    if (!estimated) {
        return WarnAndFallbackToZero(input, result, "unsupported op/cols parameter combination");
    }
    (void)predict_config.bandwidth_bytes_per_us;
    result.cycles = static_cast<double>(cycles);
    result.latency_us = evaluator::CyclesToUs(cycles, predict_config.frequency_mhz);
    return true;
}
#endif

inline bool EstimateCycles(
    const CostModelInput& input, const PredictRuntimeConfig& predict_config, CostModelResult& result)
{
    if (input.arch == CostModelArch::A5) {
        return TryEstimateA5Cycles(input, predict_config, result);
    }
#if !defined(__NPU_ARCH__) || (__NPU_ARCH__ == 2201)
    if (input.op == PtoOpcode::TLOAD || input.op == PtoOpcode::TSTORE || input.op == PtoOpcode::TMOV) {
        return TryEstimateTransferCycles(input, predict_config, result);
    }
    if (input.op == PtoOpcode::TMATMUL) {
        return TryEstimateMatmulCyclesWithResult(input, predict_config, result);
    }
    return TryEstimateElementwiseCycles(input, predict_config, result);
#else
    return WarnAndFallbackToZero(input, result, "A2/A3 formula backend is unavailable in an A5 build");
#endif
}

inline bool EstimateCycles(const CostModelInput& input, CostModelResult& result)
{
    const PredictRuntimeConfig default_config = GetDefaultPredictRuntimeConfig();
    return EstimateCycles(input, default_config, result);
}

inline CostModelResult EstimateCycles(const CostModelInput& input)
{
    CostModelResult result{};
    (void)EstimateCycles(input, result);
    return result;
}

} // namespace pto::mocker::lightweight

// Adapter for existing numeric CostModelInput callers; TileOp wrappers bypass this path.
namespace pto::mocker::lightweight::a5 {

inline std::string_view DTypeToKey(DType dtype)
{
    switch (dtype) {
        case DType::Float:
            return "fp32";
        case DType::Half:
            return "fp16";
        case DType::Float8E4M3:
            return "fp8_e4m3";
        case DType::Float8E5M2:
            return "fp8_e5m2";
        case DType::HFloat8:
            return "hif8";
        case DType::Float4E1M2:
            return "fp4_e1m2";
        case DType::Float4E2M1:
            return "fp4_e2m1";
        case DType::Int8:
            return "int8";
        case DType::Int16:
            return "int16";
        case DType::Int32:
            return "int32";
        case DType::Uint8:
            return "uint8";
        case DType::Uint16:
            return "uint16";
        case DType::Uint32:
            return "uint32";
        case DType::BFloat16:
            return "bf16";
        default:
            return "";
    }
}


// Compatibility for direct CostModelInput callers (including fit datasets).
// Real TileOp wrappers bypass this adapter and provide compile-time Tile metadata.
inline bool TryEstimateA5VfCycles(const CostModelInput& input, uint64_t& cycles)
{
    const std::string_view opcode = PtoOpcodeToString(input.op);
    if (opcode.empty()) {
        return false;
    }
    A5TileOpMetadata metadata;
    metadata.first_tile_dtype = DTypeToKey(input.op == PtoOpcode::TCVT ? input.dst_dtype : input.dtype);
    metadata.second_tile_dtype = DTypeToKey(input.dtype);
    metadata.rows = input.rows;
    metadata.cols = input.cols;
    metadata.valid_rows = input.valid_rows > 0 ? input.valid_rows : input.rows;
    metadata.valid_cols = input.valid_cols > 0 ? input.valid_cols : input.cols;
    metadata.vf_impl_kind = input.vf_impl_kind;
    metadata.round_mode = input.round_mode;
    metadata.saturation_mode = input.saturation_mode;
    metadata.tile_count = (input.op == PtoOpcode::TADD || input.op == PtoOpcode::TSUB ||
                           input.op == PtoOpcode::TMUL || input.op == PtoOpcode::TSEL) ? 3 : 2;
    bool contiguous = input.a5_shape_path_hint == A5VfShapePathHint::Path1D;
    if (input.a5_shape_path_hint == A5VfShapePathHint::Infer) {
        // Preserve the fit-data convention when Tile template metadata is absent.
        const bool scalar = input.op == PtoOpcode::TDIVS || input.op == PtoOpcode::TMINS ||
                            input.op == PtoOpcode::TNEG;
        contiguous = metadata.valid_cols == input.cols || (!scalar && input.rows == 1);
    }
    if (contiguous) {
        metadata.static_full_cols_mask = (1U << metadata.tile_count) - 1;
    }
    A5VfExecutionPath path;
    return SelectA5TileOpPath(opcode, A5TileOpOptions{input.a5_op_params}, metadata, path) &&
           fit::TryEstimateSelectedVfCycles(path, cycles);
}

} // namespace pto::mocker::lightweight::a5

#endif
