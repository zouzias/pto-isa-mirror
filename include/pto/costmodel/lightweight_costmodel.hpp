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
#include <pto/costmodel/a2a3/formula_costmodel/formula_backend_utils.hpp>
#include <pto/costmodel/arch_config.hpp>

namespace pto::mocker::lightweight {

enum class PtoOpcode {
    TADD,
    TSUB,
    TMUL,
    TDIV,
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
};

using MemLayout = ::pto::Layout;
using RoundMode = ::pto::RoundMode;

struct CostModelInput {
    PtoOpcode op;
    DType dtype;
    int64_t rows;
    int64_t cols;

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
};

struct CostModelResult {
    double cycles = 0.0;
    long double latency_us = 0.0L;
};

struct PredictRuntimeConfig {
    long double frequency_mhz = 0.0L;
    evaluator::BandwidthTable bandwidth_bytes_per_us{};
};

inline constexpr const char *DTypeToString(DType dtype)
{
    switch (dtype) {
        case DType::Float:
            return "Float";
        case DType::Half:
            return "Half";
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

inline bool WarnAndFallbackToZero(const CostModelInput &input, CostModelResult &result, std::string_view reason)
{
    result.cycles = 0.0;
    result.latency_us = 0.0L;
    std::cerr << "[WARN] lightweight::EstimateCycles fallback to 0 cycles: " << reason << ", op="
              << static_cast<int>(input.op) << ", dtype=" << DTypeToString(input.dtype) << ", rows=" << input.rows
              << ", cols=" << input.cols << '\n';
    return true;
}

inline PredictRuntimeConfig GetDefaultPredictRuntimeConfig()
{
    const auto &default_arch = evaluator::GetDefaultArchConfig();
    return {
        default_arch.frequency_hz / evaluator::kMicrosPerSecond,
        default_arch.bandwidth,
    };
}

inline bool TryParsePtoOpcode(std::string_view instrName, PtoOpcode &opcode)
{
    using Entry = std::pair<std::string_view, PtoOpcode>;
    static constexpr std::array<Entry, 43> kOpcodeMap = {{
        {"TADD", PtoOpcode::TADD},         {"TSUB", PtoOpcode::TSUB},         {"TMUL", PtoOpcode::TMUL},
        {"TDIV", PtoOpcode::TDIV},         {"TADDS", PtoOpcode::TADDS},       {"TSUBS", PtoOpcode::TSUBS},
        {"TMULS", PtoOpcode::TMULS},       {"TDIVS", PtoOpcode::TDIVS},       {"TMINS", PtoOpcode::TMINS},
        {"TMAXS", PtoOpcode::TMAXS},       {"TABS", PtoOpcode::TABS},         {"TNEG", PtoOpcode::TNEG},
        {"TEXP", PtoOpcode::TEXP},         {"TSQRT", PtoOpcode::TSQRT},       {"TRSQRT", PtoOpcode::TRSQRT},
        {"TLOG", PtoOpcode::TLOG},         {"TRELU", PtoOpcode::TRELU},       {"TLRELU", PtoOpcode::TLRELU},
        {"TNOT", PtoOpcode::TNOT},         {"TROWSUM", PtoOpcode::TROWSUM},   {"TROWMAX", PtoOpcode::TROWMAX},
        {"TROWMIN", PtoOpcode::TROWMIN},   {"TROWPROD", PtoOpcode::TROWPROD}, {"TCOLSUM", PtoOpcode::TCOLSUM},
        {"TCOLMAX", PtoOpcode::TCOLMAX},   {"TCOLMIN", PtoOpcode::TCOLMIN},   {"TCOLPROD", PtoOpcode::TCOLPROD},
        {"TMATMUL", PtoOpcode::TMATMUL},   {"TGEMV", PtoOpcode::TGEMV},       {"TCVT", PtoOpcode::TCVT},
        {"TMOV", PtoOpcode::TMOV},         {"TLOAD", PtoOpcode::TLOAD},       {"TSTORE", PtoOpcode::TSTORE},
        {"TTRANS", PtoOpcode::TTRANS},     {"TSORT32", PtoOpcode::TSORT32},   {"TMRGSORT", PtoOpcode::TMRGSORT},
        {"TSEL", PtoOpcode::TSEL},         {"TSCATTER", PtoOpcode::TSCATTER}, {"TEXTRACT", PtoOpcode::TEXTRACT},
        {"TINSERT", PtoOpcode::TINSERT},   {"TROWEXPAND", PtoOpcode::TROWEXPAND},
        {"TCOLEXPAND", PtoOpcode::TCOLEXPAND}, {"TLOADCONV", PtoOpcode::TLOADCONV},
    }};

    for (const auto &entry : kOpcodeMap) {
        if (entry.first == instrName) {
            opcode = entry.second;
            return true;
        }
    }
    return false;
}

template <typename T>
inline bool TryMapCppTypeToDType(DType &dtype)
{
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, float32_t>) {
        dtype = DType::Float;
        return true;
    } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float16_t>) {
        dtype = DType::Half;
        return true;
    } else {
        return false;
    }
}

template <typename FpType>
inline bool TryEstimateSupportedCycles(PtoOpcode op, uint64_t rows, uint64_t cols, uint64_t &cycles)
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

inline bool EstimateCycles(const CostModelInput &input, const PredictRuntimeConfig &predict_config, CostModelResult &result)
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

inline bool EstimateCycles(const CostModelInput &input, CostModelResult &result)
{
    const PredictRuntimeConfig default_config = GetDefaultPredictRuntimeConfig();
    return EstimateCycles(input, default_config, result);
}

inline CostModelResult EstimateCycles(const CostModelInput &input)
{
    CostModelResult result{};
    (void)EstimateCycles(input, result);
    return result;
}

} // namespace pto::mocker::lightweight

#endif
