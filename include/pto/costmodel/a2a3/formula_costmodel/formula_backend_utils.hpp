/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MOCKER_FORMULA_BACKEND_BASE_HPP
#define PTO_MOCKER_FORMULA_BACKEND_BASE_HPP

#include <cmath>
#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>

#include <pto/common/type.hpp>
#include <pto/costmodel/a2a3/formula_costmodel/formula_params_generated.hpp>

namespace pto::mocker::fit {

template <typename T>
inline constexpr FormulaParamDType GetFormulaParamDType()
{
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, float32_t>) {
        return FormulaParamDType::Fp32;
    } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float16_t>) {
        return FormulaParamDType::Fp16;
    } else {
        return FormulaParamDType::Any;
    }
}

inline uint64_t RoundToCycles(double value)
{
    if (value <= 0.0) {
        return 0;
    }
    return static_cast<uint64_t>(std::llround(value));
}

inline bool TryLookupFormulaParam(std::string_view op, FormulaParamDType dtype, uint64_t cols, double &slope, double &bias)
{
    if (cols > static_cast<uint64_t>(std::numeric_limits<uint16_t>::max())) {
        return false;
    }
    const uint16_t cols16 = static_cast<uint16_t>(cols);
    for (const auto &entry : kFormulaParamTable) {
        if (entry.op == op && entry.dtype == dtype && entry.cols == cols16) {
            slope = entry.slope;
            bias = entry.bias;
            return true;
        }
    }
    for (const auto &entry : kFormulaParamTable) {
        if (entry.op == op && entry.dtype == dtype && entry.cols == kFormulaAnyCols) {
            slope = entry.slope;
            bias = entry.bias;
            return true;
        }
    }
    for (const auto &entry : kFormulaParamTable) {
        if (entry.op == op && entry.dtype == FormulaParamDType::Any && entry.cols == cols16) {
            slope = entry.slope;
            bias = entry.bias;
            return true;
        }
    }
    for (const auto &entry : kFormulaParamTable) {
        if (entry.op == op && entry.dtype == FormulaParamDType::Any && entry.cols == kFormulaAnyCols) {
            slope = entry.slope;
            bias = entry.bias;
            return true;
        }
    }
    return false;
}

template <typename T>
inline bool TryEstimateFormulaCycles(std::string_view op, uint64_t rows, uint64_t cols, uint64_t &cycles)
{
    const FormulaParamDType dtype = GetFormulaParamDType<T>();
    if (dtype == FormulaParamDType::Any) {
        return false;
    }
    double slope = 0.0;
    double bias = 0.0;
    if (!TryLookupFormulaParam(op, dtype, cols, slope, bias)) {
        return false;
    }
    cycles = RoundToCycles(slope * static_cast<double>(rows) * static_cast<double>(cols) + bias);
    return true;
}

template <typename T>
inline bool TryEstimateFormulaCyclesAnyDType(std::string_view op, uint64_t rows, uint64_t cols, uint64_t &cycles)
{
    double slope = 0.0;
    double bias = 0.0;
    if (!TryLookupFormulaParam(op, GetFormulaParamDType<T>(), cols, slope, bias)) {
        return false;
    }
    cycles = RoundToCycles(slope * static_cast<double>(rows) * static_cast<double>(cols) + bias);
    return true;
}
} // namespace pto::mocker::fit

#endif
