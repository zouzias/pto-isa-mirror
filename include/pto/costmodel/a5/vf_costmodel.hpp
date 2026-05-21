/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MOCKER_A5_VF_COSTMODEL_HPP
#define PTO_MOCKER_A5_VF_COSTMODEL_HPP

#include <cstdint>
#include <string_view>

#include <pto/costmodel/a5/formula_costmodel/formula_backend_vf.hpp>

namespace pto::mocker::lightweight::a5 {

inline uint64_t CeilDivU64(uint64_t x, uint64_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}

inline std::string_view PtoOpcodeToKey(PtoOpcode op)
{
    switch (op) {
        case PtoOpcode::TADD:
            return "TADD";
        case PtoOpcode::TSUB:
            return "TSUB";
        case PtoOpcode::TMUL:
            return "TMUL";
        case PtoOpcode::TDIV:
            return "TDIV";
        case PtoOpcode::TRECIP:
            return "TRECIP";
        case PtoOpcode::TADDS:
            return "TADDS";
        case PtoOpcode::TSUBS:
            return "TSUBS";
        case PtoOpcode::TMULS:
            return "TMULS";
        case PtoOpcode::TDIVS:
            return "TDIVS";
        case PtoOpcode::TMINS:
            return "TMINS";
        case PtoOpcode::TNEG:
            return "TNEG";
        case PtoOpcode::TEXP:
            return "TEXP";
        case PtoOpcode::TSQRT:
            return "TSQRT";
        case PtoOpcode::TRSQRT:
            return "TRSQRT";
        case PtoOpcode::TLOG:
            return "TLOG";
        case PtoOpcode::TRELU:
            return "TRELU";
        case PtoOpcode::TNOT:
            return "TNOT";
        case PtoOpcode::TCVT:
            return "TCVT";
        case PtoOpcode::TSEL:
            return "TSEL";
        default:
            return "";
    }
}

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

inline bool TryGetA5ElementsPerRepeatByDTypeKey(std::string_view dtype_key, uint64_t &elements_per_repeat)
{
    if (dtype_key == "fp4_e1m2" || dtype_key == "fp4_e2m1") {
        elements_per_repeat = 256;
        return true;
    }
    if (dtype_key == "fp16" || dtype_key == "bf16") {
        elements_per_repeat = 128;
        return true;
    }
    if (dtype_key == "fp32" || dtype_key == "int32" || dtype_key == "uint32" || dtype_key == "fp8_e4m3" ||
        dtype_key == "fp8_e5m2" || dtype_key == "hif8") {
        elements_per_repeat = 64;
        return true;
    }
    if (dtype_key == "int8" || dtype_key == "uint8") {
        elements_per_repeat = 256;
        return true;
    }
    if (dtype_key == "int16" || dtype_key == "uint16") {
        elements_per_repeat = 128;
        return true;
    }
    return false;
}

inline bool TryGetA5ElementsPerRepeat(DType dtype, uint64_t &elements_per_repeat)
{
    return TryGetA5ElementsPerRepeatByDTypeKey(DTypeToKey(dtype), elements_per_repeat);
}

inline bool TryGetA5CvtElementsPerRepeat(std::string_view src_dtype, std::string_view dst_dtype,
                                         fit::ShapePath shape_path, uint64_t &elements_per_repeat)
{
    if ((src_dtype == "fp16" || src_dtype == "bf16") && dst_dtype == "fp32") {
        elements_per_repeat = 64;
        return true;
    }
    if (shape_path == fit::ShapePath::Path1D) {
        return TryGetA5ElementsPerRepeatByDTypeKey(src_dtype, elements_per_repeat);
    }
    if ((src_dtype == "fp32" && (dst_dtype == "fp16" || dst_dtype == "bf16" || dst_dtype == "fp8_e4m3" ||
                                 dst_dtype == "fp8_e5m2" || dst_dtype == "hif8")) ||
        ((src_dtype == "fp16" || src_dtype == "bf16") &&
         (dst_dtype == "fp16" || dst_dtype == "fp4_e1m2" || dst_dtype == "fp4_e2m1" || dst_dtype == "hif8"))) {
        elements_per_repeat = 128;
        return true;
    }
    return TryGetA5ElementsPerRepeatByDTypeKey(src_dtype, elements_per_repeat);
}

inline std::string_view TselOpParams(DType dtype, fit::ShapePath shape_path, uint64_t inner_iter)
{
    if (dtype == DType::Float) {
        if (shape_path == fit::ShapePath::Path1D) {
            return (inner_iter % 2 == 0) ? "TSel_b32_even" : "TSel_b32_odd";
        }
        return "TSel_b32";
    }
    if (dtype == DType::Half || dtype == DType::Int8) {
        return "TSel_b16_8";
    }
    return "";
}

inline std::string_view DefaultOpParams(PtoOpcode op)
{
    switch (op) {
        case PtoOpcode::TRECIP:
            return "default";
        default:
            return "";
    }
}

inline std::string_view CvtOpParams(RoundMode round_mode, SaturationMode saturation_mode)
{
    if (round_mode == RoundMode::CAST_NONE && saturation_mode == SaturationMode::ON) {
        return "CAST_NONE_SAT_ON_normal";
    }
    if (round_mode == RoundMode::CAST_RINT && saturation_mode == SaturationMode::ON) {
        return "CAST_RINT_SAT_ON_normal";
    }
    if (round_mode == RoundMode::CAST_TRUNC && saturation_mode == SaturationMode::OFF) {
        return "CAST_TRUNC_SAT_OFF_normal";
    }
    return "";
}

inline bool IsA5ScalarVfOp(PtoOpcode op)
{
    switch (op) {
        case PtoOpcode::TADDS:
        case PtoOpcode::TSUBS:
        case PtoOpcode::TMULS:
        case PtoOpcode::TDIVS:
        case PtoOpcode::TMINS:
        case PtoOpcode::TMAXS:
        case PtoOpcode::TNEG:
            return true;
        default:
            return false;
    }
}

inline fit::ShapePath ResolveShapePath(PtoOpcode op, int64_t rows, int64_t cols, int64_t valid_cols)
{
    if (IsA5ScalarVfOp(op)) {
        return valid_cols == cols ? fit::ShapePath::Path1D : fit::ShapePath::Path2D;
    }
    if (rows == 1 || valid_cols == cols) {
        return fit::ShapePath::Path1D;
    }
    return fit::ShapePath::Path2D;
}

inline std::string_view ResolveDefaultVfImplKind(PtoOpcode op, fit::ShapePath shape_path)
{
    if (op == PtoOpcode::TCVT) {
        return (shape_path == fit::ShapePath::Path1D) ? "NO_POST_UPDATE" : "POST_UPDATE";
    }
    if (op == PtoOpcode::TSEL) {
        return (shape_path == fit::ShapePath::Path1D) ? "NO_POST_UPDATE" : "DEFAULT";
    }
    return (shape_path == fit::ShapePath::Path1D) ? "POST_UPDATE" : "NO_POST_UPDATE";
}

inline std::string_view ResolveVfImplKind(PtoOpcode op, VFImplKind vf_impl_kind, fit::ShapePath shape_path)
{
    switch (vf_impl_kind) {
        case VFImplKind::VFIMPL_1D_NO_POST_UPDATE:
            return "NO_POST_UPDATE";
        case VFImplKind::VFIMPL_2D_NO_POST_UPDATE:
            return "NO_POST_UPDATE";
        case VFImplKind::VFIMPL_1D_POST_UPDATE:
            return "POST_UPDATE";
        case VFImplKind::VFIMPL_2D_POST_UPDATE:
            return "POST_UPDATE";
        case VFImplKind::VFIMPL_DEFAULT:
        default:
            return ResolveDefaultVfImplKind(op, shape_path);
    }
}

inline bool BuildVfCurveKeyAndLoopCount(const CostModelInput &input, fit::VfCurveKey &key, uint64_t &loop_count,
                                        uint64_t &outer_iter, uint64_t &inner_iter, uint64_t &tail_count)
{
    const int64_t valid_rows_i = input.valid_rows > 0 ? input.valid_rows : input.rows;
    const int64_t valid_cols_i = input.valid_cols > 0 ? input.valid_cols : input.cols;
    if (valid_rows_i <= 0 || valid_cols_i <= 0) {
        return false;
    }

    const std::string_view op_key = PtoOpcodeToKey(input.op);
    if (op_key.empty()) {
        return false;
    }
    const auto shape_path = ResolveShapePath(input.op, input.rows, input.cols, valid_cols_i);

    std::string_view src_dtype = DTypeToKey(input.dtype);
    std::string_view dst_dtype = "none";
    std::string_view op_params = input.a5_op_params;
    if (input.op == PtoOpcode::TCVT) {
        dst_dtype = DTypeToKey(input.dst_dtype);
        if (op_params.empty()) {
            op_params =
                (input.round_mode == RoundMode::CAST_NONE && input.saturation_mode == SaturationMode::ON)
                    ? std::string_view("CAST_RINT_SAT_ON_normal")
                    : CvtOpParams(input.round_mode, input.saturation_mode);
        }
    } else if (op_params.empty()) {
        op_params = DefaultOpParams(input.op);
    }
    if (src_dtype.empty() || dst_dtype.empty()) {
        return false;
    }

    uint64_t elements_per_repeat = 0;
    if (input.op == PtoOpcode::TCVT) {
        if (!TryGetA5CvtElementsPerRepeat(src_dtype, dst_dtype, shape_path, elements_per_repeat)) {
            return false;
        }
    } else if (!TryGetA5ElementsPerRepeat(input.dtype, elements_per_repeat)) {
        return false;
    }
    if (elements_per_repeat == 0) {
        return false;
    }

    const uint64_t valid_rows = static_cast<uint64_t>(valid_rows_i);
    const uint64_t valid_cols = static_cast<uint64_t>(valid_cols_i);
    if (shape_path == fit::ShapePath::Path1D) {
        inner_iter = CeilDivU64(valid_rows * valid_cols, elements_per_repeat);
        outer_iter = 1;
        loop_count = inner_iter;
        tail_count = (valid_rows * valid_cols) % elements_per_repeat;
    } else {
        inner_iter = CeilDivU64(valid_cols, elements_per_repeat);
        outer_iter = valid_rows;
        loop_count = outer_iter * inner_iter;
        tail_count = valid_cols % elements_per_repeat;
    }

    const auto tail_kind = tail_count == 0 ? fit::TailKind::Full : fit::TailKind::Tail;
    if (input.op == PtoOpcode::TSEL) {
        if (op_params.empty()) {
            op_params = TselOpParams(input.dtype, shape_path, inner_iter);
        }
        if (op_params.empty()) {
            return false;
        }
    }

    key = fit::VfCurveKey{
        op_key,
        src_dtype,
        dst_dtype,
        shape_path,
        ResolveVfImplKind(input.op, input.vf_impl_kind, shape_path),
        tail_kind,
        op_params,
    };
    return true;
}

inline bool TryEstimateA5VfCycles(const CostModelInput &input, uint64_t &cycles)
{
    fit::VfCurveKey key{};
    uint64_t loop_count = 0;
    uint64_t outer_iter = 0;
    uint64_t inner_iter = 0;
    uint64_t tail_count = 0;
    if (!BuildVfCurveKeyAndLoopCount(input, key, loop_count, outer_iter, inner_iter, tail_count)) {
        return false;
    }

    const fit::VfFormulaParam *param = nullptr;
    if (!fit::TryLookupVfFormulaParam(key, param)) {
        return false;
    }

    cycles = fit::EvalVfFormula(*param, loop_count, outer_iter, inner_iter, tail_count);
    return true;
}

} // namespace pto::mocker::lightweight::a5

#endif
