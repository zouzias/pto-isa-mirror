/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#pragma once

#include <iostream>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <pto/costmodel/perf_sim/recorder.hpp>
#include <pto/costmodel/a5/formula_costmodel/formula_backend_vf.hpp>

namespace pto::mocker {

struct A5TileOpOptions {
    std::string_view op_params{};
};

// Interpret template values here, not in wrapper macros or the recorder.
template <auto Value>
inline void CollectA5TemplateOption(A5TileOpOptions& options)
{
    using T = std::remove_cv_t<decltype(Value)>;
    if constexpr (
        std::is_same_v<T, RecipAlgorithm> || std::is_same_v<T, DivAlgorithm> || std::is_same_v<T, SqrtAlgorithm> ||
        std::is_same_v<T, RsqrtAlgorithm> || std::is_same_v<T, ExpAlgorithm> || std::is_same_v<T, LogAlgorithm> ||
        std::is_same_v<T, PowAlgorithm>) {
        if constexpr (Value == T::HIGH_PRECISION) {
            options.op_params = "high_precision";
        }
    }
}

// Wrapper template lists contain either values or a prefix of Tile types
// followed by values. Tile metadata is read from the actual operands.
template <auto... Values>
inline A5TileOpOptions A5OptionsFromTemplate()
{
    A5TileOpOptions options;
    (CollectA5TemplateOption<Values>(options), ...);
    return options;
}

template <typename T0, auto... Values>
inline A5TileOpOptions A5OptionsFromTemplate()
{
    return A5OptionsFromTemplate<Values...>();
}

template <typename T0, typename T1, auto... Values>
inline A5TileOpOptions A5OptionsFromTemplate()
{
    return A5OptionsFromTemplate<Values...>();
}

template <typename T0, typename T1, typename T2, auto... Values>
inline A5TileOpOptions A5OptionsFromTemplate()
{
    return A5OptionsFromTemplate<Values...>();
}

template <typename T0, typename T1, typename T2, typename T3, auto... Values>
inline A5TileOpOptions A5OptionsFromTemplate()
{
    return A5OptionsFromTemplate<Values...>();
}

struct A5TileOpMetadata {
    std::string first_tile_dtype;
    std::string second_tile_dtype;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t valid_rows = 0;
    int64_t valid_cols = 0;
    uint32_t tile_count = 0;
    uint32_t static_full_cols_mask = 0;
    uint32_t single_row_mask = 0;
    VFImplKind vf_impl_kind = VFImplKind::VFIMPL_DEFAULT;
    RoundMode round_mode = RoundMode::CAST_NONE;
    SaturationMode saturation_mode = SaturationMode::ON;
};

template <typename T>
inline void CollectA5TileOpMetadata(A5TileOpMetadata& metadata, T&& arg)
{
    using Arg = std::remove_cv_t<std::remove_reference_t<T>>;
    if constexpr (requires {
                      Arg::Rows;
                      Arg::Cols;
                      Arg::ValidCol;
                      arg.GetValidRow();
                      arg.GetValidCol();
                      ::pto::perf_sim::TileTraits<Arg>::dtype_str();
                  }) {
        const std::string dtype = ::pto::perf_sim::TileTraits<Arg>::dtype_str();
        const uint32_t tile_index = metadata.tile_count;
        if (tile_index < 32 && Arg::ValidCol == Arg::Cols) {
            metadata.static_full_cols_mask |= 1U << tile_index;
        }
        if (tile_index < 32 && Arg::Rows == 1) {
            metadata.single_row_mask |= 1U << tile_index;
        }
        if (tile_index == 0) {
            metadata.first_tile_dtype = dtype;
            metadata.rows = static_cast<int64_t>(Arg::Rows);
            metadata.cols = static_cast<int64_t>(Arg::Cols);
            metadata.valid_rows = static_cast<int64_t>(arg.GetValidRow());
            metadata.valid_cols = static_cast<int64_t>(arg.GetValidCol());
        } else if (tile_index == 1) {
            metadata.second_tile_dtype = dtype;
        }
        ++metadata.tile_count;
    } else if constexpr (std::is_same_v<Arg, VFImplKind>) {
        metadata.vf_impl_kind = arg;
    } else if constexpr (std::is_same_v<Arg, RoundMode>) {
        metadata.round_mode = arg;
    } else if constexpr (std::is_same_v<Arg, SaturationMode>) {
        metadata.saturation_mode = arg;
    }
}

inline bool FirstA5TilesUseContiguousPath(const A5TileOpMetadata& metadata, uint32_t required_tiles)
{
    if (required_tiles == 0 || required_tiles > metadata.tile_count || required_tiles >= 32) {
        return false;
    }
    const uint32_t required_mask = (1U << required_tiles) - 1U;
    return (metadata.static_full_cols_mask & required_mask) == required_mask ||
           (metadata.single_row_mask & required_mask) == required_mask;
}

template <typename... Args>
inline A5TileOpMetadata BuildA5TileOpMetadata(Args&&... args)
{
    A5TileOpMetadata metadata;
    (CollectA5TileOpMetadata(metadata, std::forward<Args>(args)), ...);
    return metadata;
}

inline constexpr uint64_t kElementsPerRepeatB32 = 64;
inline constexpr uint64_t kElementsPerRepeatB16 = 128;
inline constexpr uint64_t kElementsPerRepeatB8 = 256;

inline bool GetA5ElementsPerRepeatByDTypeKey(std::string_view dtype_key, uint64_t& elements_per_repeat)
{
    if (dtype_key == "fp4_e1m2" || dtype_key == "fp4_e2m1") {
        elements_per_repeat = kElementsPerRepeatB8;
        return true;
    }
    if (dtype_key == "fp16" || dtype_key == "bf16") {
        elements_per_repeat = kElementsPerRepeatB16;
        return true;
    }
    if (dtype_key == "fp32" || dtype_key == "int32" || dtype_key == "uint32" || dtype_key == "fp8_e4m3" ||
        dtype_key == "fp8_e5m2" || dtype_key == "hif8") {
        elements_per_repeat = kElementsPerRepeatB32;
        return true;
    }
    if (dtype_key == "int8" || dtype_key == "uint8") {
        elements_per_repeat = kElementsPerRepeatB8;
        return true;
    }
    if (dtype_key == "int16" || dtype_key == "uint16") {
        elements_per_repeat = kElementsPerRepeatB16;
        return true;
    }
    return false;
}

inline bool GetA5CvtElementsPerRepeat(
    std::string_view src_dtype, std::string_view dst_dtype, lightweight::a5::fit::ShapePath shape_path,
    uint64_t& elements_per_repeat)
{
    if ((src_dtype == "fp16" || src_dtype == "bf16") && dst_dtype == "fp32") {
        elements_per_repeat = kElementsPerRepeatB32;
        return true;
    }
    if (shape_path == lightweight::a5::fit::ShapePath::Path1D) {
        return GetA5ElementsPerRepeatByDTypeKey(src_dtype, elements_per_repeat);
    }
    if ((src_dtype == "fp32" && (dst_dtype == "fp16" || dst_dtype == "bf16" || dst_dtype == "fp8_e4m3" ||
                                 dst_dtype == "fp8_e5m2" || dst_dtype == "hif8")) ||
        ((src_dtype == "fp16" || src_dtype == "bf16") &&
         (dst_dtype == "fp16" || dst_dtype == "fp4_e1m2" || dst_dtype == "fp4_e2m1" || dst_dtype == "hif8"))) {
        elements_per_repeat = kElementsPerRepeatB16;
        return true;
    }
    return GetA5ElementsPerRepeatByDTypeKey(src_dtype, elements_per_repeat);
}

// Shared layout rule: an explicit 2D version can force a contiguous
// Tile onto the 2D path, but a non-contiguous Tile cannot be forced onto 1D.
inline void SelectA5LoopType(bool contiguous, VFImplKind version, A5VfFormulaInput& path)
{
    const bool force2D =
        version == VFImplKind::VFIMPL_2D_POST_UPDATE || version == VFImplKind::VFIMPL_2D_NO_POST_UPDATE;
    const bool noPost =
        version == VFImplKind::VFIMPL_1D_NO_POST_UPDATE || version == VFImplKind::VFIMPL_2D_NO_POST_UPDATE;
    if (contiguous && !force2D) {
        path.shape_path = lightweight::a5::fit::ShapePath::Path1D;
        path.vf_impl_kind = noPost ? "NO_POST_UPDATE" : "POST_UPDATE";
        return;
    }
    const bool post = version == VFImplKind::VFIMPL_1D_POST_UPDATE || version == VFImplKind::VFIMPL_2D_POST_UPDATE;
    path.shape_path = lightweight::a5::fit::ShapePath::Path2D;
    path.vf_impl_kind = post ? "POST_UPDATE" : "NO_POST_UPDATE";
}

// Common 1D/2D selection; callers specify the number of participating Tiles.
// Update mode is retained only to match the current formula-table keys.
inline bool SelectA5ElementwisePath(const A5TileOpMetadata& metadata, uint32_t required_tiles, A5VfFormulaInput& path)
{
    if (required_tiles == 0 || required_tiles >= 32 || metadata.tile_count < required_tiles) {
        return false;
    }
    SelectA5LoopType(FirstA5TilesUseContiguousPath(metadata, required_tiles), metadata.vf_impl_kind, path);
    return GetA5ElementsPerRepeatByDTypeKey(path.src_dtype, path.elements_per_repeat);
}

// Both default and high-precision algorithms retain 1D/2D layout selection.
// The entry point has already copied the precision option into path.op_params.
inline bool SelectA5PrecisionElementwisePath(const A5TileOpMetadata& metadata, A5VfFormulaInput& path)
{
    if (!SelectA5ElementwisePath(metadata, 2, path)) {
        return false;
    }
    if (path.opcode != "TDIVS" && path.shape_path == lightweight::a5::fit::ShapePath::Path2D) {
        // TUnaryOp/TRsqrt have one 2D update variant. Preserve its existing table key;
        // this is not a separate selector category or a different precision mode.
        path.vf_impl_kind = "NO_POST_UPDATE";
    }
    return true;
}

// Conversion has separate dtype-pair, rounding/saturation and repeat-width rules.
inline bool SelectA5CvtPath(const A5TileOpMetadata& metadata, A5VfFormulaInput& path)
{
    if (metadata.tile_count < 2) {
        return false;
    }
    path.src_dtype = metadata.second_tile_dtype;
    path.dst_dtype = metadata.first_tile_dtype;
    const bool contiguous = FirstA5TilesUseContiguousPath(metadata, 2);
    if (contiguous) {
        path.shape_path = metadata.vf_impl_kind == VFImplKind::VFIMPL_2D_NO_POST_UPDATE ?
                              lightweight::a5::fit::ShapePath::Path2D :
                              lightweight::a5::fit::ShapePath::Path1D;
        path.vf_impl_kind = "NO_POST_UPDATE";
    } else {
        const bool noPost = metadata.vf_impl_kind == VFImplKind::VFIMPL_1D_NO_POST_UPDATE ||
                            metadata.vf_impl_kind == VFImplKind::VFIMPL_2D_NO_POST_UPDATE;
        path.shape_path = lightweight::a5::fit::ShapePath::Path2D;
        path.vf_impl_kind = noPost ? "NO_POST_UPDATE" : "POST_UPDATE";
    }
    if (path.op_params.empty()) {
        if ((metadata.round_mode == RoundMode::CAST_NONE || metadata.round_mode == RoundMode::CAST_RINT) &&
            metadata.saturation_mode == SaturationMode::ON) {
            path.op_params = "CAST_RINT_SAT_ON_normal";
        } else if (metadata.round_mode == RoundMode::CAST_TRUNC && metadata.saturation_mode == SaturationMode::OFF) {
            path.op_params = "CAST_TRUNC_SAT_OFF_normal";
        } else {
            return false;
        }
    }
    return GetA5CvtElementsPerRepeat(path.src_dtype, path.dst_dtype, path.shape_path, path.elements_per_repeat);
}

// TSEL has its own default layout and a fitted variant depending on repeat parity.
inline bool SelectA5TselPath(const A5TileOpMetadata& metadata, A5VfFormulaInput& path)
{
    if (metadata.tile_count < 3 || !GetA5ElementsPerRepeatByDTypeKey(path.src_dtype, path.elements_per_repeat)) {
        return false;
    }
    const bool contiguous = FirstA5TilesUseContiguousPath(metadata, 3);
    path.shape_path = contiguous ? lightweight::a5::fit::ShapePath::Path1D : lightweight::a5::fit::ShapePath::Path2D;
    path.vf_impl_kind = contiguous ? "NO_POST_UPDATE" : "DEFAULT";
    if (metadata.vf_impl_kind != VFImplKind::VFIMPL_DEFAULT) {
        SelectA5LoopType(contiguous, metadata.vf_impl_kind, path);
    }
    if (path.op_params.empty()) {
        if (path.src_dtype == "fp32") {
            const uint64_t elems = contiguous ? path.valid_rows * path.valid_cols : path.valid_cols;
            const uint64_t repeats = (elems + path.elements_per_repeat - 1) / path.elements_per_repeat;
            path.op_params = contiguous ? (repeats % 2 == 0 ? "TSel_b32_even" : "TSel_b32_odd") : "TSel_b32";
        } else if (path.src_dtype == "fp16" || path.src_dtype == "int8") {
            path.op_params = "TSel_b16_8";
        } else {
            return false;
        }
    }
    return true;
}

// Initialize once and map each supported TileOp to its actual selection rule.
// Precision remains part of path.op_params; grouping here does not discard it.
inline bool DispatchA5TileOpSelector(
    std::string_view opcode, const A5TileOpOptions& options, const A5TileOpMetadata& metadata, A5VfFormulaInput& path)
{
    path = {};
    if (metadata.valid_rows <= 0 || metadata.valid_cols <= 0) {
        return false;
    }
    path.opcode = opcode;
    path.src_dtype = metadata.first_tile_dtype;
    path.op_params = options.op_params;
    path.valid_rows = static_cast<uint64_t>(metadata.valid_rows);
    path.valid_cols = static_cast<uint64_t>(metadata.valid_cols);

    if (opcode == "TADD" || opcode == "TSUB" || opcode == "TMUL") {
        return SelectA5ElementwisePath(metadata, 3, path);
    }
    if (opcode == "TMINS" || opcode == "TNEG") {
        return SelectA5ElementwisePath(metadata, 2, path);
    }
    // Both TDIVS operand orders share the model; TRECIP forwards as TDIVS(dst, 1, src).
    if (opcode == "TDIVS" || opcode == "TEXP" || opcode == "TSQRT" || opcode == "TRSQRT") {
        return SelectA5PrecisionElementwisePath(metadata, path);
    }
    // Dedicated rules only where dtype conversion or other parameters require them.
    if (opcode == "TCVT") {
        return SelectA5CvtPath(metadata, path);
    }
    if (opcode == "TSEL") {
        return SelectA5TselPath(metadata, path);
    }
    return false;
}

// Thin wrapper bridge: collect metadata, select once, evaluate the selected path.
inline uint64_t EstimateA5TileOpCycles(
    const char* opcode, const A5TileOpOptions& options, auto&& first_tile, auto&&... rest_tiles)
{
    const A5TileOpMetadata metadata = BuildA5TileOpMetadata(first_tile, rest_tiles...);
    A5VfFormulaInput path;
    uint64_t cycles = 0;
    if (!DispatchA5TileOpSelector(opcode, options, metadata, path) ||
        !lightweight::a5::fit::EstimateA5VfCycles(path, cycles)) {
        std::cerr << "[costmodel][ERROR][A5] Unsupported TileOp: op=" << opcode
                  << ", dtype=" << metadata.first_tile_dtype << ", valid_rows=" << metadata.valid_rows
                  << ", valid_cols=" << metadata.valid_cols << ", options=" << options.op_params << '\n';
        throw std::runtime_error("Unsupported A5 Vector costmodel prediction");
    }
    return cycles;
}

} // namespace pto::mocker
