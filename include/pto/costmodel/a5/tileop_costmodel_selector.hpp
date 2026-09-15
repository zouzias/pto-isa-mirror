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

#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "pto/costmodel/perf_sim/costmodel_provider.hpp"

namespace pto::mocker {

struct A5TileOpOptions {
    std::string_view op_params{};
};

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

inline ::pto::mocker::lightweight::A5VfShapePathHint ResolveA5VfShapePathHint(
    std::string_view opcode, const A5TileOpMetadata& metadata)
{
    using ShapePathHint = ::pto::mocker::lightweight::A5VfShapePathHint;
    uint32_t required_tiles = 0;
    if (opcode == "TADD" || opcode == "TSUB" || opcode == "TMUL") {
        required_tiles = 3;
    } else if (
        opcode == "TDIVS" || opcode == "TMINS" || opcode == "TNEG" || opcode == "TEXP" || opcode == "TSQRT" ||
        opcode == "TRSQRT" || opcode == "TRECIP" || opcode == "TCVT") {
        required_tiles = 2;
    } else {
        return ShapePathHint::Infer;
    }
    return FirstA5TilesUseContiguousPath(metadata, required_tiles) ? ShapePathHint::Path1D : ShapePathHint::Path2D;
}

template <typename... Args>
inline A5TileOpMetadata BuildA5TileOpMetadata(Args&&... args)
{
    A5TileOpMetadata metadata;
    (CollectA5TileOpMetadata(metadata, std::forward<Args>(args)), ...);
    return metadata;
}

// Resolve the NPU selection inputs without executing the selected implementation.
inline uint64_t EstimateA5TileOpCycles(
    const char* opcode, const A5TileOpOptions& options, auto&& first_tile, auto&&... rest_tiles)
{
    const A5TileOpMetadata metadata = BuildA5TileOpMetadata(first_tile, rest_tiles...);
    ::pto::perf_sim::A5VfTileOpInput input;
    input.opcode = opcode;
    input.src_dtype = metadata.first_tile_dtype;
    if (input.opcode == "TCVT") {
        input.src_dtype = metadata.second_tile_dtype;
        input.dst_dtype = metadata.first_tile_dtype;
    }
    input.rows = metadata.rows;
    input.cols = metadata.cols;
    input.valid_rows = metadata.valid_rows;
    input.valid_cols = metadata.valid_cols;
    input.vf_impl_kind = metadata.vf_impl_kind;
    input.round_mode = metadata.round_mode;
    input.saturation_mode = metadata.saturation_mode;
    input.op_params = options.op_params;
    input.shape_path_hint = ResolveA5VfShapePathHint(opcode, metadata);

    uint64_t cycles = 0;
    if (!::pto::perf_sim::TryEstimateA5VfTileOpCycles(input, cycles)) {
        std::fprintf(stderr,
                     "[costmodel][A5] Unsupported TileOp %s: dtype=%s, dst_dtype=%s, "
                     "shape=%lldx%lld, valid=%lldx%lld, options=%.*s\n",
                     opcode, input.src_dtype.c_str(), input.dst_dtype.c_str(),
                     static_cast<long long>(input.rows), static_cast<long long>(input.cols),
                     static_cast<long long>(input.valid_rows), static_cast<long long>(input.valid_cols),
                     static_cast<int>(options.op_params.size()), options.op_params.empty() ? "" : options.op_params.data());
        throw std::runtime_error("Unsupported A5 Vector costmodel prediction");
    }
    return cycles;
}

} // namespace pto::mocker
