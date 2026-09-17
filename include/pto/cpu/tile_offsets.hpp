/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TILE_OFFSETS_HPP
#define TILE_OFFSETS_HPP

#include <unistd.h>
#include <vector>

namespace pto {

template <typename T, typename TileData = void>
constexpr size_t GetC0ElemCount()
{
    constexpr int kPackedElementsPerByte = IsTwinType<T>() ? 2 : 1;
    if constexpr (!std::is_same_v<TileData, void>) {
        if constexpr (std::is_same_v<T, int32_t> && TileData::Loc == TileType::Acc) {
            return kPackedElementsPerByte * static_cast<size_t>(ACC_C0_SIZE);
        }
    }
    return kPackedElementsPerByte * static_cast<size_t>(C0_SIZE_BYTE) / sizeof(T);
}

template <typename T, typename = void>
struct HasSFractal : std::false_type {};
template <typename T>
struct HasSFractal<T, std::void_t<decltype(T::SFractal)>> : std::true_type {};

template <typename TileData>
using TypeSum = std::conditional_t<
    std::is_same_v<typename TileData::DType, half> || std::is_same_v<typename TileData::DType, bfloat16_t>, float,
    typename TileData::DType>;

template <typename TileData>
size_t inline GetTileElementOffsetSubfractals(size_t subTileR, size_t innerR, size_t subTileC, size_t innerC)
{
    if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::RowMajor)) {
        // Nz
        return subTileC * TileData::Rows * TileData::InnerCols + subTileR * TileData::InnerNumel +
               innerR * TileData::InnerCols + innerC;
    } else if constexpr (TileData::isRowMajor & (TileData::SFractal == SLayout::ColMajor)) {
        // Zn
        return subTileR * TileData::Cols * TileData::InnerRows + subTileC * TileData::InnerNumel +
               innerC * TileData::InnerRows + innerR;
    } else if constexpr (TileData::isRowMajor & (TileData::SFractal == SLayout::RowMajor)) {
        // Zz
        return subTileR * TileData::Cols * TileData::InnerRows + subTileC * TileData::InnerNumel +
               innerR * TileData::InnerCols + innerC;
    } else if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::ColMajor)) {
        // Nn
        return subTileC * TileData::Rows * TileData::InnerCols + subTileR * TileData::InnerNumel +
               innerC * TileData::InnerRows + innerR;
    } else {
        assert(false && "Invalid layout");
    }
    return 0;
}

template <typename TileData>
size_t inline GetTileElementOffsetPlain(size_t r, size_t c)
{
    if constexpr (TileData::isRowMajor) {
        return r * TileData::Cols + c;
    } else {
        return c * TileData::Rows + r;
    }
}

template <typename TileData>
size_t inline GetTileElementOffset(size_t r, size_t c)
{
    if constexpr (TileData::SFractal == SLayout::NoneBox)
        return GetTileElementOffsetPlain<TileData>(r, c);
    else {
        size_t subTileR = r / TileData::InnerRows;
        size_t innerR = r % TileData::InnerRows;
        return GetTileElementOffsetSubfractals<TileData>(
            r / TileData::InnerRows, r % TileData::InnerRows, c / TileData::InnerCols, c % TileData::InnerCols);
    }
}

// Returns the physical (padded) buffer offset of the tile element at (row, col).
// The (row, col) logical index is decomposed with the tile's VALID shape, while the
// buffer stride is taken from the PHYSICAL shape, so valid data keeps its NC1HWC0 /
// NDC1HWC0 / FRACTAL_Z slot inside the padded buffer.
template <typename ConvTile>
size_t inline GetConvTileElementOffset(size_t r, size_t c, const ConvTile& tile)
{
    constexpr size_t C0 = GetC0ElemCount<typename ConvTile::DType>();
    static_assert(C0 != 0, "Divider cannot be equal to zero");

    size_t offset = 0;
    if constexpr (ConvTile::layout == pto::Layout::NC1HWC0) {
        const size_t vh = static_cast<size_t>(tile.GetValidShape(GlobalTensorDim::DIM_2));
        const size_t vw = static_cast<size_t>(tile.GetValidShape(GlobalTensorDim::DIM_3));
        const size_t pc1 = static_cast<size_t>(tile.GetShape(GlobalTensorDim::DIM_1));
        const size_t ph = static_cast<size_t>(tile.GetShape(GlobalTensorDim::DIM_2));
        const size_t pw = static_cast<size_t>(tile.GetShape(GlobalTensorDim::DIM_3));
        const size_t i3 = r % vw;
        const size_t i2 = (r / vw) % vh;
        const size_t i0 = r / (vh * vw);
        const size_t i4 = c % C0;
        const size_t i1 = c / C0;
        offset = i0 * pc1 * ph * pw * C0 + i1 * ph * pw * C0 + i2 * pw * C0 + i3 * C0 + i4;
    } else if constexpr (ConvTile::layout == pto::Layout::NDC1HWC0) {
        const size_t vh = static_cast<size_t>(tile.GetValidShape(GlobalTensorDim::DIM_3));
        const size_t vw = static_cast<size_t>(tile.GetValidShape(GlobalTensorDim::DIM_4));
        const size_t vc1 = static_cast<size_t>(tile.GetValidShape(GlobalTensorDim::DIM_2));
        const size_t pd = static_cast<size_t>(tile.GetShape(GlobalTensorDim::DIM_1));
        const size_t pc1 = static_cast<size_t>(tile.GetShape(GlobalTensorDim::DIM_2));
        const size_t ph = static_cast<size_t>(tile.GetShape(GlobalTensorDim::DIM_3));
        const size_t pw = static_cast<size_t>(tile.GetShape(GlobalTensorDim::DIM_4));
        const size_t i4 = r % vw;
        const size_t i3 = (r / vw) % vh;
        const size_t i0 = r / (vh * vw);
        const size_t c0 = c % C0;
        const size_t i2 = (c / C0) % vc1;
        const size_t i1 = c / (vc1 * C0);
        const size_t base = i0 * pd * pc1 * ph * pw + i1 * pc1 * ph * pw + i2 * ph * pw + i3 * pw + i4;
        offset = base * C0 + c0;
    } else if constexpr (ConvTile::layout == pto::Layout::FRACTAL_Z) {
        if constexpr (ConvTile::totalDimCount == 4) {
            const size_t vc0 = static_cast<size_t>(tile.GetValidShape(GlobalTensorDim::DIM_3));
            const size_t vn = static_cast<size_t>(tile.GetValidShape(GlobalTensorDim::DIM_1));
            const size_t pn1 = static_cast<size_t>(tile.GetShape(GlobalTensorDim::DIM_1));
            const size_t pn0 = static_cast<size_t>(tile.GetShape(GlobalTensorDim::DIM_2));
            const size_t i2 = c;
            const size_t i3 = r % vc0;
            const size_t i1 = (r / vc0) % vn;
            const size_t i0 = r / (vn * vc0);
            offset = i0 * pn1 * pn0 * C0 + i1 * pn0 * C0 + i2 * C0 + i3;
        } else {
            const size_t vh = static_cast<size_t>(tile.GetValidShape(GlobalTensorDim::DIM_1));
            const size_t vw = static_cast<size_t>(tile.GetValidShape(GlobalTensorDim::DIM_2));
            const size_t ph = static_cast<size_t>(tile.GetShape(GlobalTensorDim::DIM_1));
            const size_t pw = static_cast<size_t>(tile.GetShape(GlobalTensorDim::DIM_2));
            const size_t pn = static_cast<size_t>(tile.GetShape(GlobalTensorDim::DIM_3));
            const size_t i4 = r % C0;
            const size_t i2 = (r / C0) % vw;
            const size_t i1 = (r / (vw * C0)) % vh;
            const size_t i0 = r / (vh * vw * C0);
            const size_t i3 = c;
            offset = i0 * ph * pw * pn * C0 + i1 * pw * pn * C0 + i2 * pn * C0 + i3 * C0 + i4;
        }
    }
    return offset;
}

// Builds the role-aligned valid shape vector (matched to the GLOBAL tensor dims) used
// to decode the conv (row, col) loop indices. Empty means "no valid shape override"
// (valid == physical), i.e. the global shapes are used directly.
template <typename ConvTile>
std::vector<int64_t> GetConvTileRoleValidShapes(const ConvTile& tile)
{
    constexpr size_t C0 = GetC0ElemCount<typename ConvTile::DType>();
    if constexpr (ConvTile::layout == pto::Layout::NC1HWC0) {
        return {
            tile.GetValidShape(GlobalTensorDim::DIM_0), tile.GetValidShape(GlobalTensorDim::DIM_1),
            tile.GetValidShape(GlobalTensorDim::DIM_2), tile.GetValidShape(GlobalTensorDim::DIM_3),
            static_cast<int64_t>(C0)};
    } else if constexpr (ConvTile::layout == pto::Layout::NDC1HWC0) {
        return {
            tile.GetValidShape(GlobalTensorDim::DIM_0), tile.GetValidShape(GlobalTensorDim::DIM_1),
            tile.GetValidShape(GlobalTensorDim::DIM_2), tile.GetValidShape(GlobalTensorDim::DIM_3),
            tile.GetValidShape(GlobalTensorDim::DIM_4)};
    } else if constexpr (ConvTile::layout == pto::Layout::FRACTAL_Z) {
        if constexpr (ConvTile::totalDimCount == 4) {
            return {};
        } else {
            return {
                tile.GetValidShape(GlobalTensorDim::DIM_0), tile.GetValidShape(GlobalTensorDim::DIM_1),
                tile.GetValidShape(GlobalTensorDim::DIM_2), tile.GetValidShape(GlobalTensorDim::DIM_3),
                static_cast<int64_t>(C0)};
        }
    }
    return {};
}

template <typename ConvTile>
PTO_INTERNAL int64_t CalculateValidRowFromTile(ConvTile& tile)
{
    if constexpr (ConvTile::layout == pto::Layout::NC1HWC0) {
        return tile.GetValidShape(0) * tile.GetValidShape(2) * tile.GetValidShape(3);
    } else if constexpr (ConvTile::layout == pto::Layout::NDC1HWC0) {
        return tile.GetValidShape(0) * tile.GetValidShape(3) * tile.GetValidShape(4);
    } else if constexpr (ConvTile::layout == pto::Layout::FRACTAL_Z) {
        if constexpr (ConvTile::totalDimCount == 4) {
            return tile.GetValidShape(0) * tile.GetValidShape(1) * tile.GetValidShape(3);
        } else {
            return tile.GetValidShape(0) * tile.GetValidShape(1) * tile.GetValidShape(2) * tile.GetValidShape(4);
        }
    }
    return 0;
}

template <typename ConvTile>
PTO_INTERNAL int64_t CalculateValidColFromTile(ConvTile& tile)
{
    constexpr size_t C0 = GetC0ElemCount<typename ConvTile::DType>();
    if constexpr (ConvTile::layout == pto::Layout::NC1HWC0) {
        return tile.GetValidShape(1) * C0;
    } else if constexpr (ConvTile::layout == pto::Layout::NDC1HWC0) {
        return tile.GetValidShape(1) * tile.GetValidShape(2) * C0;
    } else if constexpr (ConvTile::layout == pto::Layout::FRACTAL_Z) {
        if constexpr (ConvTile::totalDimCount == 4) {
            return tile.GetValidShape(2);
        } else {
            return tile.GetValidShape(3);
        }
    }
    return 0;
}

template <typename GlobalData>
size_t inline GetGlobalElementOffsetPlain(GlobalData& gdata, size_t r, size_t c)
{
    return r * gdata.GetStride(GlobalTensorDim::DIM_3) + c;
}

template <typename GlobalData, typename TileData = void>
size_t inline MapTileIndicesToGlobalOffset(
    size_t r, size_t c, const std::vector<int64_t>& globalShapes, const std::vector<int64_t>& globalStrides,
    const std::vector<int64_t>& validShapes = {})
{
    const size_t shape1 = static_cast<size_t>(globalShapes[GlobalTensorDim::DIM_1]);
    const size_t shape2 = static_cast<size_t>(globalShapes[GlobalTensorDim::DIM_2]);
    const size_t shape3 = static_cast<size_t>(globalShapes[GlobalTensorDim::DIM_3]);
    const size_t shape4 = static_cast<size_t>(globalShapes[GlobalTensorDim::DIM_4]);

    // Conv tiles decode the (row, col) loop indices against the VALID shape but
    // address against the global (physical) strides.
    const bool hasValidShapes = validShapes.size() > GlobalTensorDim::DIM_4;
    const size_t d1 = hasValidShapes ? static_cast<size_t>(validShapes[GlobalTensorDim::DIM_1]) : shape1;
    const size_t d2 = hasValidShapes ? static_cast<size_t>(validShapes[GlobalTensorDim::DIM_2]) : shape2;
    const size_t d3 = hasValidShapes ? static_cast<size_t>(validShapes[GlobalTensorDim::DIM_3]) : shape3;
    const size_t d4 = hasValidShapes ? static_cast<size_t>(validShapes[GlobalTensorDim::DIM_4]) : shape4;

    constexpr size_t C0 = GetC0ElemCount<typename GlobalData::DType, TileData>();
    static_assert(C0 != 0, "Divider cannot be equal to zero");

    int64_t i0, i1, i2, i3, i4, c0 = 0;
    if constexpr (GlobalData::layout == pto::Layout::ND) {
        i4 = c;
        i3 = r % d3;
        i2 = (r / d3) % d2;
        i1 = (r / (d3 * d2)) % d1;
        i0 = r / (d1 * d2 * d3);
    } else if constexpr (GlobalData::layout == pto::Layout::DN) {
        i3 = r;
        i4 = c % d4;
        i2 = (c / d4) % d2;
        i1 = (c / (d4 * d2)) % d1;
        i0 = c / (d1 * d2 * d4);
    } else if constexpr (GlobalData::layout == pto::Layout::NZ) {
        const size_t outerCol = c / d4;
        i0 = outerCol / d1;
        i1 = outerCol % d1;
        i2 = r / d3;
        i3 = r % d3;
        i4 = c % d4;
    } else if constexpr (GlobalData::layout == pto::Layout::NC1HWC0) {
        i3 = r % d3;
        i2 = (r / d3) % d2;
        i0 = r / (d2 * d3);
        i4 = c % d4;
        i1 = c / d4;
    } else if (GlobalData::layout == pto::Layout::NDC1HWC0) {
        i4 = r % d4;
        i3 = (r / d4) % d3;
        i0 = r / (d3 * d4);
        c0 = c % C0;
        i2 = (c / C0) % d2;
        i1 = c / (d2 * C0);
    } else if (GlobalData::layout == pto::Layout::FRACTAL_Z) {
        i4 = r % d4;
        i2 = (r / d4) % d2;
        i1 = (r / (d2 * d4)) % d1;
        i0 = r / (d1 * d2 * d4);
        i3 = c;
    } else if constexpr (GlobalData::layout == pto::Layout::MX_A_ZZ) {
        i0 = 0;
        i1 = r / MX_ROW_LEN;
        i2 = c / MX_COL_LEN;
        i3 = r % MX_ROW_LEN;
        i4 = c % MX_COL_LEN;
    } else if constexpr (GlobalData::layout == pto::Layout::MX_B_NN) {
        i0 = 0;
        i1 = c / MX_ROW_LEN;
        i2 = r / MX_COL_LEN;
        i3 = c % MX_ROW_LEN;
        i4 = r % MX_COL_LEN;
    } else if constexpr (GlobalData::layout == pto::Layout::MX_A_ND) {
        i0 = 0;
        i1 = 0;
        i2 = r;
        i3 = c / MX_COL_LEN;
        i4 = c % MX_COL_LEN;
    } else if constexpr (GlobalData::layout == pto::Layout::MX_A_DN) {
        i0 = 0;
        i1 = 0;
        i2 = c / MX_COL_LEN;
        i3 = r;
        i4 = c % MX_COL_LEN;
    } else if constexpr (GlobalData::layout == pto::Layout::MX_B_ND) {
        i0 = 0;
        i1 = 0;
        i2 = r / MX_COL_LEN;
        i3 = c;
        i4 = r % MX_COL_LEN;
    } else if constexpr (GlobalData::layout == pto::Layout::MX_B_DN) {
        i0 = 0;
        i1 = 0;
        i2 = c;
        i3 = r / MX_COL_LEN;
        i4 = r % MX_COL_LEN;
    } else {
        throw std::invalid_argument("Unsupported layout in MapTileIndicesToGlobalOffset");
    }

    const auto offset = i0 * globalStrides[GlobalTensorDim::DIM_0] + i1 * globalStrides[GlobalTensorDim::DIM_1] +
                        i2 * globalStrides[GlobalTensorDim::DIM_2] + i3 * globalStrides[GlobalTensorDim::DIM_3] +
                        i4 * globalStrides[GlobalTensorDim::DIM_4] + c0;
    return static_cast<size_t>(offset);
}

// TSTORE takes its GM traversal from the tile layout, as the hardware does via
// TileData::isRowMajor. Only the single-row/single-column pairing CheckStaticForVecAndMat
// allows can disagree with the GlobalTensor layout, and there the DMA writes one
// contiguous burst. TLOAD needs no equivalent: Vec loads only accept matching layouts.
template <typename GlobalData, typename TileData>
constexpr bool IsVecTransferLayoutSwapped()
{
    if constexpr (std::is_same_v<TileData, void>) {
        return false;
    } else if constexpr (!HasSFractal<TileData>::value) {
        return false;
    } else if constexpr (TileData::Loc != TileType::Vec || TileData::SFractal != SLayout::NoneBox) {
        return false;
    } else if constexpr (GlobalData::layout == pto::Layout::ND) {
        return !TileData::isRowMajor && TileData::Cols == 1;
    } else if constexpr (GlobalData::layout == pto::Layout::DN) {
        return TileData::isRowMajor && TileData::Rows == 1;
    } else {
        return false;
    }
}

// One row or one column, so only one index varies, every outer dimension is 1, and the
// vector is contiguous.
template <typename GlobalData, typename TileData>
size_t inline MapSwappedTileIndicesToGlobalOffset(
    size_t r, size_t c, const std::vector<int64_t>& globalShapes, const std::vector<int64_t>& globalStrides)
{
    (void)globalShapes;
    (void)globalStrides;
    return r + c;
}

template <typename GlobalData, typename TileData = void>
size_t inline MapTransferIndicesToGlobalOffset(
    size_t r, size_t c, const std::vector<int64_t>& globalShapes, const std::vector<int64_t>& globalStrides)
{
    if constexpr (IsVecTransferLayoutSwapped<GlobalData, TileData>()) {
        return MapSwappedTileIndicesToGlobalOffset<GlobalData, TileData>(r, c, globalShapes, globalStrides);
    } else {
        return MapTileIndicesToGlobalOffset<GlobalData, TileData>(r, c, globalShapes, globalStrides);
    }
}

template <typename DataStorage>
size_t inline GetDataElementOffset(DataStorage& storage, size_t r, size_t c)
{
    if constexpr (HasSFractal<DataStorage>::value) {
        return GetTileElementOffset<DataStorage>(r, c);
    } else {
        return GetGlobalElementOffsetPlain(storage, r, c);
    }
    return 0;
}

} // namespace pto
#endif
