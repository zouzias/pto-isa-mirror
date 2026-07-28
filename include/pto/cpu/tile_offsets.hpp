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

template <typename ConvTile>
size_t inline GetConvTileElementOffset(size_t r, size_t c, const std::vector<int64_t>& shapes)
{
    const size_t shape0 = static_cast<size_t>(shapes[GlobalTensorDim::DIM_0]);
    const size_t shape1 = static_cast<size_t>(shapes[GlobalTensorDim::DIM_1]);
    const size_t shape2 = static_cast<size_t>(shapes[GlobalTensorDim::DIM_2]);
    const size_t shape3 = static_cast<size_t>(shapes[GlobalTensorDim::DIM_3]);
    const size_t shape4 = static_cast<size_t>(shapes[GlobalTensorDim::DIM_4]);

    constexpr size_t C0 = C0_SIZE_BYTE / sizeof(typename ConvTile::DType);

    int64_t i0 = 0, i1 = 0, i2 = 0, i3 = 0, i4 = 0, c0 = 0;
    if constexpr (ConvTile::layout == pto::Layout::NC1HWC0) {
        i3 = r % shape3;
        i2 = (r / shape3) % shape2;
        i0 = r / (shape2 * shape3);
        i4 = c % shape4;
        i1 = c / shape4;
    } else if constexpr (ConvTile::layout == pto::Layout::NDC1HWC0) {
        i4 = r % shape4;
        i3 = (r / shape4) % shape3;
        i0 = r / (shape3 * shape4);
        c0 = c % C0;
        i2 = (c / C0) % shape2;
        i1 = c / (shape2 * C0);
    }
    int64_t offset = i0 * shape1 * shape2 * shape3 * shape4 + i1 * shape2 * shape3 * shape4 + i2 * shape3 * shape4 +
                     i3 * shape4 + i4;
    if constexpr (ConvTile::layout == pto::Layout::NDC1HWC0) {
        offset = offset * C0 + c0;
    }
    return offset;
}

template <typename GlobalData>
size_t inline GetGlobalElementOffsetPlain(GlobalData& gdata, size_t r, size_t c)
{
    return r * gdata.GetStride(GlobalTensorDim::DIM_3) + c;
}

template <typename GlobalData>
size_t inline MapTileIndicesToGlobalOffset(
    size_t r, size_t c, const std::vector<int64_t>& globalShapes, const std::vector<int64_t>& globalStrides)
{
    const size_t shape0 = static_cast<size_t>(shapes[GlobalTensorDim::DIM_0]);
    const size_t shape1 = static_cast<size_t>(shapes[GlobalTensorDim::DIM_1]);
    const size_t shape2 = static_cast<size_t>(shapes[GlobalTensorDim::DIM_2]);
    const size_t shape3 = static_cast<size_t>(shapes[GlobalTensorDim::DIM_3]);
    const size_t shape4 = static_cast<size_t>(shapes[GlobalTensorDim::DIM_4]);

    constexpr size_t C0 = C0_SIZE_BYTE / sizeof(typename GlobalData::DType);

    int64_t i0, i1, i2, i3, i4, c0 = 0;
    if constexpr (GlobalData::layout == pto::Layout::ND) {
        i4 = c;
        i3 = r % shape3;
        i2 = (r / shape3) % shape2;
        i1 = (r / (shape3 * shape2)) % shape1;
        i0 = r / (shape1 * shape2 * shape3);
    } else if constexpr (GlobalData::layout == pto::Layout::DN) {
        i3 = r;
        i4 = c % shape4;
        i2 = (c / shape4) % shape2;
        i1 = (c / (shape4 * shape2)) % shape1;
        i0 = c / (shape1 * shape2 * shape4);
    } else if constexpr (GlobalData::layout == pto::Layout::NZ) {
        const size_t outerCol = c / shape4;
        i0 = outerCol / shape1;
        i1 = outerCol % shape1;
        i2 = r / shape3;
        i3 = r % shape3;
        i4 = c % shape4;
    } else if constexpr (GlobalData::layout == pto::Layout::NC1HWC0) {
        i3 = r % shape3;
        i2 = (r / shape3) % shape2;
        i0 = r / (shape2 * shape3);
        i4 = c % shape4;
        i1 = c / shape4;
    } else if (GlobalData::layout == pto::Layout::NDC1HWC0) {
        i4 = r % shape4;
        i3 = (r / shape4) % shape3;
        i0 = r / (shape3 * shape4);
        c0 = c % C0;
        i2 = (c / C0) % shape2;
        i1 = c / (shape2 * C0);
    }

    const auto offset = i0 * strides[GlobalTensorDim::DIM_0] + i1 * strides[GlobalTensorDim::DIM_1] +
                        i2 * strides[GlobalTensorDim::DIM_2] + i3 * strides[GlobalTensorDim::DIM_3] +
                        i4 * strides[GlobalTensorDim::DIM_4] + c0;
    return static_cast<size_t>(offset);
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
