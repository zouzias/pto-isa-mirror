/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file logicalnot.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_LOGICALNOT__H
#define TILEOP_TILE_OPERATOR_LOGICALNOT__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include <type_traits>

template <typename U>
using select_type = std::conditional_t<
    std::is_same_v<typename U::Type, float>,
    float,
    half
>;

template <typename U>
using select_type_bool = std::conditional_t<
    std::is_same_v<typename U::Type, bool>,
    uint8_t,
    typename U::Type
>;

template <typename T, typename DstTile, typename SrcTile, typename CastTile, typename ExpTile, typename VcmpResTile>
TILEOP void LogicalNotImpl(DstTile dstTile, SrcTile srcTile, CastTile castTile,
                            ExpTile oneTile, ExpTile zeroTile, VcmpResTile vcmpResTile)
{
    if constexpr (std::is_same<T, bool>::value || std::is_same<T, uint8_t>::value ||
                std::is_same<T, int8_t>::value) {
        pto::TCVT(castTile, srcTile, pto::RoundMode::CAST_NONE);
    }
    pto::TEXPANDS(oneTile, 1.0);
    pto::TEXPANDS(zeroTile, 0.0);

    #ifdef __DAV_V220
        pipe_barrier(PIPE_V);
    #endif

    if constexpr (std::is_same<T, half>::value || std::is_same<T, float>::value) {
        pto::TCMP(vcmpResTile, srcTile, zeroTile, pto::CmpMode::EQ);
    } else if (std::is_same<T, bool>::value || std::is_same<T, uint8_t>::value ||
                std::is_same<T, int8_t>::value) {
        pto::TCMP(vcmpResTile, castTile, zeroTile, pto::CmpMode::EQ);
    }

    #ifdef __DAV_V220
        pipe_barrier(PIPE_V);
    #endif
    pto::TSEL(oneTile, vcmpResTile, oneTile, zeroTile);
    #ifdef __DAV_V220
        pipe_barrier(PIPE_V);
    #endif

    if constexpr (std::is_same<T, float>::value) {
        pto::TCVT(castTile, oneTile, pto::RoundMode::CAST_NONE);
        #ifdef __DAV_V220
            pipe_barrier(PIPE_V);
        #endif
        pto::TCVT(dstTile, castTile, pto::RoundMode::CAST_NONE);
    } else {
        pto::TCVT(dstTile, oneTile, pto::RoundMode::CAST_NONE);
    }
}

#define OP_TILE_OP_LOGICALNOT TLogicalNot
template <typename T0, typename T1, typename T2>
TILEOP void TLogicalNot(T0 dst, T1 src, T2 tmp)
{
    using ShapeValueType = typename Std::tuple_element<0, typename T0::Shape>::type;
    constexpr auto shapeSize0 = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto shapeSize1 = Std::tuple_size<typename T1::Shape>::value;
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto srcTypeSize = sizeof(typename T1::Type);

    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();

    auto srcShape0 = srcLayout.template GetShapeDim<0, expectSize>();
    auto srcShape1 = srcLayout.template GetShapeDim<1, expectSize>();
    auto srcShape2 = srcLayout.template GetShapeDim<2, expectSize>();
    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();

    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    auto dstStride3 = dstLayout.template GetStrideDim<3, expectSize>();

    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();
    auto srcStride3 = srcLayout.template GetStrideDim<3, expectSize>();

    constexpr auto dstTileH = Std::tuple_element<shapeSize0 - 2, typename T0::TileShape>::type::value;
    constexpr auto dstTileW = Std::tuple_element<shapeSize0 - 1, typename T0::TileShape>::type::value;

    constexpr auto srcTileH = Std::tuple_element<shapeSize1 - 2, typename T1::TileShape>::type::value;
    constexpr auto srcTileW = Std::tuple_element<shapeSize1 - 1, typename T1::TileShape>::type::value;

    constexpr uint32_t ALIGN_SIZE = 32;
    constexpr int64_t COUNT_MAX = 2048;
    constexpr int64_t TYPE_SIZE = std::is_same_v<typename T1::Type, float> ? 4 : 2;
    using U = select_type<T1>;
    using T = select_type_bool<T1>;
    uint32_t vcmpBitSize = (COUNT_MAX + 7) / 8;

    __ubuf__ int8_t* vcmpBitResult = reinterpret_cast<__ubuf__ int8_t*>(tmp.GetAddr());

    uintptr_t zeroCondAddr = reinterpret_cast<uintptr_t>(vcmpBitResult + vcmpBitSize);
    zeroCondAddr = (zeroCondAddr + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
    __ubuf__ int8_t* compareCondition = reinterpret_cast<__ubuf__ int8_t*>(zeroCondAddr);

    uintptr_t oneCondAddr = reinterpret_cast<uintptr_t>(compareCondition + COUNT_MAX * TYPE_SIZE);
    oneCondAddr = (oneCondAddr + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
    __ubuf__ int8_t* oneCondition = reinterpret_cast<__ubuf__ int8_t*>(oneCondAddr);

    uintptr_t castAddr = reinterpret_cast<uintptr_t>(oneCondition + COUNT_MAX * TYPE_SIZE);
    castAddr = (castAddr + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
    __ubuf__ half* castCondition = reinterpret_cast<__ubuf__ half*>(castAddr);

    uintptr_t startAddrAddr = reinterpret_cast<uintptr_t>(castCondition + COUNT_MAX);
    startAddrAddr = (startAddrAddr + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
    __ubuf__ uint64_t* startAddrUB = reinterpret_cast<__ubuf__ uint64_t*>(startAddrAddr);

    using DstTile =
        pto::Tile<pto::TileType::Vec, uint8_t, 1, COUNT_MAX, pto::BLayout::RowMajor, -1, -1>;
    using SrcTile =
        pto::Tile<pto::TileType::Vec, T, 1, COUNT_MAX, pto::BLayout::RowMajor, -1, -1>;
    using CastTile =
        pto::Tile<pto::TileType::Vec, half, 1, COUNT_MAX, pto::BLayout::RowMajor, -1, -1>;
    using ExpTile =
        pto::Tile<pto::TileType::Vec, U, 1, COUNT_MAX, pto::BLayout::RowMajor, -1, -1>;
    using VcmpResTile =
        pto::Tile<pto::TileType::Vec, uint8_t, 1, COUNT_MAX / 8, pto::BLayout::RowMajor, 1, COUNT_MAX / 8>;

    VcmpResTile vcmpResTile;
    pto::TASSIGN(vcmpResTile, (uint64_t)(vcmpBitResult));

    unsigned numLoop = dstShape4 / COUNT_MAX;
    unsigned remainAfterLoop = dstShape4 % COUNT_MAX;

    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                for (LoopVar n3Index = 0; n3Index < dstShape3; ++n3Index) {
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2 + n3Index * dstStride3;
                    auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2 + n3Index * srcStride3;
                    for (LoopVar j = 0; j < numLoop; j++) {
                        auto dstOffsetLoop = dstOffset + j * COUNT_MAX;
                        auto srcOffsetLoop = srcOffset + j * COUNT_MAX;
                        DstTile dstTile(1, COUNT_MAX);
                        SrcTile srcTile(1, COUNT_MAX);
                        CastTile castTile(1, COUNT_MAX);
                        ExpTile oneTile(1, COUNT_MAX);
                        ExpTile zeroTile(1, COUNT_MAX);
                        pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffsetLoop * dstTypeSize));
                        pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffsetLoop * srcTypeSize));
                        pto::TASSIGN(castTile, (uint64_t)(castCondition));
                        pto::TASSIGN(oneTile, (uint64_t)(oneCondition));
                        pto::TASSIGN(zeroTile, (uint64_t)(compareCondition));
                        LogicalNotImpl<T, DstTile, SrcTile, CastTile, ExpTile,
                                    VcmpResTile>(dstTile, srcTile, castTile, oneTile, zeroTile, vcmpResTile);
                    }
                    if (remainAfterLoop > 0) {
                        auto dstOffsetRemain = dstOffset + numLoop * COUNT_MAX;
                        auto srcOffsetRemain = srcOffset + numLoop * COUNT_MAX;
                        DstTile dstTile(1, remainAfterLoop);
                        SrcTile srcTile(1, remainAfterLoop);
                        CastTile castTile(1, remainAfterLoop);
                        ExpTile oneTile(1, remainAfterLoop);
                        ExpTile zeroTile(1, remainAfterLoop);
                        pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffsetRemain * dstTypeSize));
                        pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffsetRemain * srcTypeSize));
                        pto::TASSIGN(castTile, (uint64_t)(castCondition));
                        pto::TASSIGN(oneTile, (uint64_t)(oneCondition));
                        pto::TASSIGN(zeroTile, (uint64_t)(compareCondition));
                        LogicalNotImpl<T, DstTile, SrcTile, CastTile, ExpTile,
                                    VcmpResTile>(dstTile, srcTile, castTile, oneTile, zeroTile, vcmpResTile);
                    }
                }
            }
        }
    }
}
#endif