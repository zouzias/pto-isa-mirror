/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mte.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_MTE__H
#define TILEOP_TILE_OPERATOR_MTE__H
#include "../utils/layout.h"
#include "../utils/tile_tensor.h"

#define OP_TILE_OP_UB_COPY_IN TLoad
template <typename T, typename U, typename C>
__aicore__ inline void TLoad(T dst, U src, C coordinate) {
    if constexpr (T::FORMAT == Hardware::UB && U::FORMAT == Hardware::GM) {
        const auto srcLayout = src.GetLayout();
        auto srcStride0 = srcLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
        auto srcStride1 = srcLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
        auto srcStride2 = srcLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();

        const auto dstLayout = dst.GetLayout();
        auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
        auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
        auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
        auto gmOffset = srcLayout.template GetGmOffset<C, MAX_DIMS>(coordinate);

        if constexpr (TileOp::IsConstContinous<T>() == true) {
            // 对于静态整块场景，将UB合成二维，GM保持五维
            auto srcGlobal = PtoGlobal<U, typename T::Shape, typename U::Stride>(
                src.GetAddr() + gmOffset, dst.GetShape(), src.GetStride())
                                 .Data();
            auto dstTile = PtoTile<T, pto::BLayout::RowMajor, true>().Data();
            pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
            pto::TLOAD(dstTile, srcGlobal);
            return;
        }

        auto dstTile = PtoTile<T>(dst);
        auto srcGlobal = PtoGlobal<T, typename T::Shape, typename U::Stride, true>(dst.GetShape(), src.GetStride());
        for (size_t index0 = 0; index0 < dstShape0; ++index0) {
            for (size_t index1 = 0; index1 < dstShape1; ++index1) {
                for (size_t index2 = 0; index2 < dstShape2; ++index2) {
                    srcGlobal.Assign(
                        src.GetAddr() + gmOffset + index0 * srcStride0 + index1 * srcStride1 + index2 * srcStride2);
                    auto tileOffsets = TileOffset(index0, index1, index2);
                    dstTile.Assign(dst, tileOffsets);
                    pto::TLOAD(dstTile.Data(), srcGlobal.Data());
                }
            }
        }
    }
}

#define OP_TILE_OP_UB_COPY_OUT TStore
template <typename T, typename U, typename C>
__aicore__ inline void TStore(T dst, U src, C coordinate) {
    if constexpr (U::FORMAT == Hardware::UB && T::FORMAT == Hardware::GM) {
        const auto srcLayout = src.GetLayout();
        auto srcShape0 = srcLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
        auto srcShape1 = srcLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
        auto srcShape2 = srcLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
        const auto dstLayout = dst.GetLayout();
        auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
        auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
        auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
        auto gmOffset = dstLayout.template GetGmOffset<C, MAX_DIMS>(coordinate);
        using SrcDtype = std::conditional_t<std::is_same_v<typename U::Type, bool>, uint8_t, typename U::Type>;

        if constexpr (TileOp::IsConstContinous<U>() == true) {
            // 对于静态整块场景，将UB合成二维，GM保持五维
            auto dstGlobal = PtoGlobal<T, typename U::Shape, typename T::Stride>(
                dst.GetAddr() + gmOffset, src.GetShape(), dst.GetStride())
                                 .Data();
            auto srctTile = PtoTile<U, pto::BLayout::RowMajor, true>().Data();
            pto::TASSIGN(srctTile, (uint64_t)src.GetAddr());
            pto::TSTORE(dstGlobal, srctTile);
            return;
        }

        auto srctTile = PtoTile<U>(src);
        auto dstGlobal = PtoGlobal<T, typename U::Shape, typename T::Stride, true>(src.GetShape(), dst.GetStride());
        for (size_t index0 = 0; index0 < srcShape0; ++index0) {
            for (size_t index1 = 0; index1 < srcShape1; ++index1) {
                for (size_t index2 = 0; index2 < srcShape2; ++index2) {
                    dstGlobal.Assign(
                        dst.GetAddr() + gmOffset + index0 * dstStride0 + index1 * dstStride1 + index2 * dstStride2);
                    auto tileOffsets = TileOffset(index0, index1, index2);
                    srctTile.Assign(src, tileOffsets);
                    pto::TSTORE(dstGlobal.Data(), srctTile.Data());
                }
            }
        }
    }
}

template <bool copyIn, typename GlobalData, typename TileDefine>
__aicore__ inline void ProcessTransMove(GlobalData globalData, TileDefine ubData) {
    if constexpr (copyIn) {
        pto::TLOAD(ubData, globalData);
    } else {
        pto::TSTORE(globalData, ubData);
    }
}

template <typename GMType, typename UBType, bool copyIn, int tileW>
__aicore__ inline void DoTransMove(size_t *srcShape, size_t gmShape4,
    size_t *gmStride, size_t *ubStride, GMType *gmAddr, size_t gmOffset, uint64_t ubAddr) {
    using GlobalData = pto::GlobalTensor<GMType, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>>;
    for (size_t index0 = 0; index0 < srcShape[0]; ++index0) {
        for (size_t index1 = 0; index1 < srcShape[1]; ++index1) {
            for (size_t index2 = 0; index2 < srcShape[2]; ++index2) {
                for (size_t index3 = 0; index3 < srcShape[3]; ++index3) {
                    GlobalData globalData(gmAddr + gmOffset + index0 * gmStride[0] +
                        index1 * gmStride[1] + index2 * gmStride[2] + index3 * gmStride[3],
                        pto::Shape(1, 1, 1, 1, gmShape4), pto::Stride(0, 0, 0, 0, 0));
                    using TileDefine =
                        pto::Tile<pto::TileType::Vec, UBType, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
                    TileDefine ubData(1, srcShape[4]);
                    auto ubOffset = index0 * ubStride[0] + index1 * ubStride[1] +
                        index2 * ubStride[2] + index3 * ubStride[3];
                    pto::TASSIGN(ubData, ubAddr + ubOffset * sizeof(UBType));
                    ProcessTransMove<copyIn>(globalData, ubData);
                }
            }
        }
    }
}

template <unsigned axis0, unsigned axis1, bool copyIn, typename GM, typename UB, typename C>
__aicore__ inline void CallTransMove(GM gm, UB ub, C coordinate) {
    static_assert(axis0 != 4 && axis1 != 4);
    constexpr auto shapeSize = Std::tuple_size<typename UB::Shape>::value;
    constexpr auto tileW = Std::tuple_element<shapeSize - 1, typename UB::TileShape>::type::value;
    const auto gmLayout = gm.GetLayout();
    size_t gmShape4 = static_cast<size_t>(gmLayout.template GetShapeDim<4, 5>());
    size_t gmStride[] = {
        static_cast<size_t>(gmLayout.template GetStrideDim<0, 5>()),
        static_cast<size_t>(gmLayout.template GetStrideDim<1, 5>()),
        static_cast<size_t>(gmLayout.template GetStrideDim<2, 5>()),
        static_cast<size_t>(gmLayout.template GetStrideDim<3, 5>())
    };
    size_t gmOffset = static_cast<size_t>(gmLayout.template GetGmOffset<C, 5>(coordinate));
    const auto ubLayout = ub.GetLayout();
    size_t srcShape[] = {
        static_cast<size_t>(ubLayout.template GetShapeDim<0, 5>()),
        static_cast<size_t>(ubLayout.template GetShapeDim<1, 5>()),
        static_cast<size_t>(ubLayout.template GetShapeDim<2, 5>()),
        static_cast<size_t>(ubLayout.template GetShapeDim<3, 5>()),
        static_cast<size_t>(ubLayout.template GetShapeDim<4, 5>())
    };
    size_t ubStride[] = {
        static_cast<size_t>(ubLayout.template GetStrideDim<0, 5>()),
        static_cast<size_t>(ubLayout.template GetStrideDim<1, 5>()),
        static_cast<size_t>(ubLayout.template GetStrideDim<2, 5>()),
        static_cast<size_t>(ubLayout.template GetStrideDim<3, 5>())
    };
    auto exchangeAxis = [](size_t *arr) {
        auto tmp = arr[axis0];
        arr[axis0] = arr[axis1];
        arr[axis1] = tmp;
    };
    if constexpr (copyIn) {
        exchangeAxis(ubStride);
        exchangeAxis(srcShape);
    } else {
        exchangeAxis(gmStride);
    }
    using GMType = std::conditional_t<std::is_same_v<typename GM::Type, bool>, uint8_t, typename GM::Type>;
    using UBType = std::conditional_t<std::is_same_v<typename UB::Type, bool>, uint8_t, typename UB::Type>;
    DoTransMove<GMType, UBType, copyIn, tileW>(
        srcShape, gmShape4, gmStride, ubStride, gm.GetAddr(), gmOffset, ub.GetAddr());
}

#define OP_TILE_OP_TRANSPOSE_MOVEIN TTransMoveIn
template <unsigned axis0, unsigned axis1, typename DST, typename SRC, typename C>
__aicore__ inline void TTransMoveIn(DST dst, SRC src, C coordinate) {
    static_assert(DST::FORMAT == Hardware::UB && SRC::FORMAT == Hardware::GM);
    CallTransMove<axis0, axis1, true>(src, dst, coordinate);
}

#define OP_TILE_OP_TRANSPOSE_MOVEOUT TTransMoveOut
template <unsigned axis0, unsigned axis1, typename DST, typename SRC, typename C>
__aicore__ inline void TTransMoveOut(DST dst, SRC src, C coordinate) {
    static_assert(DST::FORMAT == Hardware::GM && SRC::FORMAT == Hardware::UB);
    CallTransMove<axis0, axis1, false>(dst, src, coordinate);
}

#endif