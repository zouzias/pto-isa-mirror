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
#include "utils/layout.h"
#include "utils/tile_tensor.h"

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
        for (LoopVar index0 = 0; index0 < dstShape0; ++index0) {
            for (LoopVar index1 = 0; index1 < dstShape1; ++index1) {
                for (LoopVar index2 = 0; index2 < dstShape2; ++index2) {
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
        for (LoopVar index0 = 0; index0 < srcShape0; ++index0) {
            for (LoopVar index1 = 0; index1 < srcShape1; ++index1) {
                for (LoopVar index2 = 0; index2 < srcShape2; ++index2) {
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
    for (LoopVar index0 = 0; index0 < srcShape[0]; ++index0) {
        for (LoopVar index1 = 0; index1 < srcShape[1]; ++index1) {
            for (LoopVar index2 = 0; index2 < srcShape[2]; ++index2) {
                for (LoopVar index3 = 0; index3 < srcShape[3]; ++index3) {
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

template <pto::AtomicType atomicType, typename ValuesDtype, typename ValuesTileDefine, typename ADDR>
__aicore__ inline void IndexPutCopyOut(int nBurst, int lenBurst, size_t valuesStrides[], uint64_t ubAddr, ADDR *dstAddr) {
    using DstData = pto::GlobalTensor<ADDR, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>>;
    constexpr size_t MAX_N_BURST = 4095;
    size_t repeat = nBurst / MAX_N_BURST;
    size_t lastNBurst = nBurst - repeat * MAX_N_BURST;
    ValuesTileDefine valuesData(1, lenBurst);
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
    for (LoopVar i = 0; i < repeat; ++i) {
        pto::TASSIGN(valuesData, ubAddr);
        ubAddr += MAX_N_BURST * valuesStrides[2] * sizeof(ValuesDtype);
        DstData dstData(dstAddr, pto::Shape(1, 1, 1, MAX_N_BURST, lenBurst), pto::Stride(0, 0, 0, lenBurst, 0));
        dstAddr += MAX_N_BURST * lenBurst;
        pto::TSTORE<ValuesTileDefine, DstData, atomicType>(dstData, valuesData);
    }
    if (lastNBurst != 0) {
        pto::TASSIGN(valuesData, ubAddr);
        DstData dstData(dstAddr, pto::Shape(1, 1, 1, lastNBurst, lenBurst), pto::Stride(0, 0, 0, lenBurst, 0));
        pto::TSTORE<ValuesTileDefine, DstData, atomicType>(dstData, valuesData);
    }
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
}

template <size_t current, size_t target = 3>
__aicore__ inline size_t IndexPutGetStride(size_t arr[]) {
    if constexpr (current >= target) {
        return 1;
    } else {
        return arr[current] * IndexPutGetStride<current + 1, target>(arr);
    }
}

template <pto::AtomicType atomicType, size_t dstShapeSize, size_t valuesSize, typename VAL, typename ADDR, typename IDX>
__aicore__ inline void DoIndexPut(size_t indicesShape, size_t dstShapes[], size_t valuesStrides[], uint64_t valuesAddr,
    ADDR *dstAddr, IDX indices0, IDX indices1, IDX indices2, IDX indices3) {
    using IndicesDtype = typename IDX::Type;
    using ValuesDtype = std::conditional_t<std::is_same_v<typename VAL::Type, bool>, uint8_t, typename VAL::Type>;
    constexpr auto tileW = Std::tuple_element<valuesSize - 1, typename VAL::TileShape>::type::value;
    using ValuesTileDefine = pto::Tile<pto::TileType::Vec, ValuesDtype, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
    constexpr size_t indicesSize = dstShapeSize - valuesSize + 1;
    for (LoopVar i = 0; i < indicesShape; ++i) {
        uint64_t dstOffset = 0;
        if constexpr (indicesSize >= 1) {
            dstOffset += ((__ubuf__ IndicesDtype *) indices0.GetAddr())[i] * IndexPutGetStride<4 - dstShapeSize>(dstShapes);
        }
        if constexpr (indicesSize >= 2) {
            dstOffset += ((__ubuf__ IndicesDtype *) indices1.GetAddr())[i] * IndexPutGetStride<5 - dstShapeSize>(dstShapes);
        }
        if constexpr (indicesSize >= 3) {
            dstOffset += ((__ubuf__ IndicesDtype *) indices2.GetAddr())[i] * IndexPutGetStride<6 - dstShapeSize>(dstShapes);
        }
        if constexpr (indicesSize >= 4) {
            dstOffset += ((__ubuf__ IndicesDtype *) indices3.GetAddr())[i] * IndexPutGetStride<7 - dstShapeSize>(dstShapes);
        }
        size_t nBurst = 1;
        size_t lenBurst = dstShapes[2];
        if constexpr (valuesSize == 1) {
            lenBurst = 1;
            ((__ubuf__ ValuesDtype *) valuesAddr)[0] = ((__ubuf__ ValuesDtype *) valuesAddr)[i];
        } else if constexpr (valuesSize == 3) {
            nBurst = dstShapes[1];
        } else if constexpr (valuesSize == 4) {
            nBurst = dstShapes[1] * dstShapes[0];
        }
        IndexPutCopyOut<atomicType, ValuesDtype, ValuesTileDefine>(
            nBurst, lenBurst, valuesStrides,
            valuesAddr + i * valuesStrides[4 - valuesSize] * sizeof(ValuesDtype),
            dstAddr + dstOffset
        );
    }
}

template <bool accumulate, size_t indicesSize, typename DST, typename VAL, typename IDX>
__aicore__ inline void TIndexPut(DST dst, VAL values, IDX indices0, IDX indices1, IDX indices2, IDX indices3) {
    constexpr auto atomicType = accumulate ? pto::AtomicType::AtomicAdd : pto::AtomicType::AtomicNone;
    constexpr auto dstShapeSize = Std::tuple_size<typename DST::Shape>::value;
    constexpr auto valuesSize = Std::tuple_size<typename VAL::Shape>::value;
    static_assert(dstShapeSize >= indicesSize && dstShapeSize <= 4 && dstShapeSize == valuesSize + indicesSize - 1);
    const auto indicesLayout = indices0.GetLayout();
    const auto dstLayout = dst.GetLayout();
    const auto valuesLayout = values.GetLayout();
    auto indicesShape = indicesLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    size_t dstShapes[] = {
        static_cast<size_t>(dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>()),
        static_cast<size_t>(dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>()),
        static_cast<size_t>(dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>())
    };
    size_t valuesStrides[] = {
        static_cast<size_t>(valuesLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>()),
        static_cast<size_t>(valuesLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>()),
        static_cast<size_t>(valuesLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>()),
        static_cast<size_t>(0)
    };
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    DoIndexPut<atomicType, dstShapeSize, valuesSize, VAL>(indicesShape, dstShapes, valuesStrides,
        values.GetAddr(), dst.GetAddr(), indices0, indices1, indices2, indices3);
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
}

#endif