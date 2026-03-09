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
 * \file indexadd.h
 * \brief
 */
#ifndef TILEOP_TILE_OPERATOR_INDEXADD_H
#define TILEOP_TILE_OPERATOR_INDEXADD_H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <typename T0, typename T2, typename T3, typename dstTileDefine, typename tempTileDefine,
    typename src1TileDefine, typename Scalar>
TILEOP void IndexAddNotLastAxisCompute(dstTileDefine dstTile, tempTileDefine tempTile, src1TileDefine src1Tile,
    Scalar alpha, __ubuf__ typename T0::Type *dstAddr, __ubuf__ bfloat16_t *tempAddr,
    __ubuf__ typename T2::Type *src1Addr, size_t dstOffset, size_t src1Offset) {
    pto::TASSIGN(dstTile, (uint64_t)(dstAddr + dstOffset));
    pto::TASSIGN(src1Tile, (uint64_t)(src1Addr + src1Offset));

    if constexpr (Std::is_same_v<Scalar, bfloat16_t>) {
        pto::TASSIGN(tempTile, (uint64_t)(tempAddr));
        set_flag(PIPE_S, PIPE_V, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
        if (abs(static_cast<float>(alpha) - 1) > TileOp::EPSILON) {
            pto::TMULS(src1Tile, src1Tile, alpha);
            #ifdef __DAV_V220
            pipe_barrier(PIPE_V);
            #endif
            pto::TCVT(tempTile, src1Tile, pto::RoundMode::CAST_RINT); //fp32->bf16
            #ifdef __DAV_V220
            pipe_barrier(PIPE_V);
            #endif
            pto::TCVT(src1Tile, tempTile, pto::RoundMode::CAST_NONE); //bf16->fp32
            #ifdef __DAV_V220
            pipe_barrier(PIPE_V);
            #endif
        }
        pto::TADD(dstTile, dstTile, src1Tile);
        // 当alpha不为1或index为int32类型时，需要在每一步运算后转换为bf16类型
        if (Std::is_same_v<typename T3::Type, int32_t> || abs(static_cast<float>(alpha) - 1) > TileOp::EPSILON) {
            #ifdef __DAV_V220
            pipe_barrier(PIPE_V);
            #endif
            pto::TCVT(tempTile, dstTile, pto::RoundMode::CAST_RINT); // fp32->bf16
            #ifdef __DAV_V220
            pipe_barrier(PIPE_V);
            #endif
            pto::TCVT(dstTile, tempTile, pto::RoundMode::CAST_NONE); // bf16->fp32
        }
    } else {
        set_flag(PIPE_S, PIPE_V, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
        if (abs(static_cast<float>(alpha) - 1) > TileOp::EPSILON) {
            pto::TMULS(src1Tile, src1Tile, alpha);
            #ifdef __DAV_V220
            pipe_barrier(PIPE_V);
            #endif
        }
        pto::TADD(dstTile, dstTile, src1Tile);
    }
}

template <typename T0, typename T2, typename T3, typename Scalar>
TILEOP void IndexAddLastAxisCompute(T0 dst, T2 src1, T3 src2, Scalar alpha, size_t src1Shape0, size_t src1Shape1,
    size_t src1Shape2, size_t src1Shape3, size_t src1Shape4, size_t dstStride0, size_t dstStride1, size_t dstStride2,
    size_t dstStride3, size_t src1Stride0, size_t src1Stride1, size_t src1Stride2, size_t src1Stride3) {
    auto dstAddr = (__ubuf__ typename T0::Type *)((uint64_t)(dst.GetAddr()));
    auto src1Addr = (__ubuf__ typename T2::Type *)((uint64_t)(src1.GetAddr()));
    auto idxAddr = (__ubuf__ typename T3::Type *)((uint64_t)(src2.GetAddr()));
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    uint64_t dstOffset = 0;
    uint64_t src1Offset = 0;
    if (abs(static_cast<float>(alpha) - 1) > TileOp::EPSILON) {
        for (LoopVar i = 0; i < src1Shape0; ++i) {
            for (LoopVar j = 0; j < src1Shape1; ++j) {
                for (LoopVar k = 0; k < src1Shape2; ++k) {
                    for (LoopVar l = 0; l < src1Shape3; ++l) {
                        for (LoopVar idx = 0; idx < src1Shape4; ++idx) {
                            auto index = *(idxAddr + idx);
                            dstOffset = i * dstStride0 + j * dstStride1 + k * dstStride2 + l * dstStride3 + index;
                            src1Offset =
                                i * src1Stride0 + j * src1Stride1 + k * src1Stride2 + l * src1Stride3 + idx;
                            if constexpr (Std::is_same_v<Scalar, half>) { // half
                                float mulsResult =
                                    static_cast<float>(src1Addr[src1Offset]) * static_cast<float>(alpha);
                                src1Addr[src1Offset] = static_cast<half>(mulsResult);
                            } else if constexpr (Std::is_same_v<Scalar, bfloat16_t>) { // bf16
                                float mulsResult = src1Addr[src1Offset] * TileOp::Bf16ToFp32(alpha);
                                bfloat16_t mulsResBf16 = TileOp::Fp32ToBf16R(mulsResult);
                                src1Addr[src1Offset] = TileOp::Bf16ToFp32(mulsResBf16);
                            } else { // int8,int16,int32,float32
                                Scalar mulsResult = static_cast<Scalar>(src1Addr[src1Offset]) * alpha;
                                src1Addr[src1Offset] = static_cast<typename T2::Type>(mulsResult);
                            }
                        }
                    }
                }
            }
        }
    }
    for (LoopVar i = 0; i < src1Shape0; ++i) {
        for (LoopVar j = 0; j < src1Shape1; ++j) {
            for (LoopVar k = 0; k < src1Shape2; ++k) {
                for (LoopVar l = 0; l < src1Shape3; ++l) {
                    for (LoopVar idx = 0; idx < src1Shape4; ++idx) {
                        auto index = *(idxAddr + idx);
                        dstOffset = i * dstStride0 + j * dstStride1 + k * dstStride2 + l * dstStride3 + index;
                        src1Offset = i * src1Stride0 + j * src1Stride1 + k * src1Stride2 + l * src1Stride3 + idx;
                        if constexpr (Std::is_same_v<Scalar, half>) {
                            float addResult =
                                static_cast<float>(dstAddr[dstOffset]) + static_cast<float>(src1Addr[src1Offset]);
                            if (abs(static_cast<float>(alpha) - 1) < TileOp::EPSILON &&
                                Std::is_same_v<typename T3::Type, int64_t>) {
                                dstAddr[dstOffset] = addResult; // 不需要转换
                            } else {
                                dstAddr[dstOffset] = static_cast<half>(addResult);
                            }
                        } else if constexpr (Std::is_same_v<Scalar, bfloat16_t>) {
                            float addResult = dstAddr[dstOffset] + src1Addr[src1Offset];
                            if (abs(static_cast<float>(alpha) - 1) < TileOp::EPSILON &&
                                Std::is_same_v<typename T3::Type, int64_t>) {
                                dstAddr[dstOffset] = addResult; // 不需要转换
                            } else {
                                bfloat16_t addResBf16 = TileOp::Fp32ToBf16R(addResult);
                                dstAddr[dstOffset] = TileOp::Bf16ToFp32(addResBf16);
                            }
                        } else { // int8,int16,int32,float32
                            Scalar addResult =
                                static_cast<Scalar>(dstAddr[dstOffset]) + static_cast<Scalar>(src1Addr[src1Offset]);
                            dstAddr[dstOffset] = static_cast<typename T0::Type>(addResult);
                        }
                    }
                }
            }
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

/*
src0:self
src1:source
src2:index
axis是泛化成5维后的值，实际值为 axis + shapeSize - 5
*/
template <int axis, typename T0, typename T1, typename T2, typename T3, typename T4, typename Scalar>
TILEOP void TIndexAdd(T0 dst, T1 src0, T2 src1, T3 src2, T4 tempTensor, Scalar alpha) { // T0: tileTensor
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;              // support 2-5
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>(); // validShape
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto dstStride3 = dstLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();

    const auto src1Layout = src1.GetLayout();
    auto src1Shape0 = src1Layout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto src1Shape1 = src1Layout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto src1Shape2 = src1Layout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto src1Shape3 = src1Layout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto src1Shape4 = src1Layout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    auto src1Stride0 = src1Layout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto src1Stride1 = src1Layout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto src1Stride2 = src1Layout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto src1Stride3 = src1Layout.template GetStrideDim<DIM_4TH, MAX_DIMS>();

    auto dstAddr = (__ubuf__ typename T0::Type *)((uint64_t)(dst.GetAddr()));
    auto tempAddr = (__ubuf__ typename T4::Type *)((uint64_t)(tempTensor.GetAddr()));
    auto src1Addr = (__ubuf__ typename T2::Type *)((uint64_t)(src1.GetAddr()));
    auto idxAddr = (__ubuf__ typename T3::Type *)((uint64_t)(src2.GetAddr()));
    if (!dstShape0 || !dstShape1 || !dstShape2 || !dstShape3 || !dstShape4) {
        return;
    }
    if constexpr (axis == 4) { // 尾轴
        IndexAddLastAxisCompute(dst, src1, src2, alpha, src1Shape0, src1Shape1, src1Shape2, src1Shape3, src1Shape4,
            dstStride0, dstStride1, dstStride2, dstStride3, src1Stride0, src1Stride1, src1Stride2, src1Stride3);
    } else {
        constexpr auto dstTileW =
            TileOp::GetAnyAxisMergeResult<axis + shapeSize - 3, shapeSize, typename T0::TileShape>();
        constexpr auto tempTileW =
            TileOp::GetAnyAxisMergeResult<axis + shapeSize - 3, shapeSize, typename T4::TileShape>();
        constexpr auto src1TileW =
            TileOp::GetAnyAxisMergeResult<axis + shapeSize - 3, shapeSize, typename T2::TileShape>();
        using dstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, dstTileW, pto::BLayout::RowMajor>;
        using tempTileDefine = pto::Tile<pto::TileType::Vec, bfloat16_t, 1, tempTileW, pto::BLayout::RowMajor>;
        using src1TileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, 1, src1TileW, pto::BLayout::RowMajor>;
        dstTileDefine dstTile;
        tempTileDefine tempTile;
        src1TileDefine src1Tile;
        if constexpr (axis == 0) { // 从第2轴开始合轴
            for (LoopVar i = 0; i < src1Shape0; ++i) {
                set_flag(PIPE_V, PIPE_S, EVENT_ID7);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
                auto index = *(idxAddr + i);
                auto dstOffset = index * dstStride0;
                auto src1Offset = i * src1Stride0;
                IndexAddNotLastAxisCompute<T0, T2, T3>(
                    dstTile, tempTile, src1Tile, alpha, dstAddr, tempAddr, src1Addr, dstOffset, src1Offset);
            }
        } else if constexpr (axis == 1) { // 从第3轴开始合轴
            for (LoopVar i = 0; i < src1Shape0; ++i) {
                for (LoopVar j = 0; j < src1Shape1; ++j) {
                    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
                    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
                    auto index = *(idxAddr + j);
                    auto dstOffset = i * dstStride0 + index * dstStride1;
                    auto src1Offset = i * src1Stride0 + j * src1Stride1;
                    IndexAddNotLastAxisCompute<T0, T2, T3>(
                        dstTile, tempTile, src1Tile, alpha, dstAddr, tempAddr, src1Addr, dstOffset, src1Offset);
                }
            }
        } else if constexpr (axis == 2) { // 从第4轴开始合轴
            for (LoopVar i = 0; i < src1Shape0; ++i) {
                for (LoopVar j = 0; j < src1Shape1; ++j) {
                    for (LoopVar k = 0; k < src1Shape2; ++k) {
                        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
                        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
                        auto index = *(idxAddr + k);
                        auto dstOffset = i * dstStride0 + j * dstStride1 + index * dstStride2;
                        auto src1Offset = i * src1Stride0 + j * src1Stride1 + k * src1Stride2;
                        IndexAddNotLastAxisCompute<T0, T2, T3>(
                            dstTile, tempTile, src1Tile, alpha, dstAddr, tempAddr, src1Addr, dstOffset, src1Offset);
                    }
                }
            }
        } else {
            for (LoopVar i = 0; i < src1Shape0; ++i) {
                for (LoopVar j = 0; j < src1Shape1; ++j) {
                    for (LoopVar k = 0; k < src1Shape2; ++k) {
                        for (LoopVar l = 0; l < src1Shape3; ++l) {
                            set_flag(PIPE_V, PIPE_S, EVENT_ID7);
                            wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
                            auto index = *(idxAddr + l);
                            auto dstOffset = i * dstStride0 + j * dstStride1 + k * dstStride2 + index * dstStride3;
                            auto src1Offset = i * src1Stride0 + j * src1Stride1 + k * src1Stride2 + l * src1Stride3;
                            IndexAddNotLastAxisCompute<T0, T2, T3>(
                                dstTile, tempTile, src1Tile, alpha, dstAddr, tempAddr, src1Addr, dstOffset, src1Offset);
                        }
                    }
                }
            }
        }
    }
}
#endif
