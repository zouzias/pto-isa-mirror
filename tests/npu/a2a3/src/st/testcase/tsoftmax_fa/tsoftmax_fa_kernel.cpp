/*
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"
#include "TSoftmaxFA.h"
#include <iostream>

using namespace std;
using namespace pto;

template <int seq, int headSize, int init>
__global__ AICORE void runSoftmax_dn_fusion( __gm__ aclFloat16 __out__ *x_exp, 
                                   __gm__ float __in__ *input_x, 
                                   __gm__ uint8_t __in__ *bit_mask, 
                                   __gm__ float __out__ *local_max, 
                                   __gm__ float __out__ *local_sum, 
                                   __gm__ float __out__ *new_global_max, 
                                   __gm__ float __out__ *new_global_sum, 
                                   __gm__ float __out__ *exp_max) {

    using GlobalData_F = GlobalTensor<float,Shape<1, 1, 1, headSize, seq>, pto::Stride<1, 1, 1, seq, 1>>;
    using GlobalData_U16 = GlobalTensor<uint16_t, Shape<1, 1, 1, headSize, seq>, pto::Stride<1, 1, 1, seq, 1>>;
    using GlobalData_U8 = GlobalTensor<uint8_t, Shape<1, 1, 1, headSize, seq>, pto::Stride<1, 1, 1, seq, 1>>;
    using GlobalReduce_F = GlobalTensor<float, Shape<1, 1, 1, 1, headSize>, pto::Stride<1, 1, 1, 1, 1>>;

    using TileData_F = Tile<TileType::Vec, float, headSize, seq, BLayout::RowMajor, -1, -1>;
    using TileData_U8 = Tile<TileType::Vec, uint8_t, headSize, seq, BLayout::RowMajor, -1, -1>;
    using TileData_H = Tile<TileType::Vec, half, headSize, seq, BLayout::RowMajor, -1, -1>;
    using ReduceTile_F = Tile<TileType::Vec, float, 1, headSize, BLayout::RowMajor, -1, -1>;
    TileData_F src0Tile(headSize, seq);
    TileData_U8 src1Tile(headSize, seq);
    ReduceTile_F dst0Tile(1, headSize);
    ReduceTile_F dst1Tile(1, headSize);
    ReduceTile_F dst2Tile(1, headSize);
    ReduceTile_F dst3Tile(1, headSize);
    ReduceTile_F dst4Tile(1, headSize);
    TileData_H dst5Tile(headSize, seq);

    constexpr std::size_t float_tile_bytes = static_cast<std::size_t>(headSize) * static_cast<std::size_t>(seq) * sizeof(float);
    constexpr std::size_t U16_tile_bytes = static_cast<std::size_t>(headSize) * static_cast<std::size_t>(seq) * sizeof(uint16_t);
    constexpr std::size_t full_tile_bytes = static_cast<std::size_t>(headSize) * static_cast<std::size_t>(seq) * sizeof(float);
    constexpr std::size_t reduce_tile_bytes = static_cast<std::size_t>(headSize) * sizeof(float);
    constexpr std::size_t UB_OFFSET_LIMIT = 0x3FFFFULL;
    static_assert(full_tile_bytes > 0 && reduce_tile_bytes > 0, "tile byte sizes must be > 0");

    constexpr std::size_t off0b = 0;//dst0Tile
    constexpr std::size_t off1b = off0b + reduce_tile_bytes;//dst1Tile
    constexpr std::size_t off2b = off1b + reduce_tile_bytes;//dst2Tile
    constexpr std::size_t off3b = off2b + reduce_tile_bytes;//dst3Tile
    constexpr std::size_t off4b = off3b + reduce_tile_bytes;//dst4Tile

    constexpr std::size_t off5b = off4b + reduce_tile_bytes;//src0Tile
    constexpr std::size_t off6b = off5b + float_tile_bytes;//src1Tile
    constexpr std::size_t off7b = off6b + U16_tile_bytes;//dst5Tile

    constexpr std::size_t last_tile_end = off7b - 1ULL;
    static_assert(last_tile_end <= UB_OFFSET_LIMIT, "UB offset overflow: outTile end exceeds 0x3FFFF");

    constexpr uint32_t off0 = static_cast<uint32_t>(off0b);
    constexpr uint32_t off1 = static_cast<uint32_t>(off1b);
    constexpr uint32_t off2 = static_cast<uint32_t>(off2b);
    constexpr uint32_t off3 = static_cast<uint32_t>(off3b);
    constexpr uint32_t off4 = static_cast<uint32_t>(off4b);
    constexpr uint32_t off5 = static_cast<uint32_t>(off5b);
    constexpr uint32_t off6 = static_cast<uint32_t>(off6b);
    constexpr uint32_t off7 = static_cast<uint32_t>(off7b);

    TASSIGN(dst0Tile, off0);
    TASSIGN(dst1Tile, off1);
    TASSIGN(dst2Tile, off2);
    TASSIGN(dst3Tile, off3);
    TASSIGN(dst4Tile, off4);

    TASSIGN(src0Tile, off5);
    TASSIGN(src1Tile, off6);
    TASSIGN(dst5Tile, off7);

    GlobalData_F src0Global(input_x);
    GlobalData_U8 src1Global(bit_mask);
    GlobalReduce_F dst0Global(local_max);
    GlobalReduce_F dst1Global(local_sum);
    GlobalReduce_F dst2Global(new_global_max);
    GlobalReduce_F dst3Global(new_global_sum);
    GlobalReduce_F dst4Global(exp_max);
    GlobalData_U16 dst5Global(x_exp);

    TLOAD(src0Tile, src0Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TLOAD(dst2Tile, dst2Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TLOAD(dst3Tile, dst3Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // TSOFTMAX_DN_FUSION<ReduceTile_F, TileData_H, TileData_F, TileData_U8, init>(dst5Tile, src0Tile, src1Tile, dst0Tile, dst1Tile, dst2Tile, dst3Tile, dst4Tile);
    // TSOFTMAX_DN_FUSION2<ReduceTile_F, TileData_H, TileData_F, TileData_U8, init>(dst5Tile, src0Tile, src1Tile, dst0Tile, dst1Tile, dst2Tile, dst3Tile, dst4Tile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst0Global, dst0Tile);
    local_max = dst0Global.data();
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst1Global, dst1Tile);
    local_sum = dst1Global.data();
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst2Global, dst2Tile);
    new_global_max = dst2Global.data();
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst3Global, dst3Tile);
    new_global_sum = dst3Global.data();
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst4Global, dst4Tile);
    exp_max = dst4Global.data();
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst5Global, dst5Tile);
    x_exp = dst5Global.data();
}


template <int seq, int headSize, int init>
__global__ AICORE void runSoftmax_dn_nofusion( __gm__ aclFloat16 __out__ *x_exp, 
                                   __gm__ float __in__ *input_x, 
                                   __gm__ uint8_t __in__ *bit_mask, 
                                   __gm__ float __out__ *local_max, 
                                   __gm__ float __out__ *local_sum, 
                                   __gm__ float __out__ *new_global_max, 
                                   __gm__ float __out__ *new_global_sum, 
                                   __gm__ float __out__ *exp_max) {

    using GlobalData_F = GlobalTensor<float,Shape<1, 1, 1, seq, headSize>, pto::Stride<1, 1, 1, headSize, 1>>;
    using GlobalData_U16 = GlobalTensor<uint16_t, Shape<1, 1, 1, seq, headSize>, pto::Stride<1, 1, 1, headSize, 1>>;
    using GlobalData_U8 = GlobalTensor<uint8_t, Shape<1, 1, 1, seq, headSize>, pto::Stride<1, 1, 1, headSize, 1>>;
    using GlobalReduce_F = GlobalTensor<float, Shape<1, 1, 1, 1, headSize>, pto::Stride<1, 1, 1, 1, 1>>;
    using GlobalData_NZ_U16 = GlobalTensor<uint16_t, Shape<1, 1, 1, seq, headSize>, pto::Stride<1, 1, 1, headSize, 1>, Layout::NZ>;

    using TileData_F = Tile<TileType::Vec, float, seq, headSize, BLayout::RowMajor, -1, -1>;
    using TileData_U8 = Tile<TileType::Vec, uint8_t, headSize, seq, BLayout::RowMajor, -1, -1>;
    using TileData_H = Tile<TileType::Vec, half, seq, headSize, BLayout::RowMajor, -1, -1>;
    using ReduceTile_F = Tile<TileType::Vec, float, 1, headSize, BLayout::RowMajor, -1, -1>;
    // using TileData_NZ_H = Tile<TileType::Vec, half, seq, headSize, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    using TileData_NZ_H = Tile<TileType::Vec, half, seq+1, headSize, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    TileData_F src0Tile(seq, headSize);
    TileData_U8 src1Tile(headSize, seq);
    ReduceTile_F dst0Tile(1, headSize);
    ReduceTile_F dst1Tile(1, headSize);
    ReduceTile_F dst2Tile(1, headSize);
    ReduceTile_F dst3Tile(1, headSize);
    ReduceTile_F dst4Tile(1, headSize);
    TileData_H dst5Tile(seq, headSize);
    TileData_F input_expand_tmp(seq, headSize);
    ReduceTile_F input_reduce_tmp1(1, headSize);
    ReduceTile_F input_reduce_tmp2(1, headSize);
    // TileData_NZ_H dst6Tile(seq, headSize);
    TileData_NZ_H dst6Tile(seq+1, headSize);

    constexpr std::size_t float_tile_bytes = static_cast<std::size_t>(headSize) * static_cast<std::size_t>(seq) * sizeof(float);
    constexpr std::size_t U16_tile_bytes = static_cast<std::size_t>(headSize) * static_cast<std::size_t>(seq) * sizeof(uint16_t);
    constexpr std::size_t full_tile_bytes = static_cast<std::size_t>(headSize) * static_cast<std::size_t>(seq) * sizeof(float);
    constexpr std::size_t reduce_tile_bytes = static_cast<std::size_t>(headSize) * sizeof(float);
    constexpr std::size_t UB_OFFSET_LIMIT = 0x3FFFFULL;
    static_assert(full_tile_bytes > 0 && reduce_tile_bytes > 0, "tile byte sizes must be > 0");

    constexpr std::size_t off0b = 0;//dst0Tile
    constexpr std::size_t off1b = off0b + reduce_tile_bytes;//dst1Tile
    constexpr std::size_t off2b = off1b + reduce_tile_bytes;//dst2Tile
    constexpr std::size_t off3b = off2b + reduce_tile_bytes;//dst3Tile
    constexpr std::size_t off4b = off3b + reduce_tile_bytes;//dst4Tile

    constexpr std::size_t off5b = off4b + reduce_tile_bytes;//src0Tile
    constexpr std::size_t off6b = off5b + float_tile_bytes;//src1Tile
    constexpr std::size_t off7b = off6b + U16_tile_bytes;//dst5Tile
    constexpr std::size_t off8b = off7b + float_tile_bytes;//tmp0Tile
    constexpr std::size_t off9b = off8b + reduce_tile_bytes;//tmp1Tile
    constexpr std::size_t off10b = off9b + reduce_tile_bytes;//tmp1Tile

    constexpr std::size_t last_tile_end = off10b - 1ULL;
    static_assert(last_tile_end <= UB_OFFSET_LIMIT, "UB offset overflow: outTile end exceeds 0x3FFFF");

    constexpr uint32_t off0 = static_cast<uint32_t>(off0b);
    constexpr uint32_t off1 = static_cast<uint32_t>(off1b);
    constexpr uint32_t off2 = static_cast<uint32_t>(off2b);
    constexpr uint32_t off3 = static_cast<uint32_t>(off3b);
    constexpr uint32_t off4 = static_cast<uint32_t>(off4b);
    constexpr uint32_t off5 = static_cast<uint32_t>(off5b);
    constexpr uint32_t off6 = static_cast<uint32_t>(off6b);
    constexpr uint32_t off7 = static_cast<uint32_t>(off7b);
    constexpr uint32_t off8 = static_cast<uint32_t>(off8b);
    constexpr uint32_t off9 = static_cast<uint32_t>(off9b);
    constexpr uint32_t off10 = static_cast<uint32_t>(off10b);

    TASSIGN(dst0Tile, off0);
    TASSIGN(dst1Tile, off1);
    TASSIGN(dst2Tile, off2);
    TASSIGN(dst3Tile, off3);
    TASSIGN(dst4Tile, off4);

    TASSIGN(src0Tile, off5);
    TASSIGN(src1Tile, off6);
    TASSIGN(dst5Tile, off7);
    TASSIGN(input_expand_tmp, off8);
    TASSIGN(input_reduce_tmp1, off9);
    TASSIGN(input_reduce_tmp2, off10);

    GlobalData_F src0Global(input_x);
    GlobalData_U8 src1Global(bit_mask);
    GlobalReduce_F dst0Global(local_max);
    GlobalReduce_F dst1Global(local_sum);
    GlobalReduce_F dst2Global(new_global_max);
    GlobalReduce_F dst3Global(new_global_sum);
    GlobalReduce_F dst4Global(exp_max);
    GlobalData_U16 dst5Global(x_exp);
    GlobalData_NZ_U16 dst6Global(x_exp);

    TLOAD(src0Tile, src0Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TLOAD(dst2Tile, dst2Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TLOAD(dst3Tile, dst3Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TSOFTMAX_DN_NOFUSION<ReduceTile_F, TileData_H, TileData_F, TileData_U8, init>(dst5Tile, src0Tile, src1Tile, dst0Tile, dst1Tile, dst2Tile, dst3Tile, dst4Tile, input_expand_tmp, input_reduce_tmp1, input_reduce_tmp2);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst0Global, dst0Tile);
    local_max = dst0Global.data();
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst1Global, dst1Tile);
    local_sum = dst1Global.data();
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst2Global, dst2Tile);
    new_global_max = dst2Global.data();
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst3Global, dst3Tile);
    new_global_sum = dst3Global.data();
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst4Global, dst4Tile);
    exp_max = dst4Global.data();
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst5Global, dst5Tile);
    // TMOV(dst6Tile, dst5Tile);
    // TSTORE(dst6Global, dst6Tile);
    x_exp = dst5Global.data();
}


template <int seq, int headSize, int init>
__global__ AICORE void runSoftmax_nd_fusion( __gm__ aclFloat16 __out__ *x_exp, 
                                   __gm__ float __in__ *input_x, 
                                   __gm__ uint8_t __in__ *bit_mask, 
                                   __gm__ float __out__ *local_max, 
                                   __gm__ float __out__ *local_sum, 
                                   __gm__ float __out__ *new_global_max, 
                                   __gm__ float __out__ *new_global_sum, 
                                   __gm__ float __out__ *exp_max) {

    //suppose to be 64*256
    using GlobalData_F = GlobalTensor<float, Shape<1, 1, 1, headSize, seq>, pto::Stride<1, 1, 1, seq, 1>>;
    using GlobalData_U16 = GlobalTensor<uint16_t, Shape<1, 1, 1, headSize, seq>, pto::Stride<1, 1, 1, seq, 1>>;
    using GlobalData_U8 = GlobalTensor<uint8_t, Shape<1, 1, 1, headSize, seq>, pto::Stride<1, 1, 1, seq, 1>>;
    using GlobalReduce_F = GlobalTensor<float, Shape<1, 1, 1, headSize, 1>, pto::Stride<1, 1, 1, 1, 1>, Layout::DN>;

    using TileData_F = Tile<TileType::Vec, float, headSize, seq, BLayout::RowMajor, -1, -1>;
    using TileData_U8 = Tile<TileType::Vec, uint8_t, headSize, seq, BLayout::RowMajor, -1, -1>;
    using TileData_H = Tile<TileType::Vec, half, headSize, seq, BLayout::RowMajor, -1, -1>;
    using ReduceTileF_T = Tile<TileType::Vec, float, headSize, 1, BLayout::ColMajor, -1, -1>;
    using ReduceTile_F = Tile<TileType::Vec, float, 1, headSize, BLayout::RowMajor, -1, -1>;
    TileData_F src0Tile(headSize, seq);
    TileData_U8 src1Tile(headSize, seq);
    ReduceTileF_T dst0Tile(headSize, 1);
    ReduceTileF_T dst1Tile(headSize, 1);
    ReduceTileF_T dst2Tile(headSize, 1);
    ReduceTileF_T dst3Tile(headSize, 1);
    ReduceTileF_T dst4Tile(headSize, 1);
    TileData_H dst5Tile(headSize, seq);

    constexpr std::size_t float_tile_bytes = static_cast<std::size_t>(headSize) * static_cast<std::size_t>(seq) * sizeof(float);
    constexpr std::size_t U16_tile_bytes = static_cast<std::size_t>(headSize) * static_cast<std::size_t>(seq) * sizeof(uint16_t);
    constexpr std::size_t full_tile_bytes = static_cast<std::size_t>(headSize) * static_cast<std::size_t>(seq) * sizeof(float);
    constexpr std::size_t reduce_tile_bytes = static_cast<std::size_t>(headSize) * sizeof(float);
    constexpr std::size_t UB_OFFSET_LIMIT = 0x3FFFFULL;
    static_assert(full_tile_bytes > 0 && reduce_tile_bytes > 0, "tile byte sizes must be > 0");

    constexpr std::size_t off0b = 0;//dst0Tile
    constexpr std::size_t off1b = off0b + reduce_tile_bytes;//dst1Tile
    constexpr std::size_t off2b = off1b + reduce_tile_bytes;//dst2Tile
    constexpr std::size_t off3b = off2b + reduce_tile_bytes;//dst3Tile
    constexpr std::size_t off4b = off3b + reduce_tile_bytes;//dst4Tile

    constexpr std::size_t off5b = off4b + reduce_tile_bytes;//src0Tile
    constexpr std::size_t off6b = off5b + float_tile_bytes;//src1Tile
    constexpr std::size_t off7b = off6b + U16_tile_bytes;//dst5Tile

    constexpr std::size_t last_tile_end = off7b - 1ULL;
    static_assert(last_tile_end <= UB_OFFSET_LIMIT, "UB offset overflow: outTile end exceeds 0x3FFFF");

    constexpr uint32_t off0 = static_cast<uint32_t>(off0b);  //0
    constexpr uint32_t off1 = static_cast<uint32_t>(off1b);  //100
    constexpr uint32_t off2 = static_cast<uint32_t>(off2b);  //200
    constexpr uint32_t off3 = static_cast<uint32_t>(off3b);  //300
    constexpr uint32_t off4 = static_cast<uint32_t>(off4b);  //400
    constexpr uint32_t off5 = static_cast<uint32_t>(off5b);  //500
    constexpr uint32_t off6 = static_cast<uint32_t>(off6b);  //8500
    constexpr uint32_t off7 = static_cast<uint32_t>(off7b);

    TASSIGN(dst0Tile, off0);
    TASSIGN(dst1Tile, off1);
    TASSIGN(dst2Tile, off2);
    TASSIGN(dst3Tile, off3);
    TASSIGN(dst4Tile, off4);

    TASSIGN(src0Tile, off5);
    TASSIGN(src1Tile, off6);
    TASSIGN(dst5Tile, off7);

    GlobalData_F src0Global(input_x);
    GlobalData_U8 src1Global(bit_mask);
    GlobalReduce_F dst0Global(local_max);
    GlobalReduce_F dst1Global(local_sum);
    GlobalReduce_F dst2Global(new_global_max);
    GlobalReduce_F dst3Global(new_global_sum);
    GlobalReduce_F dst4Global(exp_max);
    GlobalData_U16 dst5Global(x_exp);

    TLOAD(src0Tile, src0Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TLOAD(dst2Tile, dst2Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TLOAD(dst3Tile, dst3Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // TSOFTMAX_ND_FUSION<ReduceTileF_T, TileData_H, TileData_F, TileData_U8, init>(dst5Tile, src0Tile, src1Tile, dst0Tile, dst1Tile, dst2Tile, dst3Tile, dst4Tile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst0Global, dst0Tile);
    local_max = dst0Global.data();
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst1Global, dst1Tile);
    local_sum = dst1Global.data();
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst2Global, dst2Tile);
    new_global_max = dst2Global.data();
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst3Global, dst3Tile);
    new_global_sum = dst3Global.data();
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst4Global, dst4Tile);
    exp_max = dst4Global.data();
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst5Global, dst5Tile);
    x_exp = dst5Global.data();
}

template <int seq, int headSize, int init>
__global__ AICORE void runSoftmax_nd_nofusion( __gm__ aclFloat16 __out__ *x_exp, 
                                   __gm__ float __in__ *input_x, 
                                   __gm__ uint8_t __in__ *bit_mask, 
                                   __gm__ float __out__ *local_max, 
                                   __gm__ float __out__ *local_sum, 
                                   __gm__ float __out__ *new_global_max, 
                                   __gm__ float __out__ *new_global_sum, 
                                   __gm__ float __out__ *exp_max) {

    using GlobalData_F = GlobalTensor<float, Shape<1, 1, 1, headSize, seq>, pto::Stride<1, 1, 1, seq, 1>>;
    using GlobalData_U16 = GlobalTensor<uint16_t, Shape<1, 1, 1, headSize, seq>, pto::Stride<1, 1, 1, seq, 1>>;
    using GlobalData_U8 = GlobalTensor<uint8_t, Shape<1, 1, 1, headSize, seq>, pto::Stride<1, 1, 1, seq, 1>>;
    using GlobalReduce_F = GlobalTensor<float, Shape<1, 1, 1, headSize, 1>, pto::Stride<1, 1, 1, 1, 1>, Layout::DN>;    //ColMajor Ouput

    using TileData_F = Tile<TileType::Vec, float, headSize, seq, BLayout::RowMajor, -1, -1>;
    using TileData_U8 = Tile<TileType::Vec, uint8_t, headSize, seq, BLayout::RowMajor, -1, -1>;
    using TileData_H = Tile<TileType::Vec, half, headSize, seq, BLayout::RowMajor, -1, -1>;
    using ReduceTileF_T = Tile<TileType::Vec, float, headSize, 1, BLayout::ColMajor, -1, -1>;
    TileData_F src0Tile(headSize, seq);
    TileData_U8 src1Tile(headSize, seq);
    ReduceTileF_T dst0Tile(headSize, 1);
    ReduceTileF_T dst1Tile(headSize, 1);
    ReduceTileF_T dst2Tile(headSize, 1);
    ReduceTileF_T dst3Tile(headSize, 1);
    ReduceTileF_T dst4Tile(headSize,1);
    TileData_H dst5Tile(headSize, seq);
    TileData_F input_reduce_tmp(headSize, seq);

    constexpr std::size_t float_tile_bytes = static_cast<std::size_t>(headSize) * static_cast<std::size_t>(seq) * sizeof(float);
    constexpr std::size_t U16_tile_bytes = static_cast<std::size_t>(headSize) * static_cast<std::size_t>(seq) * sizeof(uint16_t);
    constexpr std::size_t full_tile_bytes = static_cast<std::size_t>(headSize) * static_cast<std::size_t>(seq) * sizeof(float);
    constexpr std::size_t reduce_tile_bytes = static_cast<std::size_t>(headSize) * sizeof(float);
    constexpr std::size_t UB_OFFSET_LIMIT = 0x3FFFFULL;
    static_assert(full_tile_bytes > 0 && reduce_tile_bytes > 0, "tile byte sizes must be > 0");

    constexpr std::size_t off0b = 0;//dst0Tile
    constexpr std::size_t off1b = off0b + reduce_tile_bytes;//dst1Tile
    constexpr std::size_t off2b = off1b + reduce_tile_bytes;//dst2Tile
    constexpr std::size_t off3b = off2b + reduce_tile_bytes;//dst3Tile
    constexpr std::size_t off4b = off3b + reduce_tile_bytes;//dst4Tile

    constexpr std::size_t off5b = off4b + reduce_tile_bytes;//src0Tile
    constexpr std::size_t off6b = off5b + float_tile_bytes;//src1Tile
    constexpr std::size_t off7b = off6b + U16_tile_bytes;//dst5Tile
    constexpr std::size_t off8b = off7b + float_tile_bytes;//tmp0Tile

    constexpr std::size_t last_tile_end = off8b - 1ULL;
    static_assert(last_tile_end <= UB_OFFSET_LIMIT, "UB offset overflow: outTile end exceeds 0x3FFFF");

    constexpr uint32_t off0 = static_cast<uint32_t>(off0b);
    constexpr uint32_t off1 = static_cast<uint32_t>(off1b);
    constexpr uint32_t off2 = static_cast<uint32_t>(off2b);
    constexpr uint32_t off3 = static_cast<uint32_t>(off3b);
    constexpr uint32_t off4 = static_cast<uint32_t>(off4b);
    constexpr uint32_t off5 = static_cast<uint32_t>(off5b);
    constexpr uint32_t off6 = static_cast<uint32_t>(off6b);
    constexpr uint32_t off7 = static_cast<uint32_t>(off7b);
    constexpr uint32_t off8 = static_cast<uint32_t>(off8b);

    TASSIGN(dst0Tile, off0);
    TASSIGN(dst1Tile, off1);
    TASSIGN(dst2Tile, off2);
    TASSIGN(dst3Tile, off3);
    TASSIGN(dst4Tile, off4);

    TASSIGN(src0Tile, off5);
    TASSIGN(src1Tile, off6);
    TASSIGN(dst5Tile, off7);
    TASSIGN(input_reduce_tmp, off8);

    GlobalData_F src0Global(input_x);
    GlobalData_U8 src1Global(bit_mask);
    GlobalReduce_F dst0Global(local_max);
    GlobalReduce_F dst1Global(local_sum);
    GlobalReduce_F dst2Global(new_global_max);
    GlobalReduce_F dst3Global(new_global_sum);
    GlobalReduce_F dst4Global(exp_max);
    GlobalData_U16 dst5Global(x_exp);

    TLOAD(src0Tile, src0Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TLOAD(dst2Tile, dst2Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TLOAD(dst3Tile, dst3Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TSOFTMAX_ND_NOFUSION<ReduceTileF_T, TileData_H, TileData_F, TileData_U8, init>(dst5Tile, src0Tile, src1Tile, dst0Tile, dst1Tile, dst2Tile, dst3Tile, dst4Tile, input_reduce_tmp, src0Tile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst0Global, dst0Tile);
    local_max = dst0Global.data();
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst1Global, dst1Tile);
    local_sum = dst1Global.data();
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst2Global, dst2Tile);
    new_global_max = dst2Global.data();
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst3Global, dst3Tile);
    new_global_sum = dst3Global.data();
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst4Global, dst4Tile);
    exp_max = dst4Global.data();
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst5Global, dst5Tile);
    x_exp = dst5Global.data();
}


template <int seq, int headSize, int init>
void launchTSOFTMAX_dn_fusion(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream){
    cout << "launch softmax fusion start!" << endl;
    runSoftmax_dn_fusion<seq, headSize, init><<<1, nullptr, stream>>>(x_exp, input_x, bit_mask, local_max, local_sum, new_global_max, new_global_sum, exp_max);
    cout << "launch softmax fusion end!" << endl;
}

template void launchTSOFTMAX_dn_fusion<64, 64, 0>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_dn_fusion<64, 64, 1>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_dn_fusion<128, 64, 0>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_dn_fusion<128, 64, 1>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_dn_fusion<256, 64, 0>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_dn_fusion<256, 64, 1>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);


template <int seq, int headSize, int init>
void launchTSOFTMAX_dn_nofusion(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream){
    cout << "launch softmax DN nofusion start!" << endl;
    runSoftmax_dn_nofusion<seq, headSize, init><<<1, nullptr, stream>>>(x_exp, input_x, bit_mask, local_max, local_sum, new_global_max, new_global_sum, exp_max);
    cout << "launch softmax DN nofusion end!" << endl;
}

template void launchTSOFTMAX_dn_nofusion<512, 32, 1>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_dn_nofusion<512, 32, 0>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_dn_nofusion<1024, 16, 1>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_dn_nofusion<1024, 16, 0>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_dn_nofusion<256, 64, 1>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_dn_nofusion<256, 64, 0>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_dn_nofusion<128, 64, 1>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_dn_nofusion<128, 64, 0>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_dn_nofusion<128, 128, 1>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_dn_nofusion<128, 128, 0>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);

template <int seq, int headSize, int init>
void launchTSOFTMAX_nd_fusion(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream){
    cout << "launch softmax ND fusion start!" << endl;
    runSoftmax_nd_fusion<seq, headSize, init><<<1, nullptr, stream>>>(x_exp, input_x, bit_mask, local_max, local_sum, new_global_max, new_global_sum, exp_max);
    cout << "launch softmax ND fusion end!" << endl;
}

template void launchTSOFTMAX_nd_fusion<64, 64, 1>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_nd_fusion<64, 64, 0>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_nd_fusion<128, 64, 1>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_nd_fusion<128, 64, 0>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_nd_fusion<256, 64, 1>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_nd_fusion<256, 64, 0>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_nd_fusion<64, 128, 1>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_nd_fusion<64, 128, 0>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_nd_fusion<128, 128, 1>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_nd_fusion<128, 128, 0>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);

template <int seq, int headSize, int init>
void launchTSOFTMAX_nd_nofusion(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream){
    cout << "launch softmax ND nofusion start!" << endl;
    runSoftmax_nd_nofusion<seq, headSize, init><<<1, nullptr, stream>>>(x_exp, input_x, bit_mask, local_max, local_sum, new_global_max, new_global_sum, exp_max);
    cout << "launch softmax ND nofusion end!" << endl;
}

template void launchTSOFTMAX_nd_nofusion<512, 32, 1>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_nd_nofusion<512, 32, 0>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_nd_nofusion<1024, 16, 1>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_nd_nofusion<1024, 16, 0>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_nd_nofusion<64, 64, 1>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_nd_nofusion<64, 64, 0>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_nd_nofusion<128, 64, 1>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_nd_nofusion<128, 64, 0>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_nd_nofusion<256, 64, 1>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_nd_nofusion<256, 64, 0>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_nd_nofusion<64, 128, 1>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_nd_nofusion<64, 128, 0>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_nd_nofusion<128, 128, 1>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);
template void launchTSOFTMAX_nd_nofusion<128, 128, 0>(aclFloat16 *x_exp, float *input_x, uint8_t *bit_mask, float *local_max, float *local_sum, float *new_global_max, float *new_global_sum, float *exp_max, aclrtStream stream);