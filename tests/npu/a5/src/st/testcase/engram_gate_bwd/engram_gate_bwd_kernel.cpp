/**
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
#include <pto/common/pto_tile.hpp>
#include "acl/acl.h"

using namespace pto;

template <typename T, int hidden_size, int kTRows_, int kTCols_>
__global__ AICORE void runEngramGateBwd(
    __gm__ T *grad_x,
    __gm__ T *grad_k,
    __gm__ T *grad_v,
    __gm__ float *grad_w,
    __gm__ T *grad_out,
    __gm__ T *x,
    __gm__ T *k,
    __gm__ T *v,
    __gm__ float *w,
    __gm__ float *gate,
    __gm__ float *rstd_x,
    __gm__ float *rstd_k,
    __gm__ float *raw_dot,
    float scalar,
    float clamp_value
)
{
    using DynShape = Shape<1, 1, 1, 1, hidden_size>;
    using DynStride = Stride<hidden_size, hidden_size, hidden_size, hidden_size, 1>;
    using GlobalDataT = GlobalTensor<T, DynShape, DynStride>;
    using GlobalDataF = GlobalTensor<float, DynShape, DynStride>;
    
    GlobalDataT gradOutGlobal(grad_out);
    GlobalDataT xGlobal(x);
    GlobalDataT kGlobal(k);
    GlobalDataT vGlobal(v);
    GlobalDataT gradXGlobal(grad_x);
    GlobalDataT gradKGlobal(grad_k);
    GlobalDataT gradVGlobal(grad_v);
    GlobalDataF wGlobal(w);
    GlobalDataF gradWGlobal(grad_w);
    
    using TileDataF = Tile<TileType::Vec, float, kTRows_, kTCols_>;
    using TileDataSum = Tile<TileType::Vec, float, 1, kTCols_>;
    
    TileDataF gradOutTile;
    TileDataF xTile;
    TileDataF kTile;
    TileDataF vTile;
    TileDataF wTile;
    TileDataF gradXTile;
    TileDataF gradKTile;
    TileDataF gradVTile;
    TileDataF gradWTile;
    TileDataF tmpTile0;
    TileDataF tmpTile1;
    TileDataSum sumTile;
    TileDataF dummyTile;
    
    size_t tileSize = kTRows_ * kTCols_ * sizeof(float);
    TASSIGN<0x0>(gradOutTile);
    TASSIGN<1 * tileSize>(xTile);
    TASSIGN<2 * tileSize>(kTile);
    TASSIGN<3 * tileSize>(vTile);
    TASSIGN<4 * tileSize>(wTile);
    TASSIGN<5 * tileSize>(gradXTile);
    TASSIGN<6 * tileSize>(gradKTile);
    TASSIGN<7 * tileSize>(gradVTile);
    TASSIGN<8 * tileSize>(gradWTile);
    TASSIGN<9 * tileSize>(tmpTile0);
    TASSIGN<10 * tileSize>(tmpTile1);
    TASSIGN<12 * tileSize>(sumTile);
    TASSIGN<13 * tileSize>(dummyTile);
    

    float gate_val = gate[0];
    float rstd_x_val = rstd_x[0];
    float rstd_k_val = rstd_k[0];
    float raw_dot_val = raw_dot[0];
    
    TMUL(tmpTile0, gradOutTile, vTile);
    TROWSUM(sumTile, tmpTile0, dummyTile);
    
    float abs_dot_scaled = fabs(raw_dot_val) * scalar * rstd_x_val * rstd_k_val;
    float dldg_r = 0.0f;
    if (abs_dot_scaled >= clamp_value) {
        float sqrt_factor = sqrt(scalar * rstd_x_val * rstd_k_val / fabs(raw_dot_val));
        dldg_r = gate_val * (1.0f - gate_val) * 0.5f * sqrt_factor;
    }
    
    float dot_x = raw_dot_val * rstd_x_val * rstd_x_val / hidden_size;
    float dot_k = raw_dot_val * rstd_k_val * rstd_k_val / hidden_size;

    TLOAD(gradOutTile, gradOutGlobal);
    TLOAD(xTile, xGlobal);
    TLOAD(kTile, kGlobal);
    TLOAD(vTile, vGlobal);
    TLOAD(wTile, wGlobal);
    
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    
#ifndef __PTO_AUTO__
    set_flag(PIPE_S, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
#endif

    TMULS(gradVTile, gradOutTile, gate_val);
    
    TMUL(tmpTile0, kTile, wTile);
    TMULS(tmpTile1, xTile, dot_x);
    TSUB(tmpTile0, tmpTile0, tmpTile1);
    TMULS(tmpTile0, tmpTile0, dldg_r);
    TADD(gradXTile, gradOutTile, tmpTile0);
    
    TMUL(tmpTile0, xTile, wTile);
    TMULS(tmpTile1, kTile, dot_k);
    TSUB(tmpTile0, tmpTile0, tmpTile1);
    TMULS(gradKTile, tmpTile0, dldg_r);
    
    TMUL(tmpTile0, xTile, kTile);
    TMULS(gradWTile, tmpTile0, dldg_r);
    
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    
    TSTORE(gradXGlobal, gradXTile);
    TSTORE(gradKGlobal, gradKTile);
    TSTORE(gradVGlobal, gradVTile);
    TSTORE(gradWGlobal, gradWTile);
    
    grad_x = gradXGlobal.data();
    grad_k = gradKGlobal.data();
    grad_v = gradVGlobal.data();
    grad_w = gradWGlobal.data();
}

template <bool isBf16, int hidden_size, int kTRows_, int kTCols_>
void launchEngramGateBwd(
    void *grad_x, void *grad_k, void *grad_v, void *grad_w,
    void *grad_out, void *x, void *k, void *v, void *w,
    void *gate, void *rstd_x, void *rstd_k, void *raw_dot,
    float scalar, float clamp_value, void *stream
)
{
    if constexpr (isBf16) {
        runEngramGateBwd<bfloat16_t, hidden_size, kTRows_, kTCols_>
            <<<1, nullptr, stream>>>(
                (bfloat16_t *)grad_x, (bfloat16_t *)grad_k, (bfloat16_t *)grad_v, (float *)grad_w,
                (bfloat16_t *)grad_out, (bfloat16_t *)x, (bfloat16_t *)k, (bfloat16_t *)v, (float *)w,
                (float *)gate, (float *)rstd_x, (float *)rstd_k, (float *)raw_dot, scalar, clamp_value);
    } else {
        runEngramGateBwd<float, hidden_size, kTRows_, kTCols_>
            <<<1, nullptr, stream>>>(
                (float *)grad_x, (float *)grad_k, (float *)grad_v, (float *)grad_w,
                (float *)grad_out, (float *)x, (float *)k, (float *)v, (float *)w,
                (float *)gate, (float *)rstd_x, (float *)rstd_k, (float *)raw_dot, scalar, clamp_value);
    }
}

template void launchEngramGateBwd<false, 4096, 1, 4096>(
    void *grad_x, void *grad_k, void *grad_v, void *grad_w,
    void *grad_out, void *x, void *k, void *v, void *w,
    void *gate, void *rstd_x, void *rstd_k, void *raw_dot,
    float scalar, float clamp_value, void *stream);

template void launchEngramGateBwd<true, 4096, 1, 4096>(
    void *grad_x, void *grad_k, void *grad_v, void *grad_w,
    void *grad_out, void *x, void *k, void *v, void *w,
    void *gate, void *rstd_x, void *rstd_k, void *raw_dot,
    float scalar, float clamp_value, void *stream);