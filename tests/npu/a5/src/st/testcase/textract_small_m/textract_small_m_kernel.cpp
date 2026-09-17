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
#include "textract_small_m_cases.h"

using namespace pto;

template <int Key>
__global__ AICORE void runTEXTRACTSmallM(
    __gm__ float* out, __gm__ uint8_t* a, __gm__ uint8_t* b, uint32_t m, uint32_t k)
{
    using C = TExtractSmallMCase<Key>;
    using T = std::conditional_t<C::Bf16, bfloat16_t, half>;
    constexpr int n = 32;
    constexpr int srcCols = C::Col + C::Cols;
    // Exercise Compact with a destination physically wider than the source.
    constexpr int leftCols = C::Compact ? 2 * C::Cols : C::Cols;
    constexpr int validM = C::Dynamic ? DYNAMIC : C::M;
    constexpr int validK = C::Dynamic ? DYNAMIC : C::K;
    using MatA =
        Tile<TileType::Mat, T, C::Rows, srcCols + 16, BLayout::ColMajor, C::Rows, srcCols, SLayout::RowMajor, 512>;
    using MatB = Tile<TileType::Mat, T, C::Cols, n, BLayout::ColMajor, C::Cols, n, SLayout::RowMajor, 512>;
    using Left = Tile<
        TileType::Left, T, 16, leftCols, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512, PadValue::Null,
        C::Compact ? CompactMode::Normal : CompactMode::Null>;
    using Right = TileRight<T, C::Cols, n, C::K, n>;
    using Acc = TileAcc<float, 16, n, C::M, n>;
    using GA = GlobalTensor<T, Shape<1, 1, 1, C::Rows, srcCols>, pto::Stride<1, 1, 1, srcCols, 1>>;
    using GB = GlobalTensor<T, Shape<1, 1, 1, C::Cols, n>, pto::Stride<1, 1, 1, n, 1>>;
    using GC = GlobalTensor<float, Shape<1, 1, 1, C::M, n>, pto::Stride<1, 1, 1, n, 1>>;
    MatA matA;
    MatB matB;
    Left left;
    Right right;
    Acc acc;
    if constexpr (C::Dynamic) {
        left.SetValidShape(m, k);
    }
    TASSIGN(matA, 0);
    TASSIGN(matB, 0x20000);
    TASSIGN(left, 0);
    TASSIGN(right, 0);
    TASSIGN(acc, 0);
    GA ga(reinterpret_cast<__gm__ T*>(a));
    GB gb(reinterpret_cast<__gm__ T*>(b));
    TEXPANDS(matA, T(-7));
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    TLOAD(matA, ga);
    TLOAD(matB, gb);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
    TEXTRACT(right, matB, 0, 0);
    // All four GQA head groups reuse this single Mat load.
    for (int group = 0; group < C::Groups; ++group) {
        TEXTRACT(left, matA, C::Row + group * 4, C::Col);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
        TMATMUL(acc, left, right);
#ifndef __PTO_AUTO__
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
        GC gc(out + group * C::M * n);
        TSTORE(gc, acc);
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
    }
}

template <int Key>
void launchTEXTRACTSmallM(uint8_t* out, uint8_t* a, uint8_t* b, void* stream)
{
    runTEXTRACTSmallM<Key><<<1, nullptr, stream>>>(
        reinterpret_cast<float*>(out), a, b, TExtractSmallMCase<Key>::M, TExtractSmallMCase<Key>::K);
}

#ifdef PTO_TEXTRACT_SMALL_M_CASE
template void launchTEXTRACTSmallM<PTO_TEXTRACT_SMALL_M_CASE>(uint8_t*, uint8_t*, uint8_t*, void*);
#else
template void launchTEXTRACTSmallM<1>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTEXTRACTSmallM<2>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTEXTRACTSmallM<3>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTEXTRACTSmallM<4>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTEXTRACTSmallM<5>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTEXTRACTSmallM<6>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTEXTRACTSmallM<7>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTEXTRACTSmallM<8>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTEXTRACTSmallM<9>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTEXTRACTSmallM<10>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTEXTRACTSmallM<11>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTEXTRACTSmallM<12>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTEXTRACTSmallM<13>(uint8_t*, uint8_t*, uint8_t*, void*);
#endif
