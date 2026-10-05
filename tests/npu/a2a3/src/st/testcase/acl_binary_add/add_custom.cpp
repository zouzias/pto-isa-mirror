/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Device-only add kernel. Compiled with --cce-aicore-only, so this translation unit
// contains no host code at all; the resulting object is linked by ld.lld into a
// standalone device .o that the host loads with aclrtBinaryLoadFromFile and looks up
// by the exported symbol name "add_custom".

#include <pto/pto-inst.hpp>

using namespace pto;

namespace {
constexpr unsigned kRows = 1;
constexpr unsigned kCols = 256;
constexpr unsigned kXUb = 0x0;
constexpr unsigned kYUb = 0x1000;
constexpr unsigned kZUb = 0x2000;
} // namespace

extern "C" __global__ __vector__ void add_custom(__gm__ half __in__* x, __gm__ half __in__* y, __gm__ half __out__* z)
{
    set_mask_norm();
    set_vector_mask(-1, -1);

    using Shape5 = pto::Shape<1, 1, 1, kRows, kCols>;
    using Stride5 = pto::Stride<1, 1, 1, kCols, 1>;
    using GlobalData = pto::GlobalTensor<half, Shape5, Stride5>;
    GlobalData xGlobal(x);
    GlobalData yGlobal(y);
    GlobalData zGlobal(z);

    using TileData = Tile<TileType::Vec, half, kRows, kCols, BLayout::RowMajor, -1, -1>;
    TileData xTile(kRows, kCols);
    TileData yTile(kRows, kCols);
    TileData zTile(kRows, kCols);
    TASSIGN(xTile, kXUb);
    TASSIGN(yTile, kYUb);
    TASSIGN(zTile, kZUb);

    TLOAD(xTile, xGlobal);
    TLOAD(yTile, yGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TADD(zTile, xTile, yTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(zGlobal, zTile);
}
