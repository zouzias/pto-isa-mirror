/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>

using namespace pto;

__global__ AICORE void PtoInstrFpOverloadCompileKernel(__gm__ half* out)
{
    using AccTile = TileAcc<float, 16, 16, 16, 16>;
    using MatTile = Tile<TileType::Mat, half, 16, 16, BLayout::RowMajor, 16, 16, SLayout::NoneBox>;
    using ScalingTile = Tile<TileType::Scaling, uint64_t, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox>;
    using GlobalData = GlobalTensor<half, Shape<1, 1, 1, 16, 16>, Stride<256, 256, 256, 16, 1>>;

    AccTile acc;
    MatTile mat;
    ScalingTile scaling;
    GlobalData global(out);
    Event<Op::TLOAD, Op::TSTORE_VEC, false, EVENT_ID0> waitEvent;

    TASSIGN(acc, 0x0);
    TASSIGN(mat, 0x4000);
    TASSIGN(scaling, 0x8000);

    TMOV(mat, acc, scaling, waitEvent);
    TEXTRACT(mat, acc, scaling, 0, 0, waitEvent);
    TINSERT(mat, acc, scaling, 0, 0, waitEvent);
    TSTORE(global, acc, scaling, waitEvent);

    TMOV<MatTile, AccTile, ScalingTile, ReluPreMode::NormalRelu>(mat, acc, scaling, waitEvent);
    TEXTRACT<MatTile, AccTile, ScalingTile, ReluPreMode::NormalRelu>(mat, acc, scaling, 0, 0, waitEvent);
    TINSERT<MatTile, AccTile, ScalingTile, ReluPreMode::NormalRelu>(mat, acc, scaling, 0, 0, waitEvent);
    TSTORE<AccTile, GlobalData, ScalingTile, AtomicType::AtomicNone, ReluPreMode::NormalRelu>(
        global, acc, scaling, waitEvent);

    TMOV_FP<MatTile, AccTile, ScalingTile, ReluPreMode::NoRelu>(mat, acc, scaling, waitEvent);
    TEXTRACT_FP<MatTile, AccTile, ScalingTile, ReluPreMode::NoRelu>(mat, acc, scaling, 0, 0, waitEvent);
    TINSERT_FP<MatTile, AccTile, ScalingTile, ReluPreMode::NoRelu>(mat, acc, scaling, 0, 0, waitEvent);
    TSTORE_FP<AccTile, GlobalData, ScalingTile, AtomicType::AtomicNone, ReluPreMode::NoRelu>(
        global, acc, scaling, waitEvent);
}
