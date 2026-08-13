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
#include <type_traits>
#include <utility>

using namespace pto;

namespace {

using OverloadAccTile = TileAcc<float, 16, 16, 16, 16>;
using OverloadVecTile = Tile<TileType::Vec, half, 16, 16, BLayout::RowMajor, 16, 16, SLayout::NoneBox>;
using OverloadMatTile = Tile<TileType::Mat, half, 16, 16, BLayout::RowMajor, 16, 16, SLayout::NoneBox>;
using OverloadScalingTile = Tile<TileType::Scaling, uint64_t, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox>;
using OverloadTmpTile = Tile<TileType::Vec, uint8_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512>;

static_assert(
    std::is_same_v<
        decltype(TMOV(
            std::declval<OverloadVecTile&>(), std::declval<OverloadAccTile&>(),
            std::declval<OverloadScalingTile&>())),
        RecordEvent>,
    "TMOV with a Scaling tile must resolve to the fp overload.");

static_assert(
    std::is_same_v<
        decltype(TMOV<1>(
            std::declval<OverloadMatTile&>(), std::declval<OverloadVecTile&>(), std::declval<OverloadTmpTile&>())),
        RecordEvent>,
    "TMOV with a non-Scaling temporary tile must resolve to the tmp overload.");

} // namespace

__global__ AICORE void PtoInstrFpOverloadCompileKernel(__gm__ half* out)
{
    using GlobalData = GlobalTensor<half, Shape<1, 1, 1, 16, 16>, Stride<256, 256, 256, 16, 1>>;

    OverloadAccTile acc;
    OverloadVecTile vec;
    OverloadMatTile mat;
    OverloadScalingTile scaling;
    GlobalData global(out);
    Event<Op::TLOAD, Op::TSTORE_VEC, false, EVENT_ID0> waitEvent;

    TASSIGN(acc, 0x0);
    TASSIGN(vec, 0x4000);
    TASSIGN(mat, 0x8000);
    TASSIGN(scaling, 0xc000);

    TMOV(vec, acc, scaling, waitEvent);
    TMOV<STPhase::Partial>(vec, acc, scaling, waitEvent);
    TEXTRACT(mat, acc, scaling, 0, 0, waitEvent);
    TINSERT(mat, acc, scaling, 0, 0, waitEvent);
    TSTORE(global, acc, scaling, waitEvent);

    TMOV<OverloadVecTile, OverloadAccTile, OverloadScalingTile, ReluPreMode::NormalRelu>(
        vec, acc, scaling, waitEvent);
    TMOV<STPhase::Partial, OverloadVecTile, OverloadAccTile, OverloadScalingTile, ReluPreMode::NormalRelu>(
        vec, acc, scaling, waitEvent);
    TEXTRACT<OverloadMatTile, OverloadAccTile, OverloadScalingTile, ReluPreMode::NormalRelu>(
        mat, acc, scaling, 0, 0, waitEvent);
    TINSERT<OverloadMatTile, OverloadAccTile, OverloadScalingTile, ReluPreMode::NormalRelu>(
        mat, acc, scaling, 0, 0, waitEvent);
    TSTORE<OverloadAccTile, GlobalData, OverloadScalingTile, AtomicType::AtomicNone, ReluPreMode::NormalRelu>(
        global, acc, scaling, waitEvent);

    TMOV_FP<OverloadVecTile, OverloadAccTile, OverloadScalingTile, ReluPreMode::NoRelu>(
        vec, acc, scaling, waitEvent);
    TMOV_FP<STPhase::Partial, OverloadVecTile, OverloadAccTile, OverloadScalingTile, ReluPreMode::NoRelu>(
        vec, acc, scaling, waitEvent);
    TEXTRACT_FP<OverloadMatTile, OverloadAccTile, OverloadScalingTile, ReluPreMode::NoRelu>(
        mat, acc, scaling, 0, 0, waitEvent);
    TINSERT_FP<OverloadMatTile, OverloadAccTile, OverloadScalingTile, ReluPreMode::NoRelu>(
        mat, acc, scaling, 0, 0, waitEvent);
    TSTORE_FP<OverloadAccTile, GlobalData, OverloadScalingTile, AtomicType::AtomicNone, ReluPreMode::NoRelu>(
        global, acc, scaling, waitEvent);
}
