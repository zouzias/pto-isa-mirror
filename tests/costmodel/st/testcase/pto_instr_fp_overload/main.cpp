/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>
#include <type_traits>
#include <utility>

using namespace pto;

namespace {

using AccTile = TileAcc<float, 16, 32, 16, 32>;
using MatTile = Tile<TileType::Mat, int8_t, 16, 32, BLayout::ColMajor, 16, 32, SLayout::RowMajor>;
using ScalingTile = Tile<TileType::Scaling, uint64_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox>;
using GlobalVec = GlobalTensor<uint8_t, Shape<1, 1, 1, 16, 32>, Stride<512, 512, 512, 32, 1>>;

static_assert(
    std::is_same_v<
        decltype(TSTORE(std::declval<GlobalVec&>(), std::declval<AccTile&>(), std::declval<ScalingTile&>())),
        RecordEvent>,
    "Cost Model TSTORE with an fp tile must resolve to the same-name fp overload.");
static_assert(
    std::is_same_v<
        decltype(TEXTRACT(
            std::declval<MatTile&>(), std::declval<AccTile&>(), std::declval<ScalingTile&>(), uint16_t {}, uint16_t {})),
        RecordEvent>,
    "Cost Model TEXTRACT with an fp tile must resolve to the same-name fp overload.");
static_assert(
    std::is_same_v<
        decltype(TINSERT(
            std::declval<MatTile&>(), std::declval<AccTile&>(), std::declval<ScalingTile&>(), uint16_t {}, uint16_t {})),
        RecordEvent>,
    "Cost Model TINSERT with an fp tile must resolve to the same-name fp overload.");
static_assert(
    std::is_same_v<decltype(TMOV(std::declval<MatTile&>(), std::declval<AccTile&>(), std::declval<ScalingTile&>())),
                   RecordEvent>,
    "Cost Model TMOV with a Scaling tile must resolve to the fp overload.");

template <typename T>
void InitTile(T& tile, uint32_t addr)
{
    TASSIGN(tile, addr);
}

void ExerciseNewOverloads()
{
    uint8_t dstData[512] = {};
    AccTile acc;
    MatTile mat;
    ScalingTile scaling;
    GlobalVec global(dstData);

    InitTile(acc, 0x1000);
    InitTile(mat, 0x2000);
    InitTile(scaling, 0x3000);

    TSTORE(global, acc, scaling);
    TEXTRACT(mat, acc, scaling, 0, 0);
    TINSERT(mat, acc, scaling, 0, 0);
    TMOV(mat, acc, scaling);
    TMOV<STPhase::Partial>(mat, acc, scaling);
}

void ExerciseLegacyAliases()
{
    uint8_t dstData[512] = {};
    AccTile acc;
    MatTile mat;
    ScalingTile scaling;
    GlobalVec global(dstData);

    InitTile(acc, 0x1000);
    InitTile(mat, 0x2000);
    InitTile(scaling, 0x3000);

    TSTORE_FP<AccTile, GlobalVec, ScalingTile, AtomicType::AtomicNone, ReluPreMode::NoRelu>(
        global, acc, scaling);
    TEXTRACT_FP<MatTile, AccTile, ScalingTile, ReluPreMode::NoRelu>(mat, acc, scaling, 0, 0);
    TINSERT_FP<MatTile, AccTile, ScalingTile, ReluPreMode::NoRelu>(mat, acc, scaling, 0, 0);
    TMOV_FP<STPhase::Partial, MatTile, AccTile, ScalingTile, ReluPreMode::NoRelu>(mat, acc, scaling);
}

void ExerciseWaitForwarding()
{
    uint8_t dstData[512] = {};
    AccTile acc;
    MatTile mat;
    ScalingTile scaling;
    GlobalVec global(dstData);
    Event<Op::TLOAD, Op::TSTORE_VEC, false, EVENT_ID0> waitEvent;

    InitTile(acc, 0x1000);
    InitTile(mat, 0x2000);
    InitTile(scaling, 0x3000);

    TSTORE<AccTile, GlobalVec, ScalingTile, AtomicType::AtomicNone, ReluPreMode::NormalRelu>(
        global, acc, scaling, waitEvent);
    TEXTRACT<MatTile, AccTile, ScalingTile, ReluPreMode::NormalRelu>(mat, acc, scaling, 0, 0, waitEvent);
    TINSERT<MatTile, AccTile, ScalingTile, ReluPreMode::NormalRelu>(mat, acc, scaling, 0, 0, waitEvent);
    TMOV<MatTile, AccTile, ScalingTile, ReluPreMode::NormalRelu>(mat, acc, scaling, waitEvent);
    TMOV<STPhase::Partial, MatTile, AccTile, ScalingTile, ReluPreMode::NoRelu>(mat, acc, scaling, waitEvent);

    TSTORE_FP<AccTile, GlobalVec, ScalingTile, AtomicType::AtomicNone, ReluPreMode::NormalRelu>(
        global, acc, scaling, waitEvent);
    TEXTRACT_FP<MatTile, AccTile, ScalingTile, ReluPreMode::NormalRelu>(mat, acc, scaling, 0, 0, waitEvent);
    TINSERT_FP<MatTile, AccTile, ScalingTile, ReluPreMode::NormalRelu>(mat, acc, scaling, 0, 0, waitEvent);
    TMOV_FP<MatTile, AccTile, ScalingTile, ReluPreMode::NormalRelu>(mat, acc, scaling, waitEvent);
    TMOV_FP<STPhase::Partial, MatTile, AccTile, ScalingTile, ReluPreMode::NoRelu>(mat, acc, scaling, waitEvent);
}

} // namespace

TEST(PtoInstrFpOverload, NewOverloadsCompileAndRun)
{
    ExerciseNewOverloads();
    SUCCEED();
}

TEST(PtoInstrFpOverload, LegacyAliasesCompileAndRun)
{
    ExerciseLegacyAliases();
    SUCCEED();
}

TEST(PtoInstrFpOverload, ReluModeAndWaitEventsForward)
{
    ExerciseWaitForwarding();
    SUCCEED();
}
