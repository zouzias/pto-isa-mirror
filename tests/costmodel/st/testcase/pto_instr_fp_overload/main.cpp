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

using VecTile = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, 4, 8>;
using AccTile = Tile<TileType::Acc, float, 4, 8, BLayout::RowMajor, 4, 8>;
using MatTile = Tile<TileType::Mat, float, 4, 8, BLayout::RowMajor, 4, 8, SLayout::NoneBox>;
using ScalingTile = Tile<TileType::Scaling, uint64_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox>;
using GlobalVec = GlobalTensor<float, Shape<1, 1, 1, 4, 8>, Stride<32, 32, 32, 8, 1>>;

static_assert(
    std::is_same_v<
        decltype(TSTORE(std::declval<GlobalVec&>(), std::declval<AccTile&>(), std::declval<ScalingTile&>())),
        RecordEvent>,
    "Cost Model TSTORE with an fp tile must resolve to the same-name fp overload.");
static_assert(
    std::is_same_v<
        decltype(TEXTRACT(
            std::declval<MatTile&>(), std::declval<VecTile&>(), std::declval<ScalingTile&>(), uint16_t {}, uint16_t {})),
        RecordEvent>,
    "Cost Model TEXTRACT with an fp tile must resolve to the same-name fp overload.");
static_assert(
    std::is_same_v<
        decltype(TINSERT(
            std::declval<MatTile&>(), std::declval<VecTile&>(), std::declval<ScalingTile&>(), uint16_t {}, uint16_t {})),
        RecordEvent>,
    "Cost Model TINSERT with an fp tile must resolve to the same-name fp overload.");
static_assert(
    std::is_same_v<decltype(TMOV(std::declval<VecTile&>(), std::declval<AccTile&>(), std::declval<ScalingTile&>())),
                   RecordEvent>,
    "Cost Model TMOV with a Scaling tile must resolve to the fp overload.");

template <typename T>
void InitTile(T& tile, uint32_t addr)
{
    TASSIGN(tile, addr);
}

void ExerciseNewOverloads()
{
    float dstData[32] = {};
    VecTile vec;
    AccTile acc;
    MatTile mat;
    ScalingTile scaling;
    GlobalVec global(dstData);

    InitTile(vec, 0x0);
    InitTile(acc, 0x1000);
    InitTile(mat, 0x2000);
    InitTile(scaling, 0x3000);

    TSTORE(global, acc, scaling);
    TEXTRACT(mat, vec, scaling, 0, 0);
    TINSERT(mat, vec, scaling, 0, 0);
    TMOV(vec, acc, scaling);
    TMOV<STPhase::Partial>(vec, acc, scaling);
}

void ExerciseLegacyAliases()
{
    float dstData[32] = {};
    VecTile vec;
    AccTile acc;
    MatTile mat;
    ScalingTile scaling;
    GlobalVec global(dstData);

    InitTile(vec, 0x0);
    InitTile(acc, 0x1000);
    InitTile(mat, 0x2000);
    InitTile(scaling, 0x3000);

    TSTORE_FP<VecTile, GlobalVec, ScalingTile, AtomicType::AtomicNone, ReluPreMode::NoRelu>(
        global, vec, scaling);
    TEXTRACT_FP<MatTile, VecTile, ScalingTile, ReluPreMode::NoRelu>(mat, vec, scaling, 0, 0);
    TINSERT_FP<MatTile, VecTile, ScalingTile, ReluPreMode::NoRelu>(mat, vec, scaling, 0, 0);
    TMOV_FP<STPhase::Partial, VecTile, AccTile, ScalingTile, ReluPreMode::NoRelu>(vec, acc, scaling);
}

void ExerciseWaitForwarding()
{
    float dstData[32] = {};
    VecTile vec;
    AccTile acc;
    MatTile mat;
    ScalingTile scaling;
    GlobalVec global(dstData);
    Event<Op::TLOAD, Op::TSTORE_VEC, false, EVENT_ID0> waitEvent;

    InitTile(vec, 0x0);
    InitTile(acc, 0x1000);
    InitTile(mat, 0x2000);
    InitTile(scaling, 0x3000);

    TSTORE<VecTile, GlobalVec, ScalingTile, AtomicType::AtomicNone, ReluPreMode::NormalRelu>(
        global, vec, scaling, waitEvent);
    TEXTRACT<MatTile, VecTile, ScalingTile, ReluPreMode::NormalRelu>(mat, vec, scaling, 0, 0, waitEvent);
    TINSERT<MatTile, VecTile, ScalingTile, ReluPreMode::NormalRelu>(mat, vec, scaling, 0, 0, waitEvent);
    TMOV<VecTile, AccTile, ScalingTile, ReluPreMode::NormalRelu>(vec, acc, scaling, waitEvent);
    TMOV<STPhase::Partial, VecTile, AccTile, ScalingTile, ReluPreMode::NoRelu>(vec, acc, scaling, waitEvent);

    TSTORE_FP<VecTile, GlobalVec, ScalingTile, AtomicType::AtomicNone, ReluPreMode::NormalRelu>(
        global, vec, scaling, waitEvent);
    TEXTRACT_FP<MatTile, VecTile, ScalingTile, ReluPreMode::NormalRelu>(mat, vec, scaling, 0, 0, waitEvent);
    TINSERT_FP<MatTile, VecTile, ScalingTile, ReluPreMode::NormalRelu>(mat, vec, scaling, 0, 0, waitEvent);
    TMOV_FP<VecTile, AccTile, ScalingTile, ReluPreMode::NormalRelu>(vec, acc, scaling, waitEvent);
    TMOV_FP<STPhase::Partial, VecTile, AccTile, ScalingTile, ReluPreMode::NoRelu>(vec, acc, scaling, waitEvent);
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
