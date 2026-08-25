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

#if defined(__CCE_AICORE__) && defined(__DAV_CUBE__)
using OverloadAccTile = TileAcc<float, 16, 16, 16, 16>;
using OverloadVecTile = Tile<TileType::Vec, half, 16, 16, BLayout::RowMajor, 16, 16, SLayout::NoneBox>;
using OverloadMatTile = Tile<TileType::Mat, half, 16, 16, BLayout::RowMajor, 16, 16, SLayout::NoneBox>;
using OverloadInsertMatTile = Tile<TileType::Mat, half, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor>;
using OverloadScalingTile = Tile<TileType::Scaling, uint64_t, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox>;
using OverloadTmpTile = Tile<TileType::Vec, uint8_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512>;
using OverloadGlobalData = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>>;
using OverloadWaitEvent = Event<Op::TLOAD, Op::TSTORE_VEC, false, EVENT_ID0>;

static_assert(
    std::is_same_v<
        decltype(TMOV(
            std::declval<OverloadVecTile&>(), std::declval<OverloadAccTile&>(), std::declval<OverloadScalingTile&>())),
        RecordEvent>,
    "TMOV with a Scaling tile must resolve to the fp overload.");

static_assert(
    std::is_same_v<
        decltype(TMOV<1>(
            std::declval<OverloadMatTile&>(), std::declval<OverloadVecTile&>(), std::declval<OverloadTmpTile&>())),
        RecordEvent>,
    "TMOV with a non-Scaling temporary tile must resolve to the tmp overload.");

static_assert(
    std::is_same_v<
        decltype(TMOV(
            std::declval<OverloadVecTile&>(), std::declval<OverloadAccTile&>(), std::declval<OverloadScalingTile&>(),
            std::declval<OverloadWaitEvent&>())),
        RecordEvent>,
    "TMOV fp overload with wait event must compile.");

static_assert(
    std::is_same_v<
        decltype(TMOV<STPhase::Partial, OverloadVecTile, OverloadAccTile, OverloadScalingTile, ReluPreMode::NormalRelu>(
            std::declval<OverloadVecTile&>(), std::declval<OverloadAccTile&>(), std::declval<OverloadScalingTile&>(),
            std::declval<OverloadWaitEvent&>())),
        RecordEvent>,
    "Explicit TMOV fp overload with phase and relu mode must compile.");

static_assert(
    std::is_same_v<
        decltype(TMOV<OverloadVecTile, OverloadAccTile, OverloadScalingTile, AccToVecMode::SingleModeVec0>(
            std::declval<OverloadVecTile&>(), std::declval<OverloadAccTile&>(), std::declval<OverloadScalingTile&>(),
            std::declval<OverloadWaitEvent&>())),
        RecordEvent>,
    "Explicit TMOV fp overload with AccToVecMode must compile.");

static_assert(
    std::is_same_v<
        decltype(TMOV<
                 STPhase::Partial, OverloadVecTile, OverloadAccTile, OverloadScalingTile, AccToVecMode::SingleModeVec0>(
            std::declval<OverloadVecTile&>(), std::declval<OverloadAccTile&>(), std::declval<OverloadScalingTile&>(),
            std::declval<OverloadWaitEvent&>())),
        RecordEvent>,
    "Explicit TMOV fp overload with phase and AccToVecMode must compile.");

static_assert(
    std::is_same_v<
        decltype(TMOV_FP<OverloadVecTile, OverloadAccTile, OverloadScalingTile, ReluPreMode::NoRelu>(
            std::declval<OverloadVecTile&>(), std::declval<OverloadAccTile&>(), std::declval<OverloadScalingTile&>(),
            std::declval<OverloadWaitEvent&>())),
        RecordEvent>,
    "Explicit TMOV_FP overload must compile.");

static_assert(
    std::is_same_v<
        decltype(TMOV_FP<STPhase::Partial, OverloadVecTile, OverloadAccTile, OverloadScalingTile, ReluPreMode::NoRelu>(
            std::declval<OverloadVecTile&>(), std::declval<OverloadAccTile&>(), std::declval<OverloadScalingTile&>(),
            std::declval<OverloadWaitEvent&>())),
        RecordEvent>,
    "Explicit TMOV_FP overload with phase must compile.");

static_assert(
    std::is_same_v<
        decltype(TEXTRACT(
            std::declval<OverloadMatTile&>(), std::declval<OverloadAccTile&>(), std::declval<OverloadScalingTile&>(), 0,
            0, std::declval<OverloadWaitEvent&>())),
        RecordEvent>,
    "TEXTRACT fp overload must compile.");

static_assert(
    std::is_same_v<
        decltype(TEXTRACT_FP<OverloadMatTile, OverloadAccTile, OverloadScalingTile, ReluPreMode::NoRelu>(
            std::declval<OverloadMatTile&>(), std::declval<OverloadAccTile&>(), std::declval<OverloadScalingTile&>(), 0,
            0, std::declval<OverloadWaitEvent&>())),
        RecordEvent>,
    "Explicit TEXTRACT_FP overload must compile.");

static_assert(
    std::is_same_v<
        decltype(TEXTRACT<OverloadVecTile, OverloadAccTile, OverloadScalingTile, AccToVecMode::SingleModeVec0>(
            std::declval<OverloadVecTile&>(), std::declval<OverloadAccTile&>(), std::declval<OverloadScalingTile&>(), 0,
            0, std::declval<OverloadWaitEvent&>())),
        RecordEvent>,
    "Explicit TEXTRACT fp overload with AccToVecMode must compile.");

static_assert(
    std::is_same_v<
        decltype(TINSERT(
            std::declval<OverloadInsertMatTile&>(), std::declval<OverloadAccTile&>(),
            std::declval<OverloadScalingTile&>(), 0, 0, std::declval<OverloadWaitEvent&>())),
        RecordEvent>,
    "TINSERT fp overload must compile.");

static_assert(
    std::is_same_v<
        decltype(TINSERT_FP<OverloadInsertMatTile, OverloadAccTile, OverloadScalingTile, ReluPreMode::NoRelu>(
            std::declval<OverloadInsertMatTile&>(), std::declval<OverloadAccTile&>(),
            std::declval<OverloadScalingTile&>(), 0, 0, std::declval<OverloadWaitEvent&>())),
        RecordEvent>,
    "Explicit TINSERT_FP overload must compile.");

static_assert(
    std::is_same_v<
        decltype(TINSERT<OverloadVecTile, OverloadAccTile, OverloadScalingTile, AccToVecMode::SingleModeVec0>(
            std::declval<OverloadVecTile&>(), std::declval<OverloadAccTile&>(), std::declval<OverloadScalingTile&>(), 0,
            0, std::declval<OverloadWaitEvent&>())),
        RecordEvent>,
    "Explicit TINSERT fp overload with AccToVecMode must compile.");

static_assert(
    std::is_same_v<
        decltype(TSTORE(
            std::declval<OverloadGlobalData&>(), std::declval<OverloadAccTile&>(), std::declval<OverloadScalingTile&>(),
            std::declval<OverloadWaitEvent&>())),
        RecordEvent>,
    "TSTORE fp overload must compile.");

static_assert(
    std::is_same_v<
        decltype(TSTORE_FP<
                 OverloadAccTile, OverloadGlobalData, OverloadScalingTile, AtomicType::AtomicNone, ReluPreMode::NoRelu>(
            std::declval<OverloadGlobalData&>(), std::declval<OverloadAccTile&>(), std::declval<OverloadScalingTile&>(),
            std::declval<OverloadWaitEvent&>())),
        RecordEvent>,
    "Explicit TSTORE_FP overload must compile.");

static_assert(
    std::is_same_v<
        decltype(TSTORE_FP<
                 STPhase::Partial, OverloadAccTile, OverloadGlobalData, OverloadScalingTile, AtomicType::AtomicNone,
                 ReluPreMode::NoRelu>(
            std::declval<OverloadGlobalData&>(), std::declval<OverloadAccTile&>(), std::declval<OverloadScalingTile&>(),
            std::declval<OverloadWaitEvent&>())),
        RecordEvent>,
    "Explicit TSTORE_FP phase overload must compile.");
#endif

} // namespace

__global__ AICORE void PtoInstrFpOverloadCompileKernel(__gm__ half* out)
{
#if defined(__CCE_AICORE__) && defined(__DAV_CUBE__)
    OverloadAccTile acc;
    OverloadVecTile vec;
    OverloadMatTile mat;
    OverloadInsertMatTile insertMat;
    OverloadScalingTile scaling;
    OverloadGlobalData global(out);
    OverloadWaitEvent waitEvent;

    TASSIGN(acc, 0x0);
    TASSIGN(vec, 0x4000);
    TASSIGN(mat, 0x8000);
    TASSIGN(insertMat, 0xc000);
    TASSIGN(scaling, 0x10000);

    TMOV(vec, acc, scaling, waitEvent);
    TMOV<STPhase::Partial>(vec, acc, scaling, waitEvent);
    TEXTRACT(mat, acc, scaling, 0, 0, waitEvent);
    TINSERT(insertMat, acc, scaling, 0, 0, waitEvent);
    TSTORE(global, acc, scaling, waitEvent);

    TMOV<OverloadVecTile, OverloadAccTile, OverloadScalingTile, ReluPreMode::NormalRelu>(vec, acc, scaling, waitEvent);
    TMOV<STPhase::Partial, OverloadVecTile, OverloadAccTile, OverloadScalingTile, ReluPreMode::NormalRelu>(
        vec, acc, scaling, waitEvent);
    TMOV<OverloadVecTile, OverloadAccTile, OverloadScalingTile, AccToVecMode::SingleModeVec0>(
        vec, acc, scaling, waitEvent);
    TMOV<STPhase::Partial, OverloadVecTile, OverloadAccTile, OverloadScalingTile, AccToVecMode::SingleModeVec0>(
        vec, acc, scaling, waitEvent);
    TEXTRACT<OverloadMatTile, OverloadAccTile, OverloadScalingTile, ReluPreMode::NormalRelu>(
        mat, acc, scaling, 0, 0, waitEvent);
    TEXTRACT<OverloadVecTile, OverloadAccTile, OverloadScalingTile, AccToVecMode::SingleModeVec0>(
        vec, acc, scaling, 0, 0, waitEvent);
    TINSERT<OverloadInsertMatTile, OverloadAccTile, OverloadScalingTile, ReluPreMode::NormalRelu>(
        insertMat, acc, scaling, 0, 0, waitEvent);
    TINSERT<OverloadVecTile, OverloadAccTile, OverloadScalingTile, AccToVecMode::SingleModeVec0>(
        vec, acc, scaling, 0, 0, waitEvent);
    TSTORE<OverloadAccTile, OverloadGlobalData, OverloadScalingTile, AtomicType::AtomicNone, ReluPreMode::NormalRelu>(
        global, acc, scaling, waitEvent);
    TSTORE<
        STPhase::Partial, OverloadAccTile, OverloadGlobalData, OverloadScalingTile, AtomicType::AtomicNone,
        ReluPreMode::NormalRelu>(global, acc, scaling, waitEvent);

    TMOV_FP<OverloadVecTile, OverloadAccTile, OverloadScalingTile, ReluPreMode::NoRelu>(vec, acc, scaling, waitEvent);
    TMOV_FP<STPhase::Partial, OverloadVecTile, OverloadAccTile, OverloadScalingTile, ReluPreMode::NoRelu>(
        vec, acc, scaling, waitEvent);
    TEXTRACT_FP<OverloadMatTile, OverloadAccTile, OverloadScalingTile, ReluPreMode::NoRelu>(
        mat, acc, scaling, 0, 0, waitEvent);
    TINSERT_FP<OverloadInsertMatTile, OverloadAccTile, OverloadScalingTile, ReluPreMode::NoRelu>(
        insertMat, acc, scaling, 0, 0, waitEvent);
    TSTORE_FP<OverloadAccTile, OverloadGlobalData, OverloadScalingTile, AtomicType::AtomicNone, ReluPreMode::NoRelu>(
        global, acc, scaling, waitEvent);
    TSTORE_FP<
        STPhase::Partial, OverloadAccTile, OverloadGlobalData, OverloadScalingTile, AtomicType::AtomicNone,
        ReluPreMode::NoRelu>(global, acc, scaling, waitEvent);
#else
    (void)out;
#endif
}
