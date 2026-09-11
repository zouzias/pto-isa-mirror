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
#include <gtest/gtest.h>

#include "cost_check.hpp"

using namespace pto;

namespace {

template <
    typename T, int rows, int cols, int validRow, int validCol, TileType srcLoc, BLayout srcBL, SLayout srcSL,
    TileType dstLoc, BLayout dstBL, SLayout dstSL, float profiling, float accuracy>
void runTMov()
{
    Tile<srcLoc, T, rows, cols, srcBL, validRow, validCol, srcSL> srcTile;
    Tile<dstLoc, T, rows, cols, dstBL, -1, -1, dstSL> dstTile(validRow, validCol);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x20000);

    TMOV(dstTile, srcTile);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

#define TMOV_CASE(TAG, T, R, C, VR, VC, SL_L, SL_BL, SL_SL, DL_L, DL_BL, DL_SL, PROF, ACC)                   \
    TEST(TMov, TAG)                                                                                          \
    {                                                                                                        \
        runTMov<                                                                                             \
            T, R, C, VR, VC, TileType::SL_L, BLayout::SL_BL, SLayout::SL_SL, TileType::DL_L, BLayout::DL_BL, \
            SLayout::DL_SL, PROF, ACC>();                                                                    \
    }

TMOV_CASE(
    c01_f_64_128_64_128_RM_N_RM_N, float, 64, 128, 64, 128, Vec, RowMajor, NoneBox, Vec, RowMajor, NoneBox, 1029.0f,
    0.053449f)
TMOV_CASE(
    c02_f_64_128_64_128_RM_N_CM_N, float, 64, 128, 64, 128, Vec, RowMajor, NoneBox, Vec, ColMajor, NoneBox, 1029.0f,
    0.053449f)
TMOV_CASE(
    c03_f_64_128_64_128_CM_N_CM_N, float, 64, 128, 64, 128, Vec, ColMajor, NoneBox, Vec, ColMajor, NoneBox, 1029.0f,
    0.053449f)
TMOV_CASE(
    c04_f_64_128_64_128_CM_N_RM_N, float, 64, 128, 64, 128, Vec, ColMajor, NoneBox, Vec, RowMajor, NoneBox, 1029.0f,
    0.053449f)
TMOV_CASE(
    c05_f_64_128_64_128_RM_N_CM_RM, float, 64, 128, 64, 128, Vec, RowMajor, NoneBox, Vec, ColMajor, RowMajor, 1029.0f,
    0.053449f)
TMOV_CASE(
    c06_f_64_128_64_128_CM_RM_RM_N, float, 64, 128, 64, 128, Vec, ColMajor, RowMajor, Vec, RowMajor, NoneBox, 1029.0f,
    0.053449f)
TMOV_CASE(
    c07_f_64_128_64_128_CM_N_CM_RM, float, 64, 128, 64, 128, Vec, ColMajor, NoneBox, Vec, ColMajor, RowMajor, 1029.0f,
    0.053449f)
TMOV_CASE(
    c08_f_64_128_64_128_CM_RM_CM_N, float, 64, 128, 64, 128, Vec, ColMajor, RowMajor, Vec, ColMajor, NoneBox, 1029.0f,
    0.053449f)

TMOV_CASE(
    c09_f_16_24_15_23_RM_N_RM_N, float, 16, 24, 15, 23, Vec, RowMajor, NoneBox, Vec, RowMajor, NoneBox, 120.0f, 0.0f)
TMOV_CASE(
    c10_f_64_128_63_125_RM_N_CM_N, float, 64, 128, 63, 125, Vec, RowMajor, NoneBox, Vec, ColMajor, NoneBox, 1323.0f,
    0.0f)
TMOV_CASE(
    c11_f_64_128_63_125_CM_N_CM_N, float, 64, 128, 63, 125, Vec, ColMajor, NoneBox, Vec, ColMajor, NoneBox, 1323.0f,
    0.0f)
TMOV_CASE(
    c12_f_64_128_63_125_CM_N_RM_N, float, 64, 128, 63, 125, Vec, ColMajor, NoneBox, Vec, RowMajor, NoneBox, 1323.0f,
    0.0f)
TMOV_CASE(
    c13_f_64_128_63_125_RM_N_CM_RM, float, 64, 128, 63, 125, Vec, RowMajor, NoneBox, Vec, ColMajor, RowMajor, 1323.0f,
    0.0f)
TMOV_CASE(
    c14_f_64_128_63_125_CM_RM_RM_N, float, 64, 128, 63, 125, Vec, ColMajor, RowMajor, Vec, RowMajor, NoneBox, 1323.0f,
    0.0f)
TMOV_CASE(
    c15_f_64_128_63_125_CM_N_CM_RM, float, 64, 128, 63, 125, Vec, ColMajor, NoneBox, Vec, ColMajor, RowMajor, 1323.0f,
    0.0f)
TMOV_CASE(
    c16_f_64_128_63_125_CM_RM_CM_N, float, 64, 128, 63, 125, Vec, ColMajor, RowMajor, Vec, ColMajor, NoneBox, 1323.0f,
    0.0f)

namespace {
struct FpWaitEvent : EventBaseTag {
    int waits = 0;
    void Wait() { ++waits; }
};

template <STPhase phase, bool useAlias = false, bool explicitPhase = true>
void CheckFpPhaseMove()
{
    using Acc = Tile<TileType::Acc, int32_t, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor>;
    using Mat = Tile<TileType::Mat, half, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor>;
    using Fp = Tile<TileType::Scaling, uint64_t, 1, 16, BLayout::RowMajor, 1, 16>;
    Acc acc;
    Mat mat;
    Fp fp;
    TASSIGN(acc, 0);
    TASSIGN(mat, 0x20000);
    TASSIGN(fp, 0);
    FpWaitEvent dependency;
    mocker::ResetTrace();
    auto move = [&](auto&... events) {
        if constexpr (useAlias) {
            if constexpr (explicitPhase) {
                TMOV_FP<phase>(mat, acc, fp, events...);
            } else {
                TMOV_FP(mat, acc, fp, events...);
            }
        } else {
            if constexpr (explicitPhase) {
                TMOV<phase>(mat, acc, fp, events...);
            } else {
                TMOV(mat, acc, fp, events...);
            }
        }
    };
    move();
    move(dependency);
    EXPECT_EQ(dependency.waits, 1);
    const auto& trace = mocker::GetTrace();
    ASSERT_EQ(trace.executed_pto.size(), 2);
    for (const auto& instruction : trace.executed_pto) {
        EXPECT_EQ(instruction.name, "TMOV");
        int copies = 0;
        int fpConfigs = 0;
        for (const auto& call : instruction.cce_calls) {
            if (call.name == "set_fpc") {
                ++fpConfigs;
            }
            if (call.name == "copy_matrix_cc_to_cbuf") {
                ++copies;
                ASSERT_EQ(call.args.size(), 12);
                EXPECT_EQ(call.args[7], static_cast<uint8_t>(phase));
                EXPECT_EQ(call.args[8], static_cast<uint64_t>(QuantMode_t::VDEQF16));
            }
        }
        EXPECT_EQ(copies, 1);
        EXPECT_EQ(fpConfigs, 1);
    }
}
} // namespace

TEST(TMov, fp_default_phase) { CheckFpPhaseMove<STPhase::Unspecified, false, false>(); }
TEST(TMov, fp_unspecified) { CheckFpPhaseMove<STPhase::Unspecified, false>(); }
TEST(TMov, fp_partial) { CheckFpPhaseMove<STPhase::Partial, false>(); }
TEST(TMov, fp_final) { CheckFpPhaseMove<STPhase::Final, false>(); }
TEST(TMov, fp_alias_default_phase) { CheckFpPhaseMove<STPhase::Unspecified, true, false>(); }
TEST(TMov, fp_alias_unspecified) { CheckFpPhaseMove<STPhase::Unspecified, true>(); }
TEST(TMov, fp_alias_partial) { CheckFpPhaseMove<STPhase::Partial, true>(); }
TEST(TMov, fp_alias_final) { CheckFpPhaseMove<STPhase::Final, true>(); }
