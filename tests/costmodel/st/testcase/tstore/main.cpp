/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>
#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace {
struct WaitEvent : EventBaseTag {
    int waits = 0;
    void Wait() { ++waits; }
};

enum class StoreForm { Canonical, Alias, L2Hint };

template <STPhase phase, StoreForm form = StoreForm::L2Hint, bool explicitPhase = true>
void CheckFpStore()
{
    using Acc = Tile<TileType::Acc, int32_t, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor>;
    using Fp = Tile<TileType::Scaling, uint64_t, 1, 16, BLayout::RowMajor, 1, 16>;
    using Global = GlobalTensor<half, Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>>;
    Acc acc;
    Fp fp;
    Global dst(reinterpret_cast<half*>(0x10000));
    TASSIGN(acc, 0);
    TASSIGN(fp, 0);
    WaitEvent dependency;
    mocker::ResetTrace();
    auto store = [&](auto&... events) {
        if constexpr (form == StoreForm::L2Hint) {
            if constexpr (explicitPhase) {
                TSTORE<TStoreL2Hint::NormalFirstVictim, phase>(dst, acc, fp, events...);
            } else {
                TSTORE<TStoreL2Hint::NormalFirstVictim>(dst, acc, fp, events...);
            }
        } else if constexpr (form == StoreForm::Alias) {
            if constexpr (explicitPhase) {
                TSTORE_FP<phase>(dst, acc, fp, events...);
            } else {
                TSTORE_FP(dst, acc, fp, events...);
            }
        } else {
            if constexpr (explicitPhase) {
                TSTORE<phase>(dst, acc, fp, events...);
            } else {
                TSTORE(dst, acc, fp, events...);
            }
        }
    };
    store();
    store(dependency);
    EXPECT_EQ(dependency.waits, 1);
    const auto& trace = mocker::GetTrace();
    ASSERT_EQ(trace.executed_pto.size(), 2);
    for (const auto& instruction : trace.executed_pto) {
        EXPECT_EQ(instruction.name, "TSTORE");
        int copies = 0;
        int fpConfigs = 0;
        for (const auto& call : instruction.cce_calls) {
            if (call.name == "set_fpc") {
                ++fpConfigs;
            }
            if (call.name == "copy_matrix_cc_to_gm") {
                ++copies;
                ASSERT_EQ(call.args.size(), 4);
                EXPECT_EQ((call.args[3] >> 32) & 3, static_cast<uint8_t>(phase));
                EXPECT_EQ((call.args[3] >> 34) & 0x1f, static_cast<uint64_t>(QuantMode_t::VDEQF16));
            }
        }
        EXPECT_EQ(copies, 1);
        EXPECT_EQ(fpConfigs, 1);
    }
}
} // namespace

TEST(TStore, fp_default_phase) { CheckFpStore<STPhase::Unspecified, StoreForm::Canonical, false>(); }
TEST(TStore, fp_unspecified) { CheckFpStore<STPhase::Unspecified, StoreForm::Canonical>(); }
TEST(TStore, fp_partial) { CheckFpStore<STPhase::Partial, StoreForm::Canonical>(); }
TEST(TStore, fp_final) { CheckFpStore<STPhase::Final, StoreForm::Canonical>(); }
TEST(TStore, fp_alias_default_phase) { CheckFpStore<STPhase::Unspecified, StoreForm::Alias, false>(); }
TEST(TStore, fp_alias_unspecified) { CheckFpStore<STPhase::Unspecified, StoreForm::Alias>(); }
TEST(TStore, fp_alias_partial) { CheckFpStore<STPhase::Partial, StoreForm::Alias>(); }
TEST(TStore, fp_alias_final) { CheckFpStore<STPhase::Final, StoreForm::Alias>(); }
TEST(TStore, fp_l2_default_phase) { CheckFpStore<STPhase::Unspecified, StoreForm::L2Hint, false>(); }
TEST(TStore, fp_l2_unspecified) { CheckFpStore<STPhase::Unspecified, StoreForm::L2Hint>(); }
TEST(TStore, fp_l2_partial) { CheckFpStore<STPhase::Partial, StoreForm::L2Hint>(); }
TEST(TStore, fp_l2_final) { CheckFpStore<STPhase::Final, StoreForm::L2Hint>(); }
