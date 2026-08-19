/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

namespace {

using Global4 = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 4>, pto::Stride<4, 4, 4, 4, 1>, pto::Layout::ND>;

struct SourceReadyEvent {
    int32_t* source;
    bool* waited;

    void Wait()
    {
        source[0] = 101;
        *waited = true;
    }
};

TEST(TPutAsyncNotify, SetWaitsBeforeCopyAndUpdatesSignal)
{
    int32_t source[4] = {-1, 102, 103, 104};
    int32_t destination[4] = {};
    alignas(int32_t) int32_t signalValue = -7;
    bool waited = false;

    Global4 sourceTensor(source);
    Global4 destinationTensor(destination);
    pto::comm::Signal signal(&signalValue);
    pto::comm::AsyncSession session;
    SourceReadyEvent ready{source, &waited};

    const pto::comm::AsyncEvent event = pto::comm::TPUT_ASYNC_NOTIFY<pto::comm::DmaEngine::SDMA>(
        destinationTensor, sourceTensor, signal, 9, pto::comm::NotifyOp::Set, session, 17U, ready);

    EXPECT_TRUE(waited);
    EXPECT_EQ(destination[0], 101);
    EXPECT_EQ(destination[1], 102);
    EXPECT_EQ(destination[2], 103);
    EXPECT_EQ(destination[3], 104);
    EXPECT_EQ(signalValue, 9);
    EXPECT_FALSE(event.valid());
    EXPECT_EQ(event.engine, pto::comm::DmaEngine::SDMA);
}

TEST(TPutAsyncNotify, AtomicAddUsesSignedIncrement)
{
    int32_t source[4] = {1, 2, 3, 4};
    int32_t destination[4] = {};
    alignas(int32_t) int32_t signalValue = 11;

    Global4 sourceTensor(source);
    Global4 destinationTensor(destination);
    pto::comm::Signal signal(&signalValue);
    pto::comm::AsyncSession session;

    const pto::comm::AsyncEvent event = pto::comm::TPUT_ASYNC_NOTIFY<pto::comm::DmaEngine::URMA>(
        destinationTensor, sourceTensor, signal, -3, pto::comm::NotifyOp::AtomicAdd, session, 23U);

    EXPECT_EQ(destination[0], 1);
    EXPECT_EQ(destination[1], 2);
    EXPECT_EQ(destination[2], 3);
    EXPECT_EQ(destination[3], 4);
    EXPECT_EQ(signalValue, 8);
    EXPECT_FALSE(event.valid());
    EXPECT_EQ(event.engine, pto::comm::DmaEngine::URMA);
}

} // namespace
