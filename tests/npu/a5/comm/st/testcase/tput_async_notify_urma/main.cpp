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

#include "../comm_mpi.h"
#include "tput_async_notify_urma_kernel.h"

// Verify one int32 SET notify, payload delivery, and adjacent signal canaries.
TEST(TPutAsyncNotifyUrma, Int32SetAndCanaries)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunTPutAsyncNotifyUrma(2, 2, 0, 0, UrmaNotifyStMode::Set));
}

// Verify the receiver consumes the URMA payload after observing the notification.
TEST(TPutAsyncNotifyUrma, ReceiverConsumesPayload)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunTPutAsyncNotifyUrma(2, 2, 0, 0, UrmaNotifyStMode::ReceiverConsumeSet));
}

// Verify one int32 FAA notify, payload delivery, and adjacent signal canaries.
TEST(TPutAsyncNotifyUrma, Int32AtomicAddAndCanaries)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunTPutAsyncNotifyUrma(2, 2, 0, 0, UrmaNotifyStMode::AtomicAdd));
}

// Verify 65 SET notifies repeatedly drain and reuse the eight-slot SET source ring.
TEST(TPutAsyncNotifyUrma, SixtyFiveSetBatchDrain)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunTPutAsyncNotifyUrma(2, 2, 0, 0, UrmaNotifyStMode::SetBatchDrain65));
}

// Verify 65 FAA notifies safely share the manager-owned ignored result sink.
TEST(TPutAsyncNotifyUrma, SixtyFiveFaaSharedSink)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunTPutAsyncNotifyUrma(2, 2, 0, 0, UrmaNotifyStMode::FaaSharedSink65));
}

// Verify wait-last completion across an ordered PUT, SET, GET, and FAA stream.
TEST(TPutAsyncNotifyUrma, MixedPutSetGetAddFinalWaitOnly)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunTPutAsyncNotifyUrma(2, 2, 0, 0, UrmaNotifyStMode::MixedPutSetGetAdd));
}

// SharedPool notify: one AIV SET-notifies every peer on a single shared jetty.
TEST(TPutAsyncNotifyUrma, Pool_OneJettyManyPeers_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE((RunNotifyUrmaPool<int32_t, 256, 1, 1, UrmaNotifyPoolPolicy::SingleJetty>(4, 4, 0, 0)));
}

// SharedPool notify: two AIVs each own a distinct jetty and notify every peer.
TEST(TPutAsyncNotifyUrma, Pool_MultiAivDifferentJetties_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE((RunNotifyUrmaPool<int32_t, 256, 2, 2, UrmaNotifyPoolPolicy::AivOwnsJetty>(4, 4, 0, 0)));
}

// SharedPool notify: one AIV round-robins peers across several shared jetties.
TEST(TPutAsyncNotifyUrma, Pool_OneAivManyJetties_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE((RunNotifyUrmaPool<int32_t, 256, 1, 2, UrmaNotifyPoolPolicy::RoundRobinJetty>(4, 4, 0, 0)));
}

// >256MB SET notify: the 257MB payload is auto-split by UrmaPostNotify into
// 256MB + 1MB WQEs with the SET signal still posted last, on a shared jetty.
// The receiver waits on the signal on-device and then reads the payload (sampling
// the last chunk past the 256MB boundary), so a signal that overtook the last
// chunk under RM would fail here rather than be masked by host-side timing.
TEST(TPutAsyncNotifyUrma, Pool_SetOver256MB_Chunked)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunNotifyUrmaPool<int32_t, 67371008, 1, 1, UrmaNotifyPoolPolicy::SingleJetty>(2, 2, 0, 0)));
}

int main(int argc, char** argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int ret = RUN_ALL_TESTS();
    FinalizeTPutAsyncNotifyUrma();
    CommMpiFinalize();
    return ret;
}
