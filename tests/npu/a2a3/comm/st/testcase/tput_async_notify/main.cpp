/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>

#include "tput_async_notify_kernel.h"
#include "../comm_mpi.h"

TEST(TPutAsyncNotify, SdmaSetQueue4_2Ranks) { ASSERT_TRUE(RunTPutAsyncNotifySet(2, 2, 0, 0, 4)); }

TEST(TPutAsyncNotify, SdmaAtomicAddQueue4_2Ranks) { ASSERT_TRUE(RunTPutAsyncNotifyAdd(2, 2, 0, 0, 4)); }

TEST(TPutAsyncNotify, SdmaAtomicAdd65PostsRingReuse_2Ranks)
{
    ASSERT_TRUE(RunTPutAsyncNotifyAddRingReuse(2, 2, 0, 0, 4));
}

TEST(TPutAsyncNotify, SdmaAtomicAdd700PostsSqCapacityReuse_2Ranks)
{
    ASSERT_TRUE(RunTPutAsyncNotifySqCapacityReuse(2, 2, 0, 0, 4));
}

TEST(TPutAsyncNotify, SdmaAtomicAddInterleavedOrdinaryRingReuse_2Ranks)
{
    ASSERT_TRUE(RunTPutAsyncNotifyInterleavedRingReuse(2, 2, 0, 0, 4));
}

TEST(TPutAsyncNotify, RejectNonZeroChannelGroup_2Ranks)
{
    ASSERT_TRUE(RunTPutAsyncNotifyRejectNonZeroGroup(2, 2, 0, 0, 4));
}

int main(int argc, char** argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
