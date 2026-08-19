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

TEST(TPutAsyncNotifyUrma, Set)
{
    if (CommMpiSize() != 2) {
        GTEST_SKIP() << "Requires exactly two MPI ranks";
    }
    ASSERT_TRUE(RunTPutAsyncNotifyUrma(2, 2, 0, 0, UrmaNotifyMode::Set));
}

TEST(TPutAsyncNotifyUrma, AtomicAdd)
{
    if (CommMpiSize() != 2) {
        GTEST_SKIP() << "Requires exactly two MPI ranks";
    }
    ASSERT_TRUE(RunTPutAsyncNotifyUrma(2, 2, 0, 0, UrmaNotifyMode::AtomicAdd));
}

TEST(TPutAsyncNotifyUrma, AtomicAddReusesInternalResultRing)
{
    if (CommMpiSize() != 2) {
        GTEST_SKIP() << "Requires exactly two MPI ranks";
    }
    ASSERT_TRUE(RunTPutAsyncNotifyUrma(2, 2, 0, 0, UrmaNotifyMode::AtomicAddRingReuse));
}

TEST(TPutAsyncNotifyUrma, MixedPutGetAndNotifyFinalWait)
{
    if (CommMpiSize() != 2) {
        GTEST_SKIP() << "Requires exactly two MPI ranks";
    }
    ASSERT_TRUE(RunTPutAsyncNotifyUrma(2, 2, 0, 0, UrmaNotifyMode::MixedAsyncOperations));
}

int main(int argc, char** argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    CommMpiFinalize();
    return result;
}
