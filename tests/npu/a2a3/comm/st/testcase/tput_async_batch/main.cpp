/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cerrno>
#include <cstdlib>
#include <gtest/gtest.h>

#include "../comm_mpi.h"
#include "tput_async_batch_kernel.h"

namespace {

int FirstDeviceId()
{
    const char* value = std::getenv("PTO_COMM_ST_FIRST_DEVICE_ID");
    if (value == nullptr || *value == '\0') {
        return 0;
    }
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return errno == 0 && end != value && *end == '\0' && parsed >= 0 ? static_cast<int>(parsed) : 0;
}

} // namespace

TEST(TPutAsyncBatch, SdmaAggregateFunctionalSuite)
{
    constexpr int kRanks = 2;
    if (CommMpiSize() != kRanks) {
        GTEST_SKIP() << "Requires exactly 2 MPI ranks";
    }
    if (!IsTPutAsyncBatchDeviceRangeAvailable(kRanks, FirstDeviceId())) {
        GTEST_SKIP() << "Requested device range is unavailable";
    }
    ASSERT_TRUE(RunTPutAsyncBatchFunctionalSuite(kRanks, kRanks, 0, FirstDeviceId()));
}

int main(int argc, char** argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
