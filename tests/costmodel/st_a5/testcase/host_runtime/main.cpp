/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <acl/acl.h>

#include "pto/costmodel/perf_sim/recorder.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace {

TEST(A5CostmodelHostRuntime, AllocationCopyStreamAndDeviceLifecycle)
{
    constexpr size_t kElementCount = 16;
    constexpr size_t kBufferSize = kElementCount * sizeof(uint32_t);

    EXPECT_EQ(aclInit(nullptr), ACL_SUCCESS);
    EXPECT_EQ(aclrtSetDevice(0), ACL_SUCCESS);

    int deviceId = -1;
    uint32_t deviceCount = 0;
    EXPECT_EQ(aclrtGetDevice(&deviceId), ACL_SUCCESS);
    EXPECT_EQ(deviceId, 0);
    EXPECT_EQ(aclrtGetDeviceCount(&deviceCount), ACL_SUCCESS);
    EXPECT_EQ(deviceCount, 1U);

    aclrtStream stream = nullptr;
    EXPECT_EQ(aclrtCreateStream(&stream), ACL_SUCCESS);
    ASSERT_NE(stream, nullptr);

    std::array<uint32_t, kElementCount> input{};
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<uint32_t>(i + 1);
    }

    void* hostOutput = nullptr;
    void* deviceBuffer = nullptr;
    EXPECT_EQ(aclrtMallocHost(&hostOutput, kBufferSize), ACL_SUCCESS);
    EXPECT_EQ(aclrtMalloc(&deviceBuffer, kBufferSize, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_NE(hostOutput, nullptr);
    ASSERT_NE(deviceBuffer, nullptr);

    EXPECT_EQ(
        aclrtMemcpy(deviceBuffer, kBufferSize, input.data(), kBufferSize, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    EXPECT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    EXPECT_EQ(aclrtMemcpy(hostOutput, kBufferSize, deviceBuffer, kBufferSize, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    const auto* output = static_cast<const uint32_t*>(hostOutput);
    for (size_t i = 0; i < kElementCount; ++i) {
        EXPECT_EQ(output[i], input[i]);
    }

    EXPECT_EQ(aclrtMemset(deviceBuffer, kBufferSize, 0, kBufferSize), ACL_SUCCESS);
    EXPECT_EQ(aclrtMemcpy(hostOutput, kBufferSize, deviceBuffer, kBufferSize, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    for (size_t i = 0; i < kElementCount; ++i) {
        EXPECT_EQ(output[i], 0U);
    }

    EXPECT_TRUE(::pto::perf_sim::PtoRecorder::Get().empty());
    EXPECT_TRUE(::pto::perf_sim::SyncRecorder::Get().empty());

    EXPECT_EQ(aclrtFree(deviceBuffer), ACL_SUCCESS);
    EXPECT_EQ(aclrtFreeHost(hostOutput), ACL_SUCCESS);
    EXPECT_EQ(aclrtDestroyStream(stream), ACL_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(0), ACL_SUCCESS);
    EXPECT_EQ(aclFinalize(), ACL_SUCCESS);
}

TEST(A5CostmodelHostRuntime, RejectsInvalidBufferRanges)
{
    std::array<uint8_t, 4> source{1, 2, 3, 4};
    std::array<uint8_t, 2> destination{};

    EXPECT_NE(
        aclrtMemcpy(destination.data(), destination.size(), source.data(), source.size(), ACL_MEMCPY_HOST_TO_DEVICE),
        ACL_SUCCESS);
    EXPECT_NE(aclrtMemset(destination.data(), destination.size(), 0, source.size()), ACL_SUCCESS);
}

} // namespace
