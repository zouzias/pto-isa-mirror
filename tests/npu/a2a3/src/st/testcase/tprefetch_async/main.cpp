/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

#include "tprefetch_async_kernel.h"

// ============================================================================
// Focused single-card tests for device-side async L2 prefetch.
//
// Kept minimal on purpose: each TEST_F triggers an explicit kernel template
// instantiation in tprefetch_async_kernel.cpp, and every prefetch-using kernel
// inlines the SDMA session-init cast chain into the AICORE function body. Past
// experience (see prefetch_talk.md) showed that going beyond ~3 prefetch
// kernel instances per TU pushes the Bisheng optimizer past its CastInst
// budget and triggers an ICE in tprefetch_async_kernel.cpp.o. The host vs.
// device perf comparison, static-address multi-stage and data-dependent
// multi-stage scenarios are intentionally left out for now and tracked
// separately; they can be re-added once the optimizer handles a larger
// per-TU SDMA cast graph.
// ============================================================================

class TPrefetchAsyncTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

TEST_F(TPrefetchAsyncTest, case_float_4096_globaltensor)
{
    ASSERT_TRUE((RunPrefetchAsyncCorrectness<float, 4096>(0)));
}

TEST_F(TPrefetchAsyncTest, case_int32_4096_globaltensor)
{
    ASSERT_TRUE((RunPrefetchAsyncCorrectness<int32_t, 4096>(0)));
}

TEST_F(TPrefetchAsyncTest, case_float_1mb_prefetch_reduce_tload_latency)
{
    ASSERT_TRUE((RunTloadPerf<float, 262144>(0)));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
