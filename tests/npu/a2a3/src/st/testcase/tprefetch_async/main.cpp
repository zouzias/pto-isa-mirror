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
// Single-card functional-correctness tests for device-side async L2 prefetch.
//
// Scope is intentionally limited to the public `pto::TPREFETCH_ASYNC`
// GlobalTensor API: prefetch a tile, wait on the event, then TLOAD/TSTORE
// through it and verify the output. Performance comparison test cases (host
// PTO_PREFETCH vs. device async, static-address fused multi-stage, data-
// dependent fused multi-stage) used to live here but were removed together
// with their kernels because every extra prefetch-using AICORE kernel
// instantiation embeds the SDMA session-init cast chain into the function
// body, and Bisheng's optimizer ICEs (Segmentation fault in
// llvm::CastInst::CreateBitOrPointerCast) once a single TU accumulates too
// many of those expansions. They will return as a dedicated perf TU once the
// optimizer budget is increased.
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

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
