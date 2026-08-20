/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <pto/common/fifo.hpp>
#include <thread>
#include <vector>
#include <barrier>
#include "test_common.h"

using namespace std;
using namespace pto;
using namespace PtoTestCommon;

namespace {
using T = float;

class TPUSH_A3Test : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

// Pipe and Communication
using MainPipe = TPipe<0, Direction::DIR_BOTH, 8192, 4, 4, false>;

static __aicore__ void main_incore_0_aic(__gm__ float *v1, __gm__ float *v2, __gm__ float *v3, __gm__ float *v4,
                                         int32_t v5)
{
#include "oversize_bodies/main_incore_0_aic_body.inl"
}

static __aicore__ void main_incore_0_aiv(__gm__ float *v1, __gm__ float *v2, __gm__ float *v3, __gm__ float *v4,
                                         int32_t v5)
{
#include "oversize_bodies/main_incore_0_aiv_body.inl"
}

void *g_shared_storage_ptr = nullptr;
// A simple function matching GetPipeSharedStateInjectedHookFn signature
extern "C" void *GlobalPipeHook(uint64_t key, size_t size)
{
    // We'll use a global pointer to store the allocated memory
    return g_shared_storage_ptr;
}

inline void LaunchTPush(T *out, T *A, T *B, T *C)
{
    size_t v5 = 0;
    std::cout << "Start" << std::endl;
    // 1. Allocate and zero the shared synchronization state
    size_t required_size = sizeof(MainPipe::SharedState);
    void *raw_mem = malloc(required_size);
    g_shared_storage_ptr = new (raw_mem) MainPipe::SharedState();

    T *pipe_mem;
    aclrtMalloc((void **)&pipe_mem, 2 * 65536, ACL_MEM_MALLOC_HUGE_FIRST);

    std::barrier sync_point(3);

    pto::cpu_sim::register_hooks(nullptr, (void *)GlobalPipeHook);

    auto aiv_func = [&](int32_t id) {
        pto::cpu_sim::ScopedExecutionContext ctx(0, id, 2);

        sync_point.arrive_and_wait();
        main_incore_0_aiv(C, A, B, pipe_mem, v5);
    };

    auto aic_func = [&]() {
        // Cube Core: Block 0, Subblock 0, Dim 1
        pto::cpu_sim::ScopedExecutionContext ctx(0, 0, 1);

        sync_point.arrive_and_wait();
        main_incore_0_aic(C, A, B, pipe_mem, v5);
    };

    std::thread v0(aiv_func, 0);
    std::thread v1(aiv_func, 1);
    std::thread c0(aic_func);

    v0.join();
    v1.join();
    c0.join();
}

void test_tpush()
{
#include "oversize_bodies/test_tpush_body.inl"
}

TEST_F(TPUSH_A3Test, case_1)
{
    test_tpush();
}

} // namespace
