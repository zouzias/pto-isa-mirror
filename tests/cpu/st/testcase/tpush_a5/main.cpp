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

class TPUSH_A5Test : public testing::Test {
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

static __aicore__ void main_incore_0_aic_BI_LEFTRIGHT(__gm__ float *v1, __gm__ float *v2, __gm__ float *v3, int32_t v4)
{
#include "oversize_bodies/main_incore_0_aic_bi_leftright_body.inl"
}

static __aicore__ void main_incore_0_aiv_BI_LEFTRIGHT(__gm__ float *v1, __gm__ float *v2, __gm__ float *v3, int32_t v4)
{
#include "oversize_bodies/main_incore_0_aiv_bi_leftright_body.inl"
}

static __aicore__ void main_incore_0_aic_BI_NOSPLIT(__gm__ float *v1, __gm__ float *v2, __gm__ float *v3, int32_t v4)
{
#include "oversize_bodies/main_incore_0_aic_bi_nosplit_body.inl"
}

static __aicore__ void main_incore_0_aiv_BI_NOSPLIT(__gm__ float *v1, __gm__ float *v2, __gm__ float *v3, int32_t v4)
{
#include "oversize_bodies/main_incore_0_aiv_bi_nosplit_body.inl"
}

static __aicore__ void main_incore_0_aic_BI_TOPDOWN(__gm__ float *v1, __gm__ float *v2, __gm__ float *v3, int32_t v4)
{
#include "oversize_bodies/main_incore_0_aic_bi_topdown_body.inl"
}

static __aicore__ void main_incore_0_aiv_BI_TOPDOWN(__gm__ float *v1, __gm__ float *v2, __gm__ float *v3, int32_t v4)
{
#include "oversize_bodies/main_incore_0_aiv_bi_topdown_body.inl"
}

static __aicore__ void main_incore_0_aic_C2V_NOSPLIT(__gm__ float *v1, __gm__ float *v2, __gm__ float *v3, int32_t v4)
{
#include "oversize_bodies/main_incore_0_aic_c2v_nosplit_body.inl"
}

static __aicore__ void main_incore_0_aiv_C2V_NOSPLIT(__gm__ float *v1, __gm__ float *v2, __gm__ float *v3, int32_t v4)
{
#include "oversize_bodies/main_incore_0_aiv_c2v_nosplit_body.inl"
}

inline void LaunchTPut_BI_LEFTRIGHT(T *out, T *A, T *B, T *C)
{
    pto::cpu_sim::reset_execution_context();
    using MainPipe_1 = TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, true>;
    MainPipe_1::reset_for_cpu_sim();
    size_t v5 = 0;

    std::barrier sync_point(3);

    auto aiv_func = [&](int32_t id) {
        // This sets the thread_local ExecutionContext
        pto::cpu_sim::ScopedExecutionContext ctx(0, id, 2);

        sync_point.arrive_and_wait();
        main_incore_0_aiv_BI_LEFTRIGHT(C, A, B, v5);
    };

    auto aic_func = [&]() {
        // Cube Core: Block 0, Subblock 0, Dim 1
        pto::cpu_sim::ScopedExecutionContext ctx(0, 0, 1);

        sync_point.arrive_and_wait();
        main_incore_0_aic_BI_LEFTRIGHT(C, A, B, v5);
    };

    std::thread v0(aiv_func, 0);
    std::thread v1(aiv_func, 1);
    std::thread c0(aic_func);

    v0.join();
    v1.join();
    c0.join();

    pto::cpu_sim::register_hooks(nullptr, nullptr);
}

inline void LaunchTPut_BI_NOSPLIT(T *out, T *A, T *B, T *C)
{
    pto::cpu_sim::reset_execution_context();
    using MainPipe_2 = TPipe<1, Direction::DIR_BOTH, 8192, 4, 2, true>;
    MainPipe_2::reset_for_cpu_sim();
    size_t v5 = 0;

    std::barrier sync_point(2);

    auto aiv_func = [&](int32_t id) {
        // This sets the thread_local ExecutionContext
        pto::cpu_sim::ScopedExecutionContext ctx(0, id, 1);

        sync_point.arrive_and_wait();
        main_incore_0_aiv_BI_NOSPLIT(C, A, B, v5);
    };

    auto aic_func = [&]() {
        // Cube Core: Block 0, Subblock 0, Dim 1
        pto::cpu_sim::ScopedExecutionContext ctx(0, 0, 1);

        sync_point.arrive_and_wait();
        main_incore_0_aic_BI_NOSPLIT(C, A, B, v5);
    };

    std::thread v0(aiv_func, 0);
    std::thread c0(aic_func);

    v0.join();
    c0.join();

    pto::cpu_sim::register_hooks(nullptr, nullptr);
}

inline void LaunchTPut_BI_TOPDOWN(T *out, T *A, T *B, T *C)
{
    pto::cpu_sim::reset_execution_context();
    using MainPipe_3 = TPipe<2, Direction::DIR_BOTH, 8192, 4, 2, false>;
    MainPipe_3::reset_for_cpu_sim();
    size_t v5 = 0;
    std::cout << "Start" << std::endl;

    std::barrier sync_point(3);

    auto aiv_func = [&](int32_t id) {
        // This sets the thread_local ExecutionContext
        pto::cpu_sim::ScopedExecutionContext ctx(0, id, 2);

        sync_point.arrive_and_wait();
        main_incore_0_aiv_BI_TOPDOWN(C, A, B, v5);
    };

    auto aic_func = [&]() {
        // Cube Core: Block 0, Subblock 0, Dim 1
        pto::cpu_sim::ScopedExecutionContext ctx(0, 0, 1);

        sync_point.arrive_and_wait();
        main_incore_0_aic_BI_TOPDOWN(C, A, B, v5);
    };

    std::thread v0(aiv_func, 0);
    std::thread v1(aiv_func, 1);
    std::thread c0(aic_func);

    v0.join();
    v1.join();
    c0.join();

    // 3. Cleanup
    pto::cpu_sim::register_hooks(nullptr, nullptr);
}

inline void LaunchTPut_C2V_NOSPLIT(T *out, T *A, T *B, T *C)
{
    using MainPipe = TPipe<3, Direction::DIR_C2V, 8192, 8, 2, true>;
    MainPipe::reset_for_cpu_sim();
    size_t v5 = 0;
    std::cout << "Start" << std::endl;

    std::barrier sync_point(2);

    auto aiv_func = [&](int32_t id) {
        pto::cpu_sim::ScopedExecutionContext ctx(0, id, 1);

        sync_point.arrive_and_wait();
        main_incore_0_aiv_C2V_NOSPLIT(C, B, A, v5);
    };

    auto aic_func = [&]() {
        pto::cpu_sim::ScopedExecutionContext ctx(0, 0, 1);

        sync_point.arrive_and_wait();
        main_incore_0_aic_C2V_NOSPLIT(C, B, A, v5);
    };

    std::thread v0(aiv_func, 0);
    std::thread c0(aic_func);

    c0.join();
    v0.join();

    pto::cpu_sim::register_hooks(nullptr, nullptr);
}

template <int key>
void test_tpush()
{
#include "oversize_bodies/test_tpush_body.inl"
}

TEST_F(TPUSH_A5Test, case_1)
{
    test_tpush<1>();
}

TEST_F(TPUSH_A5Test, case_2)
{
    test_tpush<2>();
}

} // namespace
