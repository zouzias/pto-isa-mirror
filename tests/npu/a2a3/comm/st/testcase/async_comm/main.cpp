/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#include "async_comm_kernel.h"
#include "../comm_mpi.h"

// ============================================================================
// Smoke test for the PTOAS-generated async_comm_kernel.
//
// The kernel exercises TPUT_ASYNC + TGET_ASYNC via the SDMA DMA engine on a
// single device.  Only MPI rank 0 runs the kernel; other ranks pass trivially.
// Naming has no "2Ranks"/"4Ranks"/"8Ranks" suffix, so run_st.py routes it to
// the mpirun -n 2 tier and the filter "*-*4Ranks*:...*" selects it.
// ============================================================================
// ----------------------------------------------------------------------------
// Test 1: 原始冒烟测试
// v2=[1000..1127] 通过 TPUT_ASYNC + TGET_ASYNC 写入 v1，验证结果正确。
// ----------------------------------------------------------------------------
TEST(AsyncComm, SmokeTest_Float128)
{
    if (CommMpiRank() != 0) { SUCCEED(); return; }
    int device_id = CommMpiRank() % 1;
    ASSERT_TRUE(RunAsyncCommTest(1, 1, 0, device_id));
}

// ----------------------------------------------------------------------------
// Test 2: 全零源数据
// v1=[1..128]（非零），v2=[0..0]（全零）。
// kernel 把 v1 覆盖成全零，验证不会残留旧数据。
// ----------------------------------------------------------------------------
TEST(AsyncComm, ZeroData_Float128)
{
    if (CommMpiRank() != 0) { SUCCEED(); return; }
    int device_id = CommMpiRank() % 1;
    ASSERT_TRUE(RunZeroDataTest(device_id));
}

// ----------------------------------------------------------------------------
// Test 3: 负数浮点数
// v2=[-1..-128]，验证 SDMA 引擎正确传输负号位，不会截断或符号错误。
// ----------------------------------------------------------------------------
TEST(AsyncComm, NegativeData_Float128)
{
    if (CommMpiRank() != 0) { SUCCEED(); return; }
    int device_id = CommMpiRank() % 1;
    ASSERT_TRUE(RunNegativeDataTest(device_id));
}

// ----------------------------------------------------------------------------
// Test 4: 同一设备连续三次启动 kernel
// 每次用不同的 v2 值（+5000 / +6000 / +7000），验证 SDMA 会话状态
// 在多次 kernel 启动之间不会残留（sqTail、完成标志槽正确重置）。
// ----------------------------------------------------------------------------
TEST(AsyncComm, MultiLaunch_Float128)
{
    if (CommMpiRank() != 0) { SUCCEED(); return; }
    int device_id = CommMpiRank() % 1;
    ASSERT_TRUE(RunMultiLaunchTest(device_id));
}

// ----------------------------------------------------------------------------
// Test 5: 跨 rank SDMA RDMA 传输
// rank 0 (device 6) 通过 SDMA RDMA TPUT_ASYNC 把 sendBuf[0..127] 推送到
// rank 1 (device 7) 的 recvBuf，rank 1 验证收到的数据是否正确。
// 不修改原始 kernel，只靠 host runner 传入 rank 1 的 RDMA 远端地址作为 v1。
// 用例名称不含 4Ranks/8Ranks，由 run_st.py 在 mpirun -n 2 层运行。
// ----------------------------------------------------------------------------
TEST(AsyncComm, CrossRank_Float128)
{
    // n_ranks=2, n_devices=2, first_rank_id=0, first_device_id=6
    // → MPI rank 0 uses device 6, MPI rank 1 uses device 7
    ASSERT_TRUE(RunCrossRankTest(2, 2, 0, 6));
}

// ----------------------------------------------------------------------------
// Test 6: 4-rank broadcast（用例名含 4Ranks → run_st.py 在 mpirun -n 4 层运行）
// rank 0 (device 4) 顺序向 rank 1/2/3 (device 5/6/7) 各发起一次 kernel。
// 每次 kernel 调用内部同时执行 TPUT_ASYNC + TGET_ASYNC（原始 kernel 固有行为）。
// 每次发送使用独立的 SdmaWorkspaceManager（避免 completion slot 污染）。
// 三个非 root rank 各自验证收到了 root 的数据。
// ----------------------------------------------------------------------------
TEST(AsyncComm, RootPut_4Ranks_Float128)
{
    // n_ranks=4, n_devices=4, first_rank_id=0, first_device_id=4
    // MPI rank 0 → device 4, rank 1 → device 5, rank 2 → device 6, rank 3 → device 7
    ASSERT_TRUE(RunRootPut4RanksTest(4, 4, 0, 4));
}

int main(int argc, char **argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
