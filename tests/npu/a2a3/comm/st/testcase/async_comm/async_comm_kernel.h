/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once

#include <cstddef>
#include <cstdint>

// Test 1: Original smoke test — v2=[1000..1127] overwritten into v1.
bool RunAsyncCommTest(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Test 2: Zero source — v2 all-zero overwrites non-zero v1.
bool RunZeroDataTest(int first_device_id);

// Test 3: Negative floats — v2=[-1..-128], verifies sign bits transfer correctly.
bool RunNegativeDataTest(int first_device_id);

// Test 4: Multi-launch — same kernel launched 3 times with different data,
//         verifies SDMA session state does not leak between launches.
bool RunMultiLaunchTest(int first_device_id);

// Test 5: Cross-rank — rank 0 (device first_device_id) pushes its sendBuf to
//         rank 1 (device first_device_id+1) via SDMA RDMA TPUT_ASYNC.
//         Rank 1 verifies the received data matches rank 0's sendBuf.
bool RunCrossRankTest(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Test 6: 4-rank root-put — rank 0 (root) sends to ranks 1/2/3 via 3
//         sequential async_comm_kernel launches (fresh workspace each time).
//         Each non-root rank verifies it received root's data.
bool RunRootPut4RanksTest(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
