/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

// ============================================================================
// tprefetch_compare (cross-rank, Scenario D) — receiver-side prefetch over a
// TPUT_ASYNC link, comparing host-initiated pto::PTO_PREFETCH against
// device-initiated pto::TPREFETCH_L2.
//
// Single-card scenarios A / B / C live under
// tests/npu/a2a3/src/st/testcase/tprefetch_compare/ since none of them need
// the HCCL test scaffold pulled in by `comm/st`.
// ============================================================================

// ---- Scenario D: cross-rank receiver-side prefetch -----------------------
// Producer-consumer pipeline over a TPUT_ASYNC link between two ranks:
//   Rank 0 (sender)   → pto::comm::TPUT_ASYNC(remote_recvBuf, sendBuf)
//   Rank 1 (receiver) → 3 prefetch modes on its own recvBuf, then TLOAD:
//     D0 no-prefetch
//     D1 host   pto::PTO_PREFETCH (host-issued aclrtCmoAsync + stream sync)
//     D2 device pto::TPREFETCH_L2 (in-kernel SQE + Wait)
// All three issue exactly one SDMA CMO prefetch (or none) against the
// freshly received recvBuf, then measure the TLOAD sweep. The receiver
// returns kernel cycles (TLOAD-only in D0/D1, prefetch+wait+TLOAD in D2)
// and host wall-clock around the "prefetch + TLOAD kernel" phase.
// Requires mpirun -n 2; sender rank = 0, first device id = 0.
template <typename T, size_t count>
bool RunScenarioDCrossRank(int nRanks, int firstDeviceId);
