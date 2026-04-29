/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_AIV_TREDUCE_HPP
#define PTO_AIV_TREDUCE_HPP

// pto::aiv::launch_treduce — Step 3 host launcher for the AIV-side gate trigger.
//
// Background
// ----------
// Step 2 of the pto-gated-reduce-kernel plan uses host-side `pto::host::HostTriggerCke`
// to release a CCU kernel parked at `WaitEvent(gateEvent_)`. That path goes through
// `RaCustomChannel(SET_CKE)` → driver ioctl, which adds host µs of latency on every
// gated kernel launch.
//
// Step 3 replaces the host trigger with an AIV-side raw MMIO store directly into
// the CCU CKE register file. The CCU resource VA (~`0xf98005000000` on the 950
// reference rig) is obtained via `pto::host::QueryCcuBaseInfo` (op=GET_BASIC_INFO,
// `out.data + 0x28`) and stuffed into `desc.mmioAddr`. `launch_treduce` then
// dispatches a 1-block AIV kernel that writes `desc.mask` at the right per-CKE
// offset within that base.
//
// The exact `(ckeId → byte_offset)` formula is determined empirically by sweeping
// `PTO_AIV_TRIGGER_STRIDE` / `PTO_AIV_TRIGGER_BYTE_OFF` (see launcher impl);
// initial candidates reflect the host-side `PTO_CKE_LAYOUT=B` byte position
// (slot[6..7]) with stride 0x40 (sizeof(union ccu_data_type_union)).
//
// Failure modes
// -------------
//   - `desc.mmioAddr == 0` → caller forgot QueryCcuBaseInfo; refuse to launch.
//   - AIV kernel writes wrong offset → CCU kernel keeps WaitEvent-ing → caller's
//     `aclrtSynchronizeStream` times out. Switch stride/byte-off candidate, retry.
//   - AIV kernel store hits unmapped device VA → NPU exception, kernel dies.
//     Bounded recovery: kill process + reset NPU.

#include <cstdint>

#include "pto/host/pto_gate_descriptor.hpp"

namespace pto {
namespace aiv {

// Status codes returned by launch_treduce. 0 = success.
enum LaunchTReduceStatus : int32_t {
    kLaunchTReduceOk            = 0,
    kLaunchTReduceMmioAddrZero  = -1,  // desc.mmioAddr not populated by caller
    kLaunchTReduceLoadFailed    = -2,  // kernel .so / symbol resolution failed
    kLaunchTReduceLaunchFailed  = -3,  // aclrtLaunchKernel returned non-zero
    kLaunchTReduceBadArg        = -4,  // null stream / mask=0 / ckeId out of range
};

// Launch the AIV trigger kernel on `stream`. The kernel runs asynchronously
// on the device; caller is expected to issue `aclrtSynchronizeStream(stream)`
// after this returns to wait for the gated CCU kernel to complete.
//
// Args:
//   stream  ACL stream the gated reduce_scatter is queued on; the trigger
//           kernel is enqueued on the SAME stream so ordering is implicit.
//   desc    Fully populated descriptor. `mmioAddr` MUST be non-zero; caller
//           is expected to have filled it via QueryCcuBaseInfo or equivalent.
//
// Returns: `kLaunchTReduceOk` (= 0) on success, or one of LaunchTReduceStatus.
inline int32_t launch_treduce(void *stream, const host::PtoGateDescriptor &desc);

} // namespace aiv
} // namespace pto

#include "pto/aiv/detail/treduce_launcher.hpp"

#endif // PTO_AIV_TREDUCE_HPP
