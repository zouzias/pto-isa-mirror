/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_HOST_HOST_TRIGGER_CKE_HPP
#define PTO_HOST_HOST_TRIGGER_CKE_HPP

#include <cstdint>

#include "pto/host/pto_gate_descriptor.hpp"

namespace pto {
namespace host {

// Status codes returned by `HostTriggerCke()`. 0 = success.
//
// Implementation detail: the codes match what `host_trigger_cke_impl.hpp`
// produces, but callers should only test `== 0` vs `!= 0`. Specific non-zero
// values are stable for logging only.
enum HostTriggerCkeStatus : int32_t {
    kHostTriggerCkeOk = 0,
    kHostTriggerCkeDlopenFailed = -1,   // libhccp.so / RaCustomChannel not found
    kHostTriggerCkeClearFailed = -2,    // chip-mandated pre-set clear step failed
    kHostTriggerCkeSetFailed = -3,      // actual mask SET step failed
    kHostTriggerCkeBadArg = -4,         // descriptor sanity check failed
};

// HostTriggerCke: host-side helper that drives the CCU CKE specified by `desc`
// to a logical "set" state, equivalent to one in-kernel `RecordEvent(gate)`.
//
// Sequence (chip rule, see `ccu_u_comm.h:99` "set前需要先清零"):
//   1. RaCustomChannel(SET_CKE, payload = all-zero)   -> clears the entry
//   2. RaCustomChannel(SET_CKE, payload = mask bits)  -> writes the mask
//
// Synchronous and self-contained: when the call returns 0, the driver has
// already acknowledged both transactions; CCU will see the gate high on its
// next read. Caller is responsible for ordering (typically: enqueue the gated
// kernel via `aclrtLaunchKernel`-equivalent, then call this helper; finally
// `aclrtSynchronizeStream`).
//
// Thread safety: the underlying `RaCustomChannel` is documented as thread-safe;
// `dlopen` of `libhccp.so` is wrapped in `std::call_once`. Multiple host threads
// targeting different (devPhyId, dieId, ckeId) tuples may call concurrently.
//
// Cost: dominated by the two driver round-trips (~µs each); negligible compared
// to a real reduce. This API is **not** intended for the hot path long-term;
// it exists to unblock Step 2 of the gated-reduce plan and gets superseded by
// the AIV-side `pto_gate_trigger` intrinsic in Step 3.
//
// Args:
//   devPhyId  Physical device id (matches driver `RaInfo.phyId`). On host the
//             caller typically obtains it from `rtGetDevicePhyIdByIndex()` or
//             from the same source it uses for `aclrtSetDevice()`.
//   desc      The gate to release. `desc.dieId`, `desc.ckeId`, `desc.mask` are
//             consumed; `desc.mmioAddr` is ignored on this path.
//
// Returns: 0 on success, or one of `HostTriggerCkeStatus` on failure.
inline int32_t HostTriggerCke(uint32_t devPhyId, const PtoGateDescriptor &desc);

} // namespace host
} // namespace pto

// Pull in the inline implementation. Kept in `detail/` so callers who only need
// the declaration (e.g. for forward inclusion in shared headers) can still
// `#include "pto/host/host_trigger_cke.hpp"` without paying the `<dlfcn.h>` /
// `<mutex>` parse cost — they would just need to define their own translation
// unit that includes the impl header. The impl header is wrapped in its own
// guards so multiple inclusion is harmless.
#include "pto/host/detail/host_trigger_cke_impl.hpp"

#endif // PTO_HOST_HOST_TRIGGER_CKE_HPP
