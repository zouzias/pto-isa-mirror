/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_HOST_PTO_GATE_DESCRIPTOR_HPP
#define PTO_HOST_PTO_GATE_DESCRIPTOR_HPP

#include <cstdint>

namespace pto {
namespace host {

// PtoGateDescriptor: minimal ABI struct describing one CKE-based gate that a
// device-side kernel is waiting on.
//
// Both transports — host trigger (via driver `CCU_U_OP_SET_CKE`, see
// `host_trigger_cke.hpp`) and the future AIV trigger (MMIO write, Step 3 in the
// pto-gated-reduce-kernel plan) — consume this same descriptor. Producers
// (typically a CCU kernel after `Translate()`) populate one descriptor per gated
// kernel instance and expose it through the kernel's `TryGetLastGateDescriptor()`
// helper.
//
// Layout is fixed (16 bytes, no padding on common 64-bit ABIs) so the struct can
// also be passed as raw bytes through SQE arguments / GM staging buffers in
// later phases.
struct PtoGateDescriptor {
    // Logical die index this gate lives on (matches driver `udie_idx`,
    // typically 0 or 1 on dual-die parts).
    uint32_t dieId;

    // Physical CKE entry index (matches driver `offsetStartIdx` in
    // `struct CustomChanInfoIn`). Valid range is `[0, ckeNum)` for the die,
    // resolved by the kernel during `Translate()`.
    uint32_t ckeId;

    // 16-bit signal-event mask (only the low 16 bits are used; per-spec each
    // CKE entry has 16 individually addressable signal bits). The host /
    // AIV producer is responsible for OR-ing concurrent producers' masks.
    uint32_t mask;

    // MMIO address used by the AIV-side `pto_gate_trigger` intrinsic (Step 3).
    // For Step 2 (host driver path) this field is unused and may be 0.
    uint64_t mmioAddr;
};

static_assert(sizeof(PtoGateDescriptor) == 24,
              "PtoGateDescriptor ABI size changed: update SQE/GM consumers");

} // namespace host
} // namespace pto

#endif // PTO_HOST_PTO_GATE_DESCRIPTOR_HPP
