/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_HOST_QUERY_CCU_BASEINFO_HPP
#define PTO_HOST_QUERY_CCU_BASEINFO_HPP

// QueryCcuBaseInfo: probe driver CCU baseinfo via RaCustomChannel(op=11).
//
// Purpose
// -------
// Step 3 of the pto-gated-reduce-kernel plan needs the CCU MMIO base address
// (a.k.a. `resourceAddr` in driver speak) so that an AIV-side
// `pto_gate_trigger` intrinsic can compute the per-CKE MMIO offset and write
// the gate bit directly without round-tripping through the host driver.
//
// The driver does not expose a typed structured-info API in any public CANN
// header, so we go through the same unstructured `RaCustomChannel` channel
// that `host_trigger_cke_impl.hpp` already uses for SET_CKE — but with
// `op = CCU_U_OP_GET_BASIC_INFO (= 11)` instead of `CCU_U_OP_SET_CKE (=254)`.
//
// The output layout was reverse-engineered from
// `hccp_test.so::test_ra_get_ccu_baseifno_test` on driver 7.0.t9.0.B806:
//
//     out.data + 0x10   (4B)  — uint32 field A (semantics unknown)
//     out.data + 0x20   (4B)  — uint32 field B (semantics unknown)
//     out.data + 0x28   (8B)  — void*  resourceAddr  ← what Step 3 needs
//
// The `(uint32_t op=11)` in the IN payload is at the natural struct offset
// inside `DriverCustomChanInfoIn` (= +0x804); we reuse that struct, so no
// special pointer arithmetic is needed here.
//
// This header is **diagnostic only** — it does not get linked into a hot
// path. Callers that want production access to the MMIO base in Step 3 should
// migrate to a typed helper once the layout is confirmed and stable.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "pto/host/detail/host_trigger_cke_impl.hpp"

namespace pto {
namespace host {

// ccu_u_opcode_t::CCU_U_OP_GET_BASIC_INFO  (driver-internal enum value).
constexpr uint32_t kDriverOpGetBasicInfo = 11;

// Result of one QueryCcuBaseInfo call. All fields are populated regardless of
// success/failure to make logs maximally useful (`rc != 0` means the driver
// call failed entirely; `opRet != 0` means the call succeeded but driver
// returned a non-zero status code in the OUT struct).
struct CcuBaseInfoProbe {
    int rc;                      // RaCustomChannel return code (0 = ok)
    int opRet;                   // out.opRet from driver (0 = ok)
    uint32_t fieldAtOffset0x10;  // see header comment
    uint32_t fieldAtOffset0x20;  // see header comment
    void *resourceAddr;          // *(void**)(out.data + 0x28)
    uint8_t firstBytes[64];      // out.data[0..63] for hex inspection
};

// QueryCcuBaseInfo: synchronous probe; performs one RaCustomChannel round-trip.
// Returns a fully populated `CcuBaseInfoProbe` (zero-initialized on dlopen
// failure — `rc == -1` in that case to mirror `kHostTriggerCkeDlopenFailed`).
//
// `dieId` is written into `in.data[0..3]` to mirror the test harness; the
// driver may or may not key off this value depending on the SoC. Pass the same
// dieId you would for HostTriggerCke.
inline CcuBaseInfoProbe QueryCcuBaseInfo(uint32_t devPhyId, uint32_t dieId) {
    CcuBaseInfoProbe probe{};

    detail::LibHccp &lib = detail::LibHccp::Instance();
    if (!lib.Ok()) {
        probe.rc = -1;  // mirror kHostTriggerCkeDlopenFailed
        return probe;
    }

    detail::DriverRaInfo info{detail::kDriverNetworkOffline, devPhyId};
    detail::DriverCustomChanInfoIn in{};
    detail::DriverCustomChanInfoOut out{};

    in.op = kDriverOpGetBasicInfo;
    *reinterpret_cast<uint32_t *>(in.data) = dieId;

    probe.rc = lib.RaCustomChannel()(info, &in, &out);
    probe.opRet = out.opRet;

    if (probe.rc == 0) {
        std::memcpy(&probe.fieldAtOffset0x10, out.data + 0x10, sizeof(uint32_t));
        std::memcpy(&probe.fieldAtOffset0x20, out.data + 0x20, sizeof(uint32_t));
        std::memcpy(&probe.resourceAddr, out.data + 0x28, sizeof(void *));
        std::memcpy(probe.firstBytes, out.data, sizeof(probe.firstBytes));
    }
    return probe;
}

// Convenience: dump the probe to stderr in a human-readable form. Format is
// ad-hoc — only intended for one-shot debug logs, not for grep-driven
// pipelines.
inline void DumpCcuBaseInfoProbe(uint32_t devPhyId, uint32_t dieId,
                                 const CcuBaseInfoProbe &probe) {
    std::fprintf(stderr,
                 "[DIAG/ccu_baseinfo] devPhyId=%u dieId=%u rc=%d opRet=%d\n",
                 devPhyId, dieId, probe.rc, probe.opRet);
    if (probe.rc != 0) {
        std::fprintf(stderr,
                     "[DIAG/ccu_baseinfo] driver call FAILED — see rc above; "
                     "no out.data to dump\n");
        return;
    }
    std::fprintf(stderr,
                 "[DIAG/ccu_baseinfo]   field@+0x10 = 0x%08x (%u)\n",
                 probe.fieldAtOffset0x10, probe.fieldAtOffset0x10);
    std::fprintf(stderr,
                 "[DIAG/ccu_baseinfo]   field@+0x20 = 0x%08x (%u)\n",
                 probe.fieldAtOffset0x20, probe.fieldAtOffset0x20);
    std::fprintf(stderr,
                 "[DIAG/ccu_baseinfo]   resourceAddr@+0x28 = %p   <-- Step 3 candidate\n",
                 probe.resourceAddr);
    std::fprintf(stderr, "[DIAG/ccu_baseinfo]   out.data[0..64]:\n");
    for (std::size_t row = 0; row < sizeof(probe.firstBytes); row += 16) {
        std::fprintf(stderr, "[DIAG/ccu_baseinfo]     +0x%02zx:", row);
        for (std::size_t b = 0; b < 16; ++b) {
            std::fprintf(stderr, " %02x", probe.firstBytes[row + b]);
        }
        std::fprintf(stderr, "\n");
    }
}

} // namespace host
} // namespace pto

#endif // PTO_HOST_QUERY_CCU_BASEINFO_HPP
