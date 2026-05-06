/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0.
 * Please refer to the License for details. You may not use this file except in
 * compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
 * See LICENSE in the root of the software repository for the full text of the
 * License.
 */

// pto::ccu::PtoGateRegistry
// =============================================================================
// Process-local descriptor registry for PTO gated CCU kernels.
//
// Producer side (CCU kernel `GeneArgs()`): after the microcode has been
// translated and the gate's `(dieId, ckeId, mask)` resolved, the kernel calls
// `pto::ccu::Publish(rankId, dieId, ckeId, mask)` to record the descriptor for
// later consumption by host trigger / AIV trigger code.
//
// Consumer side (host orchestration): example application calls
// `pto::ccu::TryGet(rankId, &out)` (or the C bridge
// `PtoCcuTryGetLastReduceScatterGateDescriptor`) after enqueuing the gated
// collective on the HCCL stream, then uses the descriptor to drive AIV trigger
// (Step 3) — host trigger path is retired post-2026-05-06 (see
// `pto-gated-reduce-kernel_36b1af78.plan.md` §3.11).
//
// History — the equivalent registry lives in
//   `hccl/src/ops/reduce/template/ccu/kernel/pto_gate_registry.{h,cc}`
// exported through `libhccl.so` as `HcclPtoTryGetLastReduceGateDescriptor`. As
// of pto-isa pilot phase 1 (light migration, 2026-05-06) this header + the
// matching `libpto_ccu_host.so` provide the same registry semantics under the
// `pto::ccu` namespace, intended to become the single source of truth in
// phase 2.
//
// Phase 1 (now) — pto-isa registry exists in parallel; example main.cc does
// a round-trip (read hccl bridge → `PtoCcuPublish_C` → `PtoCcuTryGet...`) when
// `HCCL_PTO_USE_ISA_GATED=1` to validate the lib + ABI compat. hccl source is
// not modified.
//
// Phase 2 (deferred) — once hcomm exposes `HcclThreadAcquire` and hccl exposes
// `CcuKernelArg*` types in pkg_inc, the gated CCU kernel will be rewritten as
// a pto-isa subclass and call `pto::ccu::Publish` directly, eliminating the
// hccl-side `pto_gate_registry.{h,cc}`. See pilot plan
// `.cursor/plans/pto-isa_pilot_+_d2_host_fix_0f29f560.plan.md` Part A.
#ifndef PTO_CCU_PTO_GATE_REGISTRY_HPP
#define PTO_CCU_PTO_GATE_REGISTRY_HPP

#include <cstdint>

#include "pto/host/pto_gate_descriptor.hpp"

namespace pto {
namespace ccu {

// Mask bit selected on the gate event's CKE entry. Hardware-confirmed via
// `host_trigger_cke_impl.hpp:186-203` (PTO_CKE_LAYOUT=B sweep, 2026-04-27) AND
// the AIV trigger `Exp3` (rc=0 with `byte_off=6 stride=0x40 mask=0x1`,
// 2026-05-06): chip-level CKE consumes the byte-6 LSB of the slot. Both
// transports converge on `1u << 0` written to byte 6.
constexpr uint32_t PTO_GATE_MASK = 1u << 0;
constexpr uint32_t PTO_DONE_MASK = 1u << 0;

// Env switch checked by `IsPtoGateEnabledFromEnv()`. Mirrors the legacy hccl
// constant in `pto_gate_registry.h`; kept identical so existing run scripts
// (`run_gate_tests_rs.sh`) work unchanged.
constexpr const char *PTO_GATE_ENV = "HCCL_PTO_GATE_REDUCE";

// Returns true iff `getenv(PTO_GATE_ENV) == "1"`. Kernels gate their gate
// arming on this — runtime, not compile-time, so the same libhccl /
// libpto_ccu_host can run baseline and gated workloads without rebuild.
bool IsPtoGateEnabledFromEnv();

// Producer-side: kernel publishes its resolved gate descriptor.
// Last writer wins per `rankId`; the registry is process-local and not a
// stable cross-process store. `mmioAddr` is left zero — host trigger /
// `query_ccu_baseinfo.hpp` resolves the actual VA at trigger time.
void Publish(uint32_t rankId, uint32_t dieId, uint32_t ckeId, uint32_t mask);

// Consumer-side: host pulls the most recently published descriptor for the
// given rank. Returns `false` if no kernel has published yet.
bool TryGet(uint32_t rankId, pto::host::PtoGateDescriptor &out);

} // namespace ccu
} // namespace pto

// C ABI bridge for example apps that load `libpto_ccu_host.so` via dlsym.
// Mirrors the legacy `HcclPtoTryGetLastReduceGateDescriptor` symbol but lives
// in the pto-isa lib and is named to make it obvious where it comes from.
//
// The "ReduceScatter" suffix is historical — the registry is shared across all
// gated CCU kernels (mesh1d / mesh2die / nhr1d_mem2mem / ...). Future kernels
// will publish into the same map; one `TryGet` is sufficient for any of them.
extern "C" {

// Read-side dlsym target. Mirrors the legacy hccl
// `HcclPtoTryGetLastReduceGateDescriptor` symbol semantics; lives in
// `libpto_ccu_host.so`.
bool PtoCcuTryGetLastReduceScatterGateDescriptor(uint32_t rankId,
                                                 pto::host::PtoGateDescriptor *out);

// Write-side dlsym target. Used during pilot phase 1 by example main.cc to
// "round-trip" the descriptor through pto-isa registry: read from hccl bridge,
// `PtoCcuPublish_C(...)` into pto-isa registry, then read back via
// `PtoCcuTryGetLastReduceScatterGateDescriptor` to validate the new lib.
//
// Phase 2 (deferred — see plan A.4 spike): the gated CCU kernel itself will
// call `pto::ccu::Publish` directly during `GeneArgs()`, eliminating the
// round-trip.
void PtoCcuPublish_C(uint32_t rankId, uint32_t dieId, uint32_t ckeId,
                     uint32_t mask);

} // extern "C"

#endif // PTO_CCU_PTO_GATE_REGISTRY_HPP
