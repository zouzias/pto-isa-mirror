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

// pto::ccu::PtoGatedKernel — public ABI for the pto-isa "gated CCU kernel"
// pilot operator.
// =============================================================================
//
// Architectural intent (2026-05-06 phase 2 pilot)
// -----------------------------------------------
// The "gated CCU kernel" is a pto-native operator (analogue of TPutAsync /
// TReduce in the AscendC ISA family, but living entirely on the host side as
// CCU microcode IR). It demonstrates the AIV-triggered CKE gate path end to
// end, with **zero modification to hccl or hcomm**:
//
//   ┌───────────────────────────────────────────────────────────────────────┐
//   │ pto-isa repo (ST + kernel body live here)                             │
//   ├───────────────────────────────────────────────────────────────────────┤
//   │   kernels/host/gated_reduce_scatter/                                  │
//   │     pto_gated_kernel.{h,cc}    — subclass hcomm::CcuKernel            │
//   │       Algorithm()  → builds CCU IR with WaitEvent(gate)               │
//   │       GeneArgs()   → publishes (dieId,ckeId,mask) via                 │
//   │                       pto::ccu::Publish() so ST/AIV trigger can read  │
//   │     CMakeLists.txt — links libhcomm.so (public pkg_inc only)          │
//   │                                                                       │
//   │   tests/host/gated_reduce_scatter/main.cc — ST orchestration:        │
//   │     1. HcclCommInitRootInfo   (build hccl comm)                      │
//   │     2. HcclThreadAcquire      (acquire ccu thread/stream)            │
//   │     3. HcclChannelAcquire     (acquire peer channels — optional      │
//   │                                  for single-rank pilot)              │
//   │     4. HcclCcuKernelRegister  (push KernelCreator + KernelArg)       │
//   │     5. HcclCcuKernelRegisterFinish — hcomm translates IR → microcode │
//   │     6. HcclCcuKernelLaunch    (push CCU stream → blocks on gate)     │
//   │     7. pto::ccu::TryGet       (read back published descriptor)       │
//   │     8. pto::host::QueryCcuBaseInfo — get CCU MMIO base VA            │
//   │     9. pto::aiv::launch_treduce — AIV-side MMIO write releases gate  │
//   │    10. aclrtSynchronizeStream                                         │
//   └───────────────────────────────────────────────────────────────────────┘
//
// All public APIs in steps 1–6 come from `hcomm/pkg_inc/`, which is the public
// CANN package include surface. No hccl-internal headers (`OpParam`,
// `CcuKernelAlgBase`, `CcuKernelArgReduceScatterMesh1D`, etc.) are required.
//
// Pilot scope vs future work
// --------------------------
// The pilot kernel performs a **placeholder reduce-scatter**:
//   - For nranks=1 it degenerates to "copy input → output" via LocalCopyNb,
//     which is technically a correct identity reduce-scatter (and is what hccl
//     mesh1d returns when subCommRanks_.size()==1 too).
//   - For nranks>1 the placeholder kernel does NOT replicate hccl's full
//     GroupReduce data path. That requires `CcuKernelAlgBase::GroupReduce`
//     plus its ~1500 lines of utility code, which lives in hccl-internal
//     `op_common/template/ccu/` and is outside hcomm pkg_inc. Migrating that
//     is a follow-up (phase 3 — see plan §3.13).
//
// The pilot's primary deliverable is the **gate path**, not the reduce
// semantics: prove that pto-isa can author a CCU kernel that uses
// `WaitEvent(gateEvent_)`, register/translate/launch via hcomm public APIs,
// and have an AIV-side `pto::aiv::launch_treduce` release the gate so the
// CCU stream completes. That is sufficient to validate the architectural
// pattern for future pto-native gated kernels.
//
// History
// -------
// Phase 1 (2026-05-06, retired) — `kernels/host/pto_ccu_host/` shipped the
// gate descriptor registry alone, with hccl still owning the CCU kernel body
// and example main.cc round-tripping descriptors through pto-isa as a smoke
// test. Phase 2 (this header) supersedes that approach by moving the kernel
// body itself into pto-isa. The phase 1 registry (`pto_ccu_host/`) is reused
// as-is — it's still the publish/consume side of the descriptor pipe.
//
// hccl revert path: once this pilot ST is verified, the changes from commit
// `8d100799` ("update", 2026-04-27 — which adds 35 lines of WaitEvent /
// RecordEvent / Publish to `ccu_kernel_reduce_scatter_mesh1d.{h,cc}` in hccl)
// can be reverted in full, restoring hccl to its pre-gate baseline. See plan
// §3.12 for the detailed revert checklist.

#ifndef PTO_CCU_PTO_GATED_KERNEL_HPP
#define PTO_CCU_PTO_GATED_KERNEL_HPP

#include <cstdint>
#include <memory>

// hcomm public package includes — these live under `${ASCEND_HOME_PATH}/pkg_inc`
// once CANN is installed, or `<repo>/hcomm/pkg_inc` in source builds.
#include "hcomm/ccu/ccu_kernel.h"
#include "hcomm/ccu/ccu_kernel_arg.h"
#include "hcomm/ccu/ccu_kernel_signature.h"
#include "hcomm/ccu/ccu_task_arg_v1.h"

namespace pto {
namespace ccu {

// PtoGatedKernelArg
// =================
// Static configuration handed to `HcclCcuKernelRegister`. Carries the
// information `Algorithm()` needs to lay out CCU resources (gate / done event
// masks, rank topology). Kept flat — no `OpParam` / `subCommRanks` / `OpMode`
// dependency on hccl-internal types — so this header is buildable with hcomm
// pkg_inc alone.
//
// All 5 fields are signature-relevant: two `PtoGatedKernelArg`s with different
// rank or mask values produce different microcode and therefore different
// `CcuKernelSignature`s, which is what `HcclCcuKernelRegister` keys off when
// caching translated kernels.
struct PtoGatedKernelArg : public hcomm::CcuKernelArg {
    // Rank id of THIS process within the gated kernel's communicator. For a
    // 2-rank pilot launched via `mpirun -n 2`, this is 0 or 1.
    uint32_t rankId{0};

    // Total ranks participating. Phase 2 pilot only formally supports
    // `rankSize == 1` (single-rank identity reduce-scatter); larger sizes
    // currently fall through to the same identity copy and skip the cross-die
    // GroupReduce data path. Tracked in plan §3.13.
    uint32_t rankSize{1};

    // 16-bit mask the AIV trigger writes into the CKE register slot. Default
    // matches the empirically validated layout (Exp3, 2026-05-06):
    // `byte_off=6, mask=1u<<0`.
    uint32_t gateMask{1u << 0};

    // Mask used for the kernel's "done" CompletedEvent. Currently unused by
    // host orchestration (we rely on `aclrtSynchronizeStream` for completion);
    // kept so future async consumers can WaitEvent on it.
    uint32_t doneMask{1u << 0};

    // Bytes the placeholder identity-copy will move from input to output.
    // Compile-time signature input — drives microcode buffer-loop sizing
    // even though the actual length value comes from `PtoGatedTaskArg.length`
    // at launch time.
    uint64_t payloadBytes{0};

    PtoGatedKernelArg() = default;
    PtoGatedKernelArg(uint32_t rid, uint32_t rsize, uint64_t bytes,
                      uint32_t gMask = (1u << 0), uint32_t dMask = (1u << 0))
        : rankId(rid), rankSize(rsize), gateMask(gMask), doneMask(dMask),
          payloadBytes(bytes)
    {
    }

    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature sig;
        sig.Append(std::string("pto::ccu::PtoGatedKernelArg::v1"));
        sig.Append(rankId);
        sig.Append(rankSize);
        sig.Append(gateMask);
        sig.Append(doneMask);
        sig.Append(payloadBytes);
        return sig;
    }
};

// PtoGatedTaskArg
// ===============
// Per-launch dynamic args handed to `HcclCcuKernelLaunch`. `Algorithm()`'s
// `Load()` calls reserve 4 GeneArgs slots in this exact order:
//   [0] inputAddr, [1] outputAddr, [2] token, [3] length.
// `GeneArgs()` packs these slots from the fields below.
struct PtoGatedTaskArg : public hcomm::CcuTaskArg {
    uint64_t inputAddr{0};   // device VA of input buffer (this rank's slice)
    uint64_t outputAddr{0};  // device VA of output buffer (this rank's slice)
    uint64_t length{0};      // bytes to copy this launch (must be ≤ payloadBytes)
    uint64_t token{0};       // mem-reg token; `GetTokenInfo(va, size)` if needed

    PtoGatedTaskArg() = default;
    PtoGatedTaskArg(uint64_t in, uint64_t out, uint64_t len, uint64_t tok)
        : inputAddr(in), outputAddr(out), length(len), token(tok)
    {
    }
};

// MakeGatedKernelCreator
// ======================
// Returns a `hcomm::KernelCreator` (= `std::function<unique_ptr<CcuKernel>(
//   const CcuKernelArg&)>`) suitable for passing as the `kernelCreator`
// argument of `HcclCcuKernelRegister(comm, &handle, &creator, &arg)`.
//
// The lambda dyn-casts `arg` to `PtoGatedKernelArg` and constructs a
// `detail::PtoGatedReduceScatterMesh1D` — the actual CCU kernel impl. The
// implementation lives in `kernels/host/gated_reduce_scatter/`; this factory
// is the only entry point exposed to ST code.
hcomm::KernelCreator MakeGatedKernelCreator();

}  // namespace ccu
}  // namespace pto

#endif  // PTO_CCU_PTO_GATED_KERNEL_HPP
