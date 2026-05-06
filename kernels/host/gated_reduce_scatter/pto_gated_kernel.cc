/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0.
 */

#include "pto_gated_kernel.h"

#include <cstdio>
#include <memory>

#include "pto/ccu/pto_gate_registry.hpp"

namespace pto {
namespace ccu {
namespace detail {

using namespace hcomm;

// Per-rank stderr trace prefix. Mirrors hccl mesh1d's `[PTO_GATE/kernel/...]`
// tag so log filters that key off `PTO_GATE/` keep working.
//
// All trace prints are stderr+fflush so they survive process abort during
// AIV-trigger debugging (timeouts, sanitizer trips, etc).
namespace {
inline void TracePrintf(const char *tag, uint32_t rankId, const char *body)
{
    std::fprintf(stderr, "[PTO_GATE/kernel/%s] rank=%u %s\n", tag, rankId, body);
    std::fflush(stderr);
}
}  // namespace

PtoGatedReduceScatterMesh1D::PtoGatedReduceScatterMesh1D(const CcuKernelArg &arg)
    : CcuKernel(arg)
{
    const auto *kArg = dynamic_cast<const PtoGatedKernelArg *>(&arg);
    if (kArg != nullptr) {
        rankId_       = kArg->rankId;
        rankSize_     = kArg->rankSize;
        gateMask_     = kArg->gateMask;
        doneMask_     = kArg->doneMask;
        payloadBytes_ = kArg->payloadBytes;
    } else {
        // Defensive — hccl/hcomm should never feed us a non-PtoGatedKernelArg
        // because the `KernelCreator` we hand back from
        // `MakeGatedKernelCreator()` only constructs us. Log loudly so the
        // server-side run flags this as an ABI bug rather than silently
        // running with default zero-init values.
        std::fprintf(stderr,
            "[PTO_GATE/kernel/gated_rs_mesh1d] CTOR FAILED dynamic_cast — got "
            "non-PtoGatedKernelArg, falling back to defaults (rankId=0, "
            "rankSize=1, payloadBytes=0). This kernel will translate but "
            "cannot meaningfully run.\n");
    }
    std::fprintf(stderr,
        "[PTO_GATE/kernel/gated_rs_mesh1d] CTOR rank=%u rankSize=%u "
        "gateMask=0x%x doneMask=0x%x payloadBytes=%llu\n",
        rankId_, rankSize_, gateMask_, doneMask_,
        static_cast<unsigned long long>(payloadBytes_));
    std::fflush(stderr);
}

HcclResult PtoGatedReduceScatterMesh1D::Algorithm()
{
    TracePrintf("gated_rs_mesh1d", rankId_, "Algorithm() entry");

    // -------------------------------------------------------------------------
    // Step 1: allocate CCU IR resources.
    //
    // `CreateVariable()` returns an unbound CCU IR variable that microcode
    // sees as a uint64 slot — `Load()` calls below bind these slots to the
    // GeneArgs() return positions. Order matters; GeneArgs() must return
    // values in the same order Load() is called here.
    // -------------------------------------------------------------------------
    inputVar_  = CreateVariable();
    outputVar_ = CreateVariable();
    tokenVar_  = CreateVariable();
    lengthVar_ = CreateVariable();

    // -------------------------------------------------------------------------
    // Step 2: arm the gate event.
    //
    // `gateEvent_` is the CompletedEvent the kernel will WaitEvent on. Its
    // `(dieId, ckeId)` are resolved by hcomm's `Translate()` step inside
    // `HcclCcuKernelRegisterFinish` and become readable on this object after
    // that returns. `SetMask(gateMask_)` records which signal bits of the
    // CKE register the kernel cares about — the AIV trigger writes these
    // same bits, and the kernel resumes when the AND of (event.mask &
    // bits-set-on-CKE) matches the mask.
    // -------------------------------------------------------------------------
    gateEvent_ = CreateCompletedEvent();
    gateEvent_.SetMask(gateMask_);

    doneEvent_ = CreateCompletedEvent();
    doneEvent_.SetMask(doneMask_);

    // Per-step event for the placeholder LocalCopyNb. Independent from
    // gate/done so reordering across the gate boundary is well-defined.
    copyEvent_ = CreateCompletedEvent();
    copyEvent_.SetMask(1u << 0);

    TracePrintf("gated_rs_mesh1d", rankId_, "InitResources — gate path armed");

    // -------------------------------------------------------------------------
    // Step 3: bind runtime args.
    //
    // Each `Load()` reserves one slot in the GeneArgs() vector and tells the
    // microcode "this Variable's value comes from the Nth uint64 the launcher
    // passes in via taskArgs". The order [input, output, token, length] is
    // the contract `GeneArgs()` must follow exactly.
    // -------------------------------------------------------------------------
    Load(inputVar_);   // [0] inputAddr
    Load(outputVar_);  // [1] outputAddr
    Load(tokenVar_);   // [2] token
    Load(lengthVar_);  // [3] length

    // -------------------------------------------------------------------------
    // Step 4: stall on the gate.
    //
    // After this point the CCU stream is parked at WaitEvent(gateEvent_)
    // until something writes the right mask bits to (gateEvent_.dieId,
    // gateEvent_.ckeId, gateEvent_.mask). For the pilot, that "something"
    // is `pto::aiv::launch_treduce` issued from the host on a parallel AIV
    // stream, which executes an MMIO store to the CKE register. See the
    // ST main.cc for the full release sequence.
    // -------------------------------------------------------------------------
    TracePrintf("gated_rs_mesh1d", rankId_, "about to WaitEvent(gateEvent_)");
    WaitEvent(gateEvent_);
    TracePrintf("gated_rs_mesh1d", rankId_,
                "past WaitEvent(gateEvent_) — gate released");

    // -------------------------------------------------------------------------
    // Step 5: placeholder identity copy (input → output).
    //
    // For nranks=1 a reduce-scatter degenerates to identity copy, which is
    // what hccl mesh1d emits when subCommRanks_.size()==1 too. For nranks>1
    // this is a stand-in — we skip the cross-die GroupReduce path because it
    // depends on hccl-internal `CcuKernelAlgBase` (~1500 lines of utility
    // code outside hcomm pkg_inc). The pilot's primary deliverable is the
    // gate path, not the reduce semantics — see `pto/ccu/pto_gated_kernel.hpp`
    // for the full rationale.
    //
    // The `LocalCopyNb(LocalAddr, LocalAddr, ...)` overload comes straight
    // from hcomm pkg_inc (`ccu_kernel.h:152`), and its semantics on the CCU
    // engine are "scheduled async copy between two local addresses, signal
    // event when done". We `WaitEvent(copyEvent_)` immediately after to keep
    // the kernel sequential and easy to reason about.
    // -------------------------------------------------------------------------
    CcuRep::LocalAddr inputAddr  = CreateLocalAddr();
    inputAddr.addr  = inputVar_;
    inputAddr.token = tokenVar_;

    CcuRep::LocalAddr outputAddr = CreateLocalAddr();
    outputAddr.addr  = outputVar_;
    outputAddr.token = tokenVar_;

    LocalCopyNb(outputAddr, inputAddr, lengthVar_, copyEvent_);
    WaitEvent(copyEvent_);

    // -------------------------------------------------------------------------
    // Step 6: signal completion.
    //
    // Currently no consumer waits on doneEvent_ — the ST relies on
    // `aclrtSynchronizeStream(stream)` for completion ordering. Kept for
    // future async consumers and for parity with hccl mesh1d's gated path.
    // -------------------------------------------------------------------------
    RecordEvent(doneEvent_);
    TracePrintf("gated_rs_mesh1d", rankId_,
                "RecordEvent(doneEvent_) — Algorithm() complete");

    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> PtoGatedReduceScatterMesh1D::GeneArgs(const CcuTaskArg &arg)
{
    const auto *tArg = dynamic_cast<const PtoGatedTaskArg *>(&arg);
    if (tArg == nullptr) {
        std::fprintf(stderr,
            "[PTO_GATE/kernel/gated_rs_mesh1d] GeneArgs FAILED dynamic_cast — "
            "got non-PtoGatedTaskArg. Returning empty args; launch will fail "
            "downstream.\n");
        std::fflush(stderr);
        return {};
    }

    // -------------------------------------------------------------------------
    // Publish gate descriptor.
    //
    // After `HcclCcuKernelRegisterFinish` runs `Translate()`, `gateEvent_`'s
    // `(dieId, ckeId)` are resolved to actual hardware indices. By
    // `Publish()`-ing them on every `Launch()` we keep the registry up to
    // date even if the kernel is launched multiple times with different
    // taskArgs (rare in pilot, but the cost is just a map insert).
    //
    // The AIV trigger reads these values via `pto::ccu::TryGet(rankId, &out)`
    // (or the `extern "C"` bridge `PtoCcuTryGetLastReduceScatterGateDescriptor`
    // for dlopen consumers).
    // -------------------------------------------------------------------------
    const uint32_t dieId = static_cast<uint32_t>(gateEvent_.DieId());
    const uint32_t ckeId = static_cast<uint32_t>(gateEvent_.Id());
    pto::ccu::Publish(rankId_, dieId, ckeId, gateMask_);

    std::fprintf(stderr,
        "[PTO_GATE/kernel/gated_rs_mesh1d] GeneArgs rank=%u — published "
        "descriptor (dieId=%u, ckeId=%u, mask=0x%x); inputAddr=0x%llx "
        "outputAddr=0x%llx length=%llu token=0x%llx\n",
        rankId_, dieId, ckeId, gateMask_,
        static_cast<unsigned long long>(tArg->inputAddr),
        static_cast<unsigned long long>(tArg->outputAddr),
        static_cast<unsigned long long>(tArg->length),
        static_cast<unsigned long long>(tArg->token));
    std::fflush(stderr);

    // Order MUST match the Load() sequence in Algorithm():
    // [0] inputAddr, [1] outputAddr, [2] token, [3] length.
    return {tArg->inputAddr, tArg->outputAddr, tArg->token, tArg->length};
}

}  // namespace detail
}  // namespace ccu
}  // namespace pto

// =============================================================================
// Public factory exposed via `pto/ccu/pto_gated_kernel.hpp`.
// =============================================================================
namespace pto {
namespace ccu {

hcomm::KernelCreator MakeGatedKernelCreator()
{
    return [](const hcomm::CcuKernelArg &arg)
        -> std::unique_ptr<hcomm::CcuKernel> {
        return std::make_unique<detail::PtoGatedReduceScatterMesh1D>(arg);
    };
}

}  // namespace ccu
}  // namespace pto
