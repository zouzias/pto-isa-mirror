/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0.
 */

#include "pto_gated_kernel.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    // Step-by-step trace so a SEGV in the body localises to a single line.
    // setvbuf(stderr, _IONBF) in the ST main forces these to flush even if
    // the process aborts before normal exit.
    std::fprintf(stderr,
        "[PTO_GATE/kernel/gated_rs_mesh1d] CTOR T0 — base CcuKernel(arg) "
        "returned; entering derived ctor body. arg=%p\n",
        static_cast<const void *>(&arg));

    const auto *kArg = dynamic_cast<const PtoGatedKernelArg *>(&arg);
    std::fprintf(stderr,
        "[PTO_GATE/kernel/gated_rs_mesh1d] CTOR T1 — dynamic_cast result kArg=%p\n",
        static_cast<const void *>(kArg));

    if (kArg != nullptr) {
        rankId_       = kArg->rankId;
        rankSize_     = kArg->rankSize;
        gateMask_     = kArg->gateMask;
        doneMask_     = kArg->doneMask;
        payloadBytes_ = kArg->payloadBytes;
    } else {
        std::fprintf(stderr,
            "[PTO_GATE/kernel/gated_rs_mesh1d] CTOR FAILED dynamic_cast — got "
            "non-PtoGatedKernelArg, falling back to defaults.\n");
    }
    std::fprintf(stderr,
        "[PTO_GATE/kernel/gated_rs_mesh1d] CTOR T2 — fields captured: rank=%u "
        "rankSize=%u gateMask=0x%x doneMask=0x%x payloadBytes=%llu\n",
        rankId_, rankSize_, gateMask_, doneMask_,
        static_cast<unsigned long long>(payloadBytes_));
}

HcclResult PtoGatedReduceScatterMesh1D::Algorithm()
{
    TracePrintf("gated_rs_mesh1d", rankId_, "Algorithm() entry");

    // -------------------------------------------------------------------------
    // Step 0: dieId pinning.
    //
    // §3.13.K — when `channels_` is non-empty (acquired via
    // `HcclChannelAcquire(comm, COMM_ENGINE_CCU, ...)` in the ST main.cc
    // step 4.5 BEFORE `HcclCcuKernelRegister`), hcomm's `CcuKernel::Init()`
    // resolves the kernel's dieId from `channels_[0]` via
    // `GetDieIdByChannel` — automatically pinning to the physically-enabled
    // die without a manual `SetDieId(1)` override.
    //
    // §3.13.H fallback — if `channels_` is empty (standalone caller skipped
    // the channel acquire step, or no peers in the comm), Init()'s
    // `GetDieIdByChannels(empty)` falls back to `dieId=0`, which mismatches
    // the physically-enabled die (typically 1) and leads to AllocRes
    // returning HCCL_E_UNAVAIL=7. Override here BEFORE any
    // CreateCompletedEvent/CreateVariable.
    //
    // Override hierarchy (only fires when channels_ is empty):
    //   1. `HCCL_PTO_GATE_DIE_ID` env (decimal, e.g. "1") for runtime tuning.
    //   2. Hardcoded default `1` — Atlas A5 / 800 chassis convention.
    // -------------------------------------------------------------------------
    if (channels_.empty()) {
        uint32_t pinDieId = 1U;
        const char *envDieId = std::getenv("HCCL_PTO_GATE_DIE_ID");
        if (envDieId != nullptr && *envDieId != '\0') {
            char *end = nullptr;
            unsigned long parsed = std::strtoul(envDieId, &end, 10);
            if (end != envDieId && parsed < 64U) {
                pinDieId = static_cast<uint32_t>(parsed);
            }
        }
        SetDieId(pinDieId);
        std::fprintf(stderr,
            "[PTO_GATE/kernel/gated_rs_mesh1d] rank=%u — channels_ EMPTY, "
            "fallback SetDieId(%u) (override: HCCL_PTO_GATE_DIE_ID=%s). "
            "WARNING: empty channels_ also means hcomm ResPack will reserve "
            "spare-slot CKE registers (driver shadow, not AIV-writable). "
            "AIV trigger WILL trap with RT 507035. See plan §3.13.K.\n",
            rankId_, pinDieId, envDieId == nullptr ? "<unset>" : envDieId);
    } else {
        std::fprintf(stderr,
            "[PTO_GATE/kernel/gated_rs_mesh1d] rank=%u — channels_size=%zu, "
            "die auto-pinned by hcomm Init()::GetDieIdByChannel(channels_[0])\n",
            rankId_, channels_.size());
    }
    std::fflush(stderr);

    // -------------------------------------------------------------------------
    // Step 1: per-rank Variable build (mirrors hccl mesh1d:48-69).
    //
    // For local rank: plain `CreateVariable()`. For remote ranks: bind to
    // channel slots via `CreateVariable(channels_[idx], xnId, &var)` so the
    // IR references channel-acquired XN slots.
    //
    // §3.13.K — referencing channels_ in the IR is what makes hcomm ResPack
    // reserve real AIV-writable CKE register pool for `gateEvent_`. Without
    // this (placeholder identity-copy IR + no channel references), ResPack
    // gives `gateEvent_.Id()` a spare-slot ckeId whose physical register
    // is in driver shadow → AIV write traps.
    //
    // Convention matches hccl mesh1d (`ccu_kernel_reduce_scatter_mesh1d.cc:
    // 19-22`):
    //   INPUT_XN_ID  = 0  // input buffer slot on each peer channel
    //   TOKEN_XN_ID  = 1  // memory token slot
    //   POST_SYNC_ID = 2  // post-sync notify slot
    //   CKE_IDX_0    = 0
    // -------------------------------------------------------------------------
    constexpr uint32_t INPUT_XN_ID  = 0;
    constexpr uint32_t TOKEN_XN_ID  = 1;
    constexpr uint32_t POST_SYNC_ID = 2;
    constexpr uint32_t CKE_IDX_0    = 0;

    std::vector<CcuRep::Variable> inputs;
    std::vector<CcuRep::Variable> tokens;
    inputs.reserve(rankSize_);
    tokens.reserve(rankSize_);

    uint32_t channelIdx = 0;
    for (uint32_t peerId = 0; peerId < rankSize_; ++peerId) {
        if (peerId == rankId_) {
            inputs.push_back(CreateVariable());
            tokens.push_back(CreateVariable());
        } else if (channelIdx < channels_.size()) {
            CcuRep::Variable inputVar;
            CcuRep::Variable tokenVar;
            (void)CreateVariable(channels_[channelIdx], INPUT_XN_ID, &inputVar);
            (void)CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar);
            inputs.push_back(inputVar);
            tokens.push_back(tokenVar);
            ++channelIdx;
        } else {
            // Insufficient channels (degenerate fallback, e.g. user passed
            // channels_.size() < rankSize_-1). Push local Variables — won't
            // help with peer comm but keeps the loop well-formed.
            inputs.push_back(CreateVariable());
            tokens.push_back(CreateVariable());
        }
    }

    inputVar_  = inputs[rankId_];
    outputVar_ = CreateVariable();
    tokenVar_  = tokens[rankId_];
    lengthVar_ = CreateVariable();

    // -------------------------------------------------------------------------
    // Step 2: arm the gate / done / per-step events.
    //
    // `gateEvent_.Id()` (= ckeId) is resolved by hcomm `Translate()` inside
    // `HcclCcuKernelRegisterFinish`. With non-empty channels_ above, the
    // assigned ckeId comes from the channel-acquired CKE register pool —
    // physically reachable from AIV core via MMIO. With empty channels_,
    // the assigned ckeId comes from the spare-slot pool (driver shadow,
    // unreachable from AIV) and the trigger will trap.
    // -------------------------------------------------------------------------
    gateEvent_ = CreateCompletedEvent();
    gateEvent_.SetMask(gateMask_);

    doneEvent_ = CreateCompletedEvent();
    doneEvent_.SetMask(doneMask_);

    copyEvent_ = CreateCompletedEvent();
    copyEvent_.SetMask(1u << 0);

    TracePrintf("gated_rs_mesh1d", rankId_, "InitResources — gate path armed");

    // -------------------------------------------------------------------------
    // Step 3: bind runtime args.
    //
    // 4-slot order: [0] inputAddr, [1] outputAddr, [2] token, [3] length.
    // GeneArgs() must return values in this exact order.
    // -------------------------------------------------------------------------
    Load(inputVar_);
    Load(outputVar_);
    Load(tokenVar_);
    Load(lengthVar_);

    // -------------------------------------------------------------------------
    // Step 4: stall on the gate. CCU stream parks here until AIV trigger
    // fires the gateEvent_'s CKE.
    // -------------------------------------------------------------------------
    TracePrintf("gated_rs_mesh1d", rankId_, "about to WaitEvent(gateEvent_)");
    WaitEvent(gateEvent_);
    TracePrintf("gated_rs_mesh1d", rankId_,
                "past WaitEvent(gateEvent_) — gate released");

    // -------------------------------------------------------------------------
    // Step 5: pre-reduce channel-bound sync IR (mirrors hccl
    // mesh1d:100-108).
    //
    // §3.13.K — even without a real GroupReduce data path (phase 3),
    // referencing channels_ in NotifyRecord/Wait is what makes hcomm
    // ResPack provision the AIV-writable CKE register pool. Skipping these
    // would let ResPack treat the kernel as channel-less and pick spare-
    // slot registers for gateEvent_.
    // -------------------------------------------------------------------------
    if (!channels_.empty()) {
        for (auto ch : channels_) {
            (void)NotifyRecord(ch, CKE_IDX_0, INPUT_XN_ID, inputs[rankId_],
                               1u << INPUT_XN_ID);
            (void)NotifyRecord(ch, CKE_IDX_0, TOKEN_XN_ID, tokens[rankId_],
                               1u << TOKEN_XN_ID);
        }
        const uint32_t allBit = (1u << INPUT_XN_ID) | (1u << TOKEN_XN_ID);
        for (auto ch : channels_) {
            (void)NotifyWait(ch, CKE_IDX_0, allBit);
        }
        TracePrintf("gated_rs_mesh1d", rankId_,
                    "pre-reduce sync IR emitted (NotifyRecord/Wait per channel)");
    }

    // -------------------------------------------------------------------------
    // Step 6: placeholder identity copy (input → output, rank-local).
    //
    // For nranks=1 this is a correct identity reduce-scatter (degenerate
    // case). For nranks>1 we skip the cross-die GroupReduce data path —
    // that requires hccl-internal `CcuKernelAlgBase` (~1500 lines of
    // utility code outside hcomm pkg_inc). The pilot's deliverable is the
    // gate path; full reduce semantics is phase 3 (see plan §3.13.K).
    // -------------------------------------------------------------------------
    CcuRep::LocalAddr inputAddr = CreateLocalAddr();
    inputAddr.addr  = inputVar_;
    inputAddr.token = tokenVar_;

    CcuRep::LocalAddr outputAddr = CreateLocalAddr();
    outputAddr.addr  = outputVar_;
    outputAddr.token = tokenVar_;

    LocalCopyNb(outputAddr, inputAddr, lengthVar_, copyEvent_);
    WaitEvent(copyEvent_);

    // -------------------------------------------------------------------------
    // Step 7: post-reduce sync (mirrors hccl mesh1d:135-140). Same purpose
    // as step 5 — keep channels_ referenced in the post-section IR too.
    // -------------------------------------------------------------------------
    if (!channels_.empty()) {
        for (auto ch : channels_) {
            (void)NotifyRecord(ch, CKE_IDX_0, 1u << POST_SYNC_ID);
        }
        for (auto ch : channels_) {
            (void)NotifyWait(ch, CKE_IDX_0, 1u << POST_SYNC_ID);
        }
        TracePrintf("gated_rs_mesh1d", rankId_, "post-sync IR emitted");
    }

    // -------------------------------------------------------------------------
    // Step 8: signal completion via doneEvent_. Currently no consumer waits
    // on it — ST relies on aclrtSynchronizeStream(stream). Kept for parity
    // with hccl mesh1d's gated path and for future async consumers.
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
