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

#if defined(__CCE_KT_TEST__)
#error "ccu_reduce_broadcast_kernel.hpp is a host-only header."
#endif
//
// Single persistent kernel: per work item Pull Reduce then Push Broadcast.
// Built with public AscendC::ccu APIs (no CcuKernel::*Nb / ABI shims).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "hcomm/ccu/ccu_primitives.hpp"
#include "hcomm/ccu/ccu_launch.h"
#include "hcomm/ccu/ccu_assist_pub.h"
#include "hccl/hccl_ccu_res.h"
#include "hcomm/ccu/ccu_datatype_v1.h"

#include "pto/comm/async/ccu/ccu_gate_registry.hpp"
#include "pto/comm/async/ccu/ccu_loopgroup_utils.hpp"
#include "pto/comm/async/ccu/ccu_types.hpp"

#include "config.h"

// Remnant path uses a single Loop (totalLoopNum=1). That encoding is valid only
// when the tail after m*loopSize is a multiple of memSlice (p==0). Tile payloads
// guarantee this; do not feed arbitrary byte lengths into this kernel.
static_assert(
    G_TILE_BYTES % pto::comm::ccu::CCU_MS_SIZE == 0, "ccu_gemm_ar remnant path assumes tile-aligned payloads (p==0)");

namespace hcomm {

// Minimal link-time surface against libhcomm (not in public cann-9.2 headers).
// Used only by this demo fused kernel for CompletedEvent / Wait / Record.
class CcuKernel {
public:
    CcuRep::CompletedEvent CreateCompletedEvent();
    HcclResult WaitEvent(CcuRep::CompletedEvent event, unsigned int mask);
    HcclResult RecordEvent(CcuRep::CompletedEvent event, unsigned int mask);
    HcclResult GetEventByHandle(unsigned long handle, CcuRep::CompletedEvent** out);
};

class CcuKernelMgr {
public:
    static CcuKernelMgr& GetInstance(int deviceId);
    CcuKernel* GetCurrentKernel();
    CcuKernel* GetKernel(unsigned long handle);
};

} // namespace hcomm

namespace accu = AscendC::ccu;

using pto::comm::ccu::CalcGoSize;
using pto::comm::ccu::CalcLoopCount;
using pto::comm::ccu::CCU_DONE_MASK;
using pto::comm::ccu::CCU_GATE_MASK;
using pto::comm::ccu::CCU_MS_INTERLEAVE;
using pto::comm::ccu::CCU_MS_SIZE;
using pto::comm::ccu::EncodeLoopParam;
using pto::comm::ccu::EncodeOffsetParam;
using pto::comm::ccu::EncodeParallelParam;
using pto::comm::ccu::GoConfig;
using pto::comm::ccu::GoSize;

// ============================================================================
// Kernel argument
// ============================================================================

struct CcuFusedReduceBroadcastKernelArg {
    uint32_t rankId{0};
    uint32_t rankSize{1};
    uint32_t rootId{0};

    HcclDataType dataType{HcclDataType::HCCL_DATA_TYPE_FP32};
    HcclDataType outputDataType{HcclDataType::HCCL_DATA_TYPE_FP32};
    HcclReduceOp reduceOp{HcclReduceOp::HCCL_REDUCE_SUM};

    uint32_t gateMask{CCU_GATE_MASK};
    uint32_t doneMask{CCU_DONE_MASK};
    uint32_t progressMask{1u << 1};

    uint64_t payloadBytes{0};
    uint32_t loopCount{0}; // 0 = auto

    uint32_t missionWorkItems{0};
    uint64_t itemsDoneAddr{0};
    uint64_t kernelReadyAddr{0};
    uint64_t baseOffsetBytes{0};

    uint8_t oneShotBaselineOnly{0};

    std::vector<ChannelHandle> channels;
};

static constexpr uint32_t kMaxFusedRanks = 16;

struct CcuFusedReduceBroadcastTaskArg {
    uint64_t inputAddr{0};
    uint64_t outputAddr{0};
    uint64_t length{0};

    uint64_t peerInput[kMaxFusedRanks]{};
    uint64_t peerOutput[kMaxFusedRanks]{};
    uint64_t peerToken[kMaxFusedRanks]{};

    void SetPeerAddrs(uint32_t rankSize, const uint64_t* inputs, const uint64_t* outputs, const uint64_t* tokens)
    {
        for (uint32_t i = 0; i < rankSize && i < kMaxFusedRanks; ++i) {
            peerInput[i] = inputs[i];
            peerOutput[i] = outputs[i];
            peerToken[i] = tokens[i];
        }
    }
};

// ============================================================================
// Kernel implementation
// ============================================================================

namespace detail {

// Opt-in only (PTO_CCU_FUSED_TRACE=1). Default quiet like gemm_ar.
inline bool FusedTraceEnabled()
{
    static const bool enabled = []() {
        const char* env = std::getenv("PTO_CCU_FUSED_TRACE");
        if (env == nullptr || env[0] == '\0')
            return false;
        return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0 && std::strcmp(env, "off") != 0;
    }();
    return enabled;
}

inline void FusedTrace(const char* tag, uint32_t rank, const char* msg)
{
    if (!FusedTraceEnabled())
        return;
    std::fprintf(stderr, "[CCU_FUSED/%s] rank=%u %s\n", tag, rank, msg);
}

// Keep CompletedEvent copies (shared phyRes) across RegisterEnd/Translate, then
// read physical DieId/Id — not AscendC Event.handle.
// Multi-entry: seq + pipelined kernels register in one RegisterStart/End batch.
struct FusedCkePublishPending {
    bool valid{false};
    bool hasProgress{false};
    bool isSeqOneShot{false};
    uint32_t rankId{0};
    uint32_t gateMask{0};
    uint32_t progressMask{0};
    hcomm::CcuRep::CompletedEvent gateEv;
    hcomm::CcuRep::CompletedEvent progressEv[CCU_MAX_PROGRESS_SLOTS];

    // CompletedEvent's ctor is explicit: aggregate/array {} is copy-initialization and
    // fails. A user ctor default-inits omitted members (direct ctor call), which is OK.
    FusedCkePublishPending() : gateEv() {}
};

inline std::vector<FusedCkePublishPending>& FusedCkePublishTlsList()
{
    static thread_local std::vector<FusedCkePublishPending> pending;
    return pending;
}

inline void ClearFusedCkePublishTls() { FusedCkePublishTlsList().clear(); }

inline hcomm::CcuKernel* CurrentCcuKernel()
{
    int32_t deviceId = 0;
    (void)aclrtGetDevice(&deviceId);
    return hcomm::CcuKernelMgr::GetInstance(static_cast<int>(deviceId)).GetCurrentKernel();
}

// Progress CKE slot in ccu_gate_registry (gate uses Publish/TryGet, not a progress slot).
inline constexpr uint32_t kProgressCkeSlot = 0;

inline void PublishOneCke(
    const hcomm::CcuRep::CompletedEvent& ev, uint32_t rankId, uint32_t slot, uint32_t mask, bool isGate,
    bool isSeqOneShot = false)
{
    const uint32_t dieId = static_cast<uint32_t>(ev.DieId());
    const uint32_t ckeId = static_cast<uint32_t>(ev.Id());
    const char* kind = "progress";
    if (isGate) {
        if (isSeqOneShot) {
            pto::comm::ccu::PublishSeqGate(rankId, dieId, ckeId, mask);
            kind = "seqGate";
        } else {
            // Pipelined gate registry entry (same as mesh). Do not mirror into progress[].
            pto::comm::ccu::Publish(rankId, dieId, ckeId, mask);
            kind = "gate";
        }
    } else {
        pto::comm::ccu::PublishProgress(rankId, slot, dieId, ckeId, mask);
        kind = "progress";
    }
    if (FusedTraceEnabled()) {
        std::fprintf(
            stderr, "[CCU_FUSED/publish] rank=%u %s die=%u cke=%u mask=0x%x\n", rankId, kind, dieId, ckeId, mask);
    }
}

class CcuFusedReduceBroadcastKernel {
public:
    inline explicit CcuFusedReduceBroadcastKernel(const CcuFusedReduceBroadcastKernelArg& kernelArg)
    {
        rankId_ = kernelArg.rankId;
        rankSize_ = kernelArg.rankSize;
        rootId_ = kernelArg.rootId;
        dataType_ = kernelArg.dataType;
        outputDataType_ = kernelArg.outputDataType;
        reduceOp_ = kernelArg.reduceOp;
        gateMask_ = kernelArg.gateMask;
        doneMask_ = kernelArg.doneMask;
        payloadBytes_ = kernelArg.payloadBytes;
        goConfig_.loopCount = kernelArg.loopCount != 0 ? kernelArg.loopCount : CalcLoopCount(payloadBytes_);
        goConfig_.msInterleave = CCU_MS_INTERLEAVE;
        goConfig_.memSlice = CCU_MS_SIZE;
        progressMask_ = kernelArg.progressMask;
        missionWorkItems_ = kernelArg.missionWorkItems;
        itemsDoneAddr_ = kernelArg.itemsDoneAddr;
        kernelReadyAddr_ = kernelArg.kernelReadyAddr;
        baseOffsetBytes_ = kernelArg.baseOffsetBytes;
        oneShotBaselineOnly_ = kernelArg.oneShotBaselineOnly != 0;
        ownChannels_ = kernelArg.channels;

        if (FusedTraceEnabled()) {
            std::fprintf(
                stderr,
                "[CCU_FUSED/ctor] rank=%u rankSize=%u payloadBytes=%llu loopCount=%u "
                "missionWorkItems=%u baseOff=%llu oneShot=%u channels=%zu\n",
                rankId_, rankSize_, static_cast<unsigned long long>(payloadBytes_), goConfig_.loopCount,
                missionWorkItems_, static_cast<unsigned long long>(baseOffsetBytes_), oneShotBaselineOnly_ ? 1u : 0u,
                ownChannels_.size());
        }
    }

    inline HcclResult Algorithm()
    {
        FusedTrace("algo", rankId_, "Algorithm() entry");

        try {
            InitResource();

            hcomm::CcuKernel* ker = CurrentCcuKernel();
            if (ker == nullptr) {
                FusedTrace("algo", rankId_, "GetCurrentKernel null");
                return HcclResult::HCCL_E_INTERNAL;
            }

            if (gateOnly_) {
                (void)ker->WaitEvent(gateEvent_, gateMask_);
                (void)ker->RecordEvent(doneEvent_, doneMask_);
                StashCkeForPublish();
                FusedTrace("algo", rankId_, "done (gate-only)");
                return HcclResult::HCCL_SUCCESS;
            }

            LoadArgs();

            // Both paths wait on gate (treduce-style). One-shot has no progress CKE;
            // pipelined additionally waits on a single progress CKE per group.
            (void)ker->WaitEvent(gateEvent_, gateMask_);
            if (oneShotBaselineOnly_) {
                FusedTrace("algo", rankId_, "one-shot gate released → fused RS→AG");
                DoOneShotDataPath();
            } else {
                // AIV polls rsKernelReady!=0 before progress CKE.
                if (kernelReadyAddr_ != 0) {
                    accu::Variable ready = MakeImmediate(1);
                    (void)accu::Store(kernelReadyAddr_, ready);
                }
                FusedTrace("algo", rankId_, "gate released");
                DoPipelinedDataPath();
            }

            (void)ker->RecordEvent(doneEvent_, doneMask_);
            // Defer Publish until after RegisterEnd/Translate (physical DieId/Id).
            StashCkeForPublish();
            FusedTrace("algo", rankId_, "Algorithm() complete");
            return HcclResult::HCCL_SUCCESS;
        } catch (const accu::detail::CcuException& ex) {
            std::fprintf(stderr, "[CCU_FUSED/algo] rank=%u exception: %s\n", rankId_, ex.what());
            return HcclResult::HCCL_E_INTERNAL;
        }
    }

    static inline std::vector<uint64_t> PackFusedLaunchArgs(
        const CcuFusedReduceBroadcastTaskArg& taskArg, uint32_t rankSize, const GoConfig& goConfig, bool gateOnly)
    {
        GoSize goSize = CalcGoSize(taskArg.length, goConfig);

        if (FusedTraceEnabled()) {
            std::fprintf(
                stderr,
                "[CCU_FUSED/gene] input=0x%llx output=0x%llx len=%llu "
                "goSize={off=%llu, iter=%llu, para=0x%llx, res=%llu}\n",
                static_cast<unsigned long long>(taskArg.inputAddr), static_cast<unsigned long long>(taskArg.outputAddr),
                static_cast<unsigned long long>(taskArg.length), static_cast<unsigned long long>(goSize.addrOffset),
                static_cast<unsigned long long>(goSize.loopIterNum),
                static_cast<unsigned long long>(goSize.parallelParam),
                static_cast<unsigned long long>(goSize.residual));
        }

        if (gateOnly)
            return {};

        std::vector<uint64_t> args;
        args.reserve(3 * rankSize + 5);
        for (uint32_t i = 0; i < rankSize; ++i)
            args.push_back(taskArg.peerInput[i]);
        for (uint32_t i = 0; i < rankSize; ++i)
            args.push_back(taskArg.peerOutput[i]);
        for (uint32_t i = 0; i < rankSize; ++i)
            args.push_back(taskArg.peerToken[i]);
        args.push_back(taskArg.length);
        args.push_back(goSize.addrOffset);
        args.push_back(goSize.loopIterNum);
        args.push_back(goSize.parallelParam);
        args.push_back(goSize.residual);
        return args;
    }

private:
    inline accu::Variable MakeImmediate(uint64_t value)
    {
        accu::Variable v;
        v = value;
        return v;
    }

    inline accu::Func MakeReduceLoopBody(accu::Variable& lenVar)
    {
        return accu::Func([this, &lenVar]() {
            const uint32_t channelSize = static_cast<uint32_t>(ownChannels_.size());
            const uint32_t size = channelSize + 1;
            const uint16_t allMask = static_cast<uint16_t>((1u << size) - 1);

            // Base event for this Loop body. LoopGroup parallel expansion strides
            // CKE id by offsetCfg.ckeOffset (=1), so lane k uses loopEvents_[k].
            accu::Event& event = (*loopEvents_)[0];
            for (uint32_t i = 0; i < channelSize; ++i) {
                (void)accu::Read(
                    ownChannels_[i], (*loopBufs_)[i], rsSrcAddr_[i], lenVar, event, static_cast<uint16_t>(1u << i));
            }
            (void)accu::LocalCopy(
                (*loopBufs_)[size - 1], rsSelfLocalAddr_, lenVar, event, static_cast<uint16_t>(1u << channelSize));

            (void)accu::EventWait(event, allMask);

            if (size > 1) {
                (void)accu::LocalReduce(
                    loopBufs_->data(), size, dataType_, outputDataType_, reduceOp_, lenVar, event, 1u);
                (void)accu::EventWait(event, 1u);
            }

            (void)accu::LocalCopy(rsDstAddr_, (*loopBufs_)[0], lenVar, event, 1u);
            (void)accu::EventWait(event, 1u);
        });
    }

    // Keep CompletedEvent copies so post-Translate DieId/Id stay reachable.
    // Pushes one entry per registered kernel (seq + pipelined share one RegisterEnd).
    inline void StashCkeForPublish()
    {
        FusedCkePublishPending pending;
        pending.valid = true;
        pending.isSeqOneShot = oneShotBaselineOnly_;
        pending.rankId = rankId_;
        pending.gateMask = gateMask_;
        pending.gateEv = gateEvent_;
        // One-shot: seqGate only. Pipelined: gate + single progress CKE.
        if (oneShotBaselineOnly_) {
            pending.hasProgress = false;
            FusedCkePublishTlsList().push_back(pending);
            return;
        }
        pending.hasProgress = true;
        pending.progressMask = progressMask_;
        for (uint32_t slot = 0; slot < CCU_PROGRESS_SLOTS; ++slot) {
            pending.progressEv[slot] = progressEvent_[slot];
        }
        FusedCkePublishTlsList().push_back(pending);
    }

    inline void InitResource()
    {
        gateOnly_ = ownChannels_.empty();
        if (gateOnly_) {
            FusedTrace("init", rankId_, "InitResource done (gate-only)");
            return;
        }

        for (uint32_t i = 0; i < rankSize_; ++i) {
            rsInput_.emplace_back();
            rsOutput_.emplace_back();
            rsToken_.emplace_back();
        }

        rsLengthVar_ = accu::Variable{};
        rsGoSizeOffset_ = accu::Variable{};
        rsGoSizeLoopIter_ = accu::Variable{};
        rsGoSizeParallel_ = accu::Variable{};
        rsGoSizeResidual_ = accu::Variable{};

        rsDstAddr_ = accu::LocalAddr{};
        rsSelfLocalAddr_ = accu::LocalAddr{};
        rsSrcAddr_.reserve(rankSize_);
        for (uint32_t i = 0; i < rankSize_; ++i)
            rsSrcAddr_.emplace_back();

        agSrcAddr_ = accu::LocalAddr{};
        for (uint32_t i = 0; i + 1 < rankSize_; ++i)
            agDstAddr_.emplace_back();

        // Same as old fused/mesh: CreateCompletedEvent + SetMask in InitResource
        // (not AscendC Event default-ctors). AIV poke protocol binds to these.
        hcomm::CcuKernel* ker = CurrentCcuKernel();
        if (ker == nullptr) {
            FusedTrace("init", rankId_, "GetCurrentKernel null");
            return;
        }
        // SetMask symbol is not exported on 9.2; mask is a public field (and
        // WaitEvent/RecordEvent take mask as an explicit argument).
        gateEvent_ = ker->CreateCompletedEvent();
        gateEvent_.mask = gateMask_;
        doneEvent_ = ker->CreateCompletedEvent();
        doneEvent_.mask = doneMask_;
        if (!oneShotBaselineOnly_) {
            // One CompletedEvent per in-flight group. Distinct events may reuse the
            // same mask value (gate and done already both use bit 0).
            for (uint32_t slot = 0; slot < CCU_PROGRESS_SLOTS; ++slot) {
                progressEvent_[slot] = ker->CreateCompletedEvent();
                progressEvent_[slot].mask = progressMask_;
            }
        }

        // Constant across groups; WaitBroadcast() may be emitted before the first
        // IssueBroadcast() when CCU_PIPE_DEPTH == 2, so derive it here.
        bcastPeerCount_ = 0;
        for (uint32_t r = 0; r < rankSize_ && bcastPeerCount_ < kMaxFusedRanks; ++r) {
            if (r != rootId_)
                ++bcastPeerCount_;
        }

        // MS and Event resources for LoopGroup parallel expand (see microcode:
        // each expanded lane offsets MSId by msInterleave and CKEId by ckeOffset).
        // Must be a contiguous BlockAlloc of size loopCount — same as hcomm
        // AllocGoResource — not a single Event() / Array(1).
        const uint32_t totalBufs = goConfig_.loopCount * goConfig_.msInterleave;
        loopBufs_ = std::make_unique<accu::Array<accu::CcuBuffer>>(totalBufs);
        loopEvents_ = std::make_unique<accu::Array<accu::Event>>(goConfig_.loopCount);

        FusedTrace(
            "init", rankId_,
            oneShotBaselineOnly_ ? "InitResource done (one-shot, gate only)" :
                                   "InitResource done (fused LoopGroup + Write)");
    }

    inline void LoadArgs()
    {
        uint32_t argId = 0;
        for (uint32_t i = 0; i < rankSize_; ++i)
            (void)accu::LoadArg(rsInput_[i], argId++);
        for (uint32_t i = 0; i < rankSize_; ++i)
            (void)accu::LoadArg(rsOutput_[i], argId++);
        for (uint32_t i = 0; i < rankSize_; ++i)
            (void)accu::LoadArg(rsToken_[i], argId++);
        (void)accu::LoadArg(rsLengthVar_, argId++);
        (void)accu::LoadArg(rsGoSizeOffset_, argId++);
        (void)accu::LoadArg(rsGoSizeLoopIter_, argId++);
        (void)accu::LoadArg(rsGoSizeParallel_, argId++);
        (void)accu::LoadArg(rsGoSizeResidual_, argId++);
    }

    // GoSize split: m * (loopCount·MS) via the first LoopGroup, then an optional
    // remnant of n·MS. Payloads are tile-aligned (static_assert above) so p==0 and
    // the remnant is always a single Loop of length residual(=MS) repeated n times.
    inline void DoReduce()
    {
        CCU_IF(rsGoSizeLoopIter_ != 0)
        {
            accu::Variable loopParam = MakeImmediate(0);
            loopParam = EncodeLoopParam(0, goConfig_.memSlice * goConfig_.loopCount, 0);
            loopParam += rsGoSizeLoopIter_;

            accu::Variable sliceSize = MakeImmediate(goConfig_.memSlice);
            accu::Func body = MakeReduceLoopBody(sliceSize);
            accu::Loop loop(loopParam, body);

            accu::Variable paraCfg = MakeImmediate(EncodeParallelParam(goConfig_.loopCount - 1, 0, 1));
            accu::Variable offsetCfg = MakeImmediate(EncodeOffsetParam(goConfig_.memSlice, goConfig_.msInterleave, 1));
            // maxLoopNum = number of AddLoop calls in this group (not tiling loopCount).
            accu::LoopGroup group(paraCfg, offsetCfg, /*maxLoopNum=*/1u, {loop});
        }

        CCU_IF(rsGoSizeParallel_ != 0)
        {
            const uint32_t channelSize = static_cast<uint32_t>(ownChannels_.size());
            for (uint32_t i = 0; i < channelSize; ++i)
                rsSrcAddr_[i].addr += rsGoSizeOffset_;
            rsSelfLocalAddr_.addr += rsGoSizeOffset_;
            rsDstAddr_.addr += rsGoSizeOffset_;

            accu::Variable loopCfg = MakeImmediate(EncodeLoopParam(0, 0, 1));
            accu::Func body = MakeReduceLoopBody(rsGoSizeResidual_);
            accu::Loop loop(loopCfg, body);

            accu::Variable offsetCfg = MakeImmediate(EncodeOffsetParam(goConfig_.memSlice, goConfig_.msInterleave, 1));
            accu::LoopGroup group(rsGoSizeParallel_, offsetCfg, /*maxLoopNum=*/1u, {loop});
        }
    }

    // Issue the group's peer writes without waiting. Pairs with WaitBroadcast().
    inline void IssueBroadcast(accu::Variable& groupOffset, std::vector<accu::Variable>& baseOutput, uint32_t startIdx)
    {
        agSrcAddr_.addr = rsOutput_[rankId_];
        agSrcAddr_.addr += groupOffset;
        agSrcAddr_.token = rsToken_[rankId_];

        uint32_t peers[kMaxFusedRanks];
        uint32_t nPeers = 0;
        for (uint32_t r = 0; r < rankSize_ && nPeers < kMaxFusedRanks; ++r) {
            if (r == rootId_)
                continue;
            peers[nPeers++] = r;
        }
        if (nPeers == 0)
            return;

        const uint32_t start = startIdx % nPeers;
        for (uint32_t k = 0; k < nPeers; ++k) {
            const uint32_t idx = (start + k) % nPeers;
            const uint32_t r = peers[idx];
            agDstAddr_[idx].addr = baseOutput[r];
            agDstAddr_[idx].addr += groupOffset;
            agDstAddr_[idx].token = rsToken_[r];
            (void)accu::Write(
                ownChannels_[idx], agDstAddr_[idx], agSrcAddr_, rsLengthVar_, agOpEvent_,
                static_cast<uint16_t>(1u << idx));
        }
    }

    inline void WaitBroadcast()
    {
        if (bcastPeerCount_ == 0)
            return;
        (void)accu::EventWait(agOpEvent_, static_cast<uint16_t>((1u << bcastPeerCount_) - 1));
    }

    inline void DoBroadcast(accu::Variable& groupOffset, std::vector<accu::Variable>& baseOutput, uint32_t startIdx)
    {
        IssueBroadcast(groupOffset, baseOutput, startIdx);
        WaitBroadcast();
    }

    // Bind peer/self RS source addresses and owner reduce destination for one group.
    inline void BindRsAddrsForGroup(
        const std::vector<accu::Variable>& baseInput, const accu::Variable& baseOutput,
        const accu::Variable& groupOffset)
    {
        uint32_t curId = 0;
        for (uint32_t r = 0; r < rankSize_; r++) {
            if (r != rootId_) {
                rsSrcAddr_[curId].addr = baseInput[r];
                rsSrcAddr_[curId].addr += groupOffset;
                rsSrcAddr_[curId].token = rsToken_[r];
                curId++;
            }
        }
        rsSelfLocalAddr_.addr = baseInput[rankId_];
        rsSelfLocalAddr_.addr += groupOffset;
        rsSelfLocalAddr_.token = rsToken_[rankId_];

        rsDstAddr_.addr = baseOutput;
        rsDstAddr_.addr += groupOffset;
        rsDstAddr_.token = rsToken_[rankId_];
    }

    inline void DoOneShotDataPath()
    {
        std::vector<accu::Variable> baseInput;
        accu::Variable baseOutput;
        std::vector<accu::Variable> baseOutputAll;
        LoadBaseIoAddrs(baseInput, baseOutput, baseOutputAll);

        accu::Variable groupOffset = MakeImmediate(baseOffsetBytes_);
        BindRsAddrsForGroup(baseInput, baseOutput, groupOffset);
        DoReduce();

        const uint32_t nPeers = (rankSize_ > 1) ? (rankSize_ - 1) : 1;
        const uint32_t start = rankId_ % nPeers;
        DoBroadcast(groupOffset, baseOutputAll, start);

        FusedTrace("algo", rankId_, "DoOneShotDataPath done");
    }

    inline void LoadBaseIoAddrs(
        std::vector<accu::Variable>& baseInput, accu::Variable& baseOutput, std::vector<accu::Variable>& baseOutputAll)
    {
        baseInput.resize(rankSize_);
        baseOutputAll.resize(rankSize_);
        for (uint32_t i = 0; i < rankSize_; ++i) {
            baseInput[i] = accu::Variable{};
            baseInput[i] = rsInput_[i];
            baseOutputAll[i] = accu::Variable{};
            baseOutputAll[i] = rsOutput_[i];
        }
        baseOutput = rsOutput_[rankId_];
    }

    inline void WaitNextProgressCke(hcomm::CcuKernel* ker) { (void)ker->WaitEvent(progressEvent_[0], progressMask_); }

    // Depth 2 consumes progress slots round-robin, matching the AIV poke order.
    // CCU general registers support only assignment and addition -- there is no bit
    // operation, so `counter & 1` is unavailable and parity must be carried explicitly.
    inline void WaitNextProgressCkeAlternating(
        hcomm::CcuKernel* ker, accu::Variable& parity, accu::Variable& zero, accu::Variable& one)
    {
        CCU_IF(parity != 0)
        {
            (void)ker->WaitEvent(progressEvent_[1], progressMask_);
            parity = zero;
        }
        CCU_ELSE
        {
            (void)ker->WaitEvent(progressEvent_[0], progressMask_);
            parity = one;
        }
    }

    // Per-group peer-order rotation needs CCU_IF specialization. For 2-rank
    // (nPeers==1) start is always 0 — emit a single Write path.
    // `groupSelector` picks the per-group peer rotation and must therefore track the
    // group being issued, not the group being retired (they differ at depth 2).
    // waitInline=false leaves the completion wait to the caller.
    inline void BroadcastPipelinedGroup(
        accu::Variable& groupOffset, std::vector<accu::Variable>& baseOutputAll, accu::Variable& groupSelector,
        bool waitInline)
    {
        const uint32_t nPeers = (rankSize_ > 1) ? (rankSize_ - 1) : 1;
        if (nPeers <= 1) {
            IssueBroadcast(groupOffset, baseOutputAll, 0);
            if (waitInline) {
                WaitBroadcast();
            }
            return;
        }
        for (uint32_t gi = 0; gi < missionWorkItems_; ++gi) {
            CCU_IF(groupSelector == static_cast<uint64_t>(gi))
            {
                const uint32_t start = (rankId_ + gi) % nPeers;
                IssueBroadcast(groupOffset, baseOutputAll, start);
                if (waitInline) {
                    WaitBroadcast();
                }
            }
        }
    }

    inline void DoPipelinedDataPath()
    {
        const uint32_t channelSize = static_cast<uint32_t>(ownChannels_.size());
        std::vector<accu::Variable> baseInput;
        accu::Variable baseOutput;
        std::vector<accu::Variable> baseOutputAll;
        LoadBaseIoAddrs(baseInput, baseOutput, baseOutputAll);

        accu::Variable groupOffset = MakeImmediate(baseOffsetBytes_);
        accu::Variable groupBytes = MakeImmediate(payloadBytes_);
        accu::Variable one = MakeImmediate(1);

        hcomm::CcuKernel* ker = CurrentCcuKernel();
        if (ker == nullptr) {
            FusedTrace("algo", rankId_, "DoPipelinedDataPath: GetCurrentKernel null");
            return;
        }

        if constexpr (CCU_PIPE_DEPTH >= 2) {
            // Cross-group pipelining: the Broadcast of group g is retired only after
            // the Reduce of group g+1, so the two overlap. Reduce owns the on-chip MS
            // slices while Broadcast streams HBM to remote HBM, so they do not contend
            // for the same CCU resource. `issuedCounter` drives the loop and the peer
            // rotation; `doneCounter` reports retired groups to the AIV and therefore
            // lags by one -- which is exactly why CCU_PIPE_DEPTH must be >= 2 on the
            // AIV side too, or the backpressure would deadlock against this lag.
            accu::Variable issuedCounter = MakeImmediate(0);
            accu::Variable doneCounter = MakeImmediate(0);
            accu::Variable parity = MakeImmediate(0);
            accu::Variable zero = MakeImmediate(0);

            CCU_WHILE(issuedCounter != static_cast<uint64_t>(missionWorkItems_))
            {
                WaitNextProgressCkeAlternating(ker, parity, zero, one);
                BindRsAddrsForGroup(baseInput, baseOutput, groupOffset);
                DoReduce();
                CCU_IF(issuedCounter != 0)
                {
                    WaitBroadcast();
                    doneCounter += one;
                    (void)accu::Store(itemsDoneAddr_, doneCounter);
                }
                BroadcastPipelinedGroup(groupOffset, baseOutputAll, issuedCounter, /*waitInline=*/false);
                groupOffset += groupBytes;
                issuedCounter += one;
            }

            WaitBroadcast();
            doneCounter += one;
            (void)accu::Store(itemsDoneAddr_, doneCounter);
        } else {
            accu::Variable itemsDoneCounter = MakeImmediate(0);

            CCU_WHILE(itemsDoneCounter != static_cast<uint64_t>(missionWorkItems_))
            {
                WaitNextProgressCke(ker);
                BindRsAddrsForGroup(baseInput, baseOutput, groupOffset);
                DoReduce();
                BroadcastPipelinedGroup(groupOffset, baseOutputAll, itemsDoneCounter, /*waitInline=*/true);
                groupOffset += groupBytes;
                itemsDoneCounter += one;
                (void)accu::Store(itemsDoneAddr_, itemsDoneCounter);
            }
        }

        FusedTrace("algo", rankId_, "DoPipelinedDataPath done (fused)");
        (void)channelSize;
    }

    uint32_t rankId_{0};
    uint32_t rankSize_{1};
    uint32_t rootId_{0};
    uint32_t gateMask_{1u << 0};
    uint32_t doneMask_{1u << 0};
    uint64_t payloadBytes_{0};
    bool gateOnly_{false};
    std::vector<ChannelHandle> ownChannels_;
    HcclDataType dataType_{HcclDataType::HCCL_DATA_TYPE_FP32};
    HcclDataType outputDataType_{HcclDataType::HCCL_DATA_TYPE_FP32};
    HcclReduceOp reduceOp_{HcclReduceOp::HCCL_REDUCE_SUM};

    uint32_t progressMask_{1u << 1};
    uint32_t progressWaitSlot_{0}; // host-side cursor while emitting microcode
    uint32_t missionWorkItems_{0};
    uint64_t itemsDoneAddr_{0};
    uint64_t kernelReadyAddr_{0};
    uint64_t baseOffsetBytes_{0};
    bool oneShotBaselineOnly_{false};

    GoConfig goConfig_;

    std::vector<accu::Variable> rsInput_;
    std::vector<accu::Variable> rsOutput_;
    std::vector<accu::Variable> rsToken_;
    accu::Variable rsLengthVar_;
    accu::Variable rsGoSizeOffset_;
    accu::Variable rsGoSizeLoopIter_;
    accu::Variable rsGoSizeParallel_;
    accu::Variable rsGoSizeResidual_;
    accu::LocalAddr rsDstAddr_;
    accu::LocalAddr rsSelfLocalAddr_;
    std::vector<accu::RemoteAddr> rsSrcAddr_;

    accu::LocalAddr agSrcAddr_;
    std::vector<accu::RemoteAddr> agDstAddr_;
    accu::Event agOpEvent_;
    // Number of peers every Broadcast writes to; constant across groups. Only one
    // Broadcast is ever in flight, so a single event/address set suffices even at
    // CCU_PIPE_DEPTH == 2: the wait for group g happens before group g+1 is issued.
    uint32_t bcastPeerCount_{0};

    // Gate/progress/done: old CKE-poke protocol via CompletedEvent (not AscendC Event).
    hcomm::CcuRep::CompletedEvent gateEvent_;
    hcomm::CcuRep::CompletedEvent doneEvent_;
    hcomm::CcuRep::CompletedEvent progressEvent_[CCU_MAX_PROGRESS_SLOTS];

    std::unique_ptr<accu::Array<accu::CcuBuffer>> loopBufs_;
    std::unique_ptr<accu::Array<accu::Event>> loopEvents_;
};

inline CcuFusedReduceBroadcastKernelArg& FusedRegisterTlsArg()
{
    static thread_local CcuFusedReduceBroadcastKernelArg arg{};
    return arg;
}

// Hcomm CcuKernelMgr::Register calls this via blr and treats a non-zero w0 as failure.
// Must return CcuResult (0 == success); a void entry leaves garbage in w0.
inline CcuResult FusedSynthEntry()
{
    try {
        CcuFusedReduceBroadcastKernel runner(FusedRegisterTlsArg());
        HcclResult ret = runner.Algorithm();
        if (ret != HcclResult::HCCL_SUCCESS) {
            std::fprintf(stderr, "[CCU_FUSED/synth] Algorithm failed ret=%d\n", static_cast<int>(ret));
            return CcuResult::CCU_E_INTERNAL;
        }
        return CcuResult::CCU_SUCCESS;
    } catch (const accu::detail::CcuException& ex) {
        std::fprintf(stderr, "[CCU_FUSED/synth] CcuException: %s\n", ex.what());
        return CcuResult::CCU_E_INTERNAL;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[CCU_FUSED/synth] std::exception: %s\n", ex.what());
        return CcuResult::CCU_E_INTERNAL;
    } catch (...) {
        std::fprintf(stderr, "[CCU_FUSED/synth] unknown exception\n");
        return CcuResult::CCU_E_INTERNAL;
    }
}

} // namespace detail

inline void ClearFusedCkePublishTls() { detail::ClearFusedCkePublishTls(); }

// Call only AFTER HcommCcuKernelRegisterEnd: Translate assigns physical DieId/Id there.
// Publishing between Register and RegisterEnd uses virtual ids; AIV poke will miss WaitEvent.
// Publishes every stashed kernel (seq seqGate + pipelined gate/progress) from one batch.
inline CcuResult PublishStashedCkeAfterRegisterEnd()
{
    auto& list = detail::FusedCkePublishTlsList();
    for (auto& pending : list) {
        if (!pending.valid)
            continue;
        detail::PublishOneCke(pending.gateEv, pending.rankId, /*slot=*/0, pending.gateMask, true, pending.isSeqOneShot);
        if (pending.hasProgress) {
            for (uint32_t slot = 0; slot < CCU_PROGRESS_SLOTS; ++slot) {
                detail::PublishOneCke(
                    pending.progressEv[slot], pending.rankId, detail::kProgressCkeSlot + slot, pending.progressMask,
                    false);
            }
        }
    }
    ClearFusedCkePublishTls();
    return CcuResult::CCU_SUCCESS;
}

inline CcuResult QueryPrimaryCcuIns(HcclComm comm, CcuInsHandle* outIns)
{
    if (outIns == nullptr)
        return CcuResult::CCU_E_PTR;
    CcuInsHandle handles[8]{};
    uint32_t insNum = 0;
    HcclResult hr = HcclCommQueryCcuIns(comm, handles, &insNum);
    if (hr != HCCL_SUCCESS || insNum == 0) {
        std::fprintf(stderr, "[CCU_FUSED] HcclCommQueryCcuIns failed hr=%d insNum=%u\n", static_cast<int>(hr), insNum);
        return CcuResult::CCU_E_INTERNAL;
    }
    *outIns = handles[0];
    return CcuResult::CCU_SUCCESS;
}

inline CcuResult RegisterFusedReduceBroadcastKernel(
    CcuInsHandle insHandle, uint32_t dieId, const CcuFusedReduceBroadcastKernelArg& kernelArg,
    CcuKernelHandle* outHandle)
{
    if (outHandle == nullptr)
        return CcuResult::CCU_E_PTR;
    detail::FusedRegisterTlsArg() = kernelArg;
    const char* name = kernelArg.oneShotBaselineOnly != 0 ? "pto_fused_rs_ag_oneshot" : "pto_fused_rs_ag_pipelined";
    using SynthFn = CcuResult (*)();
    SynthFn synth = &detail::FusedSynthEntry;
    // Do NOT Publish here — physical CKE ids are assigned in RegisterEnd/Translate.
    return HcommCcuKernelRegister(insHandle, dieId, name, reinterpret_cast<const void*>(synth), nullptr, 0, outHandle);
}

inline std::vector<uint64_t> PackFusedReduceBroadcastLaunchArgs(
    const CcuFusedReduceBroadcastKernelArg& kernelArg, const CcuFusedReduceBroadcastTaskArg& taskArg)
{
    GoConfig go{};
    go.loopCount = kernelArg.loopCount != 0 ? kernelArg.loopCount : CalcLoopCount(kernelArg.payloadBytes);
    go.msInterleave = CCU_MS_INTERLEAVE;
    go.memSlice = CCU_MS_SIZE;
    const bool gateOnly = kernelArg.channels.empty();
    return detail::CcuFusedReduceBroadcastKernel::PackFusedLaunchArgs(taskArg, kernelArg.rankSize, go, gateOnly);
}

inline CcuResult LaunchFusedReduceBroadcastKernel(
    ThreadHandle threadHandle, CcuKernelHandle kernelHandle, const CcuFusedReduceBroadcastKernelArg& kernelArg,
    const CcuFusedReduceBroadcastTaskArg& taskArg)
{
    std::vector<uint64_t> args = PackFusedReduceBroadcastLaunchArgs(kernelArg, taskArg);
    return HcommCcuKernelLaunch(threadHandle, kernelHandle, args.data(), static_cast<uint32_t>(args.size()));
}
