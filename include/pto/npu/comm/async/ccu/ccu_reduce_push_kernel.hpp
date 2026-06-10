/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_CCU_CCU_REDUCE_PUSH_KERNEL_HPP
#define PTO_COMM_ASYNC_CCU_CCU_REDUCE_PUSH_KERNEL_HPP

// Host-only header — must NOT be included from device (bisheng -xcce) code.
#if defined(__CCE_KT_TEST__)
#error "ccu_reduce_push_kernel.hpp is a host-only header and cannot be included in device code."
#endif

// CCU-native Push-Reduce kernel (alternative to ccu_reduce_kernel.hpp).
//
// CONTRAST WITH THE PULL VERSION
// ------------------------------
// The original CcuReduceMesh1D ("pull" mode) has root ReadNb all peer
// inputs into local block buffers, then LocalReduceNb across them.
// That requires 3*N+1 Load slots and serializes the N-1 ReadNb calls at
// root, which is both a load-slot scalability and a per-collective
// latency problem.
//
// This kernel uses the inverse data path ("push" mode):
//   - Non-root ranks each issue WriteReduceNb directly to root's output
//     buffer.  hcomm translates this to WQE opcode 0x70 ("Write with
//     atomic store add"); the hardware accumulator at root's HBM
//     guarantees atomic add even when N-1 peers write concurrently.
//   - Root issues a single LocalCopyNb to seed its output with its own
//     input; peers' atomic adds layer on top.
//
// CONSEQUENCES
// ------------
// 1. Load slots per rank are CONSTANT (3 on root, 4 on non-root) — the
//    SQE auto-fragmentation machinery (CCU_SQE_ARGS_LEN=13) is never
//    triggered regardless of N.  kCcuMeshMaxRanks=16 stops being a load
//    ceiling here; it's only retained for the host-side broadcast of
//    root's (output, token).
// 2. Per-collective parallelism rises from 1 (root sequential ReadNb)
//    to N-1 (peers push concurrently).  Latency lower-bound becomes the
//    HBM ingress port saturation at root, not the chain of ReadNb.
// 3. Root no longer needs N-1 channels for the data phase, just for the
//    notify barriers.  All channels are still created so PreSync /
//    PostSync stay symmetric (and so the gate-only fallback works).
// 4. Numerical accumulation order is NON-DETERMINISTIC.  For floating
//    SUM the ULP-level result depends on which peer's write the HBM
//    accumulator absorbs first.  Identity-element ops (SUM/MAX/MIN) are
//    safe; PROD with non-trivial values may show platform-specific
//    drift.  Bit-for-bit reproducibility seekers should stay on the
//    pull kernel.
//
// SEQUENCING (the only subtle part)
// ---------------------------------
//   gateEvent_   — each rank's AIV has produced & flushed its input
//   ┌── root ──────────────────────┐    ┌── non-root ──────────────┐
//   │  WaitEvent(gateEvent_)       │    │  WaitEvent(gateEvent_)   │
//   │  LocalCopyNb(out, in)        │    │  (nothing yet)           │
//   │  WaitEvent(opEvent_)         │    │                          │
//   │  PreSync NotifyRecord ─────► │    │ ◄───── PreSync NotifyWait │
//   │  PreSync NotifyWait          │    │   PreSync NotifyRecord ─► │
//   │  (nothing further)           │    │  WriteReduceNb(root.out) │
//   │                              │    │  WaitEvent(opEvent_)     │
//   │  PostSync NotifyRecord/Wait  │    │  PostSync NotifyRecord/W │
//   │  RecordEvent(doneEvent_)     │    │  RecordEvent(doneEvent_) │
//   └──────────────────────────────┘    └──────────────────────────┘
//
// The CRITICAL ordering is that root's LocalCopyNb completes BEFORE its
// PreSync NotifyRecord fires (which it does, via the explicit
// WaitEvent(opEvent_)).  Non-root only issues WriteReduceNb AFTER the
// PreSync barrier returns, so it is guaranteed to see root's seeded
// output.  PostSync ensures root's AIV (downstream consumer of output)
// does not see partial-sum state.
//
// REQUIREMENTS ON THE HOST SIDE
// -----------------------------
// 1. Root's output buffer does NOT need to be pre-zeroed by AIV / host:
//    LocalCopyNb seeds it with root's own input.
// 2. All non-root ranks must know root's outputAddr and token (root's
//    HBM-side memory identifier).  These can be either (a) AllGathered
//    along with everyone's input/output/token as in the pull kernel's
//    test, or (b) Bcast'd as a single (outputAddr, token) pair from
//    root.  The Task arg below exposes them as plain uint64 so callers
//    can pick either path.
//
// Dependencies: hcomm pkg_inc only (libhcomm.so). No hccl dependency.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "hcomm/ccu/ccu_kernel.h"
#include "hcomm/ccu/ccu_kernel_arg.h"
#include "hcomm/ccu/ccu_kernel_signature.h"
#include "hcomm/ccu/ccu_task_arg_v1.h"

#include "pto/npu/comm/async/ccu/ccu_gate_registry.hpp"
#include "pto/npu/comm/async/ccu/ccu_mesh_common.hpp"

namespace pto {
namespace comm {
namespace ccu {

// ============================================================================
// Kernel argument types
// ============================================================================

struct CcuReducePushKernelArg : public hcomm::CcuKernelArg {
    uint32_t rankId{0};
    uint32_t rankSize{1};
    uint32_t rootId{0};

    HcclDataType dataType{HcclDataType::HCCL_DATA_TYPE_FP32};
    HcclDataType outputDataType{HcclDataType::HCCL_DATA_TYPE_FP32};
    HcclReduceOp reduceOp{HcclReduceOp::HCCL_REDUCE_SUM};

    uint32_t gateMask{1u << 0};
    uint32_t doneMask{1u << 0};

    uint64_t payloadBytes{0};

    CcuReducePushKernelArg() = default;
    CcuReducePushKernelArg(uint32_t rid, uint32_t rsize, uint32_t root, HcclDataType dt, HcclReduceOp op, uint64_t bytes,
                           uint32_t gMask = (1u << 0), uint32_t dMask = (1u << 0))
        : rankId(rid),
          rankSize(rsize),
          rootId(root),
          dataType(dt),
          outputDataType(dt),
          reduceOp(op),
          gateMask(gMask),
          doneMask(dMask),
          payloadBytes(bytes)
    {}

    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature sig;
        sig.Append(std::string("pto::comm::ccu::CcuReducePushKernelArg::v1"));
        sig.Append(rankId);
        sig.Append(rankSize);
        sig.Append(rootId);
        sig.Append(static_cast<uint32_t>(dataType));
        sig.Append(static_cast<uint32_t>(outputDataType));
        sig.Append(static_cast<uint32_t>(reduceOp));
        sig.Append(gateMask);
        sig.Append(doneMask);
        sig.Append(payloadBytes);
        return sig;
    }
};

// Push-mode TaskArg.  Note the absence of any per-peer arrays — push
// reduce needs only this rank's own (input, output, token) plus root's
// destination (rootOutputAddr, rootToken).  Total O(1) per rank
// regardless of rankSize.
//
// CRITICAL: `token` is required on BOTH root and non-root.  CCU's
// internal memory descriptor for any LocalAddr (source of LocalCopyNb,
// source of WriteReduceNb) carries a token Xn register; the hardware
// resolves the local MR by (addr, token).  Setting it to 0 produces a
// non-existent MR lookup and the DMA hangs forever (manifests as
// aclrtSynchronizeStream timeout on the CCU stream, error 507057).
//
// Layout convention:
//   - root rank:     rootOutputAddr / rootToken IGNORED (kernel uses
//                    outputAddr + token to do LocalCopyNb).
//   - non-root rank: outputAddr IGNORED (kernel uses
//                    rootOutputAddr + rootToken for the remote side,
//                    inputAddr + token for the local source).
struct CcuReducePushTaskArg : public hcomm::CcuTaskArg {
    uint64_t inputAddr{0};       // this rank's input VA (always used)
    uint64_t outputAddr{0};      // this rank's output VA (root only)
    uint64_t token{0};           // this rank's HBM token (always used — local src/dst)
    uint64_t rootOutputAddr{0};  // root's output VA (non-root only)
    uint64_t rootToken{0};       // root's HBM token (non-root only)
    uint64_t length{0};

    CcuReducePushTaskArg() = default;
    CcuReducePushTaskArg(uint64_t in, uint64_t out, uint64_t tok, uint64_t rootOut, uint64_t rootTok, uint64_t len)
        : inputAddr(in), outputAddr(out), token(tok), rootOutputAddr(rootOut), rootToken(rootTok), length(len)
    {}
};

// ============================================================================
// Kernel implementation (detail)
// ============================================================================

namespace detail {

constexpr uint32_t PUSH_PRE_SYNC_ID = 0;
constexpr uint32_t PUSH_POST_SYNC_ID = 3;
constexpr uint32_t PUSH_CKE_IDX_0 = 0;

inline void ReducePushTrace(const char *tag, uint32_t rank, const char *msg)
{
    std::fprintf(stderr, "[CCU_REDUCE_PUSH/%s] rank=%u %s\n", tag, rank, msg);
    std::fflush(stderr);
}

class CcuReducePushMesh1D : public CcuMeshKernelBase {
public:
    inline explicit CcuReducePushMesh1D(const hcomm::CcuKernelArg &arg) : CcuMeshKernelBase(arg)
    {
        const auto *kArg = dynamic_cast<const CcuReducePushKernelArg *>(&arg);
        if (kArg != nullptr) {
            rankId_ = kArg->rankId;
            rankSize_ = kArg->rankSize;
            rootId_ = kArg->rootId;
            dataType_ = kArg->dataType;
            outputDataType_ = kArg->outputDataType;
            reduceOp_ = kArg->reduceOp;
            gateMask_ = kArg->gateMask;
            doneMask_ = kArg->doneMask;
            payloadBytes_ = kArg->payloadBytes;
        }
        // CcuKernel::channels_ is framework-managed (see ccu-pitfalls #2).
        // Save a private copy from the arg for InitResource().
        ownChannels_ = arg.channels;
        std::fprintf(stderr,
                     "[CCU_REDUCE_PUSH/ctor] rank=%u rankSize=%u rootId=%u isRoot=%d "
                     "dataType=%d reduceOp=%d payloadBytes=%llu "
                     "ownChannels=%zu channels_=%zu\n",
                     rankId_, rankSize_, rootId_, static_cast<int>(rankId_ == rootId_),
                     static_cast<int>(dataType_), static_cast<int>(reduceOp_),
                     static_cast<unsigned long long>(payloadBytes_), ownChannels_.size(), channels_.size());
    }

    ~CcuReducePushMesh1D() override = default;

    inline HcclResult Algorithm() override
    {
        ReducePushTrace("algo", rankId_, "Algorithm() entry");

        HcclResult ret = InitResource();
        if (ret != HcclResult::HCCL_SUCCESS) {
            std::fprintf(stderr, "[CCU_REDUCE_PUSH/algo] rank=%u InitResource FAILED ret=%d\n", rankId_,
                         static_cast<int>(ret));
            return ret;
        }

        if (gateOnly_) {
            WaitEvent(gateEvent_);
            ReducePushTrace("algo", rankId_, "gate released (gate-only)");
            RecordEvent(doneEvent_);
            ReducePushTrace("algo", rankId_, "Algorithm() complete (gate-only)");
            return HcclResult::HCCL_SUCCESS;
        }

        LoadArgs();

        WaitEvent(gateEvent_);
        ReducePushTrace("algo", rankId_, "gate released");

        if (rankId_ == rootId_) {
            // Root seeds output BEFORE PreSync so peers wait for the
            // copy to be visible at the HBM accumulator before adding.
            DoRootSeed();
        }

        PreSync();

        if (rankId_ != rootId_) {
            DoPeerPush();
        }

        PostSync();

        RecordEvent(doneEvent_);
        ReducePushTrace("algo", rankId_, "Algorithm() complete");
        return HcclResult::HCCL_SUCCESS;
    }

    inline std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override
    {
        const auto *tArg = dynamic_cast<const CcuReducePushTaskArg *>(&arg);
        if (tArg == nullptr) {
            std::fprintf(stderr, "[CCU_REDUCE_PUSH/gene] GeneArgs FAILED dynamic_cast\n");
            return {};
        }

        const uint32_t dieId = static_cast<uint32_t>(gateEvent_.DieId());
        const uint32_t ckeId = static_cast<uint32_t>(gateEvent_.Id());
        pto::comm::ccu::Publish(rankId_, dieId, ckeId, gateMask_);

        std::fprintf(stderr,
                     "[CCU_REDUCE_PUSH/gene] rank=%u isRoot=%d published (die=%u, cke=%u, mask=0x%x) "
                     "input=0x%llx output=0x%llx token=0x%llx rootOutput=0x%llx rootToken=0x%llx "
                     "len=%llu gateOnly=%d\n",
                     rankId_, static_cast<int>(rankId_ == rootId_), dieId, ckeId, gateMask_,
                     static_cast<unsigned long long>(tArg->inputAddr),
                     static_cast<unsigned long long>(tArg->outputAddr),
                     static_cast<unsigned long long>(tArg->token),
                     static_cast<unsigned long long>(tArg->rootOutputAddr),
                     static_cast<unsigned long long>(tArg->rootToken),
                     static_cast<unsigned long long>(tArg->length), static_cast<int>(gateOnly_));

        if (gateOnly_) {
            return {};
        }

        // Args layout MUST match LoadArgs() exactly (ccu-pitfalls #5).
        // Root:     [inputAddr, outputAddr, ownToken, length]                    (4 entries)
        // Non-root: [inputAddr, ownToken, rootOutputAddr, rootToken, length]     (5 entries)
        //
        // `ownToken` is mandatory for both — LocalAddr.token is the local-side
        // memory descriptor token used by CCU/CTP to resolve the MR; setting
        // it to 0 produces a non-existent MR lookup and the DMA hangs (CCU
        // stream times out with ACL error 507057).
        if (rankId_ == rootId_) {
            return {tArg->inputAddr, tArg->outputAddr, tArg->token, tArg->length};
        }
        return {tArg->inputAddr, tArg->token, tArg->rootOutputAddr, tArg->rootToken, tArg->length};
    }

private:
    inline HcclResult InitResource()
    {
        gateOnly_ = ownChannels_.empty();
        if (gateOnly_) {
            return InitResourceGateOnly();
        }
        return InitResourceWithChannels();
    }

    inline HcclResult InitResourceGateOnly()
    {
        std::fprintf(stderr,
                     "[CCU_REDUCE_PUSH/init] rank=%u — no channels, "
                     "gate-only mode (SetDieId fallback).\n",
                     rankId_);
        uint32_t pinDieId = 1U;
        const char *env = std::getenv("HCCL_PTO_GATE_DIE_ID");
        if (env != nullptr && *env != '\0') {
            try {
                unsigned long v = std::stoul(std::string(env), nullptr, 10);
                if (v < 64U)
                    pinDieId = static_cast<uint32_t>(v);
            } catch (...) {
            }
        }
        SetDieId(pinDieId);

        gateEvent_ = CreateCompletedEvent();
        gateEvent_.SetMask(gateMask_);
        doneEvent_ = CreateCompletedEvent();
        doneEvent_.SetMask(doneMask_);

        ReducePushTrace("init", rankId_, "InitResource done (gate-only)");
        return HcclResult::HCCL_SUCCESS;
    }

    inline HcclResult InitResourceWithChannels()
    {
        inputVar_ = CreateVariable();
        ownTokenVar_ = CreateVariable();  // both root and non-root need the local-side token
        lengthVar_ = CreateVariable();

        if (rankId_ == rootId_) {
            // Root: own (input, output) on the local side, no remote
            outputVar_ = CreateVariable();
            localSrc_ = CreateLocalAddr();
            localDst_ = CreateLocalAddr();
        } else {
            // Non-root: own input on local side, root's (output, token) on remote side
            rootOutputVar_ = CreateVariable();
            rootTokenVar_ = CreateVariable();
            localSrc_ = CreateLocalAddr();
            remoteDst_ = CreateRemoteAddr();

            // Find the channel pointing to root.  ownChannels_ enumerates
            // peers in rank-index order excluding self, so:
            //   for peer < self : channel index == peer
            //   for peer > self : channel index == peer - 1
            rootChIdx_ = (rootId_ < rankId_) ? rootId_ : (rootId_ - 1);
            if (rootChIdx_ >= ownChannels_.size()) {
                std::fprintf(stderr,
                             "[CCU_REDUCE_PUSH/init] rank=%u rootChIdx=%u out of bounds (channels=%zu)\n",
                             rankId_, rootChIdx_, ownChannels_.size());
                return HcclResult::HCCL_E_INTERNAL;
            }
        }

        gateEvent_ = CreateCompletedEvent();
        gateEvent_.SetMask(gateMask_);

        doneEvent_ = CreateCompletedEvent();
        doneEvent_.SetMask(doneMask_);

        opEvent_ = CreateCompletedEvent();
        opEvent_.SetMask(1u);

        ReducePushTrace("init", rankId_, "InitResource done");
        return HcclResult::HCCL_SUCCESS;
    }

    inline void LoadArgs()
    {
        // Order MUST match GeneArgs return vector (ccu-pitfalls #5).
        // Root:     4 Loads — [inputAddr, outputAddr, ownToken, length]
        // Non-root: 5 Loads — [inputAddr, ownToken, rootOutputAddr, rootToken, length]
        if (rankId_ == rootId_) {
            Load(inputVar_);
            Load(outputVar_);
            Load(ownTokenVar_);
            Load(lengthVar_);
        } else {
            Load(inputVar_);
            Load(ownTokenVar_);
            Load(rootOutputVar_);
            Load(rootTokenVar_);
            Load(lengthVar_);
        }
    }

    // Symmetric barrier: ensures all ranks have reached the post-LocalCopy
    // point on root + post-gate point on non-root before peers begin atomic
    // add into root's output.  Identical bit allocation to the pull kernel
    // so the two paths can share notify-bit conventions if mixed in tests.
    inline void PreSync()
    {
        NotifyBarrier(ownChannels_, PUSH_CKE_IDX_0, PUSH_PRE_SYNC_ID);
        ReducePushTrace("sync", rankId_, "PreSync done");
    }

    inline void PostSync()
    {
        NotifyBarrier(ownChannels_, PUSH_CKE_IDX_0, PUSH_POST_SYNC_ID);
        ReducePushTrace("sync", rankId_, "PostSync done");
    }

    // Root path: seed own output with own input via LocalCopyNb, then wait
    // for the op to complete BEFORE the PreSync NotifyRecord fires.  The
    // explicit WaitEvent ensures peers see the seeded value at HBM before
    // their atomic adds layer on top.
    //
    // Both src and dst use the OWN rank's token — CCU's local memory
    // descriptor pairs (addr, token) just like the remote side.  See
    // CcuBroadcastMesh1D::DoBroadcast() for the analogous pattern.
    inline void DoRootSeed()
    {
        localSrc_.addr = inputVar_;
        localSrc_.token = ownTokenVar_;

        localDst_.addr = outputVar_;
        localDst_.token = ownTokenVar_;

        opEvent_.SetMask(1u);
        (void)LocalCopyNb(localDst_, localSrc_, lengthVar_, opEvent_);
        WaitEvent(opEvent_);

        ReducePushTrace("seed", rankId_, "root LocalCopyNb done (output seeded with own input)");
    }

    // Non-root path: WriteReduceNb own input -> root's output buffer with
    // hardware atomic accumulate (WQE opcode 0x70).  N-1 peers issue this
    // concurrently; the HBM-side accumulator at root serializes the adds.
    //
    // localSrc.token must be THIS rank's own token (loaded from SQE).
    // remoteDst.token is ROOT's token (also loaded from SQE).  Setting
    // either to a constant 0 causes the CCU local/remote MR lookup to
    // fail and the WaitEvent below hangs the CCU stream.
    inline void DoPeerPush()
    {
        localSrc_.addr = inputVar_;
        localSrc_.token = ownTokenVar_;

        remoteDst_.addr = rootOutputVar_;
        remoteDst_.token = rootTokenVar_;

        opEvent_.SetMask(1u);
        (void)WriteReduceNb(ownChannels_[rootChIdx_], remoteDst_, localSrc_, lengthVar_, dataType_, reduceOp_,
                            opEvent_);
        WaitEvent(opEvent_);

        ReducePushTrace("push", rankId_, "non-root WriteReduceNb done");
    }

    uint32_t rankId_{0};
    uint32_t rankSize_{1};
    uint32_t rootId_{0};
    uint32_t gateMask_{1u << 0};
    uint32_t doneMask_{1u << 0};
    uint64_t payloadBytes_{0};
    bool gateOnly_{false};
    uint32_t rootChIdx_{0};
    decltype(std::declval<hcomm::CcuKernelArg>().channels) ownChannels_;

    HcclDataType dataType_{HcclDataType::HCCL_DATA_TYPE_FP32};
    HcclDataType outputDataType_{HcclDataType::HCCL_DATA_TYPE_FP32};
    HcclReduceOp reduceOp_{HcclReduceOp::HCCL_REDUCE_SUM};

    // Variables loaded from SQE args.
    hcomm::CcuRep::Variable inputVar_;
    hcomm::CcuRep::Variable outputVar_;       // root only
    hcomm::CcuRep::Variable ownTokenVar_;     // both root and non-root (local-side token)
    hcomm::CcuRep::Variable rootOutputVar_;   // non-root only
    hcomm::CcuRep::Variable rootTokenVar_;    // non-root only
    hcomm::CcuRep::Variable lengthVar_;

    // Per-instruction address handles.
    hcomm::CcuRep::LocalAddr localSrc_;
    hcomm::CcuRep::LocalAddr localDst_;       // root only
    hcomm::CcuRep::RemoteAddr remoteDst_;     // non-root only

    hcomm::CcuRep::CompletedEvent gateEvent_;
    hcomm::CcuRep::CompletedEvent doneEvent_;
    hcomm::CcuRep::CompletedEvent opEvent_;
};

} // namespace detail

// ============================================================================
// Public factory
// ============================================================================

inline hcomm::KernelCreator MakeCcuReducePushCreator()
{
    return [](const hcomm::CcuKernelArg &arg) -> std::unique_ptr<hcomm::CcuKernel> {
        return std::make_unique<detail::CcuReducePushMesh1D>(arg);
    };
}

} // namespace ccu
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_CCU_CCU_REDUCE_PUSH_KERNEL_HPP
