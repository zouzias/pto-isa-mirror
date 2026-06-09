/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_CCU_CCU_SCATTER_KERNEL_HPP
#define PTO_COMM_ASYNC_CCU_CCU_SCATTER_KERNEL_HPP

// Host-only header — must NOT be included from device (bisheng -xcce) code.
#if defined(__CCE_KT_TEST__)
#error "ccu_scatter_kernel.hpp is a host-only header and cannot be included in device code."
#endif

// CCU-native Gated Scatter kernel — header-only.
//
// The scatter data path is built from CcuRep primitives:
//   Root: WriteNb to each remote peer + LocalCopyNb for self
// Non-root ranks participate only in pre/post sync.
//
// Per-rank slice addressing:
//   Host pre-computes rootInputBase + r * payloadBytes for each rank r
//   and MPI-broadcasts these slice VAs. Each rank passes its received
//   slice VA as inputAddr in CcuScatterTaskArg. The host then MPI-AllGathers
//   every rank's (inputAddr, outputAddr, token) at launch time and packs them
//   into CcuScatterTaskArg::peer* so the kernel receives them through
//   GeneArgs/Load — no runtime PreSync needed. Root obtains
//   input_[r] = rootInputBase + r * payloadBytes, which is a valid local
//   address on root's device. Root uses it as WriteNb source.
//
// Dependencies: hcomm pkg_inc only (libhcomm.so). No hccl dependency.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
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

struct CcuScatterKernelArg : public hcomm::CcuKernelArg {
    uint32_t rankId{0};
    uint32_t rankSize{1};
    uint32_t rootId{0};

    uint32_t gateMask{1u << 0};
    uint32_t doneMask{1u << 0};

    uint64_t payloadBytes{0};

    CcuScatterKernelArg() = default;
    CcuScatterKernelArg(uint32_t rid, uint32_t rsize, uint32_t root, uint64_t bytes, uint32_t gMask = (1u << 0),
                        uint32_t dMask = (1u << 0))
        : rankId(rid), rankSize(rsize), rootId(root), gateMask(gMask), doneMask(dMask), payloadBytes(bytes)
    {}

    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature sig;
        sig.Append(std::string("pto::comm::ccu::CcuScatterKernelArg::v1"));
        sig.Append(rankId);
        sig.Append(rankSize);
        sig.Append(rootId);
        sig.Append(gateMask);
        sig.Append(doneMask);
        sig.Append(payloadBytes);
        return sig;
    }
};

// Scatter on CCU v100 uses Address rolling-add to derive per-slice source
// VAs inside the microcode instead of shipping N individual slice VAs:
//   currentSlice.addr = rootInputBase           (snapshot)
//   for r in [0..N-1]:
//       WriteNb(...) / LocalCopyNb(...)         (uses currentSlice)
//       currentSlice.addr += sliceStep          (roll for next iter)
// CcuRep v100 exposes ADDITION only — see ccu_datatype_v1.h: Variable and
// Address both have operator+=(Variable).  The rolling pattern itself is
// production-validated in hcomm's all-to-all-v mesh1d (look at
// `src_[rankIdx].addr += xnMaxTransportSize_` after each Write).
//
// This collapses the wire-format from the old 3N+1 (per-rank slice array)
// to 2N+3 (base + step + per-peer output/token + length).  At N=16 that is
// 35 Loads instead of 49 — buys back 14 hcomm Load slots.
//
// CCU v160 will expose a full arithmetic set (sub / mul); once available
// the entire 2N+3 packing can collapse to constant-size Loads driven by
// rankId, but that is a separate kernel rewrite.
struct CcuScatterTaskArg : public hcomm::CcuTaskArg {
    uint64_t inputAddr{0};
    uint64_t outputAddr{0};
    uint64_t length{0};
    uint64_t token{0};

    // Set by SetWritePeers.  rootInputBase is broadcast from root via the
    // host (every rank carries the same value); only root actually uses it
    // in DoScatter, peers ignore it.
    uint32_t peerCount{0};
    uint64_t rootInputBase{0};
    uint64_t sliceStep{0};
    uint64_t peerOutput[kCcuMeshMaxRanks]{};
    uint64_t peerToken[kCcuMeshMaxRanks]{};

    CcuScatterTaskArg() = default;
    CcuScatterTaskArg(uint64_t in, uint64_t out, uint64_t len, uint64_t tok)
        : inputAddr(in), outputAddr(out), length(len), token(tok)
    {}

    // Write-side packing for Scatter.  Returns false on overflow rather
    // than silently truncating peer arrays.
    [[nodiscard]] bool SetWritePeers(uint32_t rankSize, uint64_t inputBase, uint64_t step, const uint64_t *outputs,
                                     const uint64_t *tokens)
    {
        if (!EnsurePeerCapacity("SCATTER", rankSize)) {
            return false;
        }
        peerCount = rankSize;
        rootInputBase = inputBase;
        sliceStep = step;
        for (uint32_t i = 0; i < rankSize; ++i) {
            peerOutput[i] = outputs[i];
            peerToken[i] = tokens[i];
        }
        return true;
    }
};

// ============================================================================
// Kernel implementation (detail)
// ============================================================================

namespace detail {

constexpr uint32_t SC_POST_SYNC_ID = 3;
constexpr uint32_t SC_CKE_IDX_0 = 0;

inline void ScatterTrace(const char *tag, uint32_t rank, const char *msg)
{
    std::fprintf(stderr, "[CCU_SCATTER/%s] rank=%u %s\n", tag, rank, msg);
    std::fflush(stderr);
}

class CcuScatterMesh1D : public CcuMeshKernelBase {
public:
    inline explicit CcuScatterMesh1D(const hcomm::CcuKernelArg &arg) : CcuMeshKernelBase(arg)
    {
        const auto *kArg = dynamic_cast<const CcuScatterKernelArg *>(&arg);
        if (kArg != nullptr) {
            rankId_ = kArg->rankId;
            rankSize_ = kArg->rankSize;
            rootId_ = kArg->rootId;
            gateMask_ = kArg->gateMask;
            doneMask_ = kArg->doneMask;
            payloadBytes_ = kArg->payloadBytes;
        }
        // CcuKernel::channels_ is framework-managed and cannot be modified
        // directly.  Save a private copy from the arg for InitResource().
        ownChannels_ = arg.channels;
        std::fprintf(stderr,
                     "[CCU_SCATTER/ctor] rank=%u rankSize=%u rootId=%u "
                     "payloadBytes=%llu "
                     "ownChannels=%zu channels_=%zu\n",
                     rankId_, rankSize_, rootId_, static_cast<unsigned long long>(payloadBytes_), ownChannels_.size(),
                     channels_.size());
    }

    ~CcuScatterMesh1D() override = default;

    inline HcclResult Algorithm() override
    {
        ScatterTrace("algo", rankId_, "Algorithm() entry");

        HcclResult ret = InitResource();
        if (ret != HcclResult::HCCL_SUCCESS) {
            std::fprintf(stderr, "[CCU_SCATTER/algo] rank=%u InitResource FAILED ret=%d\n", rankId_,
                         static_cast<int>(ret));
            return ret;
        }

        if (gateOnly_) {
            WaitEvent(gateEvent_);
            ScatterTrace("algo", rankId_, "gate released (gate-only)");
            RecordEvent(doneEvent_);
            ScatterTrace("algo", rankId_, "Algorithm() complete (gate-only)");
            return HcclResult::HCCL_SUCCESS;
        }

        LoadArgs();

        WaitEvent(gateEvent_);
        ScatterTrace("algo", rankId_, "gate released");

        if (rankId_ == rootId_) {
            DoScatter();
        }

        PostSync();

        RecordEvent(doneEvent_);
        ScatterTrace("algo", rankId_, "Algorithm() complete");
        return HcclResult::HCCL_SUCCESS;
    }

    inline std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override
    {
        const auto *tArg = dynamic_cast<const CcuScatterTaskArg *>(&arg);
        if (tArg == nullptr) {
            std::fprintf(stderr, "[CCU_SCATTER/gene] GeneArgs FAILED dynamic_cast\n");
            return {};
        }

        const uint32_t dieId = static_cast<uint32_t>(gateEvent_.DieId());
        const uint32_t ckeId = static_cast<uint32_t>(gateEvent_.Id());
        pto::comm::ccu::Publish(rankId_, dieId, ckeId, gateMask_);

        std::fprintf(stderr,
                     "[CCU_SCATTER/gene] rank=%u published (die=%u, cke=%u, mask=0x%x) "
                     "input=0x%llx output=0x%llx len=%llu token=0x%llx peerCount=%u "
                     "rootInputBase=0x%llx sliceStep=%llu gateOnly=%d\n",
                     rankId_, dieId, ckeId, gateMask_, static_cast<unsigned long long>(tArg->inputAddr),
                     static_cast<unsigned long long>(tArg->outputAddr), static_cast<unsigned long long>(tArg->length),
                     static_cast<unsigned long long>(tArg->token), tArg->peerCount,
                     static_cast<unsigned long long>(tArg->rootInputBase),
                     static_cast<unsigned long long>(tArg->sliceStep), static_cast<int>(gateOnly_));

        if (gateOnly_) {
            return {};
        }
        return PackScatterArgs(rankSize_, tArg->rootInputBase, tArg->sliceStep, tArg->peerOutput, tArg->peerToken,
                               tArg->length);
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
                     "[CCU_SCATTER/init] rank=%u — no channels, "
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

        ScatterTrace("init", rankId_, "InitResource done (gate-only)");
        return HcclResult::HCCL_SUCCESS;
    }

    inline HcclResult InitResourceWithChannels()
    {
        rootInputBase_ = CreateVariable();
        sliceStep_ = CreateVariable();
        currentSlice_ = CreateVariable();
        for (uint32_t peerId = 0; peerId < rankSize_; peerId++) {
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        }
        lengthVar_ = CreateVariable();

        // Per-channel source LocalAddrs (snapshot rolling slice into each) +
        // per-channel remote destinations.  Sized N-1 because root never
        // WriteNb's to itself.
        for (uint32_t i = 0; i + 1 < rankSize_; i++) {
            srcSliceAddrs_.push_back(CreateLocalAddr());
            dstAddrs_.push_back(CreateRemoteAddr());
        }
        selfSrc_ = CreateLocalAddr();
        selfDst_ = CreateLocalAddr();

        gateEvent_ = CreateCompletedEvent();
        gateEvent_.SetMask(gateMask_);

        doneEvent_ = CreateCompletedEvent();
        doneEvent_.SetMask(doneMask_);

        opEvent_ = CreateCompletedEvent();

        ScatterTrace("init", rankId_, "InitResource done");
        return HcclResult::HCCL_SUCCESS;
    }

    inline void LoadArgs()
    {
        // 2N+3 layout — see PackScatterArgs in ccu_mesh_common.hpp.
        LoadScatterArgs(rootInputBase_, sliceStep_, output_, token_, lengthVar_);
    }

    inline void PostSync()
    {
        NotifyBarrier(ownChannels_, SC_CKE_IDX_0, SC_POST_SYNC_ID);
        ScatterTrace("sync", rankId_, "PostSync done");
    }

    inline void DoScatter()
    {
        // Rolling source slice: keep a Variable cursor (`currentSlice_`),
        // snapshot its current value into each per-channel LocalAddr.addr
        // (Address = Variable, the proven snapshot pattern), then advance
        // the cursor with Variable += Variable so iteration r+1 sees
        // `rootInputBase + (r+1) * sliceStep`.
        //
        // CcuRep v100 only supplies ADDITION (see ccu_datatype_v1.h —
        // `enum CcuArithmeticOperatorType { ADDITION, INVALID }`), hence
        // the cumulative form.  Production reference: hcomm all-to-all-v
        // mesh1d does `src_[r].addr += xnMaxTransportSize_` between Writes
        // in the same fashion.
        currentSlice_ = rootInputBase_;

        uint32_t chIdx = 0;
        for (uint32_t r = 0; r < rankSize_; r++) {
            opEvent_.SetMask(1u << r);
            if (r == rootId_) {
                selfSrc_.addr = currentSlice_;
                selfSrc_.token = token_[rootId_];
                selfDst_.addr = output_[r];
                selfDst_.token = token_[r];
                LocalCopyNb(selfDst_, selfSrc_, lengthVar_, opEvent_);
            } else {
                srcSliceAddrs_[chIdx].addr = currentSlice_;
                srcSliceAddrs_[chIdx].token = token_[rootId_];
                dstAddrs_[chIdx].addr = output_[r];
                dstAddrs_[chIdx].token = token_[r];
                (void)WriteNb(ownChannels_[chIdx], dstAddrs_[chIdx], srcSliceAddrs_[chIdx], lengthVar_, opEvent_);
                chIdx++;
            }
            // Advance for next iteration (skip after the last one to avoid a
            // useless terminal add).
            if (r + 1 < rankSize_) {
                currentSlice_ += sliceStep_;
            }
        }

        opEvent_.SetMask((1u << rankSize_) - 1);
        WaitEvent(opEvent_);

        ScatterTrace("scatter", rankId_, "DoScatter done");
    }

    uint32_t rankId_{0};
    uint32_t rankSize_{1};
    uint32_t rootId_{0};
    uint32_t gateMask_{1u << 0};
    uint32_t doneMask_{1u << 0};
    uint64_t payloadBytes_{0};
    bool gateOnly_{false};
    decltype(std::declval<hcomm::CcuKernelArg>().channels) ownChannels_;

    hcomm::CcuRep::Variable rootInputBase_;
    hcomm::CcuRep::Variable sliceStep_;
    std::vector<hcomm::CcuRep::Variable> output_;
    std::vector<hcomm::CcuRep::Variable> token_;

    hcomm::CcuRep::Variable lengthVar_;

    // Rolling source-slice cursor: holds `rootInputBase + r * sliceStep`,
    // snapshotted into each per-channel LocalAddr.addr (Address = Variable)
    // before the WriteNb, then advanced by `currentSlice_ += sliceStep_`
    // (Variable += Variable) for the next iteration.
    hcomm::CcuRep::Variable currentSlice_;

    std::vector<hcomm::CcuRep::LocalAddr> srcSliceAddrs_;
    std::vector<hcomm::CcuRep::RemoteAddr> dstAddrs_;
    hcomm::CcuRep::LocalAddr selfSrc_;
    hcomm::CcuRep::LocalAddr selfDst_;

    hcomm::CcuRep::CompletedEvent gateEvent_;
    hcomm::CcuRep::CompletedEvent doneEvent_;
    hcomm::CcuRep::CompletedEvent opEvent_;
};

} // namespace detail

// ============================================================================
// Public factory
// ============================================================================

inline hcomm::KernelCreator MakeCcuScatterCreator()
{
    return [](const hcomm::CcuKernelArg &arg) -> std::unique_ptr<hcomm::CcuKernel> {
        return std::make_unique<detail::CcuScatterMesh1D>(arg);
    };
}

} // namespace ccu
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_CCU_CCU_SCATTER_KERNEL_HPP
