/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_CCU_CCU_ALL_GATHER_KERNEL_HPP
#define PTO_COMM_ASYNC_CCU_CCU_ALL_GATHER_KERNEL_HPP

// Host-only header — must NOT be included from device (bisheng -xcce) code.
#if defined(__CCE_KT_TEST__)
#error "ccu_all_gather_kernel.hpp is a host-only header and cannot be included in device code."
#endif

// CCU-native Gated AllGather kernel — header-only.
//
// Based on HCCL CcuKernelAllGatherMesh1D (mesh_1D topology).
//
// AllGather semantics: each rank provides sliceSize bytes of input.
// After the op, every rank holds the full concatenation in output.
//
// Implementation: src = input[0] (local input addr), dst = output[r] + offset
// for each rank r. Uses WriteNb for remote peers, LocalCopyNb for self.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "hcomm/ccu/ccu_kernel.h"
#include "hcomm/ccu/ccu_kernel_arg.h"
#include "hcomm/ccu/ccu_kernel_signature.h"
#include "hcomm/ccu/ccu_task_arg_v1.h"

#include "pto/npu/comm/async/ccu/ccu_gate_registry.hpp"

namespace pto {
namespace comm {
namespace ccu {

// ============================================================================
// Kernel argument types
// ============================================================================

struct CcuAllGatherKernelArg : public hcomm::CcuKernelArg {
    uint32_t rankId{0};
    uint32_t rankSize{1};

    uint32_t gateMask{1u << 0};
    uint32_t doneMask{1u << 0};

    uint64_t payloadBytes{0};

    CcuAllGatherKernelArg() = default;
    CcuAllGatherKernelArg(uint32_t rid, uint32_t rsize, uint64_t bytes, uint32_t gMask = (1u << 0),
                          uint32_t dMask = (1u << 0))
        : rankId(rid), rankSize(rsize), gateMask(gMask), doneMask(dMask), payloadBytes(bytes)
    {}

    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature sig;
        sig.Append(std::string("pto::comm::ccu::CcuAllGatherKernelArg::v1"));
        sig.Append(rankId);
        sig.Append(rankSize);
        sig.Append(gateMask);
        sig.Append(doneMask);
        sig.Append(payloadBytes);
        return sig;
    }
};

struct CcuAllGatherTaskArg : public hcomm::CcuTaskArg {
    uint64_t inputAddr{0};
    uint64_t outputAddr{0};
    uint64_t offset{0};
    uint64_t sliceSize{0};
    uint64_t token{0};

    CcuAllGatherTaskArg() = default;
    CcuAllGatherTaskArg(uint64_t in, uint64_t out, uint64_t off, uint64_t slice, uint64_t tok)
        : inputAddr(in), outputAddr(out), offset(off), sliceSize(slice), token(tok)
    {}
};

// ============================================================================
// Kernel implementation (detail)
// ============================================================================

namespace detail {

constexpr uint32_t AG_OUTPUT_XN_ID = 1;
constexpr uint32_t AG_TOKEN_XN_ID = 2;
constexpr uint32_t AG_POST_SYNC_ID = 3;
constexpr uint32_t AG_CKE_IDX_0 = 0;

inline void AgTrace(const char *tag, uint32_t rank, const char *msg)
{
    std::fprintf(stderr, "[CCU_AG/%s] rank=%u %s\n", tag, rank, msg);
    std::fflush(stderr);
}

class CcuAllGatherMesh1D : public hcomm::CcuKernel {
public:
    inline explicit CcuAllGatherMesh1D(const hcomm::CcuKernelArg &arg) : CcuKernel(arg)
    {
        const auto *kArg = dynamic_cast<const CcuAllGatherKernelArg *>(&arg);
        if (kArg != nullptr) {
            rankId_ = kArg->rankId;
            rankSize_ = kArg->rankSize;
            gateMask_ = kArg->gateMask;
            doneMask_ = kArg->doneMask;
            payloadBytes_ = kArg->payloadBytes;
        }
        ownChannels_ = arg.channels;
        std::fprintf(stderr,
                     "[CCU_AG/ctor] rank=%u rankSize=%u "
                     "payloadBytes=%llu ownChannels=%zu channels_=%zu\n",
                     rankId_, rankSize_, static_cast<unsigned long long>(payloadBytes_), ownChannels_.size(),
                     channels_.size());
    }

    ~CcuAllGatherMesh1D() override = default;

    inline HcclResult Algorithm() override
    {
        AgTrace("algo", rankId_, "Algorithm() entry");

        HcclResult ret = InitResource();
        if (ret != HcclResult::HCCL_SUCCESS) {
            std::fprintf(stderr, "[CCU_AG/algo] rank=%u InitResource FAILED ret=%d\n", rankId_, static_cast<int>(ret));
            return ret;
        }

        if (gateOnly_) {
            WaitEvent(gateEvent_);
            AgTrace("algo", rankId_, "gate released (gate-only)");
            RecordEvent(doneEvent_);
            AgTrace("algo", rankId_, "Algorithm() complete (gate-only)");
            return HcclResult::HCCL_SUCCESS;
        }

        LoadArgs();

        WaitEvent(gateEvent_);
        AgTrace("algo", rankId_, "gate released");

        PreSync();
        DoAllGather();
        PostSync();

        RecordEvent(doneEvent_);
        AgTrace("algo", rankId_, "Algorithm() complete");
        return HcclResult::HCCL_SUCCESS;
    }

    inline std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override
    {
        const auto *tArg = dynamic_cast<const CcuAllGatherTaskArg *>(&arg);
        if (tArg == nullptr) {
            std::fprintf(stderr, "[CCU_AG/gene] GeneArgs FAILED dynamic_cast\n");
            return {};
        }

        const uint32_t dieId = static_cast<uint32_t>(gateEvent_.DieId());
        const uint32_t ckeId = static_cast<uint32_t>(gateEvent_.Id());
        pto::comm::ccu::Publish(rankId_, dieId, ckeId, gateMask_);

        std::fprintf(stderr,
                     "[CCU_AG/gene] rank=%u published (die=%u, cke=%u, mask=0x%x) "
                     "input=0x%llx output=0x%llx offset=%llu slice=%llu token=0x%llx gateOnly=%d\n",
                     rankId_, dieId, ckeId, gateMask_, static_cast<unsigned long long>(tArg->inputAddr),
                     static_cast<unsigned long long>(tArg->outputAddr), static_cast<unsigned long long>(tArg->offset),
                     static_cast<unsigned long long>(tArg->sliceSize), static_cast<unsigned long long>(tArg->token),
                     static_cast<int>(gateOnly_));

        if (gateOnly_) {
            return {};
        }
        return {tArg->inputAddr, tArg->outputAddr, tArg->token, tArg->offset, tArg->sliceSize};
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
        uint32_t pinDieId = 1U;
        const char *env = std::getenv("HCCL_PTO_GATE_DIE_ID");
        if (env != nullptr && *env != '\0') {
            char *end = nullptr;
            unsigned long v = std::strtoul(env, &end, 10);
            if (end != env && v < 64U)
                pinDieId = static_cast<uint32_t>(v);
        }
        SetDieId(pinDieId);

        gateEvent_ = CreateCompletedEvent();
        gateEvent_.SetMask(gateMask_);
        doneEvent_ = CreateCompletedEvent();
        doneEvent_.SetMask(doneMask_);

        AgTrace("init", rankId_, "InitResource done (gate-only)");
        return HcclResult::HCCL_SUCCESS;
    }

    inline HcclResult InitResourceWithChannels()
    {
        input_ = CreateVariable();

        uint16_t channelIdx = 0;
        for (uint32_t peerId = 0; peerId < rankSize_; peerId++) {
            if (peerId == rankId_) {
                output_.push_back(CreateVariable());
                token_.push_back(CreateVariable());
            } else {
                hcomm::CcuRep::Variable outputVar, tokenVar;
                (void)CreateVariable(ownChannels_[channelIdx], AG_OUTPUT_XN_ID, &outputVar);
                output_.push_back(outputVar);
                (void)CreateVariable(ownChannels_[channelIdx], AG_TOKEN_XN_ID, &tokenVar);
                token_.push_back(tokenVar);
                channelIdx++;
            }
        }

        offsetVar_ = CreateVariable();
        sliceSizeVar_ = CreateVariable();

        srcAddr_ = CreateLocalAddr();
        dstAddrs_.reserve(rankSize_);
        for (uint32_t i = 0; i < rankSize_; i++) {
            dstAddrs_.push_back(CreateRemoteAddr());
        }
        localDstAddr_ = CreateLocalAddr();

        gateEvent_ = CreateCompletedEvent();
        gateEvent_.SetMask(gateMask_);
        doneEvent_ = CreateCompletedEvent();
        doneEvent_.SetMask(doneMask_);
        opEvent_ = CreateCompletedEvent();

        AgTrace("init", rankId_, "InitResource done");
        return HcclResult::HCCL_SUCCESS;
    }

    inline void LoadArgs()
    {
        Load(input_);
        Load(output_[rankId_]);
        Load(token_[rankId_]);
        Load(offsetVar_);
        Load(sliceSizeVar_);
    }

    inline void PreSync()
    {
        for (auto ch : ownChannels_) {
            (void)NotifyRecord(ch, AG_CKE_IDX_0, AG_OUTPUT_XN_ID, output_[rankId_], 1u << AG_OUTPUT_XN_ID);
            (void)NotifyRecord(ch, AG_CKE_IDX_0, AG_TOKEN_XN_ID, token_[rankId_], 1u << AG_TOKEN_XN_ID);
        }
        uint32_t allBit = (1u << AG_OUTPUT_XN_ID) | (1u << AG_TOKEN_XN_ID);
        for (auto ch : ownChannels_) {
            (void)NotifyWait(ch, AG_CKE_IDX_0, allBit);
        }
        AgTrace("sync", rankId_, "PreSync done");
    }

    inline void PostSync()
    {
        for (auto &ch : ownChannels_) {
            (void)NotifyRecord(ch, AG_CKE_IDX_0, 1u << AG_POST_SYNC_ID);
        }
        for (auto &ch : ownChannels_) {
            (void)NotifyWait(ch, AG_CKE_IDX_0, 1u << AG_POST_SYNC_ID);
        }
        AgTrace("sync", rankId_, "PostSync done");
    }

    // AllGather: write local input to all peers' output[r] + offset, plus local copy.
    inline void DoAllGather()
    {
        srcAddr_.addr = input_;
        srcAddr_.token = token_[rankId_];

        uint32_t chIdx = 0;
        for (uint32_t r = 0; r < rankSize_; r++) {
            if (r != rankId_) {
                dstAddrs_[chIdx].addr = output_[r];
                dstAddrs_[chIdx].addr += offsetVar_;
                dstAddrs_[chIdx].token = token_[r];
                opEvent_.SetMask(1u << chIdx);
                (void)WriteNb(ownChannels_[chIdx], dstAddrs_[chIdx], srcAddr_, sliceSizeVar_, opEvent_);
                chIdx++;
            }
        }

        localDstAddr_.addr = output_[rankId_];
        localDstAddr_.addr += offsetVar_;
        localDstAddr_.token = token_[rankId_];
        opEvent_.SetMask(1u << chIdx);
        LocalCopyNb(localDstAddr_, srcAddr_, sliceSizeVar_, opEvent_);

        opEvent_.SetMask((1u << (chIdx + 1)) - 1);
        WaitEvent(opEvent_);

        AgTrace("gather", rankId_, "DoAllGather done");
    }

    uint32_t rankId_{0};
    uint32_t rankSize_{1};
    uint32_t gateMask_{1u << 0};
    uint32_t doneMask_{1u << 0};
    uint64_t payloadBytes_{0};
    bool gateOnly_{false};
    decltype(std::declval<hcomm::CcuKernelArg>().channels) ownChannels_;

    hcomm::CcuRep::Variable input_;
    std::vector<hcomm::CcuRep::Variable> output_;
    std::vector<hcomm::CcuRep::Variable> token_;

    hcomm::CcuRep::Variable offsetVar_;
    hcomm::CcuRep::Variable sliceSizeVar_;

    hcomm::CcuRep::LocalAddr srcAddr_;
    std::vector<hcomm::CcuRep::RemoteAddr> dstAddrs_;
    hcomm::CcuRep::LocalAddr localDstAddr_;

    hcomm::CcuRep::CompletedEvent gateEvent_;
    hcomm::CcuRep::CompletedEvent doneEvent_;
    hcomm::CcuRep::CompletedEvent opEvent_;
};

} // namespace detail

// ============================================================================
// Public factory
// ============================================================================

inline hcomm::KernelCreator MakeCcuAllGatherCreator()
{
    return [](const hcomm::CcuKernelArg &arg) -> std::unique_ptr<hcomm::CcuKernel> {
        return std::make_unique<detail::CcuAllGatherMesh1D>(arg);
    };
}

} // namespace ccu
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_CCU_CCU_ALL_GATHER_KERNEL_HPP
