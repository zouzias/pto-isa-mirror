/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_CCU_CCU_ALL_TO_ALL_KERNEL_HPP
#define PTO_COMM_ASYNC_CCU_CCU_ALL_TO_ALL_KERNEL_HPP

// Host-only header — must NOT be included from device (bisheng -xcce) code.
#if defined(__CCE_KT_TEST__)
#error "ccu_all_to_all_kernel.hpp is a host-only header and cannot be included in device code."
#endif

// CCU-native Gated AllToAll kernel — header-only.
//
// Based on HCCL CcuKernelAlltoAllMesh1D (mesh_1D topology, uniform slices).
//
// AllToAll semantics (non-V, uniform):
//   - Input is divided into nRanks slices of sliceSize bytes each.
//   - Rank r sends input slice i (at input + i*sliceSize) to rank i.
//   - After the op: output[i*sliceSize..(i+1)*sliceSize-1] = data rank i sent to this rank.
//
// Implementation: WriteNb from local input[r] to remote output[r] at the
// appropriate offset for each peer r. LocalCopyNb for self.

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

struct CcuAllToAllKernelArg : public hcomm::CcuKernelArg {
    uint32_t rankId{0};
    uint32_t rankSize{1};

    uint32_t gateMask{1u << 0};
    uint32_t doneMask{1u << 0};

    uint64_t payloadBytes{0};

    CcuAllToAllKernelArg() = default;
    CcuAllToAllKernelArg(uint32_t rid, uint32_t rsize, uint64_t bytes, uint32_t gMask = (1u << 0),
                         uint32_t dMask = (1u << 0))
        : rankId(rid), rankSize(rsize), gateMask(gMask), doneMask(dMask), payloadBytes(bytes)
    {}

    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature sig;
        sig.Append(std::string("pto::comm::ccu::CcuAllToAllKernelArg::v1"));
        sig.Append(rankId);
        sig.Append(rankSize);
        sig.Append(gateMask);
        sig.Append(doneMask);
        sig.Append(payloadBytes);
        return sig;
    }
};

struct CcuAllToAllTaskArg : public hcomm::CcuTaskArg {
    uint64_t inputAddr{0};
    uint64_t outputAddr{0};
    uint64_t sliceSize{0};
    uint64_t token{0};
    uint64_t srcStride{0};
    uint64_t srcOffset{0};
    uint64_t dstOffset{0};

    CcuAllToAllTaskArg() = default;
    CcuAllToAllTaskArg(uint64_t in, uint64_t out, uint64_t slice, uint64_t tok, uint64_t sstride, uint64_t soff,
                       uint64_t doff)
        : inputAddr(in),
          outputAddr(out),
          sliceSize(slice),
          token(tok),
          srcStride(sstride),
          srcOffset(soff),
          dstOffset(doff)
    {}
};

// ============================================================================
// Kernel implementation (detail)
// ============================================================================

namespace detail {

constexpr uint32_t A2A_OUTPUT_XN_ID = 1;
constexpr uint32_t A2A_TOKEN_XN_ID = 2;
constexpr uint32_t A2A_POST_SYNC_ID = 5;
constexpr uint32_t A2A_CKE_IDX_0 = 0;
constexpr uint32_t A2A_CKE_IDX_1 = 1;

inline void A2aTrace(const char *tag, uint32_t rank, const char *msg)
{
    std::fprintf(stderr, "[CCU_A2A/%s] rank=%u %s\n", tag, rank, msg);
    std::fflush(stderr);
}

class CcuAllToAllMesh1D : public hcomm::CcuKernel {
public:
    inline explicit CcuAllToAllMesh1D(const hcomm::CcuKernelArg &arg) : CcuKernel(arg)
    {
        const auto *kArg = dynamic_cast<const CcuAllToAllKernelArg *>(&arg);
        if (kArg != nullptr) {
            rankId_ = kArg->rankId;
            rankSize_ = kArg->rankSize;
            gateMask_ = kArg->gateMask;
            doneMask_ = kArg->doneMask;
            payloadBytes_ = kArg->payloadBytes;
        }
        ownChannels_ = arg.channels;
        std::fprintf(stderr,
                     "[CCU_A2A/ctor] rank=%u rankSize=%u "
                     "payloadBytes=%llu ownChannels=%zu channels_=%zu\n",
                     rankId_, rankSize_, static_cast<unsigned long long>(payloadBytes_), ownChannels_.size(),
                     channels_.size());
    }

    ~CcuAllToAllMesh1D() override = default;

    inline HcclResult Algorithm() override
    {
        A2aTrace("algo", rankId_, "Algorithm() entry");

        HcclResult ret = InitResource();
        if (ret != HcclResult::HCCL_SUCCESS) {
            std::fprintf(stderr, "[CCU_A2A/algo] rank=%u InitResource FAILED ret=%d\n", rankId_, static_cast<int>(ret));
            return ret;
        }

        if (gateOnly_) {
            WaitEvent(gateEvent_);
            A2aTrace("algo", rankId_, "gate released (gate-only)");
            RecordEvent(doneEvent_);
            A2aTrace("algo", rankId_, "Algorithm() complete (gate-only)");
            return HcclResult::HCCL_SUCCESS;
        }

        LoadArgs();

        WaitEvent(gateEvent_);
        A2aTrace("algo", rankId_, "gate released");

        PreSync();
        DoAllToAll();
        PostSync();

        RecordEvent(doneEvent_);
        A2aTrace("algo", rankId_, "Algorithm() complete");
        return HcclResult::HCCL_SUCCESS;
    }

    inline std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override
    {
        const auto *tArg = dynamic_cast<const CcuAllToAllTaskArg *>(&arg);
        if (tArg == nullptr) {
            std::fprintf(stderr, "[CCU_A2A/gene] GeneArgs FAILED dynamic_cast\n");
            return {};
        }

        const uint32_t dieId = static_cast<uint32_t>(gateEvent_.DieId());
        const uint32_t ckeId = static_cast<uint32_t>(gateEvent_.Id());
        pto::comm::ccu::Publish(rankId_, dieId, ckeId, gateMask_);

        std::fprintf(stderr,
                     "[CCU_A2A/gene] rank=%u published (die=%u, cke=%u, mask=0x%x) "
                     "input=0x%llx output=0x%llx slice=%llu srcStride=%llu "
                     "srcOff=%llu dstOff=%llu token=0x%llx gateOnly=%d\n",
                     rankId_, dieId, ckeId, gateMask_, static_cast<unsigned long long>(tArg->inputAddr),
                     static_cast<unsigned long long>(tArg->outputAddr),
                     static_cast<unsigned long long>(tArg->sliceSize), static_cast<unsigned long long>(tArg->srcStride),
                     static_cast<unsigned long long>(tArg->srcOffset), static_cast<unsigned long long>(tArg->dstOffset),
                     static_cast<unsigned long long>(tArg->token), static_cast<int>(gateOnly_));

        if (gateOnly_) {
            return {};
        }
        return {tArg->inputAddr, tArg->outputAddr, tArg->token,    tArg->sliceSize,
                tArg->srcStride, tArg->srcOffset,  tArg->dstOffset};
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

        A2aTrace("init", rankId_, "InitResource done (gate-only)");
        return HcclResult::HCCL_SUCCESS;
    }

    inline HcclResult InitResourceWithChannels()
    {
        uint16_t channelIdx = 0;
        for (uint32_t peerId = 0; peerId < rankSize_; peerId++) {
            if (peerId == rankId_) {
                input_.push_back(CreateVariable());
                output_.push_back(CreateVariable());
                token_.push_back(CreateVariable());
            } else {
                hcomm::CcuRep::Variable inputVar, outputVar, tokenVar;
                (void)CreateVariable(ownChannels_[channelIdx], A2A_OUTPUT_XN_ID, &outputVar);
                output_.push_back(outputVar);
                (void)CreateVariable(ownChannels_[channelIdx], A2A_TOKEN_XN_ID, &tokenVar);
                token_.push_back(tokenVar);
                input_.push_back(hcomm::CcuRep::Variable{});
                channelIdx++;
            }
        }

        sliceSizeVar_ = CreateVariable();
        srcStrideVar_ = CreateVariable();
        srcOffsetVar_ = CreateVariable();
        dstOffsetVar_ = CreateVariable();

        srcAddrs_.reserve(rankSize_);
        for (uint32_t i = 0; i < rankSize_; i++) {
            srcAddrs_.push_back(CreateLocalAddr());
        }
        dstAddrs_.reserve(rankSize_);
        for (uint32_t i = 0; i < rankSize_; i++) {
            if (i == rankId_) {
                myDst_ = CreateLocalAddr();
                dstAddrs_.push_back(hcomm::CcuRep::RemoteAddr{});
            } else {
                dstAddrs_.push_back(CreateRemoteAddr());
            }
        }

        gateEvent_ = CreateCompletedEvent();
        gateEvent_.SetMask(gateMask_);
        doneEvent_ = CreateCompletedEvent();
        doneEvent_.SetMask(doneMask_);
        opEvent_ = CreateCompletedEvent();

        A2aTrace("init", rankId_, "InitResource done");
        return HcclResult::HCCL_SUCCESS;
    }

    inline void LoadArgs()
    {
        Load(input_[rankId_]);
        Load(output_[rankId_]);
        Load(token_[rankId_]);
        Load(sliceSizeVar_);
        Load(srcStrideVar_);
        Load(srcOffsetVar_);
        Load(dstOffsetVar_);

        srcOffsetVar_ += input_[rankId_];
    }

    inline void PreSync()
    {
        for (auto ch : ownChannels_) {
            (void)NotifyRecord(ch, A2A_CKE_IDX_0, A2A_OUTPUT_XN_ID, output_[rankId_], 1u << A2A_OUTPUT_XN_ID);
            (void)NotifyRecord(ch, A2A_CKE_IDX_0, A2A_TOKEN_XN_ID, token_[rankId_], 1u << A2A_TOKEN_XN_ID);
        }
        uint32_t allBit = (1u << A2A_OUTPUT_XN_ID) | (1u << A2A_TOKEN_XN_ID);
        for (auto ch : ownChannels_) {
            (void)NotifyWait(ch, A2A_CKE_IDX_0, allBit);
        }
        A2aTrace("sync", rankId_, "PreSync done");
    }

    inline void PostSync()
    {
        for (auto &ch : ownChannels_) {
            (void)NotifyRecord(ch, A2A_CKE_IDX_1, 1u << A2A_POST_SYNC_ID);
        }
        for (auto &ch : ownChannels_) {
            (void)NotifyWait(ch, A2A_CKE_IDX_1, 1u << A2A_POST_SYNC_ID);
        }
        A2aTrace("sync", rankId_, "PostSync done");
    }

    // AllToAll: rank sends input[r*stride + srcOffset] to peer r's output[r] + dstOffset.
    // Each peer r receives at output[r] + dstOffset the data this rank sent.
    // For self: LocalCopyNb.
    inline void DoAllToAll()
    {
        for (uint32_t r = 0; r < rankSize_; r++) {
            srcAddrs_[r].addr = srcOffsetVar_;
            for (uint32_t i = 0; i < r; i++) {
                srcAddrs_[r].addr += srcStrideVar_;
            }
            srcAddrs_[r].token = token_[rankId_];

            if (r == rankId_) {
                myDst_.addr = output_[r];
                myDst_.addr += dstOffsetVar_;
                myDst_.token = token_[r];
            } else {
                dstAddrs_[r].addr = output_[r];
                dstAddrs_[r].addr += dstOffsetVar_;
                dstAddrs_[r].token = token_[r];
            }
        }

        uint32_t chIdx = 0;
        for (uint32_t r = 0; r < rankSize_; r++) {
            opEvent_.SetMask(1u << r);
            if (r == rankId_) {
                LocalCopyNb(myDst_, srcAddrs_[r], sliceSizeVar_, opEvent_);
            } else {
                (void)WriteNb(ownChannels_[chIdx], dstAddrs_[r], srcAddrs_[r], sliceSizeVar_, opEvent_);
                chIdx++;
            }
        }

        opEvent_.SetMask((1u << rankSize_) - 1);
        WaitEvent(opEvent_);

        A2aTrace("a2a", rankId_, "DoAllToAll done");
    }

    uint32_t rankId_{0};
    uint32_t rankSize_{1};
    uint32_t gateMask_{1u << 0};
    uint32_t doneMask_{1u << 0};
    uint64_t payloadBytes_{0};
    bool gateOnly_{false};
    decltype(std::declval<hcomm::CcuKernelArg>().channels) ownChannels_;

    std::vector<hcomm::CcuRep::Variable> input_;
    std::vector<hcomm::CcuRep::Variable> output_;
    std::vector<hcomm::CcuRep::Variable> token_;

    hcomm::CcuRep::Variable sliceSizeVar_;
    hcomm::CcuRep::Variable srcStrideVar_;
    hcomm::CcuRep::Variable srcOffsetVar_;
    hcomm::CcuRep::Variable dstOffsetVar_;

    std::vector<hcomm::CcuRep::LocalAddr> srcAddrs_;
    std::vector<hcomm::CcuRep::RemoteAddr> dstAddrs_;
    hcomm::CcuRep::LocalAddr myDst_;

    hcomm::CcuRep::CompletedEvent gateEvent_;
    hcomm::CcuRep::CompletedEvent doneEvent_;
    hcomm::CcuRep::CompletedEvent opEvent_;
};

} // namespace detail

// ============================================================================
// Public factory
// ============================================================================

inline hcomm::KernelCreator MakeCcuAllToAllCreator()
{
    return [](const hcomm::CcuKernelArg &arg) -> std::unique_ptr<hcomm::CcuKernel> {
        return std::make_unique<detail::CcuAllToAllMesh1D>(arg);
    };
}

} // namespace ccu
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_CCU_CCU_ALL_TO_ALL_KERNEL_HPP
