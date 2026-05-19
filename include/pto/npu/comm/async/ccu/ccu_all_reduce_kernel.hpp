/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_CCU_CCU_ALL_REDUCE_KERNEL_HPP
#define PTO_COMM_ASYNC_CCU_CCU_ALL_REDUCE_KERNEL_HPP

// Host-only header — must NOT be included from device (bisheng -xcce) code.
#if defined(__CCE_KT_TEST__)
#error "ccu_all_reduce_kernel.hpp is a host-only header and cannot be included in device code."
#endif

// CCU-native Gated AllReduce kernel — header-only.
//
// Based on HCCL CcuKernelAllReduceMesh1D (mesh_1D topology).
// AllReduce = ReduceScatter + AllGather in a single kernel launch.
//
// Phase 1 (ReduceScatter): Each rank reduces all peers' input slices at
//   input[r] + offset into output[self] + offset.
// Phase 2 (AllGather): Each rank broadcasts its reduced slice at
//   output[self] + offset to all peers' output[r] + offset.
//
// After both phases every rank holds the full reduced result in output.
//
// Dependencies: hcomm pkg_inc only (libhcomm.so). No hccl dependency.

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

struct CcuAllReduceKernelArg : public hcomm::CcuKernelArg {
    uint32_t rankId{0};
    uint32_t rankSize{1};

    HcclDataType dataType{HcclDataType::HCCL_DATA_TYPE_FP32};
    HcclDataType outputDataType{HcclDataType::HCCL_DATA_TYPE_FP32};
    HcclReduceOp reduceOp{HcclReduceOp::HCCL_REDUCE_SUM};

    uint32_t gateMask{1u << 0};
    uint32_t doneMask{1u << 0};

    uint64_t payloadBytes{0};

    CcuAllReduceKernelArg() = default;
    CcuAllReduceKernelArg(uint32_t rid, uint32_t rsize, HcclDataType dt, HcclReduceOp op, uint64_t bytes,
                          uint32_t gMask = (1u << 0), uint32_t dMask = (1u << 0))
        : rankId(rid),
          rankSize(rsize),
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
        sig.Append(std::string("pto::comm::ccu::CcuAllReduceKernelArg::v1"));
        sig.Append(rankId);
        sig.Append(rankSize);
        sig.Append(static_cast<uint32_t>(dataType));
        sig.Append(static_cast<uint32_t>(outputDataType));
        sig.Append(static_cast<uint32_t>(reduceOp));
        sig.Append(gateMask);
        sig.Append(doneMask);
        sig.Append(payloadBytes);
        return sig;
    }
};

struct CcuAllReduceTaskArg : public hcomm::CcuTaskArg {
    uint64_t inputAddr{0};
    uint64_t outputAddr{0};
    uint64_t offset{0};
    uint64_t sliceSize{0};
    uint64_t token{0};

    CcuAllReduceTaskArg() = default;
    CcuAllReduceTaskArg(uint64_t in, uint64_t out, uint64_t off, uint64_t slice, uint64_t tok)
        : inputAddr(in), outputAddr(out), offset(off), sliceSize(slice), token(tok)
    {}
};

// ============================================================================
// Kernel implementation (detail)
// ============================================================================

namespace detail {

constexpr uint32_t AR_INPUT_XN_ID = 0;
constexpr uint32_t AR_OUTPUT_XN_ID = 1;
constexpr uint32_t AR_TOKEN_XN_ID = 2;
constexpr uint32_t AR_POST_SYNC_ID = 3;
constexpr uint32_t AR_CKE_IDX_0 = 0;

inline void ArTrace(const char *tag, uint32_t rank, const char *msg)
{
    std::fprintf(stderr, "[CCU_AR/%s] rank=%u %s\n", tag, rank, msg);
    std::fflush(stderr);
}

class CcuAllReduceMesh1D : public hcomm::CcuKernel {
public:
    inline explicit CcuAllReduceMesh1D(const hcomm::CcuKernelArg &arg) : CcuKernel(arg)
    {
        const auto *kArg = dynamic_cast<const CcuAllReduceKernelArg *>(&arg);
        if (kArg != nullptr) {
            rankId_ = kArg->rankId;
            rankSize_ = kArg->rankSize;
            dataType_ = kArg->dataType;
            outputDataType_ = kArg->outputDataType;
            reduceOp_ = kArg->reduceOp;
            gateMask_ = kArg->gateMask;
            doneMask_ = kArg->doneMask;
            payloadBytes_ = kArg->payloadBytes;
        }
        ownChannels_ = arg.channels;
        std::fprintf(stderr,
                     "[CCU_AR/ctor] rank=%u rankSize=%u "
                     "dataType=%d reduceOp=%d payloadBytes=%llu "
                     "ownChannels=%zu channels_=%zu\n",
                     rankId_, rankSize_, static_cast<int>(dataType_), static_cast<int>(reduceOp_),
                     static_cast<unsigned long long>(payloadBytes_), ownChannels_.size(), channels_.size());
    }

    ~CcuAllReduceMesh1D() override = default;

    inline HcclResult Algorithm() override
    {
        ArTrace("algo", rankId_, "Algorithm() entry");

        HcclResult ret = InitResource();
        if (ret != HcclResult::HCCL_SUCCESS) {
            std::fprintf(stderr, "[CCU_AR/algo] rank=%u InitResource FAILED ret=%d\n", rankId_,
                         static_cast<int>(ret));
            return ret;
        }

        if (gateOnly_) {
            WaitEvent(gateEvent_);
            ArTrace("algo", rankId_, "gate released (gate-only)");
            RecordEvent(doneEvent_);
            ArTrace("algo", rankId_, "Algorithm() complete (gate-only)");
            return HcclResult::HCCL_SUCCESS;
        }

        LoadArgs();

        WaitEvent(gateEvent_);
        ArTrace("algo", rankId_, "gate released");

        PreSync();
        DoReduceScatterPhase();
        DoAllGatherPhase();
        PostSync();

        RecordEvent(doneEvent_);
        ArTrace("algo", rankId_, "Algorithm() complete");
        return HcclResult::HCCL_SUCCESS;
    }

    inline std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override
    {
        const auto *tArg = dynamic_cast<const CcuAllReduceTaskArg *>(&arg);
        if (tArg == nullptr) {
            std::fprintf(stderr, "[CCU_AR/gene] GeneArgs FAILED dynamic_cast\n");
            return {};
        }

        const uint32_t dieId = static_cast<uint32_t>(gateEvent_.DieId());
        const uint32_t ckeId = static_cast<uint32_t>(gateEvent_.Id());
        pto::comm::ccu::Publish(rankId_, dieId, ckeId, gateMask_);

        std::fprintf(stderr,
                     "[CCU_AR/gene] rank=%u published (die=%u, cke=%u, mask=0x%x) "
                     "input=0x%llx output=0x%llx offset=%llu slice=%llu token=0x%llx gateOnly=%d\n",
                     rankId_, dieId, ckeId, gateMask_,
                     static_cast<unsigned long long>(tArg->inputAddr),
                     static_cast<unsigned long long>(tArg->outputAddr),
                     static_cast<unsigned long long>(tArg->offset),
                     static_cast<unsigned long long>(tArg->sliceSize),
                     static_cast<unsigned long long>(tArg->token),
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

        ArTrace("init", rankId_, "InitResource done (gate-only)");
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
                (void)CreateVariable(ownChannels_[channelIdx], AR_INPUT_XN_ID, &inputVar);
                input_.push_back(inputVar);
                (void)CreateVariable(ownChannels_[channelIdx], AR_OUTPUT_XN_ID, &outputVar);
                output_.push_back(outputVar);
                (void)CreateVariable(ownChannels_[channelIdx], AR_TOKEN_XN_ID, &tokenVar);
                token_.push_back(tokenVar);
                channelIdx++;
            }
        }

        offsetVar_ = CreateVariable();
        sliceSizeVar_ = CreateVariable();

        rsSrcAddr_.reserve(rankSize_);
        for (uint32_t i = 0; i < rankSize_; i++) {
            rsSrcAddr_.push_back(CreateRemoteAddr());
        }
        rsDstAddr_ = CreateLocalAddr();

        agSrcAddr_ = CreateLocalAddr();
        agDstAddrs_.reserve(rankSize_);
        for (uint32_t i = 0; i < rankSize_; i++) {
            agDstAddrs_.push_back(CreateRemoteAddr());
        }
        agLocalDstAddr_ = CreateLocalAddr();

        gateEvent_ = CreateCompletedEvent();
        gateEvent_.SetMask(gateMask_);
        doneEvent_ = CreateCompletedEvent();
        doneEvent_.SetMask(doneMask_);
        opEvent_ = CreateCompletedEvent();

        ArTrace("init", rankId_, "InitResource done");
        return HcclResult::HCCL_SUCCESS;
    }

    inline void LoadArgs()
    {
        Load(input_[rankId_]);
        Load(output_[rankId_]);
        Load(token_[rankId_]);
        Load(offsetVar_);
        Load(sliceSizeVar_);
    }

    inline void PreSync()
    {
        for (auto ch : ownChannels_) {
            (void)NotifyRecord(ch, AR_CKE_IDX_0, AR_INPUT_XN_ID, input_[rankId_], 1u << AR_INPUT_XN_ID);
            (void)NotifyRecord(ch, AR_CKE_IDX_0, AR_OUTPUT_XN_ID, output_[rankId_], 1u << AR_OUTPUT_XN_ID);
            (void)NotifyRecord(ch, AR_CKE_IDX_0, AR_TOKEN_XN_ID, token_[rankId_], 1u << AR_TOKEN_XN_ID);
        }
        uint32_t allBit = (1u << AR_INPUT_XN_ID) | (1u << AR_OUTPUT_XN_ID) | (1u << AR_TOKEN_XN_ID);
        for (auto ch : ownChannels_) {
            (void)NotifyWait(ch, AR_CKE_IDX_0, allBit);
        }
        ArTrace("sync", rankId_, "PreSync done");
    }

    inline void PostSync()
    {
        for (auto &ch : ownChannels_) {
            (void)NotifyRecord(ch, AR_CKE_IDX_0, 1u << AR_POST_SYNC_ID);
        }
        for (auto &ch : ownChannels_) {
            (void)NotifyWait(ch, AR_CKE_IDX_0, 1u << AR_POST_SYNC_ID);
        }
        ArTrace("sync", rankId_, "PostSync done");
    }

    // Phase 1: ReduceScatter — read all peers' input slices, reduce, store to
    // local output[self] + offset.
    inline void DoReduceScatterPhase()
    {
        std::vector<hcomm::CcuRep::CcuBuf> bufs(rankSize_);
        (void)CreateBlockCcuBuf(rankSize_, bufs.data());

        uint32_t curId = 0;
        for (uint32_t r = 0; r < rankSize_; r++) {
            if (r != rankId_) {
                rsSrcAddr_[curId].addr = input_[r];
                rsSrcAddr_[curId].addr += offsetVar_;
                rsSrcAddr_[curId].token = token_[r];
                curId++;
            }
        }
        rsSrcAddr_[rankSize_ - 1].addr = input_[rankId_];
        rsSrcAddr_[rankSize_ - 1].addr += offsetVar_;
        rsSrcAddr_[rankSize_ - 1].token = token_[rankId_];

        rsDstAddr_.addr = output_[rankId_];
        rsDstAddr_.addr += offsetVar_;
        rsDstAddr_.token = token_[rankId_];

        uint32_t chIdx = 0;
        for (uint32_t r = 0; r < rankSize_; r++) {
            if (r != rankId_) {
                opEvent_.SetMask(1u << chIdx);
                (void)ReadNb(ownChannels_[chIdx], bufs[chIdx], rsSrcAddr_[chIdx], sliceSizeVar_, opEvent_);
                chIdx++;
            }
        }
        uint32_t localBufIdx = rankSize_ - 1;
        opEvent_.SetMask(1u << localBufIdx);
        LocalCopyNb(bufs[localBufIdx], *reinterpret_cast<hcomm::CcuRep::LocalAddr *>(&rsSrcAddr_[localBufIdx]),
                    sliceSizeVar_, opEvent_);

        opEvent_.SetMask((1u << rankSize_) - 1);
        WaitEvent(opEvent_);

        if (rankSize_ > 1) {
            opEvent_.SetMask(1u);
            LocalReduceNb(bufs.data(), rankSize_, dataType_, outputDataType_, reduceOp_, sliceSizeVar_, opEvent_);
            WaitEvent(opEvent_);
        }

        opEvent_.SetMask(1u);
        LocalCopyNb(rsDstAddr_, bufs[0], sliceSizeVar_, opEvent_);
        WaitEvent(opEvent_);

        ArTrace("reduce", rankId_, "DoReduceScatterPhase done");
    }

    // Phase 2: AllGather — broadcast local reduced slice to all peers' output.
    inline void DoAllGatherPhase()
    {
        agSrcAddr_.addr = output_[rankId_];
        agSrcAddr_.addr += offsetVar_;
        agSrcAddr_.token = token_[rankId_];

        uint32_t chIdx = 0;
        for (uint32_t r = 0; r < rankSize_; r++) {
            if (r != rankId_) {
                agDstAddrs_[chIdx].addr = output_[r];
                agDstAddrs_[chIdx].addr += offsetVar_;
                agDstAddrs_[chIdx].token = token_[r];
                opEvent_.SetMask(1u << chIdx);
                (void)WriteNb(ownChannels_[chIdx], agDstAddrs_[chIdx], agSrcAddr_, sliceSizeVar_, opEvent_);
                chIdx++;
            }
        }
        // Local copy for self (output already has the reduced data at offset,
        // but the HCCL convention writes it through the same path for symmetry).
        agLocalDstAddr_.addr = output_[rankId_];
        agLocalDstAddr_.addr += offsetVar_;
        agLocalDstAddr_.token = token_[rankId_];
        opEvent_.SetMask(1u << chIdx);
        LocalCopyNb(agLocalDstAddr_, agSrcAddr_, sliceSizeVar_, opEvent_);

        opEvent_.SetMask((1u << (chIdx + 1)) - 1);
        WaitEvent(opEvent_);

        ArTrace("gather", rankId_, "DoAllGatherPhase done");
    }

    uint32_t rankId_{0};
    uint32_t rankSize_{1};
    uint32_t gateMask_{1u << 0};
    uint32_t doneMask_{1u << 0};
    uint64_t payloadBytes_{0};
    bool gateOnly_{false};
    decltype(std::declval<hcomm::CcuKernelArg>().channels) ownChannels_;
    HcclDataType dataType_{HcclDataType::HCCL_DATA_TYPE_FP32};
    HcclDataType outputDataType_{HcclDataType::HCCL_DATA_TYPE_FP32};
    HcclReduceOp reduceOp_{HcclReduceOp::HCCL_REDUCE_SUM};

    std::vector<hcomm::CcuRep::Variable> input_;
    std::vector<hcomm::CcuRep::Variable> output_;
    std::vector<hcomm::CcuRep::Variable> token_;

    hcomm::CcuRep::Variable offsetVar_;
    hcomm::CcuRep::Variable sliceSizeVar_;

    // Phase 1 (ReduceScatter)
    std::vector<hcomm::CcuRep::RemoteAddr> rsSrcAddr_;
    hcomm::CcuRep::LocalAddr rsDstAddr_;

    // Phase 2 (AllGather)
    hcomm::CcuRep::LocalAddr agSrcAddr_;
    std::vector<hcomm::CcuRep::RemoteAddr> agDstAddrs_;
    hcomm::CcuRep::LocalAddr agLocalDstAddr_;

    hcomm::CcuRep::CompletedEvent gateEvent_;
    hcomm::CcuRep::CompletedEvent doneEvent_;
    hcomm::CcuRep::CompletedEvent opEvent_;
};

} // namespace detail

// ============================================================================
// Public factory
// ============================================================================

inline hcomm::KernelCreator MakeCcuAllReduceCreator()
{
    return [](const hcomm::CcuKernelArg &arg) -> std::unique_ptr<hcomm::CcuKernel> {
        return std::make_unique<detail::CcuAllReduceMesh1D>(arg);
    };
}

} // namespace ccu
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_CCU_CCU_ALL_REDUCE_KERNEL_HPP
