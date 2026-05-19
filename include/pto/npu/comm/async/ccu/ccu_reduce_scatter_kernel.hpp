/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_CCU_CCU_REDUCE_SCATTER_KERNEL_HPP
#define PTO_COMM_ASYNC_CCU_CCU_REDUCE_SCATTER_KERNEL_HPP

// Host-only header — must NOT be included from device (bisheng -xcce) code.
#if defined(__CCE_KT_TEST__)
#error "ccu_reduce_scatter_kernel.hpp is a host-only header and cannot be included in device code."
#endif

// CCU-native Gated ReduceScatter kernel — header-only.
//
// Based on HCCL CcuKernelReduceScatterMesh1D (mesh_1D topology).
// Every rank participates in data reduction. Each rank obtains a slice of
// the reduced result. The input buffer of each rank contains the full data;
// each rank reduces all peers' data at `inputAddr + offset` (where offset
// selects this rank's slice) into its own `outputAddr`.
//
// Data path (per rank): ReadNb from peers + LocalCopyNb for self →
//   LocalReduceNb → LocalCopyNb to output.
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

struct CcuReduceScatterKernelArg : public hcomm::CcuKernelArg {
    uint32_t rankId{0};
    uint32_t rankSize{1};

    HcclDataType dataType{HcclDataType::HCCL_DATA_TYPE_FP32};
    HcclDataType outputDataType{HcclDataType::HCCL_DATA_TYPE_FP32};
    HcclReduceOp reduceOp{HcclReduceOp::HCCL_REDUCE_SUM};

    uint32_t gateMask{1u << 0};
    uint32_t doneMask{1u << 0};

    uint64_t payloadBytes{0};

    CcuReduceScatterKernelArg() = default;
    CcuReduceScatterKernelArg(uint32_t rid, uint32_t rsize, HcclDataType dt, HcclReduceOp op, uint64_t bytes,
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
        sig.Append(std::string("pto::comm::ccu::CcuReduceScatterKernelArg::v1"));
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

struct CcuReduceScatterTaskArg : public hcomm::CcuTaskArg {
    uint64_t inputAddr{0};
    uint64_t outputAddr{0};
    uint64_t offset{0};
    uint64_t sliceSize{0};
    uint64_t token{0};

    CcuReduceScatterTaskArg() = default;
    CcuReduceScatterTaskArg(uint64_t in, uint64_t out, uint64_t off, uint64_t slice, uint64_t tok)
        : inputAddr(in), outputAddr(out), offset(off), sliceSize(slice), token(tok)
    {}
};

// ============================================================================
// Kernel implementation (detail)
// ============================================================================

namespace detail {

constexpr uint32_t RS_INPUT_XN_ID = 0;
constexpr uint32_t RS_TOKEN_XN_ID = 1;
constexpr uint32_t RS_POST_SYNC_ID = 2;
constexpr uint32_t RS_CKE_IDX_0 = 0;

inline void RsTrace(const char *tag, uint32_t rank, const char *msg)
{
    std::fprintf(stderr, "[CCU_RS/%s] rank=%u %s\n", tag, rank, msg);
    std::fflush(stderr);
}

class CcuReduceScatterMesh1D : public hcomm::CcuKernel {
public:
    inline explicit CcuReduceScatterMesh1D(const hcomm::CcuKernelArg &arg) : CcuKernel(arg)
    {
        const auto *kArg = dynamic_cast<const CcuReduceScatterKernelArg *>(&arg);
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
                     "[CCU_RS/ctor] rank=%u rankSize=%u "
                     "dataType=%d reduceOp=%d payloadBytes=%llu "
                     "ownChannels=%zu channels_=%zu\n",
                     rankId_, rankSize_, static_cast<int>(dataType_), static_cast<int>(reduceOp_),
                     static_cast<unsigned long long>(payloadBytes_), ownChannels_.size(), channels_.size());
    }

    ~CcuReduceScatterMesh1D() override = default;

    inline HcclResult Algorithm() override
    {
        RsTrace("algo", rankId_, "Algorithm() entry");

        HcclResult ret = InitResource();
        if (ret != HcclResult::HCCL_SUCCESS) {
            std::fprintf(stderr, "[CCU_RS/algo] rank=%u InitResource FAILED ret=%d\n", rankId_,
                         static_cast<int>(ret));
            return ret;
        }

        if (gateOnly_) {
            WaitEvent(gateEvent_);
            RsTrace("algo", rankId_, "gate released (gate-only)");
            RecordEvent(doneEvent_);
            RsTrace("algo", rankId_, "Algorithm() complete (gate-only)");
            return HcclResult::HCCL_SUCCESS;
        }

        LoadArgs();

        WaitEvent(gateEvent_);
        RsTrace("algo", rankId_, "gate released");

        PreSync();
        DoReduceScatter();
        PostSync();

        RecordEvent(doneEvent_);
        RsTrace("algo", rankId_, "Algorithm() complete");
        return HcclResult::HCCL_SUCCESS;
    }

    inline std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override
    {
        const auto *tArg = dynamic_cast<const CcuReduceScatterTaskArg *>(&arg);
        if (tArg == nullptr) {
            std::fprintf(stderr, "[CCU_RS/gene] GeneArgs FAILED dynamic_cast\n");
            return {};
        }

        const uint32_t dieId = static_cast<uint32_t>(gateEvent_.DieId());
        const uint32_t ckeId = static_cast<uint32_t>(gateEvent_.Id());
        pto::comm::ccu::Publish(rankId_, dieId, ckeId, gateMask_);

        std::fprintf(stderr,
                     "[CCU_RS/gene] rank=%u published (die=%u, cke=%u, mask=0x%x) "
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
        std::fprintf(stderr,
                     "[CCU_RS/init] rank=%u — no channels, "
                     "gate-only mode (SetDieId fallback).\n",
                     rankId_);
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

        RsTrace("init", rankId_, "InitResource done (gate-only)");
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
                hcomm::CcuRep::Variable inputVar, tokenVar;
                (void)CreateVariable(ownChannels_[channelIdx], RS_INPUT_XN_ID, &inputVar);
                input_.push_back(inputVar);
                (void)CreateVariable(ownChannels_[channelIdx], RS_TOKEN_XN_ID, &tokenVar);
                token_.push_back(tokenVar);
                channelIdx++;
            }
        }

        offsetVar_ = CreateVariable();
        sliceSizeVar_ = CreateVariable();

        dstAddr_ = CreateLocalAddr();
        srcAddr_.reserve(rankSize_);
        for (uint32_t i = 0; i < rankSize_; i++) {
            srcAddr_.push_back(CreateRemoteAddr());
        }

        gateEvent_ = CreateCompletedEvent();
        gateEvent_.SetMask(gateMask_);

        doneEvent_ = CreateCompletedEvent();
        doneEvent_.SetMask(doneMask_);

        opEvent_ = CreateCompletedEvent();

        RsTrace("init", rankId_, "InitResource done");
        return HcclResult::HCCL_SUCCESS;
    }

    inline void LoadArgs()
    {
        Load(input_[rankId_]);
        Load(output_[0]);
        Load(token_[rankId_]);
        Load(offsetVar_);
        Load(sliceSizeVar_);
    }

    inline void PreSync()
    {
        for (auto ch : ownChannels_) {
            (void)NotifyRecord(ch, RS_CKE_IDX_0, RS_INPUT_XN_ID, input_[rankId_], 1u << RS_INPUT_XN_ID);
            (void)NotifyRecord(ch, RS_CKE_IDX_0, RS_TOKEN_XN_ID, token_[rankId_], 1u << RS_TOKEN_XN_ID);
        }
        uint32_t allBit = (1u << RS_INPUT_XN_ID) | (1u << RS_TOKEN_XN_ID);
        for (auto ch : ownChannels_) {
            (void)NotifyWait(ch, RS_CKE_IDX_0, allBit);
        }
        RsTrace("sync", rankId_, "PreSync done");
    }

    inline void PostSync()
    {
        for (auto &ch : ownChannels_) {
            (void)NotifyRecord(ch, RS_CKE_IDX_0, 1u << RS_POST_SYNC_ID);
        }
        for (auto &ch : ownChannels_) {
            (void)NotifyWait(ch, RS_CKE_IDX_0, 1u << RS_POST_SYNC_ID);
        }
        RsTrace("sync", rankId_, "PostSync done");
    }

    inline void DoReduceScatter()
    {
        std::vector<hcomm::CcuRep::CcuBuf> bufs(rankSize_);
        (void)CreateBlockCcuBuf(rankSize_, bufs.data());

        // Build src addresses: remote peers via ReadNb, self via LocalCopyNb.
        // All src addresses are offset by `offsetVar_` to select this rank's slice.
        uint32_t curId = 0;
        for (uint32_t r = 0; r < rankSize_; r++) {
            if (r != rankId_) {
                srcAddr_[curId].addr = input_[r];
                srcAddr_[curId].addr += offsetVar_;
                srcAddr_[curId].token = token_[r];
                curId++;
            }
        }
        // Self input placed at array end (mirrors HCCL convention)
        srcAddr_[rankSize_ - 1].addr = input_[rankId_];
        srcAddr_[rankSize_ - 1].addr += offsetVar_;
        srcAddr_[rankSize_ - 1].token = token_[rankId_];

        // dst: this rank's output buffer (output_ has only 1 element)
        dstAddr_.addr = output_[0];
        dstAddr_.token = token_[rankId_];

        // ReadNb from remote peers
        uint32_t chIdx = 0;
        for (uint32_t r = 0; r < rankSize_; r++) {
            if (r != rankId_) {
                opEvent_.SetMask(1u << chIdx);
                (void)ReadNb(ownChannels_[chIdx], bufs[chIdx], srcAddr_[chIdx], sliceSizeVar_, opEvent_);
                chIdx++;
            }
        }
        // LocalCopyNb for self
        uint32_t localBufIdx = rankSize_ - 1;
        opEvent_.SetMask(1u << localBufIdx);
        LocalCopyNb(bufs[localBufIdx], *reinterpret_cast<hcomm::CcuRep::LocalAddr *>(&srcAddr_[localBufIdx]),
                    sliceSizeVar_, opEvent_);

        // Wait all reads
        opEvent_.SetMask((1u << rankSize_) - 1);
        WaitEvent(opEvent_);

        // Reduce
        if (rankSize_ > 1) {
            opEvent_.SetMask(1u);
            LocalReduceNb(bufs.data(), rankSize_, dataType_, outputDataType_, reduceOp_, sliceSizeVar_, opEvent_);
            WaitEvent(opEvent_);
        }

        // Copy result to output
        opEvent_.SetMask(1u);
        LocalCopyNb(dstAddr_, bufs[0], sliceSizeVar_, opEvent_);
        WaitEvent(opEvent_);

        RsTrace("reduce", rankId_, "DoReduceScatter done");
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

    hcomm::CcuRep::LocalAddr dstAddr_;
    std::vector<hcomm::CcuRep::RemoteAddr> srcAddr_;

    hcomm::CcuRep::CompletedEvent gateEvent_;
    hcomm::CcuRep::CompletedEvent doneEvent_;
    hcomm::CcuRep::CompletedEvent opEvent_;
};

} // namespace detail

// ============================================================================
// Public factory
// ============================================================================

inline hcomm::KernelCreator MakeCcuReduceScatterCreator()
{
    return [](const hcomm::CcuKernelArg &arg) -> std::unique_ptr<hcomm::CcuKernel> {
        return std::make_unique<detail::CcuReduceScatterMesh1D>(arg);
    };
}

} // namespace ccu
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_CCU_CCU_REDUCE_SCATTER_KERNEL_HPP
