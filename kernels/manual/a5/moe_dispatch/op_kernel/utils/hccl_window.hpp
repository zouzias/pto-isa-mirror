/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_DISPATCH_HCCL_WINDOW_HPP_
#define MOE_DISPATCH_HCCL_WINDOW_HPP_

#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

#include "const_args.hpp"
#include "hccl_context.hpp"

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t *
#endif

#define FORCE_INLINE_AICORE AICORE inline __attribute__((always_inline))

namespace moe_dispatch {

template <typename T>
FORCE_INLINE_AICORE void GmStore(__gm__ T *addr, T val)
{
    *((__gm__ T *)addr) = val;
}

struct PtoRemoteWindow {
    FORCE_INLINE_AICORE void Init(GM_ADDR remoteWindowContext)
    {
        context_ = reinterpret_cast<__gm__ PtoRemoteWindowContext *>(remoteWindowContext);
        rank_ = static_cast<int32_t>(context_->rank);
        rankSize_ = static_cast<int32_t>(context_->rankSize);
        segmentBytes_ = static_cast<uint64_t>(context_->windowBytes);
    }

    FORCE_INLINE_AICORE GM_ADDR LocalWindowBase() const
    {
        return reinterpret_cast<GM_ADDR>(context_->windowIn[rank_]);
    }

    FORCE_INLINE_AICORE GM_ADDR RankWindowBase(int32_t rankId) const
    {
        return reinterpret_cast<GM_ADDR>(context_->windowIn[rankId]);
    }

    FORCE_INLINE_AICORE GM_ADDR operator()(int64_t offset, int32_t rankId) const
    {
        if (offset < 0 || offset >= static_cast<int64_t>(segmentBytes_) || rankId < 0 || rankId >= rankSize_) {
            return nullptr;
        }
        return RankWindowBase(rankId) + offset;
    }

    FORCE_INLINE_AICORE uint64_t SegmentSize() const
    {
        return segmentBytes_;
    }

    FORCE_INLINE_AICORE int32_t Rank() const
    {
        return rank_;
    }

    FORCE_INLINE_AICORE int32_t RankSize() const
    {
        return rankSize_;
    }

    FORCE_INLINE_AICORE void NotifyRemoteTokenReady(int32_t rankId)
    {
        pipe_barrier(PIPE_ALL);
        pto::SYNCALL_SOFT_DCCI(reinterpret_cast<__gm__ void *>(LocalSignalBase()));
        pto::comm::Signal signal(RemoteTokenReadyCounter(rankId, rank_));
        pto::comm::TNOTIFY(signal, 1, pto::comm::NotifyOp::AtomicAdd);
    }

    FORCE_INLINE_AICORE void WaitTokenReady(int32_t srcRank)
    {
        pto::comm::Signal signal(LocalTokenReadyCounter(srcRank));
        pto::comm::TWAIT(signal, 1, pto::comm::WaitCmp::GE);
    }

private:
    FORCE_INLINE_AICORE uint64_t SignalRegionOffsetBytes() const
    {
        return segmentBytes_ - MOE_DISPATCH_SIGNAL_BYTES;
    }

    FORCE_INLINE_AICORE __gm__ int32_t *LocalSignalBase() const
    {
        return reinterpret_cast<__gm__ int32_t *>(LocalWindowBase() + SignalRegionOffsetBytes());
    }

    FORCE_INLINE_AICORE __gm__ int32_t *RemoteSignalBase(int32_t rankId) const
    {
        return reinterpret_cast<__gm__ int32_t *>((*this)(SignalRegionOffsetBytes(), rankId));
    }

    FORCE_INLINE_AICORE __gm__ int32_t *LocalTokenReadyCounter(int32_t srcRank) const
    {
        return LocalSignalBase() + MOE_DISPATCH_TOKEN_READY_BASE_INDEX + srcRank * MOE_DISPATCH_SIGNAL_STRIDE_I32;
    }

    FORCE_INLINE_AICORE __gm__ int32_t *RemoteTokenReadyCounter(int32_t rankId, int32_t srcRank) const
    {
        return RemoteSignalBase(rankId) + MOE_DISPATCH_TOKEN_READY_BASE_INDEX +
               srcRank * MOE_DISPATCH_SIGNAL_STRIDE_I32;
    }

    __gm__ PtoRemoteWindowContext *context_ = nullptr;
    int32_t rank_ = 0;
    int32_t rankSize_ = 0;
    uint64_t segmentBytes_ = 0;
};

} // namespace moe_dispatch

#endif // MOE_DISPATCH_HCCL_WINDOW_HPP_
