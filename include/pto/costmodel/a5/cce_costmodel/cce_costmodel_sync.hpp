/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#pragma once

#include <pto/costmodel/perf_sim/recorder.hpp>

#include "vf_trace.hpp"

#ifndef VST_VLD
#define VST_VLD 0
#endif
#ifndef VLD_VST
#define VLD_VST 1
#endif

namespace pto::mocker::a5::sync {

inline const char* VfMemBarName(int64_t barrierType)
{
    return barrierType == static_cast<int64_t>(VLD_VST) ? "VLD_VST" : "VST_VLD";
}

inline void RecordVfMemBar(int64_t barrierType) { ::pto::mocker::vf::trace::RecordMemBar(VfMemBarName(barrierType)); }

inline void RecordPerfBarrier(int pipe)
{
    ::pto::perf_sim::SyncRecorder::Barrier(pipe);
    if (pipe == PIPE_V) {
        ::pto::mocker::FlushPendingTail(::pto::mocker::evaluator::PipeKey::VECTOR);
    } else if (pipe == PIPE_ALL) {
        ::pto::mocker::FlushAllPendingTails();
    }
}

} // namespace pto::mocker::a5::sync

template <typename BarrierType>
inline void mem_bar(BarrierType barrierType)
{
    if (::pto::mocker::vf::trace::Armed()) {
        ::pto::mocker::a5::sync::RecordVfMemBar(static_cast<int64_t>(barrierType));
        return;
    }
    ::pto::mocker::a5::sync::RecordPerfBarrier(PIPE_V);
}

template <typename Pipe>
inline void pto_costmodel_pipe_barrier(Pipe pipe)
{
    if (::pto::mocker::vf::trace::Armed()) {
        ::pto::mocker::a5::sync::RecordVfMemBar(VST_VLD);
        return;
    }
    ::pto::mocker::a5::sync::RecordPerfBarrier(static_cast<int>(pipe));
}

template <typename SrcPipe, typename DstPipe, typename Token>
inline void pto_costmodel_set_flag(SrcPipe srcPipe, DstPipe dstPipe, Token token)
{
    if (!::pto::mocker::vf::trace::Armed()) {
        ::pto::perf_sim::SyncRecorder::Signal(
            static_cast<int>(token), static_cast<int>(srcPipe), static_cast<int>(dstPipe));
    }
}

template <typename SrcPipe, typename DstPipe, typename Token>
inline void pto_costmodel_wait_flag(SrcPipe srcPipe, DstPipe dstPipe, Token token)
{
    if (!::pto::mocker::vf::trace::Armed()) {
        ::pto::perf_sim::SyncRecorder::Wait(
            static_cast<int>(token), static_cast<int>(srcPipe), static_cast<int>(dstPipe));
    }
}

template <typename SrcPipe, typename Message>
inline void ffts_cross_core_sync(SrcPipe srcPipe, Message message)
{
    if (!::pto::mocker::vf::trace::Armed()) {
        const int flagId = (static_cast<int>(message) >> 8) & 0xf;
        ::pto::perf_sim::SyncRecorder::Signal(flagId, static_cast<int>(srcPipe), -1, true);
    }
}

template <typename FlagId>
inline void wait_flag_dev(FlagId flagId)
{
    if (!::pto::mocker::vf::trace::Armed()) {
        ::pto::perf_sim::SyncRecorder::Wait(static_cast<int>(flagId), -1, -1, true);
    }
}

template <typename SrcPipe, typename FlagId>
inline void wait_flag_dev(SrcPipe srcPipe, FlagId flagId)
{
    if (!::pto::mocker::vf::trace::Armed()) {
        ::pto::perf_sim::SyncRecorder::Wait(static_cast<int>(flagId), static_cast<int>(srcPipe), -1, true);
    }
}

template <typename SrcPipe, typename FlagId>
inline void set_intra_block(SrcPipe srcPipe, FlagId flagId)
{
    if (!::pto::mocker::vf::trace::Armed()) {
        ::pto::perf_sim::SyncRecorder::Signal(static_cast<int>(flagId), static_cast<int>(srcPipe), -1, true);
    }
}

template <typename SrcPipe, typename FlagId>
inline void wait_intra_block(SrcPipe srcPipe, FlagId flagId)
{
    if (!::pto::mocker::vf::trace::Armed()) {
        ::pto::perf_sim::SyncRecorder::Wait(static_cast<int>(flagId), static_cast<int>(srcPipe), -1, true);
    }
}

template <typename BarrierType>
inline void dsb(BarrierType)
{
    if (::pto::mocker::vf::trace::Armed()) {
        ::pto::mocker::a5::sync::RecordVfMemBar(VST_VLD);
        return;
    }
    ::pto::mocker::a5::sync::RecordPerfBarrier(PIPE_ALL);
}

// Supply the A5 costmodel SyncAll implementation before the real A5 header is
// reached by pto_instr_impl.hpp. This prevents costmodel builds from entering
// syncall_soft.hpp and its device-only atomic/load/store intrinsics.
#ifndef PTO_NPU_A5_SYNCALL_HPP
#define PTO_NPU_A5_SYNCALL_HPP

namespace pto {

inline uint16_t CostmodelFftsMessage(uint16_t eventId) { return static_cast<uint16_t>(1U + ((eventId & 0xfU) << 8U)); }

template <SyncCoreType CoreType = SyncCoreType::AIVOnly>
inline void SYNCALL_IMPL()
{
    pto_costmodel_pipe_barrier(PIPE_ALL);
    if constexpr (CoreType == SyncCoreType::AIVOnly) {
        ffts_cross_core_sync(PIPE_MTE3, CostmodelFftsMessage(SYNC_AIV_ONLY_ALL));
        wait_flag_dev(PIPE_S, SYNC_AIV_ONLY_ALL);
    } else if constexpr (CoreType == SyncCoreType::AICOnly) {
        ffts_cross_core_sync(PIPE_FIX, CostmodelFftsMessage(SYNC_AIC_FLAG));
        wait_flag_dev(PIPE_S, SYNC_AIC_FLAG);
    } else {
        set_intra_block(PIPE_MTE3, SYNC_AIV_FLAG);
        wait_intra_block(PIPE_S, SYNC_AIC_AIV_FLAG);
    }
}

template <SyncCoreType CoreType = SyncCoreType::AIVOnly, typename T>
inline void SYNCALL_SOFT_IMPL(T*, int32_t)
{
    SYNCALL_IMPL<CoreType>();
}

template <typename T>
inline void SYNCALL_SOFT_AIC_IMPL(T*, int32_t)
{
    SYNCALL_IMPL<SyncCoreType::AICOnly>();
}

template <SyncCoreType CoreType = SyncCoreType::Mix, typename T>
inline void SYNCALL_SOFT_MIX_IMPL(T*, int32_t)
{
    SYNCALL_IMPL<CoreType>();
}

} // namespace pto

#endif // PTO_NPU_A5_SYNCALL_HPP
