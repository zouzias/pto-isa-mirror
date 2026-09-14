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

#include "a5_vf_stub.hpp"

#ifndef VST_VLD
#define VST_VLD 0
#endif
#ifndef VLD_VST
#define VLD_VST 1
#endif
#ifndef VST_VST
#define VST_VST 2
#endif

namespace pto::mocker::a5::sync {
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
inline void mem_bar(BarrierType)
{
    if (::pto::mocker::a5::host::InVfScope()) {
        return;
    }
    ::pto::mocker::a5::sync::RecordPerfBarrier(PIPE_V);
}

template <typename Pipe>
inline void pto_costmodel_pipe_barrier(Pipe pipe)
{
    if (::pto::mocker::a5::host::InVfScope()) {
        return;
    }
    ::pto::mocker::a5::sync::RecordPerfBarrier(static_cast<int>(pipe));
}

inline void set_atomic_add() { ::pto::mocker::RecordCceCall("set_atomic_add", 1); }
inline void set_atomic_bf16() { ::pto::mocker::RecordCceCall("set_atomic_bf16", 1); }
inline void set_atomic_f16() { ::pto::mocker::RecordCceCall("set_atomic_f16", 1); }
inline void set_atomic_f32() { ::pto::mocker::RecordCceCall("set_atomic_f32", 1); }
inline void set_atomic_none() { ::pto::mocker::RecordCceCall("set_atomic_none", 1); }
inline void set_atomic_s16() { ::pto::mocker::RecordCceCall("set_atomic_s16", 1); }
inline void set_atomic_s32() { ::pto::mocker::RecordCceCall("set_atomic_s32", 1); }
inline void set_atomic_s8() { ::pto::mocker::RecordCceCall("set_atomic_s8", 1); }
inline void set_fpc(auto addr) { ::pto::mocker::RecordCceCall("set_fpc", 1, addr); }
inline void set_quant_pre(auto mode) { ::pto::mocker::RecordCceCall("set_quant_pre", 1, mode); }

template <typename SrcPipe, typename DstPipe, typename Token>
inline void set_flag(SrcPipe srcPipe, DstPipe dstPipe, Token token)
{
    if (!::pto::mocker::a5::host::InVfScope()) {
        ::pto::perf_sim::SyncRecorder::Signal(
            static_cast<int>(token), static_cast<int>(srcPipe), static_cast<int>(dstPipe));
    }
}

template <typename SrcPipe, typename DstPipe, typename Token>
inline void wait_flag(SrcPipe srcPipe, DstPipe dstPipe, Token token)
{
    if (!::pto::mocker::a5::host::InVfScope()) {
        ::pto::perf_sim::SyncRecorder::Wait(
            static_cast<int>(token), static_cast<int>(srcPipe), static_cast<int>(dstPipe));
        ::pto::mocker::FlushPendingTail(::pto::mocker::evaluator::PipeKey::VECTOR);
    }
}

template <typename SrcPipe, typename Message>
inline void ffts_cross_core_sync(SrcPipe srcPipe, Message message)
{
    if (!::pto::mocker::a5::host::InVfScope()) {
        const int flagId = (static_cast<int>(message) >> 8) & 0xf;
        ::pto::perf_sim::SyncRecorder::Signal(flagId, static_cast<int>(srcPipe), -1, true);
    }
}

template <typename FlagId>
inline void wait_flag_dev(FlagId flagId)
{
    if (!::pto::mocker::a5::host::InVfScope()) {
        ::pto::perf_sim::SyncRecorder::Wait(static_cast<int>(flagId), -1, -1, true);
    }
}

template <typename SrcPipe, typename FlagId>
inline void wait_flag_dev(SrcPipe srcPipe, FlagId flagId)
{
    if (!::pto::mocker::a5::host::InVfScope()) {
        ::pto::perf_sim::SyncRecorder::Wait(static_cast<int>(flagId), static_cast<int>(srcPipe), -1, true);
    }
}

template <typename SrcPipe, typename FlagId>
inline void set_intra_block(SrcPipe srcPipe, FlagId flagId)
{
    if (!::pto::mocker::a5::host::InVfScope()) {
        ::pto::perf_sim::SyncRecorder::Signal(static_cast<int>(flagId), static_cast<int>(srcPipe), -1, true);
    }
}

template <typename SrcPipe, typename FlagId>
inline void wait_intra_block(SrcPipe srcPipe, FlagId flagId)
{
    if (!::pto::mocker::a5::host::InVfScope()) {
        ::pto::perf_sim::SyncRecorder::Wait(static_cast<int>(flagId), static_cast<int>(srcPipe), -1, true);
    }
}

template <typename BarrierType>
inline void dsb(BarrierType)
{
    if (::pto::mocker::a5::host::InVfScope()) {
        return;
    }
    ::pto::mocker::a5::sync::RecordPerfBarrier(PIPE_ALL);
}
