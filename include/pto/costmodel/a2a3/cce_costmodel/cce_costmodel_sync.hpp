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

#include <pto/costmodel/a2a3/cce_costmodel/cce_costmodel_core.hpp>
#include <pto/costmodel/perf_sim/recorder.hpp>

inline void dsb(auto barrierType)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("dsb", cycles, barrierType);
}
inline void ffts_cross_core_sync(auto srcPipe, auto msg)
{
    uint16_t flag_id = (msg >> 8) & 0xf;
    ::pto::perf_sim::SyncRecorder::Signal(static_cast<int>(flag_id), srcPipe, -1, true);
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("ffts_cross_core_sync", cycles, srcPipe, msg);
}
inline void pto_costmodel_pipe_barrier(auto pipe)
{
    ::pto::perf_sim::SyncRecorder::Barrier(pipe);
    FlushTailsForPipe(pipe);
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("pipe_barrier", cycles, pipe);
}
inline void set_atomic_add()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_atomic_add", cycles);
}
inline void set_atomic_bf16()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_atomic_bf16", cycles);
}
inline void set_atomic_f16()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_atomic_f16", cycles);
}
inline void set_atomic_f32()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_atomic_f32", cycles);
}
inline void set_atomic_none()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_atomic_none", cycles);
}
inline void set_atomic_s16()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_atomic_s16", cycles);
}
inline void set_atomic_s32()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_atomic_s32", cycles);
}
inline void set_atomic_s8()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_atomic_s8", cycles);
}
inline void set_cmpmask(auto cmpMaskPtr)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_cmpmask", cycles, cmpMaskPtr);
}
inline void set_ctrl(auto ctrl)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_ctrl", cycles, ctrl);
}
inline void set_deqscale(auto scale)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_deqscale", cycles, scale);
}
inline void set_ffts_base_addr(auto fftsAddr)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_ffts_base_addr", cycles, fftsAddr);
}
inline void set_flag(auto srcPipe, auto dstPipe, auto token)
{
    ::pto::perf_sim::SyncRecorder::Signal(static_cast<int>(token), srcPipe, dstPipe);
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_flag", cycles, srcPipe, dstPipe, token);
}
inline void set_fmatrix(auto regFmatrix)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_fmatrix", cycles, regFmatrix);
}
inline void set_fmatrix_b(auto regFmatrix)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_fmatrix_b", cycles, regFmatrix);
}
inline void set_fpc(auto deqTensorAddr)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_fpc", cycles, deqTensorAddr);
}
inline void set_mask_count()
{
    ::pto::mocker::SetVectorCountMode(true);
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_mask_count", cycles);
}
inline void set_mask_norm()
{
    ::pto::mocker::SetVectorCountMode(false);
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_mask_norm", cycles);
}
inline void set_mov_pad_val(auto value)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_mov_pad_val", cycles, value);
}
inline void set_nd_para(auto ndParaSPR)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_nd_para", cycles, ndParaSPR);
}
inline void set_padding(auto paddingValue)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_padding", cycles, paddingValue);
}
inline void set_quant_pre(auto preQuantScalar)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_quant_pre", cycles, preQuantScalar);
}
inline void set_va_reg_sb(auto vaReg, auto addrArray)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_va_reg_sb", cycles, vaReg, addrArray);
}
inline void set_vector_mask(auto mask0, auto mask1)
{
    // A full-mask restore (SetFullVecMaskByDType -> -1,-1) signals norm mode; any
    // partial/count mask (SetVectorCount -> 0,n; SetContinuousMask for cols<epr ->
    // 0,(1<<n)-1) signals count-mode dispatch. Signed -1 normalizes to all-ones
    // under uint64 conversion, so this matches regardless of the literal's type.
    const bool full_mask = (static_cast<uint64_t>(mask0) == ~0ULL) && (static_cast<uint64_t>(mask1) == ~0ULL);
    ::pto::mocker::SetVectorCountMode(!full_mask);
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_vector_mask", cycles, mask0, mask1);
}
inline void wait_flag(auto srcPipe, auto dstPipe, auto token)
{
    ::pto::perf_sim::SyncRecorder::Wait(static_cast<int>(token), srcPipe, dstPipe);
    FlushTailsForPipe(srcPipe);
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("wait_flag", cycles, srcPipe, dstPipe, token);
}
inline void wait_flag_dev(auto flagId)
{
    ::pto::perf_sim::SyncRecorder::Wait(static_cast<int>(flagId), -1, -1, true);
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("wait_flag_dev", cycles, flagId);
}

// Supply costmodel SyncAll implementations before pto_instr.hpp defines the
// public SYNCALL wrappers. A2/A3 costmodel should stay on the mock path and must
// not enter syncall_soft.hpp, which depends on device-only intrinsics.
#ifndef PTO_NPU_A2A3_SYNCALL_HPP
#define PTO_NPU_A2A3_SYNCALL_HPP

namespace pto {

inline uint16_t CostmodelFftsMessage(uint16_t eventId)
{
    return static_cast<uint16_t>(1U + ((eventId & 0xfU) << 8U));
}

template <SyncCoreType CoreType = SyncCoreType::AIVOnly>
inline void SYNCALL_IMPL()
{
    ::pto_costmodel_pipe_barrier(PIPE_ALL);
    if constexpr (CoreType == SyncCoreType::AIVOnly) {
        ::ffts_cross_core_sync(PIPE_MTE3, CostmodelFftsMessage(SYNC_AIV_ONLY_ALL));
        ::wait_flag_dev(SYNC_AIV_ONLY_ALL);
    } else if constexpr (CoreType == SyncCoreType::AICOnly) {
        ::ffts_cross_core_sync(PIPE_FIX, CostmodelFftsMessage(SYNC_AIC_FLAG));
        ::wait_flag_dev(SYNC_AIC_FLAG);
    } else {
        ::ffts_cross_core_sync(PIPE_MTE3, CostmodelFftsMessage(SYNC_AIV_FLAG));
        ::wait_flag_dev(SYNC_AIC_AIV_FLAG);
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

#endif // PTO_NPU_A2A3_SYNCALL_HPP
