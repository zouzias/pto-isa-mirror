/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_FFN_COMBINE_H
#define DISPATCH_FFN_COMBINE_H

#include "kernel_operator.h"

#include "device_debug.h"
#include "dispatch_ffn_combine_tiling.h"
#if defined(__DAV_VEC__)
#include "combine.h"
#include "dispatch.h"
#define DISPATCH_FFN_COMBINE_V8_FRONT_REORDER_PROCESS_IMPLEMENTATION
#include "front_reorder.h"
#undef DISPATCH_FFN_COMBINE_V8_FRONT_REORDER_PROCESS_IMPLEMENTATION
#endif
#if defined(__DAV_CUBE__)
#include "gmm1.h"
#include "gmm2.h"
#endif
#include "kernel_launch.hpp"
#include "profile_debug_config.h"
#if defined(__DAV_VEC__)
#include "swiglu.h"
#include "unpermute.h"
#endif

#define TemplateMMA2AClass typename AType_, typename BType_, typename CType_, bool TB_, bool Nz_
#define TemplateMMA2ACFunc AType_, BType_, CType_, TB_, Nz_

template <TemplateMMA2AClass>
class DispatchFFNCombine {
public:
    __aicore__ inline void Init(GM_ADDR xGM, GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR expertIdGM, GM_ADDR scale1GM,
                                GM_ADDR scale2GM, GM_ADDR probs, GM_ADDR xActiveMaskGM, GM_ADDR outGM,
                                GM_ADDR expertTokenNums, GM_ADDR workspaceGM,
                                const __gm__ DispatchFFNCombineTilingData *tilingData,
                                __gm__ uint64_t *profileEntry = nullptr);
    __aicore__ inline void Process();

private:
    friend struct dispatch_ffn_combine_v8::DeviceDebug;

    __aicore__ inline void RecordProfile(size_t index) const
    {
#if DISPATCH_FFN_COMBINE_V8_ENABLE_INNER_PROFILE
        if (profileEntry_ != nullptr && index < kDispatchFFNCombineProfileEntryU64Count) {
            profileEntry_[index] = get_sys_cnt();
        }
#else
        (void)index;
#endif
    }

    __aicore__ inline void RecordStageStart(size_t stage) const
    {
        RecordProfile(kDispatchFFNCombineProfileStageBase + stage * 2U);
    }

    __aicore__ inline void RecordStageEnd(size_t stage) const
    {
        RecordProfile(kDispatchFFNCombineProfileStageBase + stage * 2U + 1U);
    }

    GM_ADDR xGM_ = nullptr;
    GM_ADDR weight1GM_ = nullptr;
    GM_ADDR weight2GM_ = nullptr;
    GM_ADDR scale1GM_ = nullptr;
    GM_ADDR scale2GM_ = nullptr;
    GM_ADDR expertIdGM_ = nullptr;
    GM_ADDR expertTokenNumsGM_ = nullptr;
    GM_ADDR workspaceGM_ = nullptr;
    GM_ADDR probsGM_ = nullptr;
    GM_ADDR outGM_ = nullptr;
    const __gm__ DispatchFFNCombineTilingData *tilingData_ = nullptr;
    volatile __gm__ uint64_t *profileEntry_ = nullptr;
};

template <TemplateMMA2AClass>
__aicore__ inline void DispatchFFNCombine<TemplateMMA2ACFunc>::Init(
    GM_ADDR xGM, GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR expertIdGM, GM_ADDR scale1GM, GM_ADDR scale2GM,
    GM_ADDR probs, GM_ADDR xActiveMaskGM, GM_ADDR outGM, GM_ADDR expertTokenNums, GM_ADDR workspaceGM,
    const __gm__ DispatchFFNCombineTilingData *tilingData, __gm__ uint64_t *profileEntry)
{
    (void)xActiveMaskGM;
    xGM_ = xGM;
    weight1GM_ = weight1GM;
    weight2GM_ = weight2GM;
    scale1GM_ = scale1GM;
    scale2GM_ = scale2GM;
    expertIdGM_ = expertIdGM;
    expertTokenNumsGM_ = expertTokenNums;
    workspaceGM_ = workspaceGM;
    probsGM_ = probs;
    outGM_ = outGM;
    tilingData_ = tilingData;
    profileEntry_ = profileEntry;
}

#define DISPATCH_FFN_COMBINE_V8_DEVICE_DEBUG_IMPLEMENTATION
#include "device_debug.h"
#undef DISPATCH_FFN_COMBINE_V8_DEVICE_DEBUG_IMPLEMENTATION

template <TemplateMMA2AClass>
__aicore__ inline void DispatchFFNCombine<TemplateMMA2ACFunc>::Process()
{
    using OutputElement = half;
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DEVICE_DEBUG
    dispatch_ffn_combine_v8::DeviceDebug::DispatchFfnLogProcessDebug(*this);
#endif
    RecordStageStart(DISPATCH_FFN_COMBINE_PROFILE_STAGE_FRONT);
#if defined(__DAV_VEC__)
    const bool useFront = dispatch_ffn_combine_v8::FrontReorderProcess<CType_>(
        xGM_, expertIdGM_, expertTokenNumsGM_, workspaceGM_, tilingData_, profileEntry_);
#else
    const bool useFront = false;
#endif
    RecordStageEnd(DISPATCH_FFN_COMBINE_PROFILE_STAGE_FRONT);

#if defined(__DAV_VEC__)
    if (tilingData_->frontReorderTiling.stageNum >= 9U) {
        RecordStageStart(DISPATCH_FFN_COMBINE_PROFILE_STAGE_DISPATCH);
        if (useFront && tilingData_->dispatchTiling.dispatchGatherMode != 0U) {
            dispatch_ffn_combine_v8::DispatchGather<CType_> dispatchGather;
            dispatchGather.Init(expertTokenNumsGM_, workspaceGM_, tilingData_, profileEntry_);
            dispatchGather.Process();
        }
        RecordStageEnd(DISPATCH_FFN_COMBINE_PROFILE_STAGE_DISPATCH);
    }
#endif
#if defined(__DAV_CUBE__)
    if (tilingData_->frontReorderTiling.stageNum >= 10U) {
        dispatch_ffn_combine_v8::Gmm1<CType_> gmm1;
        gmm1.Init(weight1GM_, scale1GM_, expertTokenNumsGM_, workspaceGM_, tilingData_, profileEntry_);
        RecordStageStart(DISPATCH_FFN_COMBINE_PROFILE_STAGE_GMM1);
        gmm1.Process();
        RecordStageEnd(DISPATCH_FFN_COMBINE_PROFILE_STAGE_GMM1);
    }
#endif
#if defined(__DAV_VEC__)
    if (tilingData_->frontReorderTiling.stageNum >= 11U) {
        dispatch_ffn_combine_v8::Swiglu<CType_> swiglu;
        swiglu.Init(expertTokenNumsGM_, workspaceGM_, tilingData_, profileEntry_);
        RecordStageStart(DISPATCH_FFN_COMBINE_PROFILE_STAGE_SWIGLU);
        swiglu.Process();
        RecordStageEnd(DISPATCH_FFN_COMBINE_PROFILE_STAGE_SWIGLU);
    }
#endif
#if defined(__DAV_CUBE__)
    if (tilingData_->frontReorderTiling.stageNum >= 12U) {
        dispatch_ffn_combine_v8::Gmm2<CType_> gmm2;
        gmm2.Init(weight2GM_, scale2GM_, expertTokenNumsGM_, workspaceGM_, tilingData_, profileEntry_);
        RecordStageStart(DISPATCH_FFN_COMBINE_PROFILE_STAGE_GMM2);
        gmm2.Process();
        RecordStageEnd(DISPATCH_FFN_COMBINE_PROFILE_STAGE_GMM2);
    }
#endif
#if defined(__DAV_VEC__)
    if (tilingData_->frontReorderTiling.stageNum >= 13U) {
        RecordStageStart(DISPATCH_FFN_COMBINE_PROFILE_STAGE_COMBINE);
        dispatch_ffn_combine_v8::Combine<OutputElement> combine;
        combine.Init(workspaceGM_, tilingData_, profileEntry_);
        combine.Process();
        RecordStageEnd(DISPATCH_FFN_COMBINE_PROFILE_STAGE_COMBINE);
    }
    if (tilingData_->frontReorderTiling.stageNum >= 14U) {
        dispatch_ffn_combine_v8::Unpermute<OutputElement> unpermute;
        unpermute.Init(workspaceGM_, probsGM_, outGM_, tilingData_, profileEntry_);
        RecordStageStart(DISPATCH_FFN_COMBINE_PROFILE_STAGE_UNPERMUTE);
        unpermute.Process();
        RecordStageEnd(DISPATCH_FFN_COMBINE_PROFILE_STAGE_UNPERMUTE);
    }
#endif
}

#endif // DISPATCH_FFN_COMBINE_H
