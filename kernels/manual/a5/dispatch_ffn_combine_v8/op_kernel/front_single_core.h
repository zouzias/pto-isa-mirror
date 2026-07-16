/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_FFN_COMBINE_V8_FRONT_SINGLE_CORE_H
#define DISPATCH_FFN_COMBINE_V8_FRONT_SINGLE_CORE_H

#include "front_reorder.h"

namespace dispatch_ffn_combine_v8 {

constexpr uint32_t kFrontSingleCoreSortOwnerCore = 0U;
constexpr uint32_t kFrontSingleCoreSortAlignElems = 32U;

template <typename InputElement>
class FrontReorderSingleCore : public FrontReorderCommonState {
public:
    AICORE inline void Init(GM_ADDR xGM, GM_ADDR expertIdGM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
                            const __gm__ DispatchFFNCombineTilingData *tilingData,
                            volatile __gm__ uint64_t *profileEntry = nullptr)
    {
        xPtr_ = xGM;
        expertIdPtr_ = reinterpret_cast<__gm__ int32_t *>(expertIdGM);
        expertTokenNumsPtr_ = reinterpret_cast<__gm__ int32_t *>(expertTokenNumsGM);
        workspaceGM_ = workspaceGM;
        tilingData_ = tilingData;
        profileEntry_ = profileEntry;

        const auto &info = tilingData_->dispatchFFNCombineInfo;
        problemM_ = info.M;
        problemK_ = info.K;
        topK_ = info.topK;
        expertPerRank_ = info.expertPerRank;
        maxOutputSize_ = info.maxOutputSize;
        rank_ = tilingData_->runtimeInfo.rank;
        rankSize_ = tilingData_->runtimeInfo.rankSize;

        const auto &front = tilingData_->frontReorderTiling;
        stageNum_ = front.stageNum;
        expertNum_ = front.expertNum;
        expertNumAligned_ = front.expertNumAligned;
        routeElems_ = front.routeElems;
        alignedRouteElems_ = front.alignedRouteElems;
        frontCase_ = front.frontCase;
        sortLoopMaxElement_ = front.sortLoopMaxElement;
        sortNeedCoreNum_ = front.sortNeedCoreNum;
        sortPerCoreElems_ = front.sortPerCoreElems;
        sortLastCoreElems_ = front.sortLastCoreElems;
        sortPerCoreLoops_ = front.sortPerCoreLoops;
        sortPerCorePerLoopElems_ = front.sortPerCorePerLoopElems;
        sortPerCoreLastLoopElems_ = front.sortPerCoreLastLoopElems;
        sortLastCoreLoops_ = front.sortLastCoreLoops;
        sortLastCorePerLoopElems_ = front.sortLastCorePerLoopElems;
        sortLastCoreLastLoopElems_ = front.sortLastCoreLastLoopElems;
        sortOutLoopMaxElems_ = front.sortOutLoopMaxElems;

        coreIdx_ = get_block_idx();
        coreNum_ = get_block_num();
        if ASCEND_IS_AIV {
            coreIdx_ = get_block_idx() + get_subblockid() * get_block_num();
            coreNum_ = get_block_num() * get_subblockdim();
        }

        expandedRowIdxPtr_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM_ + front.expandedRowIdxOffset);
        frontExpandedExpertPtr_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM_ + front.frontExpandedExpertOffset);
        frontExpandDstToSrcPtr_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM_ + front.frontExpandDstToSrcOffset);
        localTokenPerExpertPtr_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM_ + front.localTokenPerExpertOffset);
        frontCountScratchPtr_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM_ + front.frontCountScratchOffset);
        cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM_ + front.cumsumMMOffset);
        preSumBeforeRankPtr_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM_ + front.preSumBeforeRankOffset);

        remoteWindow_.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
        peerMemoryLayout_.Init(remoteWindow_);
        offsetAPtr_ = reinterpret_cast<__gm__ int8_t *>(remoteWindow_() + peerMemoryLayout_.offsetA);
        tokenPerExpertPtr_ =
            reinterpret_cast<__gm__ int32_t *>(remoteWindow_() + peerMemoryLayout_.offsetPeerTokenPerExpert);
    }

    friend struct DeviceDebug;

    AICORE inline uint32_t SingleSortTileLength() const
    {
        const uint32_t vbsLastCoreElems = sortLastCorePerLoopElems_ == 0U ? routeElems_ : sortLastCorePerLoopElems_;
        return static_cast<uint32_t>(alignUp(vbsLastCoreElems, static_cast<uint32_t>(sizeof(int32_t))));
    }

    AICORE inline uint32_t SingleSortNum() const
    {
        return static_cast<uint32_t>(alignUp(SingleSortTileLength(), kFrontSingleCoreSortAlignElems));
    }

    AICORE inline void BuildSingleCoreSort() const
    {
        const uint32_t sortNum = SingleSortNum();
        const uint64_t sortIntBytes = AlignBytes<int32_t>(static_cast<uint64_t>(sortNum) * sizeof(int32_t));
        const uint64_t sortPackedBytes = AlignBytes<float>(static_cast<uint64_t>(sortNum) * 2U * sizeof(float));
        const uint64_t sortKeyBytes = AlignBytes<float>(static_cast<uint64_t>(sortNum) * sizeof(float));
        const uint64_t expertUb = 0U;
        const uint64_t payloadUb = sortIntBytes;
        const uint64_t packedUb = payloadUb + sortIntBytes;
        const uint64_t mergeTmpUb = packedUb + sortPackedBytes;
        const uint64_t sortKeyUb = mergeTmpUb + sortPackedBytes;
        const uint64_t actualUbBytes = sortKeyUb + sortKeyBytes;
        constexpr uint32_t kFrontSortBufferCount = 4U;
        constexpr uint32_t kFrontSortPayloadAndKey = 2U;
        const uint64_t requiredUbBytes =
            static_cast<uint64_t>(sortNum) * sizeof(int32_t) * kFrontSortPayloadAndKey * kFrontSortBufferCount;
        if (coreIdx_ != kFrontSingleCoreSortOwnerCore || routeElems_ == 0U || sortNum > kFrontSortMaxElems ||
            actualUbBytes > AtlasA5::UB_SIZE || requiredUbBytes > AtlasA5::UB_SIZE) {
            return;
        }

        const uint32_t totalLength = routeElems_;
        PtoLoadVector<int32_t>(expertUb, expertIdPtr_, totalLength);
        pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
        PtoFillArithProgressionInt32(payloadUb, 0, 1, sortNum);
        pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();

        FrontSortInt32ToPackedUb(expertUb, payloadUb, packedUb, mergeTmpUb, sortKeyUb, totalLength, sortNum);
        pipe_barrier(PIPE_ALL);
        FrontExtractPackedSortResult(expertUb, payloadUb, sortKeyUb, packedUb, totalLength);
        pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
        PtoStoreVector<int32_t>(frontExpandedExpertPtr_, expertUb, totalLength);
        PtoStoreVector<int32_t>(frontExpandDstToSrcPtr_, payloadUb, totalLength);
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
    }

};

} // namespace dispatch_ffn_combine_v8

#endif // DISPATCH_FFN_COMBINE_V8_FRONT_SINGLE_CORE_H
