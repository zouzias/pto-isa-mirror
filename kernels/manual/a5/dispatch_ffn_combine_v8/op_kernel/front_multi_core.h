/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_FFN_COMBINE_V8_FRONT_MULTI_CORE_H
#define DISPATCH_FFN_COMBINE_V8_FRONT_MULTI_CORE_H

#include "front_reorder.h"

namespace dispatch_ffn_combine_v8 {

template <typename InputElement>
class FrontReorderMultiCore : public FrontReorderCommonState {
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
        sortNeedCoreNum_ = front.sortNeedCoreNum;
        sortLoopMaxElement_ = front.sortLoopMaxElement;
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

    struct SortMergeState {
        uint32_t srcWsIndex = 0;
        uint32_t listNum = 0;
        uint32_t perListElements = 0;
        uint32_t lastListElements = 0;
    };

    AICORE inline uint64_t MergeOutPackedBytes(uint32_t elemNum) const
    {
        return AlignBytes<float>(static_cast<uint64_t>(FrontPtoGetSortLen<float>(elemNum)) * sizeof(float));
    }

    AICORE inline uint64_t MergeOutIntBytes(uint32_t elemNum) const
    {
        return AlignBytes<int32_t>(static_cast<uint64_t>(elemNum) * sizeof(int32_t));
    }

    AICORE inline uint64_t MergeOutInputUb(uint32_t slot, uint32_t perListElems) const
    {
        return static_cast<uint64_t>(slot) * MergeOutPackedBytes(perListElems);
    }

    AICORE inline uint64_t MergeOutMergedUb(uint32_t activeListNum, uint32_t perListElems) const
    {
        return static_cast<uint64_t>(activeListNum) * MergeOutPackedBytes(perListElems);
    }

    AICORE inline uint64_t MergeOutTmpUb(uint32_t activeListNum, uint32_t perListElems) const
    {
        return MergeOutMergedUb(activeListNum, perListElems) + MergeOutPackedBytes(activeListNum * perListElems);
    }

    AICORE inline uint64_t MergeOutExpertUb(uint32_t activeListNum, uint32_t perListElems) const
    {
        return MergeOutTmpUb(activeListNum, perListElems) + MergeOutPackedBytes(activeListNum * perListElems);
    }

    AICORE inline uint64_t MergeOutPayloadUb(uint32_t activeListNum, uint32_t perListElems) const
    {
        return MergeOutExpertUb(activeListNum, perListElems) + MergeOutIntBytes(activeListNum * perListElems);
    }

    AICORE inline uint64_t MergeOnlyRequiredUbBytes(uint32_t activeListNum, uint32_t perListElems) const
    {
        return MergeOutTmpUb(activeListNum, perListElems) + MergeOutPackedBytes(activeListNum * perListElems);
    }

    AICORE inline uint64_t SortOutPayloadUb(uint32_t activeListNum, uint32_t perListElems) const
    {
        return MergeOutIntBytes(activeListNum * perListElems);
    }

    AICORE inline uint64_t SortOutScratchUb(uint32_t activeListNum, uint32_t perListElems) const
    {
        return MergeOutTmpUb(activeListNum, perListElems);
    }

    AICORE inline uint64_t SortOutRequiredUbBytes(uint32_t activeListNum, uint32_t perListElems) const
    {
        const uint64_t totalElems = static_cast<uint64_t>(activeListNum) * perListElems;
        const uint64_t mergeBytes = MergeOnlyRequiredUbBytes(activeListNum, perListElems);
        const uint64_t payloadBytes = SortOutPayloadUb(activeListNum, perListElems) + MergeOutIntBytes(totalElems);
        const uint64_t scratchBytes = SortOutScratchUb(activeListNum, perListElems) + MergeOutIntBytes(totalElems);
        uint64_t requiredBytes = mergeBytes > payloadBytes ? mergeBytes : payloadBytes;
        requiredBytes = requiredBytes > scratchBytes ? requiredBytes : scratchBytes;
        return requiredBytes;
    }

    AICORE inline uint64_t FrontSortWorkspaceBytes() const
    {
        const uint64_t sortedIntBytes =
            AlignBytes<int32_t>(static_cast<uint64_t>(alignedRouteElems_) * 2U * sizeof(int32_t));
        const uint64_t packedRunBytes =
            AlignBytes<float>(static_cast<uint64_t>(alignedRouteElems_) * 2U * sizeof(float));
        return sortedIntBytes > packedRunBytes ? sortedIntBytes : packedRunBytes;
    }

    AICORE inline __gm__ float *FrontSortWsPtr(uint32_t srcWsIndex) const
    {
        const auto &front = tilingData_->frontReorderTiling;
        const uint64_t offset = srcWsIndex == 0U ? front.frontSortWs0Offset : front.frontSortWs1Offset;
        return reinterpret_cast<__gm__ float *>(workspaceGM_ + offset);
    }

    AICORE inline uint32_t BuildOneCoreVmsSrcWsIndex(uint32_t listNum) const
    {
        uint32_t srcWsIndex = 0U;
        while (listNum > 1U) {
            listNum = (listNum + kFrontMergeOutMaxFanIn - 1U) / kFrontMergeOutMaxFanIn;
            srcWsIndex = 1U - srcWsIndex;
        }
        return srcWsIndex;
    }

    AICORE inline SortMergeState BuildLocalSortState() const
    {
        SortMergeState state;
        state.srcWsIndex = sortPerCoreLoops_ > 1U ? BuildOneCoreVmsSrcWsIndex(sortPerCoreLoops_) : 0U;
        state.listNum = sortNeedCoreNum_;
        state.perListElements = sortPerCoreElems_;
        state.lastListElements = sortLastCoreElems_;
        return state;
    }

    AICORE inline SortMergeState BuildFinalSortState() const
    {
        SortMergeState state = BuildLocalSortState();
        while (state.listNum > kFrontMergeOutMaxFanIn) {
            const uint32_t currentStageNeedCoreNum =
                (state.listNum + kFrontMergeOutMaxFanIn - 1U) / kFrontMergeOutMaxFanIn;
            const uint32_t remainListNum = state.listNum - (currentStageNeedCoreNum - 1U) * kFrontMergeOutMaxFanIn;
            state.lastListElements = state.perListElements * (remainListNum - 1U) + state.lastListElements;
            state.perListElements *= kFrontMergeOutMaxFanIn;
            state.listNum = currentStageNeedCoreNum;
            state.srcWsIndex = 1U - state.srcWsIndex;
        }
        return state;
    }

    /*
        srcPtr             输入 packed workspace
        dstPtr             输出 packed workspace
        inputBaseElem      输入 run 组在全局 record 空间里的起始 record 下标
        outputBaseElem     输出位置的起始 record 下标
        listNum            要归并几路，1~4
        perListElements    普通每一路有多少 record
        lastListElements   最后一路有多少 record
        srcWorkspaceBytes  输入 workspace 字节数
        dstWorkspaceBytes  输出 workspace 字节数
        extractToFinal     是否最终抽取为 int32 结果 */
    AICORE inline bool MergePackedListGroupImpl(__gm__ float *srcPtr, __gm__ float *dstPtr, uint32_t inputBaseElem,
                                                uint32_t outputBaseElem, uint32_t listNum, uint32_t perListElements,
                                                uint32_t lastListElements, uint64_t srcWorkspaceBytes,
                                                uint64_t dstWorkspaceBytes, bool extractToFinal) const
    {
        if (listNum == 0U || listNum > kFrontMergeOutMaxFanIn || perListElements == 0U || lastListElements == 0U) {
            return false;
        }

        uint32_t listRemainElements[kFrontMergeOutMaxFanIn] = {0U, 0U, 0U, 0U};
        uint32_t offsets[kFrontMergeOutMaxFanIn] = {0U, 0U, 0U, 0U};
        uint32_t allRemainElements = 0U;
        for (uint32_t listIdx = 0U; listIdx < listNum; ++listIdx) {
            const uint32_t elems = listIdx == listNum - 1U ? lastListElements : perListElements;
            const uint32_t elemBase = inputBaseElem + listIdx * perListElements;
            const uint32_t packedOffset = FrontPtoGetSortOffset<float>(elemBase);
            const uint32_t packedLen = FrontPtoGetSortLen<float>(elems);
            if (static_cast<uint64_t>(packedOffset + packedLen) * sizeof(float) > srcWorkspaceBytes) {
                return false;
            }
            listRemainElements[listIdx] = elems;
            offsets[listIdx] = packedOffset;
            allRemainElements += elems;
        }
        if (extractToFinal) {
            if (allRemainElements != routeElems_) {
                return false;
            }
        }
        //中间模式写 packed workspace，所以要转成 packed float offset。最终模式写 int32 输出表
        uint32_t outOffset = extractToFinal ? outputBaseElem : FrontPtoGetSortOffset<float>(outputBaseElem);
        while (allRemainElements > 0U) {
            uint32_t activeListNum = 0U;
            for (uint32_t listIdx = 0U; listIdx < listNum; ++listIdx) {
                activeListNum += listRemainElements[listIdx] > 0U ? 1U : 0U;
            }
            if (activeListNum == 0U) {
                return false;
            }

            uint32_t perListElems = sortOutLoopMaxElems_; //当前是 2040。意思是每一路一次最多搬 2040 条 packed record 进 UB。
            const uint64_t requiredUbBytes = extractToFinal ? SortOutRequiredUbBytes(activeListNum, perListElems) :
                                                              MergeOnlyRequiredUbBytes(activeListNum, perListElems);
            if (perListElems == 0U || requiredUbBytes > AtlasA5::UB_SIZE) {
                return false;
            }

            //准备本轮临时数组
            uint16_t elementCountList[kFrontMergeOutMaxFanIn] = {0U, 0U, 0U, 0U};
            uint32_t listSortedNums[kFrontMergeOutMaxFanIn] = {0U, 0U, 0U, 0U};
            uint64_t tmpUbInputs[kFrontMergeOutMaxFanIn] = {0U, 0U, 0U, 0U};
            uint32_t activeToList[kFrontMergeOutMaxFanIn] = {0U, 0U, 0U, 0U};
            uint32_t loadedElems = 0U;
            uint32_t activeIdx = 0U;
            pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_MTE2>();
            for (uint32_t listIdx = 0U; listIdx < listNum; ++listIdx) {
                if (listRemainElements[listIdx] == 0U) {
                    continue;
                }
                const uint32_t curElems =
                    listRemainElements[listIdx] > perListElems ? perListElems : listRemainElements[listIdx];
                const uint32_t packedLen = FrontPtoGetSortLen<float>(curElems);
                const uint64_t inputUb = MergeOutInputUb(activeIdx, perListElems);
                PtoLoadVector<float>(inputUb, srcPtr + offsets[listIdx], packedLen);
                tmpUbInputs[activeIdx] = inputUb;
                elementCountList[activeIdx] = static_cast<uint16_t>(curElems);
                activeToList[activeIdx] = listIdx;
                loadedElems += curElems;
                ++activeIdx;
            }
            if (activeIdx != activeListNum || loadedElems == 0U) {
                return false;
            }

            const uint64_t mergedUb = MergeOutMergedUb(activeListNum, perListElems);
            pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_V>();
            if (activeListNum == 1U) { //1路不需要排序，直接拷贝到输出
                PtoMoveUb<float>(mergedUb, tmpUbInputs[0], FrontPtoGetSortLen<float>(elementCountList[0]));
                listSortedNums[0] = elementCountList[0];
                if (extractToFinal) {
                    pipe_barrier(PIPE_ALL);
                }
            } else { //多路归并排序
                FrontMergePackedSortRecordsChecked(mergedUb, MergeOutTmpUb(activeListNum, perListElems), tmpUbInputs[0],
                                               tmpUbInputs[1], activeListNum >= 3U ? tmpUbInputs[2] : 0U,
                                               activeListNum >= 4U ? tmpUbInputs[3] : 0U, elementCountList,
                                               activeListNum, listSortedNums);
            }

            uint32_t curLoopSortedNum = 0U;
            for (uint32_t idx = 0U; idx < activeListNum; ++idx) {
                uint32_t sortedNum = listSortedNums[idx];
                if (extractToFinal) {
                    if (sortedNum > elementCountList[idx]) {
                        sortedNum = elementCountList[idx];
                    }
                }
                const uint32_t listIdx = activeToList[idx];
                listRemainElements[listIdx] -= sortedNum;
                offsets[listIdx] += FrontPtoGetSortOffset<float>(sortedNum);
                curLoopSortedNum += sortedNum;
            }
            if (curLoopSortedNum == 0U || curLoopSortedNum > loadedElems ||
                curLoopSortedNum > activeListNum * perListElems) {
                return false;
            }
            allRemainElements -= curLoopSortedNum; //更新总剩余

            if (extractToFinal) {
                FrontExtractPackedSortResult(0U, SortOutPayloadUb(activeListNum, perListElems),
                                             SortOutScratchUb(activeListNum, perListElems), mergedUb,
                                             curLoopSortedNum);
                pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
                PtoStoreVector<int32_t>(frontExpandedExpertPtr_ + outOffset, 0U, curLoopSortedNum);
                PtoStoreVector<int32_t>(frontExpandDstToSrcPtr_ + outOffset,
                                        SortOutPayloadUb(activeListNum, perListElems), curLoopSortedNum);
                outOffset += curLoopSortedNum;
            } else {
                const uint32_t outLen = FrontPtoGetSortLen<float>(curLoopSortedNum);
                if (static_cast<uint64_t>(outOffset + outLen) * sizeof(float) > dstWorkspaceBytes) {
                    return false;
                }
                pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
                PtoStoreVector<float>(dstPtr + outOffset, mergedUb, outLen);
                pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
                outOffset += outLen;
            }
        }
        return true;
    }

    AICORE inline void BuildOneCoreVmsProcess() const
    {
        uint32_t runStart = 0U;
        uint32_t runElems = 0U;
        GetVbsSortRunRange(runStart, runElems);
        if (runElems == 0U) {
            return;
        }

        const bool isLastSortCore = coreIdx_ == sortNeedCoreNum_ - 1U;
        uint32_t listNum = isLastSortCore ? sortLastCoreLoops_ : sortPerCoreLoops_;
        uint32_t perListElements = isLastSortCore ? sortLastCorePerLoopElems_ : sortPerCorePerLoopElems_;
        uint32_t lastListElements = isLastSortCore ? sortLastCoreLastLoopElems_ : sortPerCoreLastLoopElems_;
        if (listNum <= 1U || perListElements == 0U || lastListElements == 0U) {
            return;
        }

        const uint64_t workspaceBytes = FrontSortWorkspaceBytes();
        uint32_t srcWsIndex = 0U;
        while (listNum > 1U) {
            const uint32_t loops = (listNum + kFrontMergeOutMaxFanIn - 1U) / kFrontMergeOutMaxFanIn; //kFrontMergeOutMaxFanIn=4路归并
            const uint32_t remainListNum = listNum - (loops - 1U) * kFrontMergeOutMaxFanIn;
            __gm__ float *srcPtr = FrontSortWsPtr(srcWsIndex);
            __gm__ float *dstPtr = FrontSortWsPtr(1U - srcWsIndex);
            for (uint32_t loop = 0U; loop < loops; ++loop) {
                const uint32_t groupListNum = loop == loops - 1U ? remainListNum : kFrontMergeOutMaxFanIn;
                const uint32_t groupLastListElements = loop == loops - 1U ? lastListElements : perListElements;
                const uint32_t baseElem = runStart + loop * kFrontMergeOutMaxFanIn * perListElements;
                if (!MergePackedListGroupImpl(srcPtr, dstPtr, baseElem, baseElem, groupListNum, perListElements,
                                              groupLastListElements, workspaceBytes, workspaceBytes, false)) {
                    return;
                }
            }
            lastListElements = perListElements * (remainListNum - 1U) + lastListElements;
            perListElements *= kFrontMergeOutMaxFanIn;
            listNum = loops; //继续下一轮的4路归并
            srcWsIndex = 1U - srcWsIndex;
        }
    }

    AICORE inline void BuildVmsProcess() const
    {
        if (sortNeedCoreNum_ <= kFrontMergeOutMaxFanIn || sortPerCoreElems_ == 0U || sortLastCoreElems_ == 0U ||
            routeElems_ == 0U) {
            return;
        }

        SortMergeState state = BuildLocalSortState();
        const uint64_t workspaceBytes = FrontSortWorkspaceBytes();
        while (state.listNum > kFrontMergeOutMaxFanIn) {
            const uint32_t currentStageNeedCoreNum =
                (state.listNum + kFrontMergeOutMaxFanIn - 1U) / kFrontMergeOutMaxFanIn;
            const uint32_t remainListNum = state.listNum - (currentStageNeedCoreNum - 1U) * kFrontMergeOutMaxFanIn;  //4路归并
            __gm__ float *srcPtr = FrontSortWsPtr(state.srcWsIndex);
            __gm__ float *dstPtr = FrontSortWsPtr(1U - state.srcWsIndex);
            if (coreIdx_ < currentStageNeedCoreNum) {
                const uint32_t groupListNum =
                    coreIdx_ == currentStageNeedCoreNum - 1U ? remainListNum : kFrontMergeOutMaxFanIn;
                const uint32_t groupLastListElements =
                    coreIdx_ == currentStageNeedCoreNum - 1U ? state.lastListElements : state.perListElements;
                const uint32_t baseElem = coreIdx_ * kFrontMergeOutMaxFanIn * state.perListElements;
                (void)MergePackedListGroupImpl(srcPtr, dstPtr, baseElem, baseElem, groupListNum, state.perListElements,
                                               groupLastListElements, workspaceBytes, workspaceBytes, false);
            }
            state.lastListElements = state.perListElements * (remainListNum - 1U) + state.lastListElements;
            state.perListElements *= kFrontMergeOutMaxFanIn;
            state.listNum = currentStageNeedCoreNum;  //继续下一轮
            state.srcWsIndex = 1U - state.srcWsIndex;
            pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
        }
    }

    AICORE inline void GetVbsSortRunRange(uint32_t &runStart, uint32_t &runElems) const
    {
        runStart = coreIdx_ * sortPerCoreElems_;
        runElems = 0U;
        if (coreIdx_ >= sortNeedCoreNum_ || runStart >= routeElems_) {
            return;
        }
        runElems = (coreIdx_ == sortNeedCoreNum_ - 1U) ? sortLastCoreElems_ : sortPerCoreElems_;
        if (runStart + runElems > routeElems_) {
            runElems = routeElems_ - runStart;
        }
    }

    AICORE inline bool VbsSortRunFitsUb(uint32_t runAlignedElems) const
    {
        const uint64_t runSortBytes = AlignBytes<int32_t>(static_cast<uint64_t>(runAlignedElems) * sizeof(int32_t));
        const uint64_t runSortPackedBytes =
            AlignBytes<float>(static_cast<uint64_t>(runAlignedElems) * 2U * sizeof(float));
        const uint64_t requiredBytes = runSortBytes * 3U + runSortPackedBytes * 2U;
        return runAlignedElems <= kFrontSortMaxElems && runAlignedElems <= sortLoopMaxElement_ &&
               requiredBytes <= AtlasA5::UB_SIZE;
    }

    AICORE inline void BuildVbsSortRuns() const
    {
        uint32_t runStart = 0U;
        uint32_t runElems = 0U;
        GetVbsSortRunRange(runStart, runElems); //多核切分，获取本核负责的总行数
        if (runElems == 0U) {
            return;
        }

        const bool isLastSortCore = coreIdx_ == sortNeedCoreNum_ - 1U;
        const uint32_t coreLoops = isLastSortCore ? sortLastCoreLoops_ : sortPerCoreLoops_;  //本核分到的总行数，需要分成几轮来处理，每轮处理6144条
        const uint32_t corePerLoopElems = isLastSortCore ? sortLastCorePerLoopElems_ : sortPerCorePerLoopElems_;
        const uint32_t coreLastLoopElems = isLastSortCore ? sortLastCoreLastLoopElems_ : sortPerCoreLastLoopElems_;
        if (coreLoops == 0U || corePerLoopElems == 0U || coreLastLoopElems == 0U) {
            return;
        }

        const auto &front = tilingData_->frontReorderTiling;
        __gm__ float *packedPtr = reinterpret_cast<__gm__ float *>(workspaceGM_ + front.frontSortWs0Offset);
        for (uint32_t loop = 0U; loop < coreLoops; ++loop) {  //按照多轮6144遍历
            const uint32_t loopStart = runStart + loop * corePerLoopElems;
            if (loopStart >= runStart + runElems || loopStart >= routeElems_) {
                return;
            }
            uint32_t loopElems = (loop == coreLoops - 1U) ? coreLastLoopElems : corePerLoopElems;
            if (loopStart + loopElems > runStart + runElems) {
                loopElems = runStart + runElems - loopStart;
            }
            if (loopStart + loopElems > routeElems_) {
                loopElems = routeElems_ - loopStart;
            }
            if (loopElems == 0U) {
                return;
            }

            const uint32_t loopAlignedElems = FrontAlignSortBlock(loopElems);  //32个元素对齐，TSORT32限制
            if (!VbsSortRunFitsUb(loopAlignedElems)) {
                return;
            }

            //计算各种片上变量的UBsize 偏移
            const uint64_t runSortBytes =
                AlignBytes<int32_t>(static_cast<uint64_t>(loopAlignedElems) * sizeof(int32_t));
            const uint64_t runPackedSortBytes =
                AlignBytes<float>(static_cast<uint64_t>(loopAlignedElems) * 2U * sizeof(float));
            const uint64_t runExpertUb = 0U;
            const uint64_t runPayloadUb = runSortBytes;
            const uint64_t runSortKeyUb = runSortBytes * 2U;
            const uint64_t runPackedSortUb = runSortBytes * 3U;
            const uint64_t runMergeTmpUb = runPackedSortUb + runPackedSortBytes;

            //批量加载expertId到UB，准备排序
            PtoLoadVector<int32_t>(runExpertUb, expertIdPtr_ + loopStart, loopElems);
            pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
            /*  route:     0  1  2  3  4
                expertId: 7  2  7  1  2
                payload:  0  1  2  3  4  -- //初始化 runPayloadUb，用于保存route的原始索引在排序中不丢失
                按 expertId 排序后变成：
                expertId: 1  2  2  7  7
                payload:  3  1  4  0  2
            */
            PtoFillArithProgressionInt32(runPayloadUb, static_cast<int32_t>(loopStart), 1, loopElems);
            pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();

            FrontSortInt32ToPackedUb(runExpertUb, runPayloadUb, runPackedSortUb, runMergeTmpUb, runSortKeyUb,
                                     loopElems, loopAlignedElems);
            pipe_barrier(PIPE_ALL);

            const uint32_t packedOffset = FrontPtoGetSortOffset<float>(loopStart);
            const uint32_t packedLen = FrontPtoGetSortLen<float>(loopElems);
            if (static_cast<uint64_t>(packedOffset + packedLen) * sizeof(float) > FrontSortWorkspaceBytes()) {
                return;
            }
            pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
            // 本 run 排好序的 packed records 从 UB 写回 GM
            PtoStoreVector<float>(packedPtr + packedOffset, runPackedSortUb, packedLen);
            pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
        }
    }

    AICORE inline void BuildSortOutProcess() const
    {
        if (coreIdx_ != 0U || sortNeedCoreNum_ == 0U || sortPerCoreElems_ == 0U || routeElems_ == 0U) {
            return;
        }

        const SortMergeState finalState = BuildFinalSortState();
        const uint32_t finalListNum = finalState.listNum;
        const uint32_t finalPerListElements = finalState.perListElements;
        const uint32_t finalLastListElements = finalState.lastListElements;
        if (finalListNum == 0U || finalListNum > kFrontMergeOutMaxFanIn || finalPerListElements == 0U ||
            finalLastListElements == 0U) {
            return;
        }
        const uint64_t workspaceBytes = FrontSortWorkspaceBytes();
        __gm__ float *srcPackedPtr = FrontSortWsPtr(finalState.srcWsIndex);
        (void)MergePackedListGroupImpl(srcPackedPtr, srcPackedPtr, 0U, 0U, finalListNum, finalPerListElements,
                                       finalLastListElements, workspaceBytes, 0U, true);
    }

};

} // namespace dispatch_ffn_combine_v8

#endif // DISPATCH_FFN_COMBINE_V8_FRONT_MULTI_CORE_H
