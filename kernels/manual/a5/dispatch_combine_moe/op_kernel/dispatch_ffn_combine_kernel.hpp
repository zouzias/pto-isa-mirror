/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DISPATCH_FFN_COMBINE_KERNEL_HPP
#define DISPATCH_FFN_COMBINE_KERNEL_HPP

#include "kernel_operator.h"

#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

#include "utils/moe_pto_utils.hpp"
#include "utils/pto_vector_ops.hpp"

#include "utils/block_mmad_preload_async_fixpipe_quant.hpp"
#include "utils/block_epilogue_pertoken_row.hpp"
#include "utils/block_epilogue_pertoken_v2.hpp"
#include "utils/block_epilogue_pertoken_swiglu.hpp"
#include "utils/hccl_window.hpp"
#include "utils/const_args.hpp"
#include "utils/layout3d.hpp"
#include "token_reorder/routing/moe_init_routing_quant_tiling.h"
#include "token_reorder/routing/moe_init_routing_quant.cpp"
#include "token_reorder/routing/moe_init_routing_fullload_dynamic_quant.h"
#include "token_reorder/unpermute/moe_token_unpermute.h"

inline __gm__ struct OpSystemRunCfg g_opSystemRunCfg {
    pto_ext::support::kL2Offset
};

using namespace AscendC;

namespace pto_ext::Gemm::Kernel {
namespace pto_bridge = pto_ext::dispatch_combine_moe::pto_bridge;
namespace kernel_detail {

#define FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__

template <typename T>
FORCE_INLINE_AICORE __gm__ T *GetTensorAddr(uint32_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t *dataAddr = reinterpret_cast<__gm__ uint64_t *>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t *retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T *>(*(retPtr + index));
}

#undef FORCE_INLINE_AICORE

using pto_ext::PtoPipeBarrier;
using pto_ext::PtoSetFlag;
using pto_ext::PtoSyncAll;
using pto_ext::PtoWaitFlag;

using pto_bridge::PtoAddScalarVector;
using pto_bridge::PtoAddVector;
using pto_bridge::PtoFillVector;
using pto_bridge::PtoLoadVector;
using pto_bridge::PtoStoreVector;

} // namespace kernel_detail

constexpr uint16_t SYNCFLAGC2V = 9;
constexpr uint16_t SYNCFLAGV2C = 10;

template <class BlockMmad_, class BlockScheduler_, class ElementGroupList_, class BlockEpilogue1_,
          class BlockEpilogue2_, class BlockEpilogue3_>
class DispatchFFNCombineKernel {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;
    using ElementScale = uint64_t;
    using LayoutScale = typename layout::VectorLayout;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = typename layout::VectorLayout;
    using BlockScheduler = BlockScheduler_;

    using BlockEpilogue1 = BlockEpilogue1_;
    using BlockEpilogue2 = BlockEpilogue2_;
    using BlockEpilogue3 = BlockEpilogue3_;
    using IntPtoGlobal = pto_bridge::PtoGlobalNd<int32_t>;

    using ElementD1 = typename BlockEpilogue1::ElementD;
    using LayoutD1 = typename BlockEpilogue1::LayoutD;
    using ElementD2 = typename BlockEpilogue2::ElementD;
    using LayoutD2 = typename BlockEpilogue2::LayoutD;

    /// Parameters structure
    struct Params {
        // Data members
        PtoShape3D problemShape;
        __gm__ ElementA *ptrA;
        LayoutA layoutA;
        LayoutA layoutA2;
        GM_ADDR ptrB1;
        LayoutB layoutB1;
        GM_ADDR ptrB2;
        LayoutB layoutB2;
        GM_ADDR ptrScale1;
        LayoutScale layoutScale1;
        GM_ADDR ptrScale2;
        LayoutScale layoutScale2;
        __gm__ ElementD2 *ptrOutput;
        LayoutD1 layoutD1;
        LayoutD2 layoutD2;
        GM_ADDR ptrWorkspace;
        GM_ADDR ptrExpertTokenNums;
        int32_t EP;
        int32_t listLen;
        int32_t expertPerRank;
        uint32_t maxOutputSize;
        uint32_t rank;
        uint32_t rankSize;
        int32_t ubMoveNum;
        GM_ADDR remoteWindowContext;
        //--------------
        GM_ADDR expertIdx;
        GM_ADDR moeInitRoutingQuantScale;
        GM_ADDR moeInitRoutingQuantOffset;
        GM_ADDR expandedX;
        GM_ADDR expandedRowIdx;
        GM_ADDR expertTokensCountOrCumsum;
        GM_ADDR expertTokensBeforeCapacity;
        GM_ADDR dynamicQuantScale;
        GM_ADDR probs;
        GM_ADDR ptrXActiveMask;
        int64_t topK;
        uint64_t initRoutingQuantTilingKey;
        uint32_t epilogueCoreNum;
        uint32_t epilogueGranularity;
        optiling::MoeInitRoutingQuantTilingData moeInitRoutingQuantTilingData;
        //--------------

        // Methods
        __forceinline__[host, aicore]
        Params()
        {}

        __forceinline__[host, aicore]
        Params(PtoShape3D problemShape_, uint32_t EP_, uint32_t listLen_, uint32_t expertPerRank_,
               uint32_t maxOutputSize_, uint32_t rank_, uint32_t rankSize_, int32_t ubMoveNum_,
               GM_ADDR remoteWindowContext_, int64_t topK_, uint64_t initRoutingQuantTilingKey_,
               uint32_t epilogueCoreNum_, uint32_t epilogueGranularity_, GM_ADDR ptrA_, LayoutA layoutA_,
               LayoutA layoutA2_, GM_ADDR ptrB1_, LayoutB layoutB1_, GM_ADDR ptrB2_, LayoutB layoutB2_,
               GM_ADDR ptrScale1_, LayoutScale layoutScale1_, GM_ADDR ptrScale2_, LayoutScale layoutScale2_,
               GM_ADDR ptrOutput_, LayoutD2 layoutD1_, LayoutD2 layoutD2_, GM_ADDR expertIdx_,
               GM_ADDR moeInitRoutingQuantScale_, GM_ADDR moeInitRoutingQuantOffset_,
               GM_ADDR expertTokensBeforeCapacity_, GM_ADDR probs_, GM_ADDR ptrWorkspace_, GM_ADDR gmExpertTokenNums_,
               GM_ADDR ptrXActiveMask_, optiling::MoeInitRoutingQuantTilingData moeInitRoutingQuantTilingData_)
            : problemShape(problemShape_),
              EP(EP_),
              listLen(listLen_),
              expertPerRank(expertPerRank_),
              maxOutputSize(maxOutputSize_),
              rank(rank_),
              rankSize(rankSize_),
              ubMoveNum(ubMoveNum_),
              remoteWindowContext(remoteWindowContext_),
              topK(topK_),
              initRoutingQuantTilingKey(initRoutingQuantTilingKey_),
              epilogueCoreNum(epilogueCoreNum_),
              epilogueGranularity(epilogueGranularity_),
              ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)),
              layoutA(layoutA_),
              layoutA2(layoutA2_),
              ptrB1(ptrB1_),
              layoutB1(layoutB1_),
              ptrB2(ptrB2_),
              layoutB2(layoutB2_),
              ptrScale1(ptrScale1_),
              layoutScale1(layoutScale1_),
              ptrScale2(ptrScale2_),
              layoutScale2(layoutScale2_),
              ptrOutput(reinterpret_cast<__gm__ ElementD2 *>(ptrOutput_)),
              layoutD1(layoutD1_),
              layoutD2(layoutD2_),
              expertIdx(expertIdx_),
              moeInitRoutingQuantScale(moeInitRoutingQuantScale_),
              moeInitRoutingQuantOffset(moeInitRoutingQuantOffset_),
              expertTokensBeforeCapacity(expertTokensBeforeCapacity_),
              probs(probs_),
              ptrWorkspace(ptrWorkspace_),
              ptrExpertTokenNums(gmExpertTokenNums_),
              ptrXActiveMask(ptrXActiveMask_),
              moeInitRoutingQuantTilingData(moeInitRoutingQuantTilingData_)
        {}
    };

    // Methods
    __forceinline__ __aicore__
    DispatchFFNCombineKernel(Params const &params)
    {
        if ASCEND_IS_AIC {
            coreIdx = AscendC::GetBlockIdx();
            coreNum = AscendC::GetBlockNum();
        }

        if ASCEND_IS_AIV {
            coreIdx = get_block_idx() + get_subblockid() * get_block_num();
            coreNum = get_block_num() * get_subblockdim();
        }

        initBuffer(params);
    }

    __forceinline__ __aicore__
    ~DispatchFFNCombineKernel()
    {}

    template <int32_t CORE_TYPE = g_coreType>
    __forceinline__ __aicore__ void operator()(Params const &params);

    template <>
    __forceinline__ __aicore__ void operator()<AscendC::AIC>(Params const &params)
    {
        RunGmm1Impl(params);
        RunGmmInterlockImpl(params);
        RunGmm2Impl(params);
    }

    template <>
    __forceinline__ __aicore__ void operator()<AscendC::AIV>(Params const &params)
    {
        RunRoutingImpl(params);
        RunDispatchGatherImpl(params);
        RunSwigluImpl(params);
        RunCombineImpl(params);
        RunRestoreImpl(params);
    }

    __forceinline__ __aicore__ void RunGmm1Impl(Params const &params)
    {
        GMM1(params);
    }

    __forceinline__ __aicore__ void RunGmmInterlockImpl(Params const &)
    {
        AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGV2C);
    }

    __forceinline__ __aicore__ void RunGmm2Impl(Params const &params)
    {
        GMM2(params);
    }

    __forceinline__ __aicore__ void RunRoutingImpl(Params const &params)
    {
        icache_preload(8);
        int64_t localTokenPerExpertOffset =
            peerMemoryLayout.offsetPeerTokenPerExpert + tokenPerExpertLayout(params.rank, 0, 0) * sizeof(int32_t);
        GM_ADDR localTokenPerExpert = remoteWindow() + localTokenPerExpertOffset;
        uint32_t expandedRowIdxOffset = AlignUp(static_cast<uint32_t>(params.problemShape.shape[0]), 256) * params.topK * sizeof(int32_t);

        ApplyXActiveMask(params);
        moe_init_routing_quant<ElementD2>(
            reinterpret_cast<GM_ADDR>(params.ptrA), params.expertIdx, params.moeInitRoutingQuantScale,
            params.moeInitRoutingQuantOffset, remoteWindow() + peerMemoryLayout.offsetA, workspaceInfo.expandedRowIdx,
            localTokenPerExpert, params.expertTokensBeforeCapacity,
            remoteWindow() + peerMemoryLayout.offsetPeerPerTokenScale, params.ptrWorkspace + expandedRowIdxOffset,
            &params.moeInitRoutingQuantTilingData, params.initRoutingQuantTilingKey);

        kernel_detail::PtoSyncAll<true>();
        CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2(params, localTokenPerExpertOffset);

        if (coreIdx == 0) {
            GetCumsumForMMAIV(tokenPerExpertPtr, cumsumMMPtr, params.expertPerRank, params.rank, params.EP);
        }
        kernel_detail::PtoSyncAll<true>();

        __gm__ int32_t *expertTokenNumsPtr = reinterpret_cast<__gm__ int32_t *>(params.ptrExpertTokenNums);
        if (coreIdx == 0) {
            CopyGMToGM(expertTokenNumsPtr, cumsumMMPtr + (params.EP - 1) * params.expertPerRank, params.expertPerRank,
                       params.ubMoveNum);
        }
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0);
    }

    __forceinline__ __aicore__ void RunDispatchGatherImpl(Params const &params)
    {
        uint16_t syncgmm1Idx = 1;
        uint32_t prevGroupSum1 = 0;
        uint32_t dequantSum1 = 0;
        uint32_t dequantSum2 = 0;
        int32_t prevSum = 0;
        if (coreIdx < params.EP) {
            prevSum = gm_load(preSumBeforeRankPtr + coreIdx * params.expertPerRank);
        }

        icache_preload(8);
        kernel_detail::PtoSetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        kernel_detail::PtoSetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        int32_t pingpongIdx = 0;
        for (int32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentM = gm_load(cumsumMMPtr + (params.EP - 1) * params.expertPerRank + groupIdx);
            for (int32_t dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += coreNum) {
                uint32_t rowStart =
                    (dstEpIdx == 0 ? 0 : gm_load(cumsumMMPtr + (dstEpIdx - 1) * params.expertPerRank + groupIdx)) +
                    prevGroupSum1;
                if (rowStart < params.maxOutputSize) {
                    uint32_t rows = gm_load(tokenPerExpertPtr + tokenPerExpertLayout(dstEpIdx, params.rank, groupIdx));
                    if (rowStart + rows > params.maxOutputSize) {
                        rows = params.maxOutputSize - rowStart;
                    }
                    uint32_t rowSrc = prevSum;
                    prevSum += rows;
                    GM_ADDR otherRankPtr = remoteWindow(0, dstEpIdx);
                    __gm__ ElementA *remotePackedRows =
                        reinterpret_cast<__gm__ ElementA *>(otherRankPtr + peerMemoryLayout.offsetA);
                    auto offsetA = PtoCoord2D(rowStart, 0);
                    int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
                    int64_t gmOffsetPeer = rowSrc * (static_cast<uint32_t>(params.problemShape.shape[2]) + UB_ALIGN);
                    int32_t ubMoveNum = 2;
                    CopyGMToGMPerToken(gmAPtr + gmOffsetA, gmPerTokenScale1Ptr + rowStart,
                                       remotePackedRows + gmOffsetPeer, rows, static_cast<uint32_t>(params.problemShape.shape[2]),
                                       ubMoveNum, pingpongIdx);
                }
            }
            kernel_detail::PtoSyncAll<true>();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(syncgmm1Idx / CROSS_CORE_FLAG_MAX_SET_COUNT);
            syncgmm1Idx++;

            prevGroupSum1 += currentM;

            if (groupIdx + 1 <= params.epilogueGranularity) {
                if (dequantSum1 + currentM <= params.maxOutputSize) {
                    dequantSum1 += currentM;
                } else if (dequantSum1 < params.maxOutputSize) {
                    dequantSum1 = params.maxOutputSize;
                }
            }

            if (groupIdx + 1 > params.epilogueGranularity && dequantSum1 < params.maxOutputSize) {
                if (dequantSum1 + dequantSum2 + currentM <= params.maxOutputSize) {
                    dequantSum2 += currentM;
                } else if (dequantSum1 + dequantSum2 < params.maxOutputSize) {
                    dequantSum2 += params.maxOutputSize - dequantSum1 - dequantSum2;
                }
            }
        }
        kernel_detail::PtoWaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        kernel_detail::PtoWaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        stageDequantSum1 = dequantSum1;
        stageDequantSum2 = dequantSum2;
    }

    __forceinline__ __aicore__ void RunSwigluImpl(Params const &params)
    {
        uint32_t n = static_cast<uint32_t>(params.problemShape.shape[1]);
        BlockEpilogue1 blockEpilogue1(resource, n);

        AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGC2V);
        kernel_detail::PtoSyncAll<true>();
        if (stageDequantSum1 > 0) {
            uint32_t rowStartThisCore = 0;
            auto offsetC = PtoCoord2D(0U, 0);
            PtoShape2D shapeC(stageDequantSum1, static_cast<uint32_t>(params.problemShape.shape[1]));
            LayoutC layoutC{stageDequantSum1, static_cast<uint32_t>(params.problemShape.shape[1])};
            int64_t gmOffsetC = layoutC.GetOffset(offsetC);
            int64_t gmOffsetD = params.layoutD1.GetOffset(offsetC);
            blockEpilogue1(gmCPtr + gmOffsetC, shapeC, gmPerTokenScale1Ptr + rowStartThisCore,
                           gmPermutedTokenPtr + gmOffsetD, gmPerTokenScale2Ptr + rowStartThisCore,
                           params.epilogueCoreNum);
        }
        kernel_detail::PtoSyncAll<true>();
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNCFLAGV2C);

        if ((params.epilogueGranularity < params.expertPerRank && params.epilogueGranularity > 0)) {
            AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGC2V);
            kernel_detail::PtoSyncAll<true>();
            if (stageDequantSum2 > 0) {
                uint32_t rowStartThisCore = stageDequantSum1;
                auto offsetC = PtoCoord2D(rowStartThisCore, 0);
                PtoShape2D shapeC(stageDequantSum2, static_cast<uint32_t>(params.problemShape.shape[1]));
                LayoutC layoutC{stageDequantSum2, static_cast<uint32_t>(params.problemShape.shape[1])};
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                int64_t gmOffsetD = params.layoutD1.GetOffset(offsetC);
                blockEpilogue1(gmCPtr + gmOffsetC, shapeC, gmPerTokenScale1Ptr + rowStartThisCore,
                               gmPermutedTokenPtr + gmOffsetD, gmPerTokenScale2Ptr + rowStartThisCore, coreNum);
            }
            kernel_detail::PtoSyncAll<true>();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNCFLAGV2C);
        }

        blockEpilogue1.Finalize();
    }

    __forceinline__ __aicore__ void RunCombineImpl(Params const &params)
    {
        uint32_t n2 = static_cast<uint32_t>(params.problemShape.shape[2]);
        typename BlockEpilogue2::Params epilogueParams2{
            static_cast<int32_t>(params.EP),
            static_cast<int32_t>(params.expertPerRank),
            reinterpret_cast<__gm__ int32_t *>(remoteWindow() + peerMemoryLayout.offsetPeerTokenPerExpert),
            static_cast<int32_t>(n2),
            static_cast<int32_t>(params.rank),
            remoteWindow,
            static_cast<int32_t>(peerMemoryLayout.offsetPeerPerTokenScale)};

        typename BlockEpilogue3::Params epilogueParams3{
            static_cast<int32_t>(params.EP),
            static_cast<int32_t>(params.expertPerRank),
            static_cast<int32_t>(params.rank),
            reinterpret_cast<__gm__ int32_t *>(remoteWindow() + peerMemoryLayout.offsetPeerTokenPerExpert),
            params.layoutD2,
            static_cast<int32_t>(n2),
            static_cast<int32_t>(L1TileShape::N),
            remoteWindow,
            static_cast<int32_t>(peerMemoryLayout.offsetD),
            static_cast<int32_t>(peerMemoryLayout.offsetPeerPerTokenScale),
            tokenPerExpertLayout};

        BlockEpilogue2 blockEpilogue2(resource, epilogueParams2);
        BlockEpilogue3 blockEpilogue3(resource, epilogueParams3);
        if (isCombineV1) {
            blockEpilogue2.SetFlag();
            CombineV1(params, blockEpilogue2);
        } else {
            blockEpilogue3.SetFlag();
            CombineV2(params, blockEpilogue3);
        }
    }

    __forceinline__ __aicore__ void RunRestoreImpl(Params const &params)
    {
        uint32_t n2 = static_cast<uint32_t>(params.problemShape.shape[2]);
        kernel_detail::PtoSyncAll<true>();
        ResetTokenPerExpert(params.EP * paddedExpertNumAligned);
        remoteWindow.CrossRankSync();

        MoeTokenUnpermuteTilingData tilingData;
        MoeTokenUnpermuteTiling(static_cast<uint32_t>(params.problemShape.shape[0]) * params.topK, n2, params.topK, tilingData, coreNum);
        KernelMoeTokenUnpermute<ElementD2, int32_t, float, true> kernelMoeTokenUnpermuteOp;
        kernelMoeTokenUnpermuteOp.Init(remoteWindow() + peerMemoryLayout.offsetD, workspaceInfo.expandedRowIdx,
                                       params.probs, reinterpret_cast<GM_ADDR>(params.ptrOutput), &tilingData);
        kernelMoeTokenUnpermuteOp.Process();
    }

private:
    __forceinline__ __aicore__ void initBuffer(Params const &params)
    {
        remoteWindow.Init(params.remoteWindowContext);
        workspaceInfo = WorkspaceInfo(params);
        peerMemoryLayout = PeerMemoryLayout(params, remoteWindow);
        cumsumMMPtr = reinterpret_cast<__gm__ int32_t *>(workspaceInfo.ptrcumsumMM);
        gmAPtr = reinterpret_cast<__gm__ ElementA *>(workspaceInfo.ptrA);
        gmCPtr = reinterpret_cast<__gm__ ElementC *>(workspaceInfo.ptrC);
        gmPermutedTokenPtr = reinterpret_cast<__gm__ ElementD1 *>(workspaceInfo.ptrPermutedToken);
        gmC2Ptr = reinterpret_cast<__gm__ ElementC *>(workspaceInfo.ptrC2);
        gmPerTokenScale1Ptr = reinterpret_cast<__gm__ ElementPerTokenScale *>(workspaceInfo.ptrPerTokenScale);
        gmPerTokenScale2Ptr = reinterpret_cast<__gm__ ElementPerTokenScale *>(workspaceInfo.ptrPerTokenScale2);
        tokenPerExpertPtr =
            reinterpret_cast<__gm__ int32_t *>(remoteWindow() + peerMemoryLayout.offsetPeerTokenPerExpert);
        paddedExpertNumAligned = AlignUp(params.EP * params.expertPerRank + 1, ALIGN_128);
        tokenPerExpertLayout = Layout3D(paddedExpertNumAligned, params.expertPerRank);
        preSumBeforeRankPtr = reinterpret_cast<__gm__ int32_t *>(workspaceInfo.ptrSumBeforeRank);

        stageDequantSum1 = 0;
        stageDequantSum2 = 0;
        isCombineV1 = true;
        if (static_cast<uint32_t>(params.problemShape.shape[0]) * params.topK <= 4096) {
            isCombineV1 = false;
        }
    }

    template <typename T>
    __forceinline__ __aicore__ void CopyGMToGM(__gm__ T *dstPtr, __gm__ T *srcPtr, int32_t elemNum, int32_t ubMoveNum)
    {
        kernel_detail::PtoSetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        kernel_detail::PtoSetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

        constexpr int32_t BufferNum = 2;
        constexpr uint64_t tmpBufferOffsetPing = 0;
        constexpr uint64_t tmpBufferOffsetPong = 96 * 1024;

        // [ReduceScatter] 2. Pre Interface Sync
        int pingpongId = 0;
        auto processCount = CeilDiv(elemNum, ubMoveNum);
        for (uint32_t processIndex = 0; processIndex < processCount; ++processIndex) {
            uint32_t curProcessNum =
                (processIndex == processCount - 1) ? elemNum - ubMoveNum * (processCount - 1) : ubMoveNum;
            AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
            uint64_t ubOffsetBytes = pingpongId == 0 ? tmpBufferOffsetPing : tmpBufferOffsetPong;
            auto processOffset = processIndex * ubMoveNum;

            auto inputOffset = processOffset;
            auto outputOffset = processOffset;
            // [ReduceScatter] 2. Pre Interface Sync
            kernel_detail::PtoWaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            // [ReduceScatter] 3. Start shmem_mte_get_mem_nbi
            pto_bridge::PtoLoadVector(ubOffsetBytes, srcPtr + inputOffset, curProcessNum);
            kernel_detail::PtoSetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            kernel_detail::PtoWaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            pto_bridge::PtoStoreVector(dstPtr + outputOffset, ubOffsetBytes, curProcessNum);

            // [ReduceScatter] 4. Post Interface Sync
            kernel_detail::PtoSetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            pingpongId = (pingpongId + 1) % BufferNum;
        }
        // [ReduceScatter] 4. Post Interface Sync

        kernel_detail::PtoWaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        kernel_detail::PtoWaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }


    template <typename T>
    __forceinline__ __aicore__ void CopyGMToGMPerToken(__gm__ T *dst, __gm__ float *dstScale, __gm__ T *src, int32_t rows,
                                       int32_t hiddenSize, int32_t ubMoveNum, int32_t &pingpongId)
    {
        static_assert(sizeof(T) == 1, "CopyGMToGMPerToken expects byte-packed per-token rows");
        using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using PackedGlobal = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
        using PackedTile = pto::Tile<pto::TileType::Vec, T, 1, 1024, pto::BLayout::RowMajor, -1, -1>;

        constexpr int32_t BufferNum = 2;
        constexpr uint64_t tmpBufferOffsetPing = 0;
        constexpr uint64_t tmpBufferOffsetPong = 96 * 1024;
        constexpr int32_t packedTileCols = 1024;
        uint32_t copyInNum = hiddenSize + UB_ALIGN;
        auto processCount = CeilDiv(rows, ubMoveNum);
        __gm__ T *localPackedScratch =
            reinterpret_cast<__gm__ T *>(remoteWindow() + peerMemoryLayout.offsetPeerPerTokenScale);

        for (uint32_t processIndex = 0; processIndex < processCount; ++processIndex) {
            pingpongId = (pingpongId + 1) % BufferNum;
            AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
            uint64_t ubOffsetBytes = pingpongId == 0 ? tmpBufferOffsetPing : tmpBufferOffsetPong;
            auto inputOffset = processIndex * ubMoveNum * copyInNum;

            int32_t rowNum = ubMoveNum;
            if (processIndex == processCount - 1) {
                rowNum = rows - processIndex * ubMoveNum;
            }

            ShapeDyn packedShape(1, 1, 1, rowNum, copyInNum);
            StrideDyn packedStride(rowNum * copyInNum, rowNum * copyInNum, rowNum * copyInNum, copyInNum, 1);
            PackedGlobal localPackedG(localPackedScratch, packedShape, packedStride);
            PackedGlobal remotePackedG(src + inputOffset, packedShape, packedStride);
            PackedTile packedTile(1, copyInNum < packedTileCols ? copyInNum : packedTileCols);
            pto::TASSIGN(packedTile, ubOffsetBytes);
            pto::comm::TGET(localPackedG, remotePackedG, packedTile);
            kernel_detail::PtoPipeBarrier<PIPE_ALL>();

            kernel_detail::PtoWaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            uint32_t dataLen = rowNum * copyInNum;
            pto_bridge::PtoLoadVector(ubOffsetBytes, localPackedScratch, dataLen);

            kernel_detail::PtoSetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            kernel_detail::PtoWaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            auto outputOffset = processIndex * ubMoveNum * hiddenSize;
            pto_bridge::StorePerTokenRows(dst, ubOffsetBytes, outputOffset, static_cast<uint16_t>(rowNum),
                              static_cast<uint16_t>(hiddenSize));
            pto_bridge::StorePerTokenScales(dstScale, ubOffsetBytes, processIndex * ubMoveNum, static_cast<uint16_t>(rowNum),
                                static_cast<uint16_t>(hiddenSize));
            kernel_detail::PtoSetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }

    __forceinline__ __aicore__
    void ApplyXActiveMask(Params const &params)
    {
        if (params.ptrXActiveMask == nullptr) {
            return;
        }
        int32_t m = static_cast<int32_t>(params.problemShape.shape[0]);
        int32_t topK = params.topK;
        int32_t expertNum = params.expertPerRank * params.EP;
        auto *expertIdxPtr = reinterpret_cast<__gm__ int32_t *>(params.expertIdx);
        auto *xActiveMaskPtr = reinterpret_cast<__gm__ bool *>(params.ptrXActiveMask);

        uint32_t totalElements = static_cast<uint32_t>(m * topK);
        uint32_t base = totalElements / coreNum;
        uint32_t rem = totalElements % coreNum;

        uint32_t startCarry = (coreIdx < rem) ? coreIdx : rem;
        uint32_t endCarry = ((coreIdx + 1) < rem) ? (coreIdx + 1) : rem;
        int32_t startIdx = static_cast<int32_t>(coreIdx * base + startCarry);
        int32_t endIdx = static_cast<int32_t>((coreIdx + 1) * base + endCarry);
        int32_t copySize = endIdx - startIdx;

        using Tile = pto::Tile<pto::TileType::Vec, int32_t, 1, 1024, pto::BLayout::RowMajor, -1, -1>;
        constexpr int32_t TileElems = 1024;
        constexpr uint64_t tmpUbOffset = 0;
        for (int32_t offset = 0; offset < copySize; offset += TileElems) {
            uint32_t cur = static_cast<uint32_t>((copySize - offset > TileElems) ? TileElems : (copySize - offset));
            auto expertIdxGlobal = pto_bridge::MakeContiguousGlobalFromPtr(
                expertIdxPtr + startIdx + offset, cur);
            Tile tile(1, cur);
            pto::TASSIGN(tile, tmpUbOffset);
            pto::TLOAD(tile, expertIdxGlobal);

            kernel_detail::PtoSetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            kernel_detail::PtoWaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

            for (uint32_t i = 0; i < cur; ++i) {
                int32_t tokenIdx = (startIdx + offset + i) / topK;
                bool isActive = gm_load(xActiveMaskPtr + tokenIdx);
                if (!isActive) {
                    tile.SetValue(i, expertNum);
                }
            }

            kernel_detail::PtoSetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            kernel_detail::PtoWaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            pto::TSTORE(expertIdxGlobal, tile);
        }
        kernel_detail::PtoSyncAll<true>();
    }

    __forceinline__ __aicore__
    void GetCumsumForMMAIV(__gm__ int32_t *tokenPerExpert, __gm__ int32_t *result, uint32_t expertPerRank,
                           uint32_t rankId, uint32_t EP)
    {
        int32_t expertPerRankAligned = (expertPerRank + 8 - 1) / 8 * 8;
        constexpr uint64_t tmpBufferUbOffset = 0;
        pto_bridge::LoadExpertCountsPadded(tmpBufferUbOffset, tokenPerExpert, rankId * expertPerRank, static_cast<uint16_t>(EP),
                               static_cast<uint16_t>(expertPerRank * sizeof(int32_t)),
                               static_cast<uint16_t>((paddedExpertNumAligned - expertPerRank) * sizeof(int32_t)));

        kernel_detail::PtoSetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        kernel_detail::PtoWaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        for (uint32_t i = 1; i < EP; ++i) {
            uint64_t rowOffset = tmpBufferUbOffset + static_cast<uint64_t>(i) * expertPerRankAligned * sizeof(int32_t);
            uint64_t prevRowOffset =
                tmpBufferUbOffset + static_cast<uint64_t>(i - 1) * expertPerRankAligned * sizeof(int32_t);
            pto_bridge::PtoAddVector<int32_t>(rowOffset, rowOffset, prevRowOffset, expertPerRank);
            kernel_detail::PtoPipeBarrier<PIPE_V>();
        }

        kernel_detail::PtoSetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        kernel_detail::PtoWaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        pto_bridge::StoreExpertCountsPadded(result, tmpBufferUbOffset, static_cast<uint16_t>(EP),
                                static_cast<uint16_t>(expertPerRank * sizeof(int32_t)));
    }

    __forceinline__ __aicore__
    void GMM1(Params const &params)
    {
        icache_preload(8);
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);
        float aivFinishGroups = 0.0f;
        __gm__ float *aivFinishPtr = workspaceInfo.ptrSoftFlagBase + params.EP * FLAGSTRIDE;

        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetC = 0;
        uint32_t startCoreIdx = 0;
        uint32_t syncGroupIdx = 0;
        int64_t preCurrentmSum = 0;
        int32_t syncLoopIdx = -1;

        uint16_t syncgmmIdx = 0;
        AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx /
                                        CROSS_CORE_FLAG_MAX_SET_COUNT); // Wait for AIV to finish cumsum for matmul
        syncgmmIdx++;

        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentM = gm_load(cumsumMMPtr + (params.EP - 1) * params.expertPerRank + groupIdx);
            if (preCurrentmSum >= params.maxOutputSize) {
                currentM = 0;
            } else if (preCurrentmSum + currentM >= params.maxOutputSize) {
                currentM = params.maxOutputSize - preCurrentmSum;
            }
            int32_t arrayGroupIdx = params.listLen == 1 ? 0 : groupIdx;
            __gm__ ElementB *gmB1Ptr =
                reinterpret_cast<__gm__ ElementB *>(kernel_detail::GetTensorAddr<int8_t>(arrayGroupIdx, params.ptrB1));
            __gm__ ElementScale *gmSPtr =
                reinterpret_cast<__gm__ ElementScale *>(kernel_detail::GetTensorAddr<int64_t>(arrayGroupIdx, params.ptrScale1));
            PtoShape3D inGroupProblemShape =
                PtoShape3D(currentM, static_cast<uint32_t>(params.problemShape.shape[1]), static_cast<uint32_t>(params.problemShape.shape[2]));
            LayoutA layoutA = params.layoutA.GetTileLayout(PtoShape2D(inGroupProblemShape.shape[0], inGroupProblemShape.shape[2]));
            LayoutB layoutB1 = params.layoutB1;
            LayoutScale layoutScale = params.layoutScale1;
            LayoutC layoutC = LayoutC(static_cast<uint32_t>(inGroupProblemShape.shape[0]), static_cast<uint32_t>(inGroupProblemShape.shape[1]));
            blockScheduler.Update(inGroupProblemShape, L1TileShape::ToPtoShapeMN());
            uint32_t coreLoops = blockScheduler.GetCoreLoops();
            // Determine the starting loopIdx of the current core under the current groupIdx
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;
            // Loop through the matmul of each groupIdx

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                for (; syncGroupIdx <= groupIdx; syncGroupIdx++) {
                    AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx / CROSS_CORE_FLAG_MAX_SET_COUNT);
                    syncgmmIdx++;
                }

                auto blockCoordMN = blockScheduler.GetBlockCoordMN(loopIdx);
                auto actualBlockShapeMN = blockScheduler.GetActualBlockShapeMN(blockCoordMN);
                uint32_t blockM = static_cast<uint32_t>(blockCoordMN.shape[0]);
                uint32_t blockN = static_cast<uint32_t>(blockCoordMN.shape[1]);
                auto offsetA = PtoCoord2D(blockM * L1TileShape::M, 0);
                auto offsetB = PtoCoord2D(0, blockN * L1TileShape::N);
                auto offsetC = PtoCoord2D(blockM * L1TileShape::M, blockN * L1TileShape::N);
                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = layoutB1.GetOffset(offsetB);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                int64_t gmOffsetS =
                    blockN * L1TileShape::N + (params.listLen == 1 ? groupIdx * static_cast<uint32_t>(params.problemShape.shape[1]) : 0);
                if (currentM > 0) {
                    PtoShape3D actualBlockShape = PtoShape3D(
                        actualBlockShapeMN.shape[0], actualBlockShapeMN.shape[1], static_cast<uint32_t>(inGroupProblemShape.shape[2]));
                    blockMmad(gmAPtr + gmGroupOffsetA + gmOffsetA, layoutA, gmB1Ptr + gmGroupOffsetB + gmOffsetB,
                              layoutB1, gmCPtr + gmGroupOffsetC + gmOffsetC, layoutC, gmSPtr + gmOffsetS, layoutScale,
                              actualBlockShape);
                }
            }

            if ((groupIdx + 1) == params.epilogueGranularity && (groupIdx < params.expertPerRank - 1)) {
                syncLoopIdx++;
                if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                    blockMmad.SynchronizeBlock();
                }
                // Synchronization signal: GMM1 notifies SwiGLU [1]
                blockMmad.Finalize(syncLoopIdx, SYNCFLAGC2V);
            }

            preCurrentmSum += currentM;
            gmGroupOffsetA += static_cast<uint32_t>(inGroupProblemShape.shape[0]) * static_cast<uint32_t>(inGroupProblemShape.shape[2]);
            if (params.listLen == 1) {
                gmGroupOffsetB += static_cast<uint32_t>(inGroupProblemShape.shape[2]) * static_cast<uint32_t>(inGroupProblemShape.shape[1]);
            }
            gmGroupOffsetC += static_cast<uint32_t>(inGroupProblemShape.shape[0]) * static_cast<uint32_t>(inGroupProblemShape.shape[1]);
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }

        for (; syncGroupIdx < params.expertPerRank; syncGroupIdx++) {
            AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx / CROSS_CORE_FLAG_MAX_SET_COUNT);
            syncgmmIdx++;
        }

        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }
        // Synchronization signal: GMM1 notifies SwiGLU [2]
        blockMmad.Finalize(syncLoopIdx + 1, SYNCFLAGC2V);
    }

    __forceinline__ __aicore__
    void GMM2(Params const &params)
    {
        icache_preload(8);
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);

        uint32_t n2 = static_cast<uint32_t>(params.problemShape.shape[2]);
        uint32_t k2 = static_cast<uint32_t>(params.problemShape.shape[1]) / 2;

        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetC = 0;

        uint32_t startCoreIdx = 0;

        int64_t preCurrentmSum = 0;
        int32_t syncLoopIdx = -1;
        uint32_t lastDequantExpertNum = params.expertPerRank;

        if (params.epilogueGranularity < params.expertPerRank) {
            lastDequantExpertNum = params.expertPerRank - params.epilogueGranularity;
        }

        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentM = gm_load(cumsumMMPtr + (params.EP - 1) * params.expertPerRank + groupIdx);
            if (preCurrentmSum >= params.maxOutputSize) {
                currentM = 0;
            } else if (preCurrentmSum + currentM > params.maxOutputSize) {
                currentM = params.maxOutputSize - preCurrentmSum;
            }
            int32_t arrayGroupIdx = params.listLen == 1 ? 0 : groupIdx;
            __gm__ ElementB *gmB2Ptr =
                reinterpret_cast<__gm__ ElementB *>(kernel_detail::GetTensorAddr<int8_t>(arrayGroupIdx, params.ptrB2));
            __gm__ ElementScale *gmS2Ptr =
                reinterpret_cast<__gm__ ElementScale *>(kernel_detail::GetTensorAddr<int64_t>(arrayGroupIdx, params.ptrScale2));
            PtoShape3D inGroupProblemShape = PtoShape3D(currentM, n2, k2); // M N K

            LayoutA layoutA = params.layoutA2.GetTileLayout(PtoShape2D(inGroupProblemShape.shape[0], inGroupProblemShape.shape[2]));
            LayoutB layoutB2 = params.layoutB2;
            LayoutScale layoutScale = params.layoutScale2;
            LayoutC layoutC = LayoutC(static_cast<uint32_t>(inGroupProblemShape.shape[0]), static_cast<uint32_t>(inGroupProblemShape.shape[1]));

            blockScheduler.Update(inGroupProblemShape, L1TileShape::ToPtoShapeMN());
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            // Determine the starting loopIdx of the current core under the current groupIdx
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;
            // Loop through the matmul of each groupIdx
            if (params.expertPerRank > lastDequantExpertNum &&
                groupIdx + 1 == params.expertPerRank - lastDequantExpertNum) {
                AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGV2C);
            }

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                if (loopIdx + coreNum >= coreLoops) {
                    syncLoopIdx = groupIdx;
                }

                auto blockCoordMN = blockScheduler.GetBlockCoordMN(loopIdx);
                auto actualBlockShapeMN = blockScheduler.GetActualBlockShapeMN(blockCoordMN);
                uint32_t blockM = static_cast<uint32_t>(blockCoordMN.shape[0]);
                uint32_t blockN = static_cast<uint32_t>(blockCoordMN.shape[1]);
                auto offsetA = PtoCoord2D(blockM * L1TileShape::M, 0);
                auto offsetB = PtoCoord2D(0, blockN * L1TileShape::N);
                auto offsetC = PtoCoord2D(blockM * L1TileShape::M, blockN * L1TileShape::N);

                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = layoutB2.GetOffset(offsetB);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                int64_t gmOffsetS =
                    blockN * L1TileShape::N + (params.listLen == 1 ? groupIdx * n2 : 0); // One scale group per expert
                if (currentM > 0) {
                    PtoShape3D actualBlockShape = PtoShape3D(
                        actualBlockShapeMN.shape[0], actualBlockShapeMN.shape[1], static_cast<uint32_t>(inGroupProblemShape.shape[2]));
                    blockMmad(gmPermutedTokenPtr + gmGroupOffsetA + gmOffsetA, layoutA,
                              gmB2Ptr + gmGroupOffsetB + gmOffsetB, layoutB2, gmC2Ptr + gmGroupOffsetC + gmOffsetC,
                              layoutC, gmS2Ptr + gmOffsetS, layoutScale, actualBlockShape, syncLoopIdx, 0);
                }
            }
            preCurrentmSum += currentM;
            gmGroupOffsetA += static_cast<uint32_t>(inGroupProblemShape.shape[0]) *
                              static_cast<uint32_t>(inGroupProblemShape.shape[2]);
            if (params.listLen == 1) {
                gmGroupOffsetB += static_cast<uint32_t>(inGroupProblemShape.shape[2]) *
                                  static_cast<uint32_t>(inGroupProblemShape.shape[1]);
            }
            gmGroupOffsetC += static_cast<uint32_t>(inGroupProblemShape.shape[0]) *
                              static_cast<uint32_t>(inGroupProblemShape.shape[1]);

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }
        if (isCombineV1) {
            blockMmad.Finalize(params.expertPerRank - 1, 0);
        }
    }

    __forceinline__ __aicore__
    void InitArithProgress(Params const &params)
    {
        kernel_detail::PtoSetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        kernel_detail::PtoWaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        pto_bridge::StoreZeroPtoUbToGm<float>(workspaceInfo.ptrSoftFlagBase, 0,
                                  static_cast<uint32_t>((params.EP + 1) * FLAGSTRIDE));
        kernel_detail::PtoSetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        kernel_detail::PtoWaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    }

    __forceinline__ __aicore__
    void CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2(Params const &params,
                                                                        int64_t localTokenPerExpertOffset)
    {
        uint32_t numPerCore = paddedExpertNumAligned;
        constexpr uint64_t tmpBufferUbOffset = 0;
        uint64_t prevSumUbOffset = static_cast<uint64_t>(numPerCore) * sizeof(int32_t);

        remoteWindow.ResetLocalTokenReady();
        kernel_detail::PtoSyncAll<true>();
        remoteWindow.CrossRankSync();

        for (int32_t dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += coreNum) {
            if (dstEpIdx == params.rank) {
                continue;
            }
            __gm__ int32_t *srcAddress = reinterpret_cast<__gm__ int32_t *>(remoteWindow() + localTokenPerExpertOffset);
            __gm__ void *dstPeermemPtr = remoteWindow(localTokenPerExpertOffset, dstEpIdx);

            kernel_detail::PtoSetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
            using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
            using TputGlobal = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;
            using TputTile = pto::Tile<pto::TileType::Vec, int32_t, 1, 128, pto::BLayout::RowMajor, -1, -1>;
            int64_t scratchOffsetBytes =
                peerMemoryLayout.offsetPeerPerTokenScale + static_cast<int64_t>(coreIdx) * numPerCore * sizeof(int32_t);
            __gm__ int32_t *localScratch =
                reinterpret_cast<__gm__ int32_t *>(remoteWindow(scratchOffsetBytes, params.rank));
            ShapeDyn tputShape(1, 1, 1, 1, numPerCore);
            StrideDyn tputStride(numPerCore, numPerCore, numPerCore, numPerCore, 1);
            TputTile tputTile(1, numPerCore);

            kernel_detail::PtoWaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

            pto_bridge::PtoLoadVector(tmpBufferUbOffset, srcAddress, numPerCore);

            kernel_detail::PtoSetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            kernel_detail::PtoWaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            pto_bridge::PtoAddScalarVector<int32_t>(tmpBufferUbOffset, tmpBufferUbOffset, numPerCore, static_cast<int32_t>(0x800000));
            kernel_detail::PtoSetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            kernel_detail::PtoWaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            pto_bridge::PtoStoreVector(localScratch, tmpBufferUbOffset, numPerCore);
            TputGlobal localPackedG(localScratch, tputShape, tputStride);
            TputGlobal remotePackedG(reinterpret_cast<__gm__ int32_t *>(dstPeermemPtr), tputShape, tputStride);
            pto::comm::TPUT(remotePackedG, localPackedG, tputTile);
            kernel_detail::PtoSetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            kernel_detail::PtoWaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            remoteWindow.NotifyRemoteTokenReady(dstEpIdx);
        }
        for (int32_t dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += coreNum) {
            if (dstEpIdx != params.rank) {
                remoteWindow.WaitTokenReady(dstEpIdx);
                __gm__ int32_t *tokenBase = tokenPerExpertPtr + tokenPerExpertLayout(dstEpIdx, 0, 0);
                pto_bridge::PtoLoadVector(tmpBufferUbOffset, tokenBase, numPerCore);
                kernel_detail::PtoSetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                kernel_detail::PtoWaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                pto_bridge::PtoAddScalarVector<int32_t>(tmpBufferUbOffset, tmpBufferUbOffset, numPerCore, static_cast<int32_t>(-0x800000));
                kernel_detail::PtoPipeBarrier<PIPE_V>();
                kernel_detail::PtoSetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                kernel_detail::PtoWaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                pto_bridge::PtoStoreVector(tokenBase, tmpBufferUbOffset, numPerCore);
            } else {
                pto_bridge::PtoLoadVector(tmpBufferUbOffset, tokenPerExpertPtr + tokenPerExpertLayout(dstEpIdx, 0, 0), numPerCore);
                kernel_detail::PtoSetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                kernel_detail::PtoWaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            }
            kernel_detail::PtoPipeBarrier<PIPE_ALL>();
            int32_t prevSum = 0;
            int32_t j = 0;
            for (int32_t i = 0; i < (params.rank + 1) * params.expertPerRank; i++) {
                if (i >= params.rank * params.expertPerRank) {
                    pto_bridge::PtoSetValue<int32_t>(prevSumUbOffset, static_cast<uint32_t>(j), prevSum);
                    j++;
                }
                prevSum += pto_bridge::PtoGetValue<int32_t>(tmpBufferUbOffset, static_cast<uint32_t>(i));
            }
            kernel_detail::PtoSetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            kernel_detail::PtoWaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            pto_bridge::PtoStoreVector(preSumBeforeRankPtr + dstEpIdx * params.expertPerRank, prevSumUbOffset,
                           params.expertPerRank);
        }

        kernel_detail::PtoSyncAll<true>();
    }

    __forceinline__ __aicore__
    void ResetTokenPerExpert(int32_t num)
    {
        if (coreIdx != coreNum - 1) {
            return;
        }
        kernel_detail::PtoSetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        kernel_detail::PtoWaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        pto_bridge::StoreZeroPtoUbToGm<int32_t>(tokenPerExpertPtr, 0, static_cast<uint32_t>(num));
        kernel_detail::PtoSetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        kernel_detail::PtoWaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    }

    __forceinline__ __aicore__
    void UpdateAicFlags(const Params &params)
    {
        float flagBase = 1.0f * params.expertPerRank;
        __gm__ float *aicFinishPtr = workspaceInfo.ptrSoftFlagBase + params.EP * FLAGSTRIDE;
        float flag = 0.0f;
        float lastflag = -1.0f;
        __gm__ float *flagPtr = workspaceInfo.ptrSoftFlagBase;
        constexpr uint32_t TileElems = 1024;
        using Tile = pto::Tile<pto::TileType::Vec, float, 1, TileElems, pto::BLayout::RowMajor, -1, -1>;
        constexpr uint64_t tmpUbOffset = 0;
        uint32_t flagElemNum = static_cast<uint32_t>(params.EP * FLAGSTRIDE);
        while (flag < flagBase) {
            flag = flagBase;
            pto_bridge::PtoLoadVector<float, TileElems>(tmpUbOffset, flagPtr, flagElemNum);
            kernel_detail::PtoSetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            kernel_detail::PtoWaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

            for (int32_t ep = 0; ep < params.EP; ++ep) {
                uint32_t elemOffset = static_cast<uint32_t>(ep * FLAGSTRIDE);
                uint64_t tileOffsetBytes =
                    tmpUbOffset + static_cast<uint64_t>(elemOffset / TileElems) * TileElems * sizeof(float);
                Tile tile(1, TileElems);
                pto::TASSIGN(tile, tileOffsetBytes);
                flag = min(flag, tile.GetValue(elemOffset % TileElems));
            }

            if (flag > lastflag) {
                gm_store(aicFinishPtr, flag);
                gm_dcci(aicFinishPtr);
                lastflag = flag;
            }
        }
    }

    __forceinline__ __aicore__
    void CombineV1(Params const &params, BlockEpilogue2 &blockEpilogue)
    {
        uint32_t n2 = static_cast<uint32_t>(params.problemShape.shape[2]);
        int32_t prevGroupSum2 = 0;

        icache_preload(8);
        for (uint32_t t_groupIdx = 0; t_groupIdx < params.expertPerRank; ++t_groupIdx) {
            int32_t flagId = t_groupIdx / CROSS_CORE_FLAG_MAX_SET_COUNT;
            AscendC::CrossCoreWaitFlag<0x2>(flagId);
            kernel_detail::PtoSyncAll<true>();

            uint32_t groupIdx = t_groupIdx;

            for (int32_t dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += coreNum) {
                __gm__ void *dstPeermemPtr = remoteWindow(peerMemoryLayout.offsetD, dstEpIdx);
                uint32_t srcRowOffset =
                    (dstEpIdx == 0 ? 0 : gm_load(cumsumMMPtr + (dstEpIdx - 1) * params.expertPerRank + groupIdx)) +
                    prevGroupSum2;
                if (srcRowOffset < params.maxOutputSize) {
                    uint32_t dataRows =
                        gm_load(tokenPerExpertPtr + tokenPerExpertLayout(dstEpIdx, params.rank, groupIdx));
                    if (srcRowOffset + dataRows > params.maxOutputSize) {
                        dataRows = params.maxOutputSize - srcRowOffset;
                    }
                    uint32_t dstRowOffset = gm_load(preSumBeforeRankPtr + dstEpIdx * params.expertPerRank + groupIdx);
                    auto offsetC = PtoCoord2D(srcRowOffset, 0);
                    auto offsetPeer = PtoCoord2D(dstRowOffset, 0);
                    PtoShape2D shapeC(dataRows, n2);
                    int64_t gmOffsetC = params.layoutD2.GetOffset(offsetC);
                    int64_t gmOffsetPeer = params.layoutD2.GetOffset(offsetPeer);
                    __gm__ ElementD2 *dstPeerBase = reinterpret_cast<__gm__ ElementD2 *>(dstPeermemPtr) + gmOffsetPeer;
                    if constexpr (std::is_same_v<ElementA, int8_t>) {
                        blockEpilogue(gmC2Ptr + gmOffsetC, shapeC, gmPerTokenScale2Ptr + srcRowOffset, dstPeerBase,
                                      dstEpIdx);
                    } else {
                        blockEpilogue(gmC2Ptr + gmOffsetC, shapeC, dstPeerBase, dstEpIdx);
                    }
                }
            }
            prevGroupSum2 += gm_load(cumsumMMPtr + (params.EP - 1) * params.expertPerRank + groupIdx);
        }
        blockEpilogue.Finalize();
    }

    __forceinline__ __aicore__
    void CombineV2(Params const &params, BlockEpilogue3 &blockEpilogue)
    {
        BlockScheduler blockScheduler;
        int32_t syncLoopIdx = 0;
        uint32_t startCoreIdx = 0;
        uint32_t aicCoreNum = coreNum / 2;
        uint32_t aicCoreIdx = get_block_idx();
        uint32_t aivSubCoreIdx = get_subblockid();
        uint32_t preSrcExpertSum = 0;
        uint32_t n2 = static_cast<uint32_t>(params.problemShape.shape[2]);
        uint32_t k2 = static_cast<uint32_t>(params.problemShape.shape[1]) / 2;
        icache_preload(8);
        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentExpertM = gm_load(cumsumMMPtr + (params.EP - 1) * params.expertPerRank + groupIdx);
            if (preSrcExpertSum >= params.maxOutputSize) {
                currentExpertM = 0;
            } else if (preSrcExpertSum + currentExpertM > params.maxOutputSize) {
                currentExpertM = params.maxOutputSize - preSrcExpertSum;
            }
            PtoShape3D inGroupProblemShape = PtoShape3D(currentExpertM, n2, k2); // M N K
            blockScheduler.Update(inGroupProblemShape, L1TileShape::ToPtoShapeMN());
            uint32_t coreLoops = blockScheduler.GetCoreLoops();
            uint32_t startLoopIdx =
                ((aicCoreIdx < startCoreIdx) ? (aicCoreIdx + aicCoreNum) : aicCoreIdx) - startCoreIdx;

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += aicCoreNum) {
                auto blockCoordMN = blockScheduler.GetBlockCoordMN(loopIdx);
                auto actualBlockShapeMN = blockScheduler.GetActualBlockShapeMN(blockCoordMN);
                int32_t m0 = 16;
                uint32_t blockM = static_cast<uint32_t>(actualBlockShapeMN.shape[0]);
                uint32_t blockN = static_cast<uint32_t>(actualBlockShapeMN.shape[1]);
                //  Block count, the shape of each block is (m0, blockN)
                int32_t m_rows = (blockM + m0 - 1) / m0;
                int32_t aiv_m_rows = m_rows / 2;
                if (aivSubCoreIdx == 1 && aiv_m_rows * 2 < m_rows) {
                    aiv_m_rows += 1;
                }
                uint32_t m_offset = static_cast<uint32_t>(blockCoordMN.shape[0]) * L1TileShape::M;
                if (aivSubCoreIdx == 1) {
                    m_offset += (m_rows / 2) * m0;
                }

                for (; syncLoopIdx <= groupIdx; syncLoopIdx++) {
                    int32_t flag_id = syncLoopIdx / CROSS_CORE_FLAG_MAX_SET_COUNT;
                    AscendC::CrossCoreWaitFlag<0x2>(flag_id);
                }

                for (int32_t cur_row = 0; cur_row < aiv_m_rows; cur_row++) {
                    auto realTileCoord =
                        PtoCoord2D(m_offset, static_cast<uint32_t>(blockCoordMN.shape[1]) * L1TileShape::N);
                    uint32_t actualm = m0;
                    if (aivSubCoreIdx == 1 && cur_row == aiv_m_rows - 1) {
                        actualm = blockM - (m_rows / 2) * m0 - cur_row * m0;
                    }
                    PtoShape2D realTileShape(actualm, blockN);
                    blockEpilogue(gmC2Ptr, gmPerTokenScale2Ptr, realTileCoord, realTileShape, groupIdx, preSrcExpertSum,
                                  preSumBeforeRankPtr);
                    m_offset += m0;
                }
            }
            preSrcExpertSum += currentExpertM;
            startCoreIdx = (startCoreIdx + coreLoops) % aicCoreNum;
        }
        blockEpilogue.Finalize();
    }

private:
    struct WorkspaceInfo {
        GM_ADDR ptrA;
        GM_ADDR ptrPerTokenScale;
        GM_ADDR ptrcumsumMM;
        GM_ADDR ptrC;
        GM_ADDR ptrC2;
        GM_ADDR ptrPermutedToken;
        GM_ADDR ptrPerTokenScale2;
        GM_ADDR expandedRowIdx;
        GM_ADDR ptrTokenPerExpert;
        GM_ADDR ptrSumBeforeRank;
        __gm__ float *ptrSoftFlagBase;

        __forceinline__ __aicore__
        WorkspaceInfo()
        {}

        __forceinline__ __aicore__
        WorkspaceInfo(const Params &params)
        {
            uint32_t k2 = static_cast<uint32_t>(params.problemShape.shape[1]) / 2;
            uint32_t n2 = static_cast<uint32_t>(params.problemShape.shape[2]);
            int64_t workspaceOffset = 0;
            expandedRowIdx = params.ptrWorkspace;

            workspaceOffset += AlignUp(static_cast<uint32_t>(params.problemShape.shape[0]), 256) * params.topK * sizeof(int32_t);
            ptrcumsumMM = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += (params.EP * params.EP * params.expertPerRank) * sizeof(int32_t);

            workspaceOffset += (params.EP * params.EP * params.expertPerRank) * sizeof(int32_t);
            ptrPerTokenScale = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.maxOutputSize * sizeof(ElementPerTokenScale);
            ptrPerTokenScale2 = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.maxOutputSize * sizeof(ElementPerTokenScale);
            ptrTokenPerExpert = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += (params.EP * params.EP * params.expertPerRank) * sizeof(int32_t);
            ptrC = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.maxOutputSize * static_cast<uint32_t>(params.problemShape.shape[1]) * sizeof(ElementC);
            ptrC2 = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.maxOutputSize * n2 * sizeof(ElementC);
            ptrA = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.maxOutputSize * static_cast<uint32_t>(params.problemShape.shape[2]) * sizeof(ElementA);
            ptrPermutedToken = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.maxOutputSize * k2 * sizeof(ElementA);
            ptrSumBeforeRank = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.EP * sizeof(int32_t) * FLAGSTRIDE;
            ptrSoftFlagBase = reinterpret_cast<__gm__ float *>(params.ptrWorkspace + workspaceOffset);
        }
    };

    struct PeerMemoryLayout {
        int64_t offsetA;
        int64_t offsetPeerPerTokenScale;
        int64_t offsetPeerTokenPerExpert;
        int64_t offsetD;

        __forceinline__ __aicore__
        PeerMemoryLayout()
        {}

        __forceinline__ __aicore__
        PeerMemoryLayout(const Params &params, const PtoRemoteWindow &remoteWindow)
        {
            offsetA = 0; // Occupies one third of BUFFSIZE
            offsetPeerPerTokenScale = offsetA + AlignUp(remoteWindow.SegmentSize() / 3, 512); // Occupies 1 MB
            offsetD = offsetPeerPerTokenScale + MB_SIZE;                         // Occupies the remaining space
            offsetPeerTokenPerExpert = remoteWindow.SegmentSize() - 2 * MB_SIZE; // Occupies the final 2 MB
        }
    };

    Arch::Resource<ArchTag> resource;

    uint32_t coreIdx;
    uint32_t coreNum;

    WorkspaceInfo workspaceInfo;
    PeerMemoryLayout peerMemoryLayout;

    __gm__ ElementA *gmAPtr;
    __gm__ ElementC *gmCPtr;

    __gm__ ElementD1 *gmPermutedTokenPtr;
    __gm__ ElementC *gmC2Ptr;

    __gm__ ElementPerTokenScale *gmPerTokenScale1Ptr;
    __gm__ ElementPerTokenScale *gmPerTokenScale2Ptr;

    __gm__ int32_t *tokenPerExpertPtr;
    __gm__ int32_t *cumsumMMPtr;
    __gm__ int32_t *preSumBeforeRankPtr;
    Layout3D tokenPerExpertLayout;
    PtoRemoteWindow remoteWindow;
    int32_t paddedExpertNumAligned;
    uint32_t stageDequantSum1;
    uint32_t stageDequantSum2;
    bool isCombineV1;
};

} // namespace pto_ext::Gemm::Kernel

#endif // DISPATCH_FFN_COMBINE_KERNEL_HPP
