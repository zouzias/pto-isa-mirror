/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file dispatch_combine_moe.h
 * \brief
 */

#ifndef DISPATCH_COMBINE_MOE_H
#define DISPATCH_COMBINE_MOE_H

#include "kernel_operator.h"

#include "dispatch_combine_moe_tiling.h"

#include "utils/moe_pto_utils.hpp"
#include "utils/dispatch_policy_custom.hpp"

#include "utils/const_args.hpp"
#include "dispatch_combine_moe_kernel.hpp"
#include "token_reorder/routing/moe_init_routing_quant_tiling.h"

namespace DispatchCombineMoeImpl {
#define TemplateMMA2AClass typename AType_, typename BType_, typename CType_, bool TB_, bool Nz_
#define TemplateMMA2ACFunc AType_, BType_, CType_, TB_, Nz_

template <typename Layout, typename ElementType, typename = void>
struct LayoutBInitializer {
    __forceinline__ __aicore__ static Layout create(uint32_t k, uint32_t n)
    {
        return Layout{k, n};
    }
};

template <typename Layout, typename ElementType>
struct LayoutBInitializer<Layout, ElementType, std::enable_if_t<Layout::kTileLayout == pto::TileLayoutCustom::ZN>> {
    __forceinline__ __aicore__ static Layout create(uint32_t k, uint32_t n)
    {
        return Layout::template MakeLayout<ElementType>(k, n);
    }
};

template <TemplateMMA2AClass>
class DispatchCombineMoe {
public:
    __aicore__ inline DispatchCombineMoe(){};
    __aicore__ inline void Init(GM_ADDR xGM, GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR expertIdGM, GM_ADDR scale1GM,
                                GM_ADDR scale2GM, GM_ADDR probs, GM_ADDR xActiveMaskGM, GM_ADDR outGM,
                                GM_ADDR expertTokenNums, GM_ADDR workspaceGM,
                                const __gm__ DispatchCombineMoeTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void AssignTensorAddresses(GM_ADDR xGM, GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR expertIdGM,
                                                 GM_ADDR scale1GM, GM_ADDR scale2GM, GM_ADDR probs,
                                                 GM_ADDR xActiveMaskGM, GM_ADDR outGM, GM_ADDR expertTokenNums,
                                                 GM_ADDR workspaceGM);
    __aicore__ inline void AssignDispatchInfo(const __gm__ DispatchCombineMoeTilingData *tilingData);
    __aicore__ inline void AssignCocInfo(const __gm__ DispatchCombineMoeTilingData *tilingData);
    __aicore__ inline void AssignVbsTiling(const __gm__ optiling::MoeInitRoutingQuantTilingData *src);
    __aicore__ inline void AssignGatherTiling(optiling::InnerMoeGatherOutComputeTilingData &dst,
                                              const __gm__ optiling::InnerMoeGatherOutComputeTilingData *src);
    __aicore__ inline void AssignRoutingTiling(const __gm__ DispatchCombineMoeTilingData *tilingData);
    __aicore__ inline void AssignRuntimeInfo(const __gm__ DispatchCombineMoeTilingData *tilingData);

    using ProcessArchTag = pto_ext::Arch::AtlasA5;
    static constexpr bool PROCESS_ENABLE_UNIT_FLAG = false;
    static constexpr bool PROCESS_ENABLE_SHUFFLE_K = true;
    static constexpr uint32_t PROCESS_WORKSPACE_STAGES = 2;
    static constexpr uint32_t PROCESS_PRELOAD_STAGES = 1;
    static constexpr uint32_t PROCESS_L1_STAGES = 2;
    static constexpr uint32_t PROCESS_L0A_STAGES = 2;
    static constexpr uint32_t PROCESS_L0B_STAGES = 2;
    static constexpr uint32_t PROCESS_L0C_STAGES = 1;
    static constexpr uint32_t PROCESS_UB_STAGES = 2;

    using ProcessLayoutA = pto_ext::layout::ND;
    using ProcessLayoutB =
        typename std::conditional<Nz_, pto_ext::layout::Zn,
                                  typename std::conditional<TB_, pto_ext::layout::DN, pto_ext::layout::ND>::type>::type;
    using ProcessL1TileShape = pto_ext::GemmShape<128, 256, 512>;
    using ProcessL0TileShape = pto_ext::GemmShape<128, 256, 128>;
    using ProcessDispatchPolicy =
        pto_ext::Gemm::MmadAtlasA5PreloadAsyncFixpipe<PROCESS_PRELOAD_STAGES, PROCESS_L1_STAGES, PROCESS_L0A_STAGES,
                                                      PROCESS_L0B_STAGES, PROCESS_L0C_STAGES, PROCESS_ENABLE_UNIT_FLAG,
                                                      PROCESS_ENABLE_SHUFFLE_K>;
    using ProcessAType = pto_ext::Gemm::GemmType<int8_t, pto_ext::layout::ND>;
    using ProcessBType = pto_ext::Gemm::GemmType<int8_t, ProcessLayoutB>;
    using ProcessCType = pto_ext::Gemm::GemmType<float16_t, pto_ext::layout::ND>;
    using ProcessD1Type = pto_ext::Gemm::GemmType<int8_t, pto_ext::layout::ND>;
    using ProcessD2Type = typename std::conditional<std::is_same_v<CType_, bfloat16_t>,
                                                    pto_ext::Gemm::GemmType<bfloat16_t, pto_ext::layout::ND>,
                                                    pto_ext::Gemm::GemmType<CType_, pto_ext::layout::ND>>::type;
    using ProcessBlockMmad =
        pto_ext::Gemm::Block::BlockMmad<ProcessDispatchPolicy, ProcessL1TileShape, ProcessL0TileShape, ProcessAType,
                                        ProcessBType, ProcessCType>;
    using ProcessScaleType = pto_ext::Gemm::GemmType<uint64_t, pto_ext::layout::VectorLayout>;
    using ProcessPerTokenScaleType = pto_ext::Gemm::GemmType<float, pto_ext::layout::VectorLayout>;
    using ProcessElementMulType = pto_ext::Gemm::GemmType<float, pto_ext::layout::ND>;
    using ProcessTileElemWiseMuls = pto_ext::Epilogue::Tile::TileElemWiseMuls<ProcessArchTag, ProcessElementMulType, 0>;
    using ProcessTileCopy1 = pto_ext::Epilogue::Tile::TileCopy<ProcessArchTag, ProcessCType, ProcessScaleType,
                                                               ProcessPerTokenScaleType, ProcessD1Type>;
    using ProcessTileCopy2 = pto_ext::Epilogue::Tile::TileCopy<ProcessArchTag, ProcessCType, ProcessScaleType,
                                                               ProcessPerTokenScaleType, ProcessD2Type>;
    using ProcessBlockEpilogue1 = pto_ext::Epilogue::Block::BlockEpilogue<
        pto_ext::Epilogue::EpilogueAtlasA5PerTokenDequantSwigluQuant<PROCESS_UB_STAGES>, ProcessCType,
        ProcessPerTokenScaleType, ProcessD1Type, ProcessTileElemWiseMuls, ProcessTileCopy1>;
    using ProcessBlockEpilogue2 =
        pto_ext::Epilogue::Block::BlockEpilogue<pto_ext::Epilogue::EpilogueAtlasA5PerTokenDequant<PROCESS_UB_STAGES>,
                                                ProcessCType, ProcessPerTokenScaleType, ProcessD2Type,
                                                ProcessTileCopy2>;
    using ProcessBlockEpilogue3 =
        pto_ext::Epilogue::Block::BlockEpilogue<pto_ext::Epilogue::EpilogueAtlasA5PerTokenDequantV2<PROCESS_UB_STAGES>,
                                                ProcessCType, ProcessPerTokenScaleType, ProcessD2Type,
                                                ProcessTileCopy2>;
    using ProcessBlockScheduler = typename pto_ext::Gemm::Block::GemmIdentityBlockSwizzle<9, 1>;
    using ProcessElementGroupList = int64_t;
    using ProcessMatmulKernel =
        pto_ext::Gemm::Kernel::DispatchCombineMoeKernel<ProcessBlockMmad, ProcessBlockScheduler,
                                                        ProcessElementGroupList, ProcessBlockEpilogue1,
                                                        ProcessBlockEpilogue2, ProcessBlockEpilogue3>;

    template <typename MatmulKernel, typename LayoutAParam, typename LayoutBParam, typename LayoutScaleParam,
              typename LayoutD1Param, typename LayoutD2Param>
    __aicore__ inline void BuildAndRunKernelParams(const pto_ext::PtoShape3D &problemShape, uint32_t epilogueCoreNum,
                                                   uint32_t epilogueGranularity, const LayoutAParam &layoutA1,
                                                   const LayoutAParam &layoutA2, const LayoutBParam &layoutB1,
                                                   const LayoutBParam &layoutB2, const LayoutScaleParam &layoutScale1,
                                                   const LayoutScaleParam &layoutScale2, const LayoutD1Param &layoutD1,
                                                   const LayoutD2Param &layoutD2);

private:
    GM_ADDR xGM_;
    GM_ADDR weight1GM_;
    GM_ADDR weight2GM_;
    GM_ADDR expertIdGM_;
    GM_ADDR scale1GM_;
    GM_ADDR scale2GM_;
    GM_ADDR probs_;
    GM_ADDR xActiveMaskGM_;
    GM_ADDR outGM_;
    GM_ADDR gmExpertTokenNums_;
    GM_ADDR workspaceGM_;

    GM_ADDR moeInitRoutingQuantScale = nullptr;
    GM_ADDR moeInitRoutingQuantOffset = nullptr;
    GM_ADDR expertTokensBeforeCapacity = nullptr;

    int32_t rank;
    int32_t rankSize;
    int32_t aivNum;
    GM_ADDR remoteWindowContext_;

    int32_t m0;
    int32_t k0;
    int32_t n0;
    int32_t swizzlOffset;
    int32_t swizzlDirect;
    int32_t ubMoveNum;
    int32_t pValue;

    int32_t commNpuSplit;
    int32_t commDataSplit;
    int32_t lenPerLoop;

    int32_t m;
    int32_t k;
    int32_t n;
    int32_t topK;
    int32_t expertPerRank;
    int32_t maxOutputSize;
    int32_t EP;
    int32_t listLen;

    optiling::MoeInitRoutingQuantTilingData moeInitRoutingQuantTilingData;
    uint64_t initRoutingQuantTilingKey;

    // Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
};

template <TemplateMMA2AClass>
__aicore__ inline void DispatchCombineMoe<TemplateMMA2ACFunc>::AssignTensorAddresses(
    GM_ADDR xGM, GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR expertIdGM, GM_ADDR scale1GM, GM_ADDR scale2GM,
    GM_ADDR probs, GM_ADDR xActiveMaskGM, GM_ADDR outGM, GM_ADDR expertTokenNums, GM_ADDR workspaceGM)
{
    xGM_ = xGM;
    weight1GM_ = weight1GM;
    weight2GM_ = weight2GM;
    expertIdGM_ = expertIdGM;
    scale1GM_ = scale1GM;
    scale2GM_ = scale2GM;
    probs_ = probs;
    xActiveMaskGM_ = xActiveMaskGM;
    outGM_ = outGM;
    gmExpertTokenNums_ = expertTokenNums;
    workspaceGM_ = workspaceGM;
}

template <TemplateMMA2AClass>
__aicore__ inline void DispatchCombineMoe<TemplateMMA2ACFunc>::AssignDispatchInfo(
    const __gm__ DispatchCombineMoeTilingData *tilingData)
{
    aivNum = tilingData->dispatchCombineMoeInfo.aivNum;
    m = tilingData->dispatchCombineMoeInfo.M;
    k = tilingData->dispatchCombineMoeInfo.K;
    n = tilingData->dispatchCombineMoeInfo.N;
    EP = tilingData->dispatchCombineMoeInfo.worldSize;
    topK = tilingData->dispatchCombineMoeInfo.topK;
    expertPerRank = tilingData->dispatchCombineMoeInfo.expertPerRank;
    maxOutputSize = tilingData->dispatchCombineMoeInfo.maxOutputSize;
    listLen = tilingData->dispatchCombineMoeInfo.listLen;
}

template <TemplateMMA2AClass>
__aicore__ inline void DispatchCombineMoe<TemplateMMA2ACFunc>::AssignCocInfo(
    const __gm__ DispatchCombineMoeTilingData *tilingData)
{
    m0 = tilingData->cocTiling.m0;
    k0 = tilingData->cocTiling.k0;
    n0 = tilingData->cocTiling.n0;
    swizzlDirect = tilingData->cocTiling.swizzleDirect;
    swizzlOffset = tilingData->cocTiling.swizzleOffset;
    ubMoveNum = tilingData->cocTiling.ubMoveNum;
    pValue = tilingData->cocTiling.pValue;
    commNpuSplit = tilingData->cocTiling.commNpuSplit;
    commDataSplit = tilingData->cocTiling.commDataSplit;
    lenPerLoop = tilingData->cocTiling.lenPerLoop;
}

template <TemplateMMA2AClass>
__aicore__ inline void DispatchCombineMoe<TemplateMMA2ACFunc>::AssignVbsTiling(
    const __gm__ optiling::MoeInitRoutingQuantTilingData *src)
{
    moeInitRoutingQuantTilingData.vbsComputeParamsOp.needCoreNum = src->vbsComputeParamsOp.needCoreNum;
    moeInitRoutingQuantTilingData.vbsComputeParamsOp.perCoreElements = src->vbsComputeParamsOp.perCoreElements;
    moeInitRoutingQuantTilingData.vbsComputeParamsOp.perCoreLoops = src->vbsComputeParamsOp.perCoreLoops;
    moeInitRoutingQuantTilingData.vbsComputeParamsOp.perCorePerLoopElements =
        src->vbsComputeParamsOp.perCorePerLoopElements;
    moeInitRoutingQuantTilingData.vbsComputeParamsOp.perCoreLastLoopElements =
        src->vbsComputeParamsOp.perCoreLastLoopElements;
    moeInitRoutingQuantTilingData.vbsComputeParamsOp.lastCoreElements = src->vbsComputeParamsOp.lastCoreElements;
    moeInitRoutingQuantTilingData.vbsComputeParamsOp.lastCoreLoops = src->vbsComputeParamsOp.lastCoreLoops;
    moeInitRoutingQuantTilingData.vbsComputeParamsOp.lastCorePerLoopElements =
        src->vbsComputeParamsOp.lastCorePerLoopElements;
    moeInitRoutingQuantTilingData.vbsComputeParamsOp.lastCoreLastLoopElements =
        src->vbsComputeParamsOp.lastCoreLastLoopElements;
    moeInitRoutingQuantTilingData.vbsComputeParamsOp.oneLoopMaxElements = src->vbsComputeParamsOp.oneLoopMaxElements;
}

template <TemplateMMA2AClass>
__aicore__ inline void DispatchCombineMoe<TemplateMMA2ACFunc>::AssignGatherTiling(
    optiling::InnerMoeGatherOutComputeTilingData &dst, const __gm__ optiling::InnerMoeGatherOutComputeTilingData *src)
{
    dst.needCoreNum = src->needCoreNum;
    dst.activateRows = src->activateRows;
    dst.perCoreRows = src->perCoreRows;
    dst.perCorePerLoopRows = src->perCorePerLoopRows;
    dst.perCoreLastLoopRows = src->perCoreLastLoopRows;
    dst.lastCoreRows = src->lastCoreRows;
    dst.lastCorePerLoopRows = src->lastCorePerLoopRows;
    dst.lastCoreLastLoopRows = src->lastCoreLastLoopRows;
    dst.perCoreLoops = src->perCoreLoops;
    dst.lastCoreLoops = src->lastCoreLoops;
    dst.perLoopCols = src->perLoopCols;
    dst.lastLoopCols = src->lastLoopCols;
    dst.colLoops = src->colLoops;
}

template <TemplateMMA2AClass>
__aicore__ inline void DispatchCombineMoe<TemplateMMA2ACFunc>::AssignRoutingTiling(
    const __gm__ DispatchCombineMoeTilingData *tilingData)
{
    const __gm__ optiling::MoeInitRoutingQuantTilingData *src = &tilingData->cocTiling.moeInitRoutingQuantTilingData;
    moeInitRoutingQuantTilingData.coreNum = src->coreNum;
    moeInitRoutingQuantTilingData.n = src->n;
    moeInitRoutingQuantTilingData.cols = src->cols;
    moeInitRoutingQuantTilingData.k = src->k;
    moeInitRoutingQuantTilingData.expertCapacity = src->expertCapacity;
    moeInitRoutingQuantTilingData.expertNum = src->expertNum;
    moeInitRoutingQuantTilingData.dropPadMode = src->dropPadMode;
    moeInitRoutingQuantTilingData.expertTokensCountOrCumsumFlag = src->expertTokensCountOrCumsumFlag;
    moeInitRoutingQuantTilingData.expertTokensBeforeCapacityFlag = src->expertTokensBeforeCapacityFlag;
    moeInitRoutingQuantTilingData.smoothType = src->smoothType;
    AssignVbsTiling(src);
    moeInitRoutingQuantTilingData.vmsMiddleComputeParamsOp.needCoreNum = src->vmsMiddleComputeParamsOp.needCoreNum;
    moeInitRoutingQuantTilingData.sortOutComputeParamsOp.oneLoopMaxElements =
        src->sortOutComputeParamsOp.oneLoopMaxElements;
    AssignGatherTiling(moeInitRoutingQuantTilingData.srcToDstComputeParamsOp, &src->srcToDstComputeParamsOp);
    AssignGatherTiling(moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp,
                       &src->srcToDstCapacityComputeParamsOp);
    AssignGatherTiling(moeInitRoutingQuantTilingData.gatherOutComputeParamsOp, &src->gatherOutComputeParamsOp);
    initRoutingQuantTilingKey = tilingData->cocTiling.initRoutingQuantTilingKey;
}

template <TemplateMMA2AClass>
__aicore__ inline void DispatchCombineMoe<TemplateMMA2ACFunc>::AssignRuntimeInfo(
    const __gm__ DispatchCombineMoeTilingData *tilingData)
{
    rank = static_cast<int32_t>(tilingData->runtimeInfo.rank);
    rankSize = static_cast<int32_t>(tilingData->runtimeInfo.rankSize);
    remoteWindowContext_ = reinterpret_cast<GM_ADDR>(tilingData->runtimeInfo.remoteWindowContext);
}

template <TemplateMMA2AClass>
__aicore__ inline void DispatchCombineMoe<TemplateMMA2ACFunc>::Init(
    GM_ADDR xGM, GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR expertIdGM, GM_ADDR scale1GM, GM_ADDR scale2GM,
    GM_ADDR probs, GM_ADDR xActiveMaskGM, GM_ADDR outGM, GM_ADDR expertTokenNums, GM_ADDR workspaceGM,
    const __gm__ DispatchCombineMoeTilingData *tilingData)
{
    AssignTensorAddresses(xGM, weight1GM, weight2GM, expertIdGM, scale1GM, scale2GM, probs, xActiveMaskGM, outGM,
                          expertTokenNums, workspaceGM);
    AssignDispatchInfo(tilingData);
    AssignCocInfo(tilingData);
    AssignRoutingTiling(tilingData);
    AssignRuntimeInfo(tilingData);
}

template <TemplateMMA2AClass>
template <typename MatmulKernel, typename LayoutAParam, typename LayoutBParam, typename LayoutScaleParam,
          typename LayoutD1Param, typename LayoutD2Param>
__aicore__ inline void DispatchCombineMoe<TemplateMMA2ACFunc>::BuildAndRunKernelParams(
    const pto_ext::PtoShape3D &problemShape, uint32_t epilogueCoreNum, uint32_t epilogueGranularity,
    const LayoutAParam &layoutA1, const LayoutAParam &layoutA2, const LayoutBParam &layoutB1,
    const LayoutBParam &layoutB2, const LayoutScaleParam &layoutScale1, const LayoutScaleParam &layoutScale2,
    const LayoutD1Param &layoutD1, const LayoutD2Param &layoutD2)
{
    typename MatmulKernel::Params params{problemShape,
                                         static_cast<uint32_t>(EP),
                                         static_cast<uint32_t>(listLen),
                                         static_cast<uint32_t>(expertPerRank),
                                         static_cast<uint32_t>(maxOutputSize),
                                         static_cast<uint32_t>(rank),
                                         static_cast<uint32_t>(rankSize),
                                         ubMoveNum,
                                         remoteWindowContext_,
                                         static_cast<uint32_t>(topK),
                                         initRoutingQuantTilingKey,
                                         epilogueCoreNum,
                                         epilogueGranularity,
                                         xGM_,
                                         layoutA1,
                                         layoutA2,
                                         weight1GM_,
                                         layoutB1,
                                         weight2GM_,
                                         layoutB2,
                                         scale1GM_,
                                         layoutScale1,
                                         scale2GM_,
                                         layoutScale2,
                                         outGM_,
                                         layoutD1,
                                         layoutD2,
                                         expertIdGM_,
                                         moeInitRoutingQuantScale,
                                         moeInitRoutingQuantOffset,
                                         expertTokensBeforeCapacity,
                                         probs_,
                                         workspaceGM_,
                                         gmExpertTokenNums_,
                                         xActiveMaskGM_,
                                         moeInitRoutingQuantTilingData};
    MatmulKernel kernel(params);
    kernel(params);
}

template <TemplateMMA2AClass>
__aicore__ inline void DispatchCombineMoe<TemplateMMA2ACFunc>::Process()
{
    uint32_t k2 = n / 2;
    uint32_t n2 = k;
    ProcessLayoutB layoutB1 = LayoutBInitializer<ProcessLayoutB, BType_>::create(k, n);
    ProcessLayoutB layoutB2 = LayoutBInitializer<ProcessLayoutB, BType_>::create(k2, n2);
    ProcessLayoutA layoutA1{static_cast<uint32_t>(m), static_cast<uint32_t>(k)};
    ProcessLayoutA layoutA2{static_cast<uint32_t>(m), static_cast<uint32_t>(k2)};
    pto_ext::layout::VectorLayout layoutScale1{static_cast<uint32_t>(n)};
    pto_ext::layout::VectorLayout layoutScale2{static_cast<uint32_t>(n2)};
    pto_ext::layout::ND layoutD1{static_cast<uint32_t>(maxOutputSize), static_cast<uint32_t>(k2)};
    pto_ext::layout::ND layoutD2{static_cast<uint32_t>(m * topK), static_cast<uint32_t>(n2)};
    pto_ext::PtoShape3D problemShape{static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k)};

    uint32_t epilogueCoreNum = aivNum;
    uint32_t epilogueGranularity = expertPerRank <= 4 ? expertPerRank - 1 : expertPerRank - 3;
    BuildAndRunKernelParams<ProcessMatmulKernel>(problemShape, epilogueCoreNum, epilogueGranularity, layoutA1, layoutA2,
                                                 layoutB1, layoutB2, layoutScale1, layoutScale2, layoutD1, layoutD2);
}

} // namespace DispatchCombineMoeImpl
#endif // DISPATCH_COMBINE_MOE_H
