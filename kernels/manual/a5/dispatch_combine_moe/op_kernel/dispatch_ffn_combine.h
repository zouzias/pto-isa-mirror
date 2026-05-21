/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dispatch_ffn_combine.h
 * \brief
 */

#ifndef DISPATCH_FFN_COMBINE_H
#define DISPATCH_FFN_COMBINE_H

using namespace AscendC;

#include "kernel_operator.h"

#include "dispatch_ffn_combine_tiling.h"

#include "utils/dispatch_policy_custom.hpp"

#include "utils/const_args.hpp"
#include "dispatch_ffn_combine_kernel.hpp"
#include "moe_init_routing_quant/moe_init_routing_quant_tiling.h"

namespace DispatchFFNCombineImpl {
#define TemplateMMA2AClass typename AType_, typename BType_, typename CType_, bool TB_, bool Nz_
#define TemplateMMA2ACFunc AType_, BType_, CType_, TB_, Nz_

using namespace AscendC;

template <typename Layout, typename ElementType, typename = void>
struct LayoutBInitializer {
    PTO_DEVICE
    static Layout create(uint32_t k, uint32_t n)
    {
        return Layout{k, n};
    }
};

template <typename Layout, typename ElementType>
struct LayoutBInitializer<Layout, ElementType, std::enable_if_t<Layout::kTileLayout == pto::TileLayoutCustom::ZN>> {
    PTO_DEVICE
    static Layout create(uint32_t k, uint32_t n)
    {
        return Layout::template MakeLayout<ElementType>(k, n);
    }
};

template <TemplateMMA2AClass>
class DispatchFFNCombine {
public:
    __aicore__ inline DispatchFFNCombine(){};
    __aicore__ inline void Init(GM_ADDR xGM, GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR expertIdGM, GM_ADDR scale1GM,
                                GM_ADDR scale2GM, GM_ADDR probs, GM_ADDR xActiveMaskGM, GM_ADDR outGM,
                                GM_ADDR expertTokenNums, GM_ADDR workspaceGM,
                                const __gm__ DispatchFFNCombineTilingData *tilingData);
    __aicore__ inline void Process();

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
__aicore__ inline void DispatchFFNCombine<TemplateMMA2ACFunc>::Init(
    GM_ADDR xGM, GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR expertIdGM, GM_ADDR scale1GM, GM_ADDR scale2GM,
    GM_ADDR probs, GM_ADDR xActiveMaskGM, GM_ADDR outGM, GM_ADDR expertTokenNums, GM_ADDR workspaceGM,
    const __gm__ DispatchFFNCombineTilingData *tilingData)
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

    aivNum = tilingData->dispatchFFNCombineInfo.aivNum;

    m = tilingData->dispatchFFNCombineInfo.M;
    k = tilingData->dispatchFFNCombineInfo.K;
    n = tilingData->dispatchFFNCombineInfo.N;
    EP = tilingData->dispatchFFNCombineInfo.worldSize;
    topK = tilingData->dispatchFFNCombineInfo.topK;
    expertPerRank = tilingData->dispatchFFNCombineInfo.expertPerRank;
    maxOutputSize = tilingData->dispatchFFNCombineInfo.maxOutputSize;
    listLen = tilingData->dispatchFFNCombineInfo.listLen;

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
    moeInitRoutingQuantTilingData.coreNum = tilingData->cocTiling.moeInitRoutingQuantTilingData.coreNum;
    moeInitRoutingQuantTilingData.n = tilingData->cocTiling.moeInitRoutingQuantTilingData.n;
    moeInitRoutingQuantTilingData.cols = tilingData->cocTiling.moeInitRoutingQuantTilingData.cols;
    moeInitRoutingQuantTilingData.k = tilingData->cocTiling.moeInitRoutingQuantTilingData.k;
    moeInitRoutingQuantTilingData.expertCapacity =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.expertCapacity;
    moeInitRoutingQuantTilingData.expertNum = tilingData->cocTiling.moeInitRoutingQuantTilingData.expertNum;
    moeInitRoutingQuantTilingData.dropPadMode = tilingData->cocTiling.moeInitRoutingQuantTilingData.dropPadMode;
    moeInitRoutingQuantTilingData.expertTokensCountOrCumsumFlag =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.expertTokensCountOrCumsumFlag;
    moeInitRoutingQuantTilingData.expertTokensBeforeCapacityFlag =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.expertTokensBeforeCapacityFlag;
    moeInitRoutingQuantTilingData.smoothType = tilingData->cocTiling.moeInitRoutingQuantTilingData.smoothType;
    moeInitRoutingQuantTilingData.vbsComputeParamsOp.needCoreNum =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.vbsComputeParamsOp.needCoreNum;
    moeInitRoutingQuantTilingData.vbsComputeParamsOp.perCoreElements =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.vbsComputeParamsOp.perCoreElements;
    moeInitRoutingQuantTilingData.vbsComputeParamsOp.perCoreLoops =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.vbsComputeParamsOp.perCoreLoops;
    moeInitRoutingQuantTilingData.vbsComputeParamsOp.perCorePerLoopElements =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.vbsComputeParamsOp.perCorePerLoopElements;
    moeInitRoutingQuantTilingData.vbsComputeParamsOp.perCoreLastLoopElements =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.vbsComputeParamsOp.perCoreLastLoopElements;
    moeInitRoutingQuantTilingData.vbsComputeParamsOp.lastCoreElements =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.vbsComputeParamsOp.lastCoreElements;
    moeInitRoutingQuantTilingData.vbsComputeParamsOp.lastCoreLoops =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.vbsComputeParamsOp.lastCoreLoops;
    moeInitRoutingQuantTilingData.vbsComputeParamsOp.lastCorePerLoopElements =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.vbsComputeParamsOp.lastCorePerLoopElements;
    moeInitRoutingQuantTilingData.vbsComputeParamsOp.lastCoreLastLoopElements =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.vbsComputeParamsOp.lastCoreLastLoopElements;
    moeInitRoutingQuantTilingData.vbsComputeParamsOp.oneLoopMaxElements =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.vbsComputeParamsOp.oneLoopMaxElements;
    moeInitRoutingQuantTilingData.vmsMiddleComputeParamsOp.needCoreNum =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.vmsMiddleComputeParamsOp.needCoreNum;
    moeInitRoutingQuantTilingData.sortOutComputeParamsOp.oneLoopMaxElements =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.sortOutComputeParamsOp.oneLoopMaxElements;
    moeInitRoutingQuantTilingData.srcToDstComputeParamsOp.needCoreNum =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.srcToDstComputeParamsOp.needCoreNum;
    moeInitRoutingQuantTilingData.srcToDstComputeParamsOp.activateRows =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.srcToDstComputeParamsOp.activateRows;
    moeInitRoutingQuantTilingData.srcToDstComputeParamsOp.perCoreRows =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.srcToDstComputeParamsOp.perCoreRows;
    moeInitRoutingQuantTilingData.srcToDstComputeParamsOp.perCorePerLoopRows =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.srcToDstComputeParamsOp.perCorePerLoopRows;
    moeInitRoutingQuantTilingData.srcToDstComputeParamsOp.perCoreLastLoopRows =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.srcToDstComputeParamsOp.perCoreLastLoopRows;
    moeInitRoutingQuantTilingData.srcToDstComputeParamsOp.lastCoreRows =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.srcToDstComputeParamsOp.lastCoreRows;
    moeInitRoutingQuantTilingData.srcToDstComputeParamsOp.lastCorePerLoopRows =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.srcToDstComputeParamsOp.lastCorePerLoopRows;
    moeInitRoutingQuantTilingData.srcToDstComputeParamsOp.lastCoreLastLoopRows =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.srcToDstComputeParamsOp.lastCoreLastLoopRows;
    moeInitRoutingQuantTilingData.srcToDstComputeParamsOp.perCoreLoops =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.srcToDstComputeParamsOp.perCoreLoops;
    moeInitRoutingQuantTilingData.srcToDstComputeParamsOp.lastCoreLoops =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.srcToDstComputeParamsOp.lastCoreLoops;
    moeInitRoutingQuantTilingData.srcToDstComputeParamsOp.perLoopCols =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.srcToDstComputeParamsOp.perLoopCols;
    moeInitRoutingQuantTilingData.srcToDstComputeParamsOp.lastLoopCols =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.srcToDstComputeParamsOp.lastLoopCols;
    moeInitRoutingQuantTilingData.srcToDstComputeParamsOp.colLoops =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.srcToDstComputeParamsOp.colLoops;
    moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp.needCoreNum =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp.needCoreNum;
    moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp.activateRows =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp.activateRows;
    moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp.perCoreRows =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp.perCoreRows;
    moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp.perCorePerLoopRows =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp.perCorePerLoopRows;
    moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp.perCoreLastLoopRows =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp.perCoreLastLoopRows;
    moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp.lastCoreRows =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp.lastCoreRows;
    moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp.lastCorePerLoopRows =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp.lastCorePerLoopRows;
    moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp.lastCoreLastLoopRows =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp.lastCoreLastLoopRows;
    moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp.perCoreLoops =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp.perCoreLoops;
    moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp.lastCoreLoops =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp.lastCoreLoops;
    moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp.perLoopCols =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp.perLoopCols;
    moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp.lastLoopCols =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp.lastLoopCols;
    moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp.colLoops =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.srcToDstCapacityComputeParamsOp.colLoops;
    moeInitRoutingQuantTilingData.gatherOutComputeParamsOp.needCoreNum =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.gatherOutComputeParamsOp.needCoreNum;
    moeInitRoutingQuantTilingData.gatherOutComputeParamsOp.activateRows =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.gatherOutComputeParamsOp.activateRows;
    moeInitRoutingQuantTilingData.gatherOutComputeParamsOp.perCoreRows =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.gatherOutComputeParamsOp.perCoreRows;
    moeInitRoutingQuantTilingData.gatherOutComputeParamsOp.perCorePerLoopRows =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.gatherOutComputeParamsOp.perCorePerLoopRows;
    moeInitRoutingQuantTilingData.gatherOutComputeParamsOp.perCoreLastLoopRows =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.gatherOutComputeParamsOp.perCoreLastLoopRows;
    moeInitRoutingQuantTilingData.gatherOutComputeParamsOp.lastCoreRows =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.gatherOutComputeParamsOp.lastCoreRows;
    moeInitRoutingQuantTilingData.gatherOutComputeParamsOp.lastCorePerLoopRows =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.gatherOutComputeParamsOp.lastCorePerLoopRows;
    moeInitRoutingQuantTilingData.gatherOutComputeParamsOp.lastCoreLastLoopRows =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.gatherOutComputeParamsOp.lastCoreLastLoopRows;
    moeInitRoutingQuantTilingData.gatherOutComputeParamsOp.perCoreLoops =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.gatherOutComputeParamsOp.perCoreLoops;
    moeInitRoutingQuantTilingData.gatherOutComputeParamsOp.lastCoreLoops =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.gatherOutComputeParamsOp.lastCoreLoops;
    moeInitRoutingQuantTilingData.gatherOutComputeParamsOp.perLoopCols =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.gatherOutComputeParamsOp.perLoopCols;
    moeInitRoutingQuantTilingData.gatherOutComputeParamsOp.lastLoopCols =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.gatherOutComputeParamsOp.lastLoopCols;
    moeInitRoutingQuantTilingData.gatherOutComputeParamsOp.colLoops =
        tilingData->cocTiling.moeInitRoutingQuantTilingData.gatherOutComputeParamsOp.colLoops;
    initRoutingQuantTilingKey = tilingData->cocTiling.initRoutingQuantTilingKey;

    rank = static_cast<int32_t>(tilingData->runtimeInfo.rank);
    rankSize = static_cast<int32_t>(tilingData->runtimeInfo.rankSize);
    remoteWindowContext_ = reinterpret_cast<GM_ADDR>(tilingData->runtimeInfo.remoteWindowContext);
}

template <TemplateMMA2AClass>
__aicore__ inline void DispatchFFNCombine<TemplateMMA2ACFunc>::Process()
{
    // Define ArchTag
    using ArchTag = pto_ext::Arch::AtlasA5;
    constexpr bool enableUnitFlag = false;
    constexpr bool enableShuffleK = true;

    uint32_t k2 = n / 2;
    uint32_t n2 = k;

    int64_t activeNum = 0;
    int64_t expertCapacity = 0;
    int64_t expertNum = expertPerRank * EP;
    int64_t dropPadMode = 0;
    int64_t expertTokensCountOrCumsumFlag = 2;
    bool expertTokensBeforeCapacityFlag = false;
    int64_t quantMode = 1;

    using LayoutA = pto_ext::layout::ND;
    using LayoutB =
        typename std::conditional<Nz_, pto_ext::layout::Zn,
                                  typename std::conditional<TB_, pto_ext::layout::DN, pto_ext::layout::ND>::type>::type;

    LayoutB layoutB1 = LayoutBInitializer<LayoutB, BType_>::create(k, n);
    LayoutB layoutB2 = LayoutBInitializer<LayoutB, BType_>::create(k2, n2);
    using LayoutC = pto_ext::layout::ND;
    using L1TileShape = pto_ext::GemmShape<128, 256, 512>; // M, N, K

    constexpr uint32_t workspaceStages = 2;
    constexpr uint32_t preloadStages = 1;
    constexpr uint32_t l1Stages = 2;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;

    using DispatchPolicy = pto_ext::Gemm::MmadAtlasA5PreloadAsyncFixpipe<preloadStages, l1Stages, l0AStages, l0BStages,
                                                                         l0CStages, enableUnitFlag, enableShuffleK>;

    using L0TileShape = pto_ext::GemmShape<128, 256, 128>;
    using AType = pto_ext::Gemm::GemmType<int8_t, pto_ext::layout::ND>;
    using BType = pto_ext::Gemm::GemmType<int8_t, LayoutB>;
    using CType = pto_ext::Gemm::GemmType<float16_t, pto_ext::layout::ND>;
    using D1Type = pto_ext::Gemm::GemmType<int8_t, pto_ext::layout::ND>;

    using D2Type = typename std::conditional<std::is_same_v<CType_, bfloat16_t>,
                                             pto_ext::Gemm::GemmType<bfloat16_t, pto_ext::layout::ND>,
                                             pto_ext::Gemm::GemmType<CType_, pto_ext::layout::ND> >::type;

    using BlockMmad = pto_ext::Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    constexpr uint32_t ubStages = 2;

    using EpilogueDispatchPolicy1 = pto_ext::Epilogue::EpilogueAtlasA5PerTokenDequantSwigluQuant<ubStages>;

    using ScaleType = pto_ext::Gemm::GemmType<uint64_t, pto_ext::layout::VectorLayout>;
    using PerTokenScaleType = pto_ext::Gemm::GemmType<float, pto_ext::layout::VectorLayout>;
    using ElementMulType = pto_ext::Gemm::GemmType<float, pto_ext::layout::ND>;
    using TileElemWiseMuls = pto_ext::Epilogue::Tile::TileElemWiseMuls<ArchTag, ElementMulType, 0>;

    using TileCopy1 = pto_ext::Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, D1Type>;
    using BlockEpilogue1 = pto_ext::Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy1, CType, PerTokenScaleType,
                                                                   D1Type, TileElemWiseMuls, TileCopy1>;

    using EpilogueDispatchPolicy2 = pto_ext::Epilogue::EpilogueAtlasA5PerTokenDequant<ubStages>;
    using EpilogueDispatchPolicy3 = pto_ext::Epilogue::EpilogueAtlasA5PerTokenDequantV2<ubStages>;

    using TileCopy2 = pto_ext::Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, D2Type>;
    using BlockEpilogue2 =
        pto_ext::Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy2, CType, PerTokenScaleType, D2Type, TileCopy2>;
    using BlockEpilogue3 =
        pto_ext::Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy3, CType, PerTokenScaleType, D2Type, TileCopy2>;

    using BlockScheduler = typename pto_ext::Gemm::Block::GemmIdentityBlockSwizzle<9, 1>;
    using ElementGroupList = int64_t;
    using MatmulKernel =
        pto_ext::Gemm::Kernel::DispatchFFNCombineKernel<BlockMmad, BlockScheduler, ElementGroupList, BlockEpilogue1,
                                                        BlockEpilogue2, BlockEpilogue3>;

    LayoutA layoutA1{static_cast<uint32_t>(m), static_cast<uint32_t>(k)};
    LayoutA layoutA2{static_cast<uint32_t>(m), static_cast<uint32_t>(k2)};
    pto_ext::layout::VectorLayout layoutScale1{static_cast<uint32_t>(n)};
    pto_ext::layout::VectorLayout layoutScale2{static_cast<uint32_t>(n2)};
    pto_ext::layout::ND layoutD1{static_cast<uint32_t>(maxOutputSize), static_cast<uint32_t>(k2)};
    pto_ext::layout::ND layoutD2{static_cast<uint32_t>(m * topK), static_cast<uint32_t>(n2)};
    // Prepare params

    pto_ext::PtoShape3D problemShape =
        pto_ext::MakePtoShape3D(static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k));

    uint32_t epilogueCoreNum = aivNum;
    uint32_t epilogueGranularity = expertPerRank - 3;
    if (expertPerRank <= 4) {
        epilogueGranularity = expertPerRank - 1;
    }
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
    // Call kernel
    MatmulKernel kernel(params);
    kernel(params);
}

} // namespace DispatchFFNCombineImpl
#endif // DISPATCH_FFN_COMBINE_H