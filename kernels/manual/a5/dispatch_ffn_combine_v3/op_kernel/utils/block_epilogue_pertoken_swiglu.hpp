/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PTO_EXT_EPILOGUE_BLOCK_PER_TOKEN_SWIGLU_HPP
#define PTO_EXT_EPILOGUE_BLOCK_PER_TOKEN_SWIGLU_HPP

#include "dispatch_policy_custom.hpp"

#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

#include "pto_vector_ops.hpp"

namespace pto_ext::Epilogue::Block {
namespace swiglu_detail {

template <typename Element, int TileElems = 1024>
using PtoVecTile = pto_ext::dispatch_ffn_combine_v3::pto_bridge::PtoVecTile<Element, TileElems>;

using pto_ext::dispatch_ffn_combine_v3::pto_bridge::PtoAbsVector;
using pto_ext::dispatch_ffn_combine_v3::pto_bridge::PtoAddScalarVector;
using pto_ext::dispatch_ffn_combine_v3::pto_bridge::PtoCastVector;
using pto_ext::dispatch_ffn_combine_v3::pto_bridge::PtoDivVector;
using pto_ext::dispatch_ffn_combine_v3::pto_bridge::PtoExpVector;
using pto_ext::dispatch_ffn_combine_v3::pto_bridge::PtoGetValue;
using pto_ext::dispatch_ffn_combine_v3::pto_bridge::PtoLoadVector;
using pto_ext::dispatch_ffn_combine_v3::pto_bridge::PtoMulElementwiseVector;
using pto_ext::dispatch_ffn_combine_v3::pto_bridge::PtoMulVector;
using pto_ext::dispatch_ffn_combine_v3::pto_bridge::PtoReduceMaxVector;
using pto_ext::dispatch_ffn_combine_v3::pto_bridge::PtoSetValue;
using pto_ext::dispatch_ffn_combine_v3::pto_bridge::PtoStoreVector;

template <auto Pipe>
PTO_DEVICE void PtoPipeBarrier()
{
    AscendC::PipeBarrier<Pipe>();
}

template <AscendC::HardEvent Event>
PTO_DEVICE void PtoSetFlag(int32_t eventId)
{
    AscendC::SetFlag<Event>(eventId);
}

template <AscendC::HardEvent Event>
PTO_DEVICE void PtoWaitFlag(int32_t eventId)
{
    AscendC::WaitFlag<Event>(eventId);
}

}  // namespace swiglu_detail


// float scale, dequant per expert
template <
    uint32_t UB_STAGES_,
    class CType_,
    class LayoutPerTokenScale_,
    class DType_,
    class TileElemWiseMuls_,
    class TileCopy_
>
class BlockEpilogue <
    EpilogueAtlasA5PerTokenDequantSwigluQuant<UB_STAGES_>,
    CType_,
    Gemm::GemmType<float, LayoutPerTokenScale_>,
    DType_,
    TileElemWiseMuls_,
    TileCopy_
> {
public:
    using DispatchPolicy = EpilogueAtlasA5PerTokenDequantSwigluQuant<UB_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;

    // Data infos
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = LayoutPerTokenScale_;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;

    // Check data infos
    static_assert(
        std::is_same_v<ElementC, half> && (std::is_same_v<ElementD, float> || std::is_same_v<ElementD, int8_t>),
        "The element type template parameters of BlockEpilogue are wrong"
    );
    static_assert(
        std::is_same_v<LayoutC, layout::ND> &&
            std::is_same_v<LayoutPerTokenScale, layout::VectorLayout> && std::is_same_v<LayoutD, layout::ND>,
        "The layout template parameters of BlockEpilogue are wrong"
    );

    struct Params {
        __gm__ ElementPerTokenScale *ptrPerTokenScale{nullptr};
        LayoutPerTokenScale layoutPerTokenScale{};
        __gm__ ElementD *ptrD{nullptr};
        LayoutD layoutD{};

        PTO_DEVICE
        Params() {};

        PTO_DEVICE
        Params(__gm__ ElementPerTokenScale *ptrPerTokenScale_, LayoutPerTokenScale const &layoutPerTokenScale_,
            __gm__ ElementD *ptrD_, LayoutD const &layoutD_
        ) : ptrPerTokenScale(ptrPerTokenScale_), layoutPerTokenScale(layoutPerTokenScale_),
            ptrD(ptrD_), layoutD(layoutD_) {}
    };

    PTO_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource, int32_t n, Params const &params = Params{}) : params(params)
    {
        size_t ubOffset = 0;
        int32_t eventVMTE2 = 0;
        int32_t eventMTE2V = 0;
        int32_t eventMTE3V = 0;
        int32_t eventVMTE3 = 0;
        uint32_t blockN = n;
        uint32_t ChunkTileLen = blockN / 2;
        uint32_t HalfChunkTileLen = ChunkTileLen / 2;

        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubCOffsetList[i] = resource.ubBuf.GetBufferAddrByByte(ubOffset);
            ubOffset += blockN * sizeof(ElementC);
            ubDOffsetList[i] = resource.ubBuf.GetBufferAddrByByte(ubOffset);
            ubOffset += blockN * sizeof(ElementD);
            ubCFp32OffsetList[i] = resource.ubBuf.GetBufferAddrByByte(ubOffset);
            ubOffset += blockN * sizeof(float);
            ubCFp32ChunkNOffsetList[i] = resource.ubBuf.GetBufferAddrByByte(ubOffset);
            ubOffset += ChunkTileLen * sizeof(float);
            ubCFp32ChunkNAbsOffsetList[i] = resource.ubBuf.GetBufferAddrByByte(ubOffset);
            ubOffset += ChunkTileLen * sizeof(float);
            ubCFp32ChunkNMaxOffsetList[i] = resource.ubBuf.GetBufferAddrByByte(ubOffset);
            ubOffset += HalfChunkTileLen * sizeof(float);

            eventUbCVMTE2List[i] = eventVMTE2++;
            eventUbCMTE2VList[i] = eventMTE2V++;
            eventUbDMTE3VList[i] = eventMTE3V++;
            eventUbDVMTE3List[i] = eventVMTE3++;

            swiglu_detail::PtoSetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            swiglu_detail::PtoSetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }

        ubPerTokenScaleOutputOffset = resource.ubBuf.GetBufferAddrByByte(ubOffset);
    }
    PTO_DEVICE
    void Finalize()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            swiglu_detail::PtoWaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            swiglu_detail::PtoWaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }
    PTO_DEVICE
    ~BlockEpilogue()
    {
    }

    PTO_DEVICE
    void UpdateParams(Params const &params_)
    {
        params = params_;
    }
    // Each tile is 1x7168, and each block covers all tokens for one expert = [group[i], 7168]
    template <typename CallbackT = pto_ext::support::NoopCallback>
    PTO_DEVICE
    void operator() (
        __gm__ ElementC *gmCPtr,
        PtoShape2D const &shapeC,
        __gm__ ElementPerTokenScale *gmPerTokenScale1Ptr,
        __gm__ ElementD *gmDPtr,
        __gm__ ElementPerTokenScale *gmPerTokenScale2Ptr,

        uint32_t epilogueCoreNum = 40,
        CallbackT &&callback = CallbackT{}
    )
    {
        callback();
        uint32_t blockM = static_cast<uint32_t>(shapeC.shape[0]);
        uint32_t blockN = static_cast<uint32_t>(shapeC.shape[1]);

        uint32_t tileLoops = blockM;
        uint32_t subblockIdx = get_block_idx() + get_subblockid() * get_block_num();

        uint32_t subblockNum = get_block_num() * 2;
        uint32_t moveDataCoreNum = subblockNum - epilogueCoreNum;

        if (subblockIdx < moveDataCoreNum) {
            return;
        }
        uint32_t epilogueCoreIdx = subblockIdx - moveDataCoreNum;

        uint32_t perCoreData =  blockM / epilogueCoreNum;
        uint32_t remainderData = blockM % epilogueCoreNum;

        uint32_t tasksForIdx  = epilogueCoreIdx < remainderData ? perCoreData + 1 : perCoreData;
        uint32_t loopStartIdx = epilogueCoreIdx * perCoreData + (epilogueCoreIdx < remainderData? epilogueCoreIdx : remainderData);

        uint32_t alignedPerCoreData = RoundUp<BYTE_PER_BLK / sizeof(ElementPerTokenScale)>(perCoreData + 1);

        uint32_t ChunkTileLen = blockN / 2;
        uint32_t HalfChunkTileLen = ChunkTileLen / 2;


        for (uint32_t loopIdx = loopStartIdx; loopIdx < loopStartIdx + tasksForIdx; ++loopIdx) {

            __gm__ ElementC *gmTileC = gmCPtr + loopIdx * blockN;

            auto ubCOffset = ubCOffsetList[ubListId];
            auto ubDOffset = ubDOffsetList[ubListId];
            auto ubCFp32Offset = ubCFp32OffsetList[ubListId];
            auto ubCFp32ChunkNOffset = ubCFp32ChunkNOffsetList[ubListId];
            auto ubAbsOffset = ubCFp32ChunkNAbsOffsetList[ubListId];
            auto ubReduceMaxOffset = ubCFp32ChunkNMaxOffsetList[ubListId];
            auto ubOutputTmpOffset = ubAbsOffset;
            auto ubQuantScratchOffset = ubAbsOffset;

            __gm__ ElementD *gmTileD = gmDPtr + loopIdx * ChunkTileLen;
            // Move C from GM workspace to UB
            swiglu_detail::PtoWaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
            swiglu_detail::PtoLoadVector<ElementC>(ubCOffset, gmTileC, blockN);
            swiglu_detail::PtoSetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);

            // Cast C to FP32 in UB
            swiglu_detail::PtoWaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
            swiglu_detail::PtoCastVector<float, ElementC>(ubCFp32Offset, ubCOffset, blockN, pto::RoundMode::CAST_NONE);
            swiglu_detail::PtoSetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);

            // Get per-token scale from row loopIdx of gmPerTokenScale
            ElementPerTokenScale perTokenScale = gm_load(gmPerTokenScale1Ptr + loopIdx);

            swiglu_detail::PtoSetFlag<AscendC::HardEvent::S_V>(0);
            swiglu_detail::PtoWaitFlag<AscendC::HardEvent::S_V>(0);
            // Multiply FP32 C by the per-token scale
            swiglu_detail::PtoPipeBarrier<PIPE_V>();
            swiglu_detail::PtoMulVector(ubCFp32Offset, ubCFp32Offset, blockN, perTokenScale);
            swiglu_detail::PtoPipeBarrier<PIPE_V>();

            // Swiglu computation process
            swiglu_detail::PtoMulVector(ubCFp32ChunkNOffset, ubCFp32Offset, ChunkTileLen, -1.0f);
            swiglu_detail::PtoPipeBarrier<PIPE_V>();
            swiglu_detail::PtoExpVector<float>(ubCFp32ChunkNOffset, ubCFp32ChunkNOffset, ChunkTileLen);
            swiglu_detail::PtoPipeBarrier<PIPE_V>();
            swiglu_detail::PtoAddScalarVector(ubCFp32ChunkNOffset, ubCFp32ChunkNOffset, ChunkTileLen, 1.0f);
            swiglu_detail::PtoPipeBarrier<PIPE_V>();
            swiglu_detail::PtoDivVector<float>(ubCFp32ChunkNOffset, ubCFp32Offset, ubCFp32ChunkNOffset, ChunkTileLen);
            swiglu_detail::PtoPipeBarrier<PIPE_V>();
            swiglu_detail::PtoMulElementwiseVector<float>(ubCFp32ChunkNOffset, ubCFp32ChunkNOffset, ubCFp32Offset + static_cast<uint64_t>(ChunkTileLen) * sizeof(float), ChunkTileLen);

            // Quantization process; difference between the two approaches
            swiglu_detail::PtoPipeBarrier<PIPE_V>();
            swiglu_detail::PtoAbsVector<float>(ubAbsOffset, ubCFp32ChunkNOffset, ChunkTileLen);
            swiglu_detail::PtoPipeBarrier<PIPE_V>();

            swiglu_detail::PtoReduceMaxVector(ubReduceMaxOffset, ubAbsOffset, ubAbsOffset, ChunkTileLen);
            swiglu_detail::PtoPipeBarrier<PIPE_V>();

            swiglu_detail::PtoSetFlag<AscendC::HardEvent::V_S>(0);
            swiglu_detail::PtoWaitFlag<AscendC::HardEvent::V_S>(0);

            ElementPerTokenScale GMubDequantScale = swiglu_detail::PtoGetValue<ElementPerTokenScale>(ubReduceMaxOffset, 0);
            swiglu_detail::PtoSetFlag<AscendC::HardEvent::S_V>(0);

            auto ubPerTokenScaleOutputElemOffset = loopIdx - loopStartIdx;
            swiglu_detail::PtoSetValue<ElementPerTokenScale>(ubPerTokenScaleOutputOffset, ubPerTokenScaleOutputElemOffset, GMubDequantScale / 127.f);

            swiglu_detail::PtoWaitFlag<AscendC::HardEvent::S_V>(0);
            swiglu_detail::PtoMulVector(ubOutputTmpOffset, ubCFp32ChunkNOffset, ChunkTileLen, 127.f / GMubDequantScale);
            swiglu_detail::PtoPipeBarrier<PIPE_V>();

            swiglu_detail::PtoCastVector<int32_t, float>(ubQuantScratchOffset, ubOutputTmpOffset, ChunkTileLen, pto::RoundMode::CAST_RINT);
            swiglu_detail::PtoPipeBarrier<PIPE_V>();
            AscendC::SetDeqScale(static_cast<half>(1.0));
            swiglu_detail::PtoCastVector<half, int32_t>(ubQuantScratchOffset, ubQuantScratchOffset, ChunkTileLen, pto::RoundMode::CAST_RINT);
            swiglu_detail::PtoPipeBarrier<PIPE_V>();

            swiglu_detail::PtoWaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDVMTE3List[ubListId]);
            swiglu_detail::PtoCastVector<ElementD, half>(ubDOffset, ubQuantScratchOffset, ChunkTileLen, pto::RoundMode::CAST_RINT);
            swiglu_detail::PtoSetFlag<AscendC::HardEvent::V_MTE3>(eventUbDMTE3VList[ubListId]);

            swiglu_detail::PtoWaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
            swiglu_detail::PtoStoreVector<ElementD>(gmTileD, ubDOffset, ChunkTileLen);
            swiglu_detail::PtoSetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
        }

        if(tasksForIdx > 0){
            swiglu_detail::PtoSetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            swiglu_detail::PtoWaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);

            swiglu_detail::PtoStoreVector<ElementPerTokenScale>(gmPerTokenScale2Ptr + loopStartIdx, ubPerTokenScaleOutputOffset, tasksForIdx);
        }


    }

private:
    Params params;

    uint64_t ubCOffsetList[UB_STAGES];
    uint64_t ubDOffsetList[UB_STAGES];

    int32_t eventUbCVMTE2List[UB_STAGES];
    int32_t eventUbCMTE2VList[UB_STAGES];
    int32_t eventUbDMTE3VList[UB_STAGES];
    int32_t eventUbDVMTE3List[UB_STAGES];

    uint32_t ubListId{0};

    uint64_t ubCFp32OffsetList[UB_STAGES];
    uint64_t ubCFp32ChunkNOffsetList[UB_STAGES];
    uint64_t ubCFp32ChunkNAbsOffsetList[UB_STAGES];
    uint64_t ubCFp32ChunkNMaxOffsetList[UB_STAGES];
    uint64_t ubPerTokenScaleOutputOffset{0};

};

}  // namespace pto_ext::Epilogue::Block

#endif  // PTO_EXT_EPILOGUE_BLOCK_PER_TOKEN_SWIGLU_HPP
