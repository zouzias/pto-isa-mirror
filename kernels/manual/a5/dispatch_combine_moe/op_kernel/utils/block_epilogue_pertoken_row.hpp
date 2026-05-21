/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PTO_EXT_EPILOGUE_BLOCK_PER_TOKEN_ROW_HPP
#define PTO_EXT_EPILOGUE_BLOCK_PER_TOKEN_ROW_HPP

#include "dispatch_policy_custom.hpp"

#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

#include "hccl_window.hpp"
#include "pto_vector_ops.hpp"

namespace pto_ext::Epilogue::Block {
namespace row_detail {

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

using pto_ext::dispatch_combine_moe::pto_bridge::PtoCastVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoLoadVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoMulVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoStoreVector;

} // namespace row_detail

// float scale, dequant per expert
template <uint32_t UB_STAGES_, class CType_, class LayoutPerTokenScale_, class DType_, class TileCopy_>
class BlockEpilogue<EpilogueAtlasA5PerTokenDequant<UB_STAGES_>, CType_, Gemm::GemmType<float, LayoutPerTokenScale_>,
                    DType_, TileCopy_> {
public:
    using DispatchPolicy = EpilogueAtlasA5PerTokenDequant<UB_STAGES_>;
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
    static_assert(std::is_same_v<ElementC, half> &&
                      (std::is_same_v<ElementD, half> || std::is_same_v<ElementD, bfloat16_t>),
                  "The element type template parameters of BlockEpilogue are wrong");
    static_assert(std::is_same_v<LayoutC, layout::ND> && std::is_same_v<LayoutPerTokenScale, layout::VectorLayout> &&
                      std::is_same_v<LayoutD, layout::ND>,
                  "The layout template parameters of BlockEpilogue are wrong");

    struct Params {
        __gm__ int32_t *ptrTokenPerExpert{nullptr};
        int32_t EP;
        int32_t expertPerRank;
        int32_t n2;
        int32_t rank;
        PtoRemoteWindow remoteWindow;
        int32_t scratchOffset;

        PTO_DEVICE
        Params(){};

        PTO_DEVICE
        Params(int32_t EP_, int32_t expertPerRank_, __gm__ int32_t *ptrTokenPerExpert_, int32_t n2_, int32_t rank_,
               PtoRemoteWindow &remoteWindow_, int32_t scratchOffset_)
            : ptrTokenPerExpert(ptrTokenPerExpert_),
              EP(EP_),
              expertPerRank(expertPerRank_),
              n2(n2_),
              rank(rank_),
              remoteWindow(remoteWindow_),
              scratchOffset(scratchOffset_)
        {}
    };

    PTO_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource, Params const &params = Params{}) : params(params)
    {
        size_t ubOffset = 0;
        int32_t eventVMTE2 = 0;
        int32_t eventMTE2V = 0;
        int32_t eventMTE3V = 0;
        int32_t eventVMTE3 = 0;
        int32_t blockN = params.n2;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubCOffsetList[i] = ubOffset;
            ubOffset += blockN * sizeof(ElementC);
            ubDOffsetList[i] = ubOffset;
            ubOffset += blockN * sizeof(ElementD);

            eventUbCVMTE2List[i] = eventVMTE2++;
            eventUbCMTE2VList[i] = eventMTE2V++;
            eventUbDMTE3VList[i] = eventMTE3V++;
            eventUbDVMTE3List[i] = eventVMTE3++;

            ubCFp32OffsetList[i] = ubOffset;
            ubOffset += blockN * sizeof(float);
        }
    }
    PTO_DEVICE
    void SetFlag()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            row_detail::PtoSetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            row_detail::PtoSetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }

    PTO_DEVICE
    void Finalize()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            row_detail::PtoWaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            row_detail::PtoWaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }
    PTO_DEVICE
    ~BlockEpilogue()
    {}

    PTO_DEVICE
    void UpdateParams(Params const &params_)
    {
        params = params_;
    }

    PTO_DEVICE
    void operator()(__gm__ ElementC *gmCPtr, PtoShape2D const &shapeC, __gm__ ElementPerTokenScale *gmPerTokenScalePtr,
                    __gm__ ElementD *ptrD, int32_t dstRank)
    {
        using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using TputGlobal = pto::GlobalTensor<ElementD, ShapeDyn, StrideDyn, pto::Layout::ND>;
        using TputTile = pto::Tile<pto::TileType::Vec, ElementD, 1, 1024, pto::BLayout::RowMajor, -1, -1>;

        uint32_t blockM = static_cast<uint32_t>(shapeC.shape[0]);
        uint32_t blockN = static_cast<uint32_t>(shapeC.shape[1]);
        uint32_t tileLoops = blockM;
        constexpr uint32_t scratchCols = 1024;
        int32_t logicalSubCoreIdx = get_block_idx() + get_subblockid() * get_block_num();
        int64_t scratchOffsetBytes =
            params.scratchOffset + static_cast<int64_t>(logicalSubCoreIdx) * scratchCols * sizeof(ElementD);
        __gm__ ElementD *localScratch =
            reinterpret_cast<__gm__ ElementD *>(params.remoteWindow(scratchOffsetBytes, params.rank));

        for (uint32_t loopIdx = 0; loopIdx < tileLoops; loopIdx++) {
            __gm__ ElementC *gmTileC = gmCPtr + loopIdx * blockN;
            uint64_t ubCOffset = ubCOffsetList[ubListId];
            uint64_t ubCFp32Offset = ubCFp32OffsetList[ubListId];
            uint64_t ubDOffset = ubDOffsetList[ubListId];

            row_detail::PtoWaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
            row_detail::PtoLoadVector(ubCOffset, gmTileC, blockN);
            row_detail::PtoSetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);

            row_detail::PtoWaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
            row_detail::PtoCastVector<ElementPerTokenScale, ElementC>(ubCFp32Offset, ubCOffset, blockN,
                                                                      pto::RoundMode::CAST_NONE);
            row_detail::PtoSetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);

            ElementPerTokenScale perTokenScale = gm_load(gmPerTokenScalePtr + loopIdx);

            row_detail::PtoSetFlag<AscendC::HardEvent::S_V>(0);
            row_detail::PtoWaitFlag<AscendC::HardEvent::S_V>(0);
            row_detail::PtoPipeBarrier<PIPE_V>();
            row_detail::PtoMulVector(ubCFp32Offset, ubCFp32Offset, blockN, perTokenScale);
            row_detail::PtoPipeBarrier<PIPE_V>();

            row_detail::PtoWaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
            row_detail::PtoCastVector<ElementD, ElementPerTokenScale>(ubDOffset, ubCFp32Offset, blockN,
                                                                      pto::RoundMode::CAST_RINT);
            row_detail::PtoSetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);

            row_detail::PtoWaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
            __gm__ ElementD *dstRowBase = ptrD + loopIdx * blockN;
            if (dstRank == params.rank) {
                row_detail::PtoStoreVector(dstRowBase, ubDOffset, blockN);
            } else {
                for (uint32_t colOffset = 0; colOffset < blockN; colOffset += scratchCols) {
                    uint32_t chunkCols = (blockN - colOffset < scratchCols) ? (blockN - colOffset) : scratchCols;
                    row_detail::PtoStoreVector(
                        localScratch, ubDOffset + static_cast<uint64_t>(colOffset) * sizeof(ElementD), chunkCols);
                    ShapeDyn rowShape(1, 1, 1, 1, chunkCols);
                    StrideDyn rowStride(chunkCols, chunkCols, chunkCols, chunkCols, 1);
                    TputTile tputTile(1, chunkCols < scratchCols ? chunkCols : scratchCols);
                    TputGlobal localRowG(localScratch, rowShape, rowStride);
                    TputGlobal remoteRowG(dstRowBase + colOffset, rowShape, rowStride);
                    pto::comm::TPUT(remoteRowG, localRowG, tputTile);
                    row_detail::PtoPipeBarrier<PIPE_ALL>();
                }
            }
            row_detail::PtoSetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);

            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
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
};

} // namespace pto_ext::Epilogue::Block

#endif // PTO_EXT_EPILOGUE_BLOCK_PER_TOKEN_ROW_HPP