#ifndef PTO_EXT_EPILOGUE_BLOCK_PER_TOKEN_V2_ONLY_HPP
#define PTO_EXT_EPILOGUE_BLOCK_PER_TOKEN_V2_ONLY_HPP

#include "dispatch_policy_custom.hpp"

#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

#include "hccl_window.hpp"
#include "layout3d.hpp"
#include "pto_vector_ops.hpp"

namespace pto_ext::Epilogue::Block {
namespace detail {

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

using pto_ext::dispatch_ffn_combine_v3::pto_bridge::PtoElemOffsetBytes;

template <typename Element>
PTO_DEVICE __ubuf__ Element *PtoUbPtr(uint64_t ubOffsetBytes)
{
    return reinterpret_cast<__ubuf__ Element *>(ubOffsetBytes);
}

template <typename DstElement, typename SrcElement, int TileElems = 128>
PTO_DEVICE void PtoCastVector(uint64_t dstUbOffsetBytes, uint64_t srcUbOffsetBytes, uint32_t elemNum,
                              pto::RoundMode mode)
{
    pto_ext::dispatch_ffn_combine_v3::pto_bridge::PtoCastVector<DstElement, SrcElement, TileElems>(
        dstUbOffsetBytes, srcUbOffsetBytes, elemNum, mode);
}

template <typename Element, int TileElems = 128>
PTO_DEVICE void PtoMulVector(uint64_t dstUbOffsetBytes, uint64_t srcUbOffsetBytes, uint32_t elemNum, Element scalar)
{
    pto_ext::dispatch_ffn_combine_v3::pto_bridge::PtoMulVector<Element, TileElems>(dstUbOffsetBytes, srcUbOffsetBytes,
                                                                                   elemNum, scalar);
}

template <typename Element, int TileElems = 128>
PTO_DEVICE void PtoLoadVector(uint64_t dstUbOffsetBytes, __gm__ Element *src, uint32_t elemNum)
{
    pto_ext::dispatch_ffn_combine_v3::pto_bridge::PtoLoadVector<Element, TileElems>(dstUbOffsetBytes, src, elemNum);
}

template <typename Element, int TileElems = 128>
PTO_DEVICE void PtoStoreVector(__gm__ Element *dst, uint64_t srcUbOffsetBytes, uint32_t elemNum)
{
    pto_ext::dispatch_ffn_combine_v3::pto_bridge::PtoStoreVector<Element, TileElems>(dst, srcUbOffsetBytes, elemNum);
}

template <typename Element, int TileElems = 128>
PTO_DEVICE void PtoLoadMatrixRows(uint64_t dstUbOffsetBytes, __gm__ Element *src, uint32_t rowNum, uint32_t colNum,
                                  uint32_t dstStride, uint32_t srcStride)
{
    pto_ext::dispatch_ffn_combine_v3::pto_bridge::PtoLoadMatrixRows<Element, TileElems>(dstUbOffsetBytes, src, rowNum,
                                                                                        colNum, dstStride, srcStride);
}

template <typename Element, int TileElems = 128>
PTO_DEVICE void PtoStoreMatrixRows(__gm__ Element *dst, uint64_t srcUbOffsetBytes, uint32_t rowNum, uint32_t colNum,
                                   uint32_t dstStride, uint32_t srcStride)
{
    pto_ext::dispatch_ffn_combine_v3::pto_bridge::PtoStoreMatrixRows<Element, TileElems>(dst, srcUbOffsetBytes, rowNum,
                                                                                         colNum, dstStride, srcStride);
}

} // namespace detail

template <uint32_t UB_STAGES_, class CType_, class LayoutPerTokenScale_, class DType_, class TileCopy_>
class BlockEpilogue<EpilogueAtlasA2PerTokenDequantV2<UB_STAGES_>, CType_, Gemm::GemmType<float, LayoutPerTokenScale_>,
                    DType_, TileCopy_> {
public:
    using DispatchPolicy = EpilogueAtlasA2PerTokenDequantV2<UB_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;

    // Data infos
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = LayoutPerTokenScale_;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;

    struct Params {
        __gm__ int32_t *ptrTokenPerExpert{nullptr};
        int32_t EP;
        int32_t expertPerRank;
        int32_t n2;
        LayoutC layoutC;
        int32_t n0;
        int32_t rank;
        PtoRemoteWindow remoteWindow;
        int32_t offsetD;
        int32_t scratchOffset;
        Layout3D tokenPerExpertLayout;
        PTO_DEVICE
        Params(){};
        PTO_DEVICE
        Params(int32_t EP_, int32_t expertPerRank_, int32_t rank_, __gm__ int32_t *ptrTokenPerExpert_, LayoutC layoutC_,
               int32_t n2_, int32_t n0_, PtoRemoteWindow &remoteWindow_, int32_t offsetD_, int32_t scratchOffset_,
               Layout3D tokenPerExpertLayout_)
            : ptrTokenPerExpert(ptrTokenPerExpert_),
              EP(EP_),
              expertPerRank(expertPerRank_),
              rank(rank_),
              layoutC(layoutC_),
              n2(n2_),
              n0(n0_),
              remoteWindow(remoteWindow_),
              offsetD(offsetD_),
              scratchOffset(scratchOffset_),
              tokenPerExpertLayout(tokenPerExpertLayout_)
        {}
    };

    PTO_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource, Params const &params = Params{}) : params(params)
    {
        // ub:192KB
        n0 = params.n0;
        uint64_t ubOffset = 0;
        for (int32_t i = 0; i < 2; i++) {
            ubCOffsetList[i] = ubOffset;
            ubOffset += max_len * sizeof(ElementC);
            ubDOffsetList[i] = ubOffset;
            ubOffset += max_len * sizeof(ElementD);
            ubFp32OffsetList[i] = ubOffset;
            ubOffset += max_len * sizeof(float);
            scaleUbOffsetList[i] = ubOffset;
            ubOffset += (max_len / n0) * sizeof(float);
            source_scale_offset[i] = -1;
        }
        tokenPerExpertPtr = reinterpret_cast<__gm__ int32_t *>(params.ptrTokenPerExpert);
        tokenPerExpertLayout = params.tokenPerExpertLayout;
        is_ping = true;
    }
    PTO_DEVICE
    void SetFlag()
    {
        detail::PtoSetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        detail::PtoSetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        detail::PtoSetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        detail::PtoSetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        detail::PtoSetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
        detail::PtoSetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID3);
        detail::PtoSetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        detail::PtoSetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
    }

    PTO_DEVICE
    void Finalize()
    {
        detail::PtoWaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        detail::PtoWaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        detail::PtoWaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        detail::PtoWaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        detail::PtoWaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
        detail::PtoWaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID3);
        detail::PtoWaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        detail::PtoWaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
    }
    PTO_DEVICE
    ~BlockEpilogue()
    {}
    PTO_DEVICE
    void operator()(__gm__ ElementC *gmCPtr, __gm__ ElementPerTokenScale *gmPerTokenScalePtr,
                    PtoCoord2D const &blockCoord, PtoShape2D const &actualBlockShape, int32_t groupIdx,
                    int32_t preSrcExpertSum, __gm__ int32_t *preSumBeforeRank)
    {
        is_ping = !is_ping;
        auto event_id = is_ping ? EVENT_ID0 : EVENT_ID1;
        auto event_id_2 = is_ping ? EVENT_ID2 : EVENT_ID3;
        int32_t blockRow = static_cast<int32_t>(blockCoord.shape[0]);
        int32_t blockCol = static_cast<int32_t>(blockCoord.shape[1]);
        uint32_t actualM = static_cast<uint32_t>(actualBlockShape.shape[0]);
        uint32_t actualN = static_cast<uint32_t>(actualBlockShape.shape[1]);

        uint64_t ubCOffset = ubCOffsetList[is_ping];
        uint64_t ubDOffset = ubDOffsetList[is_ping];
        int32_t gmCOffset = preSrcExpertSum * params.n2 + blockRow * params.n2 + blockCol;
        __gm__ ElementC *gmTileC = gmCPtr + gmCOffset;
        uint64_t ubCFp32Offset = ubFp32OffsetList[is_ping];
        uint64_t scaleUbOffset = scaleUbOffsetList[is_ping];

        detail::PtoWaitFlag<AscendC::HardEvent::V_MTE2>(event_id);
        detail::PtoLoadMatrixRows(ubCOffset, gmTileC, actualM, actualN, n0, params.n2);
        detail::PtoSetFlag<AscendC::HardEvent::MTE2_V>(event_id);

        detail::PtoWaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
        for (uint32_t row = 0; row < actualM; ++row) {
            detail::PtoCastVector<float, ElementC>(ubCFp32Offset + detail::PtoElemOffsetBytes<float>(n0 * row),
                                                   ubCOffset + detail::PtoElemOffsetBytes<ElementC>(n0 * row), actualN,
                                                   pto::RoundMode::CAST_NONE);
        }
        detail::PtoSetFlag<AscendC::HardEvent::V_MTE2>(event_id);

        detail::PtoWaitFlag<AscendC::HardEvent::V_MTE2>(event_id_2);
        detail::PtoWaitFlag<AscendC::HardEvent::S_MTE2>(event_id_2);

        int32_t gmScaleOffset = preSrcExpertSum + blockRow;
        if (source_scale_offset[event_id] != gmScaleOffset) {
            source_scale_offset[event_id] = gmScaleOffset;
            detail::PtoLoadVector(scaleUbOffset, gmPerTokenScalePtr + gmScaleOffset, actualM);
        }

        detail::PtoSetFlag<AscendC::HardEvent::MTE2_S>(event_id_2);
        detail::PtoSetFlag<AscendC::HardEvent::MTE2_V>(event_id_2);

        detail::PtoWaitFlag<AscendC::HardEvent::MTE2_V>(event_id_2);
        detail::PtoWaitFlag<AscendC::HardEvent::MTE2_S>(
            event_id_2); // Note that the value must be MTE2_S instead of MTE2_V.
                         // Otherwise, 0 will be read, causing garbled characters.
        detail::PtoPipeBarrier<PIPE_V>();
        __ubuf__ float *scaleUbPtr = detail::PtoUbPtr<float>(scaleUbOffset);
        for (uint32_t row = 0; row < actualM; ++row) {
            float scale = scaleUbPtr[row];
            uint64_t rowFp32Offset = ubCFp32Offset + detail::PtoElemOffsetBytes<float>(n0 * row);
            detail::PtoMulVector<float>(rowFp32Offset, rowFp32Offset, actualN, scale);
        }
        detail::PtoPipeBarrier<PIPE_V>();
        detail::PtoWaitFlag<AscendC::HardEvent::MTE3_V>(event_id);
        for (uint32_t row = 0; row < actualM; ++row) {
            detail::PtoCastVector<ElementD, float>(ubDOffset + detail::PtoElemOffsetBytes<ElementD>(n0 * row),
                                                   ubCFp32Offset + detail::PtoElemOffsetBytes<float>(n0 * row), actualN,
                                                   pto::RoundMode::CAST_RINT);
        }
        detail::PtoSetFlag<AscendC::HardEvent::S_MTE2>(event_id_2);
        detail::PtoSetFlag<AscendC::HardEvent::V_MTE2>(event_id_2);
        detail::PtoSetFlag<AscendC::HardEvent::V_MTE3>(event_id);

        int32_t lenTile = static_cast<int32_t>(actualM);
        int32_t stTile = blockRow;
        int32_t edTile = stTile + lenTile;
        int32_t preSumRankInExpert = 0;
        int32_t tileOffset = 0;

        detail::PtoWaitFlag<AscendC::HardEvent::V_MTE3>(event_id);
        for (int32_t dstEpIdx = 0; dstEpIdx < params.EP; dstEpIdx++) {
            int32_t lenRankInExpert =
                gm_load(tokenPerExpertPtr + tokenPerExpertLayout(dstEpIdx, params.rank, groupIdx));
            int32_t dstExpertOffset = gm_load(preSumBeforeRank + dstEpIdx * params.expertPerRank + groupIdx);
            int32_t stRankInExpert = preSumRankInExpert;
            int32_t edRankInExpert = stRankInExpert + lenRankInExpert;
            preSumRankInExpert += lenRankInExpert;
            if (stRankInExpert >= edTile) {
                break;
            } else if (edRankInExpert <= stTile) {
                continue;
            }
            int32_t stData = max(stRankInExpert, stTile);
            int32_t edData = min(edRankInExpert, edTile);
            uint32_t lenData = edData - stData;
            if (lenData <= 0) {
                continue;
            }

            uint32_t dstOffsetInExpert = 0;
            if (stTile > stRankInExpert) {
                dstOffsetInExpert = stTile - stRankInExpert;
            }
            __gm__ void *dstPeermemPtr = params.remoteWindow(params.offsetD, dstEpIdx);
            __gm__ ElementD *gmRemotePeerPtr = reinterpret_cast<__gm__ ElementD *>(dstPeermemPtr);
            auto dstOffset = MakePtoCoord2D(dstOffsetInExpert + dstExpertOffset, blockCol);
            int64_t gmDstOffset = params.layoutC.GetOffset(dstOffset);
            __gm__ ElementD *gmTileD = gmRemotePeerPtr + gmDstOffset;
            if (dstEpIdx == params.rank) {
                detail::PtoStoreMatrixRows(gmTileD, ubDOffset + detail::PtoElemOffsetBytes<ElementD>(tileOffset * n0),
                                           lenData, actualN, params.n2, n0);
            } else {
                using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
                using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
                using TputGlobal = pto::GlobalTensor<ElementD, ShapeDyn, StrideDyn, pto::Layout::ND>;
                using TputTile = pto::Tile<pto::TileType::Vec, ElementD, 1, 128, pto::BLayout::RowMajor, -1, -1>;

                int32_t logicalSubCoreIdx = get_block_idx() + get_subblockid() * get_block_num();
                int64_t scratchOffsetBytes =
                    params.scratchOffset + static_cast<int64_t>(logicalSubCoreIdx) * n0 * sizeof(ElementD);
                __gm__ ElementD *localScratch =
                    reinterpret_cast<__gm__ ElementD *>(params.remoteWindow(scratchOffsetBytes, params.rank));
                ShapeDyn rowShape(1, 1, 1, 1, actualN);
                StrideDyn localStride(actualN, actualN, actualN, actualN, 1);
                StrideDyn remoteStride(params.n2, params.n2, params.n2, params.n2, 1);
                TputTile tputTile(1, actualN);
                __gm__ ElementD *remotePeerBase = reinterpret_cast<__gm__ ElementD *>(dstPeermemPtr) + gmDstOffset;

                for (uint32_t rowIdx = 0; rowIdx < lenData; ++rowIdx) {
                    detail::PtoStoreVector(localScratch,
                                           ubDOffset + detail::PtoElemOffsetBytes<ElementD>((tileOffset + rowIdx) * n0),
                                           actualN);
                    TputGlobal localRowG(localScratch, rowShape, localStride);
                    TputGlobal remoteRowG(remotePeerBase + rowIdx * params.n2, rowShape, remoteStride);
                    pto::comm::TPUT(remoteRowG, localRowG, tputTile);
                    detail::PtoPipeBarrier<PIPE_ALL>();
                }
            }
            tileOffset += lenData;
        }
        detail::PtoSetFlag<AscendC::HardEvent::MTE3_V>(event_id);
    }

private:
    Params params;
    uint64_t ubCOffsetList[UB_STAGES];
    uint64_t ubDOffsetList[UB_STAGES];
    uint64_t ubFp32OffsetList[UB_STAGES];
    uint64_t scaleUbOffsetList[UB_STAGES];
    int32_t source_scale_offset[UB_STAGES];

    int32_t max_len = 8 * 32 / 4 * 128;
    int32_t n0;
    bool is_ping = false;

    int32_t repeat = 128;

    __gm__ int32_t *tokenPerExpertPtr;
    Layout3D tokenPerExpertLayout;
};
} // namespace pto_ext::Epilogue::Block
#endif