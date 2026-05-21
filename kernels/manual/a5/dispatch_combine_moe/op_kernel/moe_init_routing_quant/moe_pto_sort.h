#ifndef INNER_MOE_PTO_SORT_H
#define INNER_MOE_PTO_SORT_H

#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

#include "moe_common.h"
#include "../utils/pto_vector_ops.hpp"

namespace MoeInitRoutingQuant {
namespace pto_detail {

constexpr uint32_t PTO_SORT_BLOCK_ELEMS = 32;
constexpr uint32_t PTO_PACKED_SORT_BLOCK_ELEMS = 64;
constexpr uint32_t MAX_V3_SORT_ELEMS = 8192;
constexpr float PTO_SORT_NEG_INF = -3.4028235e38F;

using PtoV2SortKeyTile = pto::Tile<pto::TileType::Vec, float, 1, MAX_V3_SORT_ELEMS, pto::BLayout::RowMajor, -1, -1>;
using PtoV2SortPayloadTile =
    pto::Tile<pto::TileType::Vec, uint32_t, 1, MAX_V3_SORT_ELEMS, pto::BLayout::RowMajor, -1, -1>;
using PtoV2PackedSortTile =
    pto::Tile<pto::TileType::Vec, float, 1, MAX_V3_SORT_ELEMS * 2, pto::BLayout::RowMajor, -1, -1>;
using PtoV2PackedPayloadTile =
    pto::Tile<pto::TileType::Vec, uint32_t, 1, MAX_V3_SORT_ELEMS * 2, pto::BLayout::RowMajor, -1, -1>;

template <auto Pipe>
PTO_INTERNAL void PtoPipeBarrier()
{
    AscendC::PipeBarrier<Pipe>();
}

template <AscendC::HardEvent Event>
PTO_INTERNAL void PtoSetFlag(int32_t eventId)
{
    AscendC::SetFlag<Event>(eventId);
}

template <AscendC::HardEvent Event>
PTO_INTERNAL void PtoWaitFlag(int32_t eventId)
{
    AscendC::WaitFlag<Event>(eventId);
}

template <AscendC::HardEvent Event>
PTO_INTERNAL void PtoSetWaitFlag(AscendC::HardEvent eventId)
{
    SetWaitFlag<Event>(eventId);
}

PTO_INTERNAL void PtoSyncAll()
{
    AscendC::SyncAll();
}

using pto_ext::dispatch_combine_moe::pto_bridge::PtoAbsVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoAddScalarVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoAddVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoCastVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoDivVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoFillVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoGetValue;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoLoadVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoMoveVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoMulElementwiseVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoMulVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoReduceMaxVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoSetValue;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoStoreAtomicAddVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoStoreVector;

PTO_INTERNAL void PtoFillArithProgressionInt32(uint64_t dstUb, int32_t firstValue, int32_t diffValue, uint32_t count)
{
    __ubuf__ int32_t *dstPtr = reinterpret_cast<__ubuf__ int32_t *>(dstUb);
    int32_t value = firstValue;
    for (uint32_t idx = 0; idx < count; ++idx) {
        dstPtr[idx] = value;
        value += diffValue;
    }
    PtoSetFlag<AscendC::HardEvent::S_V>(0);
    PtoWaitFlag<AscendC::HardEvent::S_V>(0);
}

PTO_INTERNAL uint32_t AlignUpSortBlock(uint32_t elemNum)
{
    return ((elemNum + PTO_SORT_BLOCK_ELEMS - 1) / PTO_SORT_BLOCK_ELEMS) * PTO_SORT_BLOCK_ELEMS;
}

PTO_INTERNAL int32_t FillTailMergeArray(int32_t *mrgArray, int32_t validCols, int32_t blockLen)
{
    int32_t arrayCount = 0;
    int32_t remainCols = validCols;
    for (int32_t curBlockLen = blockLen; curBlockLen >= static_cast<int32_t>(PTO_PACKED_SORT_BLOCK_ELEMS);
         curBlockLen /= 4) {
        int32_t count = 0;
        for (; count < remainCols / curBlockLen; ++count) {
            mrgArray[arrayCount++] = curBlockLen;
        }
        remainCols -= count * curBlockLen;
    }
    return arrayCount;
}

PTO_INTERNAL void MergeTailPackedSortRecords(PtoV2PackedSortTile &packedSortTile, PtoV2PackedSortTile &mergeTmpTile,
                                             uint32_t validCols, uint32_t blockLen)
{
    int32_t mergePlan[15] = {0};
    const int32_t mergePlanCount =
        FillTailMergeArray(mergePlan, static_cast<int32_t>(validCols), static_cast<int32_t>(blockLen));
    if (mergePlanCount <= 1) {
        return;
    }

    pto::MrgSortExecutedNumList executedNumList{};
    uint16_t mergedCols = 0;
    const uint64_t packedAddr = reinterpret_cast<uint64_t>(packedSortTile.data());
    const uint64_t tmpAddr = reinterpret_cast<uint64_t>(mergeTmpTile.data());
    for (int32_t i = 0; i < mergePlanCount - 1; ++i) {
        mergedCols += static_cast<uint16_t>(mergePlan[i]);
        PtoV2PackedSortTile src0Tile(1, mergedCols);
        PtoV2PackedSortTile src1Tile(1, static_cast<uint16_t>(mergePlan[i + 1]));
        PtoV2PackedSortTile dstTile(1, mergedCols + static_cast<uint16_t>(mergePlan[i + 1]));
        PtoV2PackedSortTile tmpTile(1, mergedCols + static_cast<uint16_t>(mergePlan[i + 1]));
        pto::TASSIGN(src0Tile, packedAddr);
        pto::TASSIGN(src1Tile, packedAddr + static_cast<uint64_t>(mergedCols) * sizeof(float));
        pto::TASSIGN(dstTile, packedAddr);
        pto::TASSIGN(tmpTile, tmpAddr);
        pto::TMRGSORT<PtoV2PackedSortTile, PtoV2PackedSortTile, PtoV2PackedSortTile, PtoV2PackedSortTile, false>(
            dstTile, executedNumList, tmpTile, src0Tile, src1Tile);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

PTO_INTERNAL void MergePackedSortRecords(PtoV2PackedSortTile &packedSortTile, PtoV2PackedSortTile &mergeTmpTile,
                                         uint32_t validCols)
{
    uint32_t blockLen = PTO_PACKED_SORT_BLOCK_ELEMS;
    const uint64_t packedAddr = reinterpret_cast<uint64_t>(packedSortTile.data());
    const uint64_t tmpAddr = reinterpret_cast<uint64_t>(mergeTmpTile.data());
    for (; blockLen * 4 <= validCols; blockLen *= 4) {
        const uint16_t cols = validCols / (blockLen * 4) * (blockLen * 4);
        PtoV2PackedSortTile srcTile(1, cols);
        PtoV2PackedSortTile tmpTile(1, cols);
        pto::TASSIGN(srcTile, packedAddr);
        pto::TASSIGN(tmpTile, tmpAddr);
        pto::TMRGSORT(tmpTile, srcTile, blockLen);
        AscendC::PipeBarrier<PIPE_V>();
        pto::TMOV(srcTile, tmpTile);
        AscendC::PipeBarrier<PIPE_V>();
    }

    if (blockLen < validCols) {
        PtoV2PackedSortTile tailTile(1, validCols);
        PtoV2PackedSortTile tailTmpTile(1, validCols);
        pto::TASSIGN(tailTile, packedAddr);
        pto::TASSIGN(tailTmpTile, tmpAddr);
        MergeTailPackedSortRecords(tailTile, tailTmpTile, validCols, blockLen);
    }
}

PTO_INTERNAL void PtoMergePackedSortRecords(uint64_t dstUb, uint64_t tmpUb, uint64_t src0Ub, uint64_t src1Ub,
                                            uint64_t src2Ub, uint64_t src3Ub, const uint16_t *elementCountList,
                                            uint32_t remainListNum, uint32_t *listSortedNums)
{
    const uint32_t src0Cols = GetSortLen<float>(elementCountList[0]);
    const uint32_t src1Cols = (remainListNum >= 2) ? GetSortLen<float>(elementCountList[1]) : 0;
    const uint32_t src2Cols = (remainListNum >= 3) ? GetSortLen<float>(elementCountList[2]) : 0;
    const uint32_t src3Cols = (remainListNum >= 4) ? GetSortLen<float>(elementCountList[3]) : 0;
    const uint32_t dstCols = src0Cols + src1Cols + src2Cols + src3Cols;
    ASCENDC_ASSERT((dstCols <= MAX_V3_SORT_ELEMS * 2),
                   { KERNEL_LOG(KERNEL_ERROR, "dstCols exceeds PTO merge capacity"); });

    PtoV2PackedSortTile dstTile(1, dstCols);
    PtoV2PackedSortTile tmpTile(1, dstCols);
    PtoV2PackedSortTile src0Tile(1, src0Cols);
    PtoV2PackedSortTile src1Tile(1, src1Cols);
    PtoV2PackedSortTile src2Tile(1, src2Cols);
    PtoV2PackedSortTile src3Tile(1, src3Cols);
    pto::TASSIGN(dstTile, dstUb);
    pto::TASSIGN(tmpTile, tmpUb);
    pto::TASSIGN(src0Tile, src0Ub);
    pto::TASSIGN(src1Tile, src1Ub);
    if (src2Cols > 0) {
        pto::TASSIGN(src2Tile, src2Ub);
    }
    if (src3Cols > 0) {
        pto::TASSIGN(src3Tile, src3Ub);
    }

    pto::MrgSortExecutedNumList executedNumList{};
    if (remainListNum == MERGE_LIST_TWO) {
        pto::TMRGSORT<PtoV2PackedSortTile, PtoV2PackedSortTile, PtoV2PackedSortTile, PtoV2PackedSortTile, true>(
            dstTile, executedNumList, tmpTile, src0Tile, src1Tile);
    } else if (remainListNum == MERGE_LIST_THREE) {
        pto::TMRGSORT<PtoV2PackedSortTile, PtoV2PackedSortTile, PtoV2PackedSortTile, PtoV2PackedSortTile,
                      PtoV2PackedSortTile, true>(dstTile, executedNumList, tmpTile, src0Tile, src1Tile, src2Tile);
    } else {
        pto::TMRGSORT<PtoV2PackedSortTile, PtoV2PackedSortTile, PtoV2PackedSortTile, PtoV2PackedSortTile,
                      PtoV2PackedSortTile, PtoV2PackedSortTile, true>(dstTile, executedNumList, tmpTile, src0Tile,
                                                                      src1Tile, src2Tile, src3Tile);
    }

    listSortedNums[0] = executedNumList.mrgSortList0;
    listSortedNums[1] = executedNumList.mrgSortList1;
    listSortedNums[2] = executedNumList.mrgSortList2;
    listSortedNums[3] = executedNumList.mrgSortList3;
}

PTO_INTERNAL void PtoSortInt32ToPackedUB(uint64_t inputValueUb, uint64_t inputPayloadUb, uint64_t packedSortUb,
                                         uint64_t mergeTmpUb, uint32_t elemNum)
{
    if (elemNum == 0) {
        return;
    }

    const uint32_t alignedElemNum = AlignUpSortBlock(elemNum);
    ASCENDC_ASSERT((alignedElemNum <= MAX_V3_SORT_ELEMS),
                   { KERNEL_LOG(KERNEL_ERROR, "alignedElemNum exceeds PTO sort capacity"); });

    const uint64_t sortKeyUb = mergeTmpUb;
    PtoCastVector<float, int32_t>(sortKeyUb, inputValueUb, elemNum, pto::RoundMode::CAST_CEIL);
    PtoMulVector<float>(sortKeyUb, sortKeyUb, elemNum, -1.0F);

    __ubuf__ float *sortKeyPtr = reinterpret_cast<__ubuf__ float *>(sortKeyUb);
    __ubuf__ uint32_t *payloadPtr = reinterpret_cast<__ubuf__ uint32_t *>(inputPayloadUb);
    for (uint32_t i = elemNum; i < alignedElemNum; ++i) {
        sortKeyPtr[i] = PTO_SORT_NEG_INF;
        payloadPtr[i] = 0;
    }

    PtoV2SortKeyTile srcTile(1, alignedElemNum);
    PtoV2SortPayloadTile payloadTile(1, alignedElemNum);
    PtoV2PackedSortTile packedTile(1, alignedElemNum * 2);
    PtoV2PackedSortTile mergeTmpTile(1, alignedElemNum * 2);
    pto::TASSIGN(srcTile, sortKeyUb);
    pto::TASSIGN(payloadTile, inputPayloadUb);
    pto::TASSIGN(packedTile, packedSortUb);
    pto::TASSIGN(mergeTmpTile, mergeTmpUb);

    pto::TSORT32(packedTile, srcTile, payloadTile);
    AscendC::PipeBarrier<PIPE_V>();
    MergePackedSortRecords(packedTile, mergeTmpTile, alignedElemNum * 2);
}

PTO_INTERNAL void PtoExtractPackedSortResult(uint64_t sortedValueUb, uint64_t sortedPayloadUb, uint64_t packedSortUb,
                                             uint32_t elemNum)
{
    if (elemNum == 0) {
        return;
    }
    if (elemNum == 1) {
        __ubuf__ const float *packedPtr = reinterpret_cast<__ubuf__ const float *>(packedSortUb);
        __ubuf__ int32_t *valueOut = reinterpret_cast<__ubuf__ int32_t *>(sortedValueUb);
        __ubuf__ uint32_t *payloadOut = reinterpret_cast<__ubuf__ uint32_t *>(sortedPayloadUb);
        __ubuf__ const uint32_t *packedPayloadPtr = reinterpret_cast<__ubuf__ const uint32_t *>(packedSortUb);
        valueOut[0] = -static_cast<int32_t>(packedPtr[0]);
        payloadOut[0] = packedPayloadPtr[1];
        return;
    }

    const uint64_t sortedValueScratchUb = sortedValueUb;
    PtoV2PackedPayloadTile packedPayloadTile(1, elemNum * 2);
    PtoV2SortPayloadTile sortedPayloadTile(1, elemNum);
    pto::TASSIGN(packedPayloadTile, packedSortUb);
    pto::TASSIGN(sortedPayloadTile, sortedPayloadUb);
    pto::TGATHER<PtoV2SortPayloadTile, PtoV2PackedPayloadTile, pto::MaskPattern::P1010>(sortedPayloadTile,
                                                                                        packedPayloadTile);
    AscendC::PipeBarrier<PIPE_V>();

    PtoV2SortKeyTile sortedKeyTile(1, elemNum);
    PtoV2PackedSortTile packedTile(1, elemNum * 2);
    pto::TASSIGN(sortedKeyTile, sortedValueScratchUb);
    pto::TASSIGN(packedTile, packedSortUb);
    pto::TGATHER<PtoV2SortKeyTile, PtoV2PackedSortTile, pto::MaskPattern::P0101>(sortedKeyTile, packedTile);
    AscendC::PipeBarrier<PIPE_V>();
    PtoMulVector<float>(sortedValueScratchUb, sortedValueScratchUb, elemNum, -1.0F);

    PtoCastVector<int32_t, float>(sortedValueUb, sortedValueScratchUb, elemNum, pto::RoundMode::CAST_CEIL);
}

PTO_INTERNAL void PtoSortInt32AscendingUB(uint64_t inputValueUb, uint64_t inputPayloadUb, uint64_t sortedValueUb,
                                          uint64_t sortedPayloadUb, uint64_t packedSortUb, uint64_t mergeTmpUb,
                                          uint32_t elemNum)
{
    if (elemNum == 0) {
        return;
    }
    if (elemNum == 1) {
        __ubuf__ const int32_t *valueIn = reinterpret_cast<__ubuf__ const int32_t *>(inputValueUb);
        __ubuf__ const uint32_t *payloadIn = reinterpret_cast<__ubuf__ const uint32_t *>(inputPayloadUb);
        __ubuf__ int32_t *valueOut = reinterpret_cast<__ubuf__ int32_t *>(sortedValueUb);
        __ubuf__ uint32_t *payloadOut = reinterpret_cast<__ubuf__ uint32_t *>(sortedPayloadUb);
        valueOut[0] = valueIn[0];
        payloadOut[0] = payloadIn[0];
        return;
    }

    PtoSortInt32ToPackedUB(inputValueUb, inputPayloadUb, packedSortUb, mergeTmpUb, elemNum);
    PtoExtractPackedSortResult(sortedValueUb, sortedPayloadUb, packedSortUb, elemNum);
}

} // namespace pto_detail
} // namespace MoeInitRoutingQuant

#endif // INNER_MOE_PTO_SORT_H
