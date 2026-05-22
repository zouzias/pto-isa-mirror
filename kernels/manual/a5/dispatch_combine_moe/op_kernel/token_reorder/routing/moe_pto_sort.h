#ifndef INNER_MOE_PTO_SORT_H
#define INNER_MOE_PTO_SORT_H

#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

#include "moe_common.h"
#include "../../utils/pto_vector_ops.hpp"

namespace MoeInitRoutingQuant {
constexpr uint32_t PTO_SORT_BLOCK_ELEMS = 32;
constexpr uint32_t PTO_PACKED_SORT_BLOCK_ELEMS = 64;
constexpr uint32_t MAX_SORT_ELEMS = 8192;
constexpr float PTO_SORT_NEG_INF = -3.4028235e38F;

using PtoSortKeyTile = pto::Tile<pto::TileType::Vec, float, 1, MAX_SORT_ELEMS, pto::BLayout::RowMajor, -1, -1>;
using PtoSortPayloadTile = pto::Tile<pto::TileType::Vec, uint32_t, 1, MAX_SORT_ELEMS, pto::BLayout::RowMajor, -1, -1>;
using PtoPackedSortTile = pto::Tile<pto::TileType::Vec, float, 1, MAX_SORT_ELEMS * 2, pto::BLayout::RowMajor, -1, -1>;
using PtoPackedPayloadTile =
    pto::Tile<pto::TileType::Vec, uint32_t, 1, MAX_SORT_ELEMS * 2, pto::BLayout::RowMajor, -1, -1>;

PTO_INTERNAL void PtoFillArithProgressionInt32(uint64_t dstUb, int32_t firstValue, int32_t diffValue, uint32_t count)
{
    __ubuf__ int32_t *dstPtr = reinterpret_cast<__ubuf__ int32_t *>(dstUb);
    int32_t value = firstValue;
    for (uint32_t idx = 0; idx < count; ++idx) {
        dstPtr[idx] = value;
        value += diffValue;
    }
    pto_detail::PtoSetFlag<AscendC::HardEvent::S_V>(0);
    pto_detail::PtoWaitFlag<AscendC::HardEvent::S_V>(0);
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

PTO_INTERNAL void MergeTailPackedSortRecords(PtoPackedSortTile &packedSortTile, PtoPackedSortTile &mergeTmpTile,
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
        PtoPackedSortTile src0Tile(1, mergedCols);
        PtoPackedSortTile src1Tile(1, static_cast<uint16_t>(mergePlan[i + 1]));
        PtoPackedSortTile dstTile(1, mergedCols + static_cast<uint16_t>(mergePlan[i + 1]));
        PtoPackedSortTile tmpTile(1, mergedCols + static_cast<uint16_t>(mergePlan[i + 1]));
        pto::TASSIGN(src0Tile, packedAddr);
        pto::TASSIGN(src1Tile, packedAddr + static_cast<uint64_t>(mergedCols) * sizeof(float));
        pto::TASSIGN(dstTile, packedAddr);
        pto::TASSIGN(tmpTile, tmpAddr);
        pto::TMRGSORT<PtoPackedSortTile, PtoPackedSortTile, PtoPackedSortTile, PtoPackedSortTile, false>(
            dstTile, executedNumList, tmpTile, src0Tile, src1Tile);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

PTO_INTERNAL void MergePackedSortRecords(PtoPackedSortTile &packedSortTile, PtoPackedSortTile &mergeTmpTile,
                                         uint32_t validCols)
{
    uint32_t blockLen = PTO_PACKED_SORT_BLOCK_ELEMS;
    const uint64_t packedAddr = reinterpret_cast<uint64_t>(packedSortTile.data());
    const uint64_t tmpAddr = reinterpret_cast<uint64_t>(mergeTmpTile.data());
    for (; blockLen * 4 <= validCols; blockLen *= 4) {
        const uint16_t cols = validCols / (blockLen * 4) * (blockLen * 4);
        PtoPackedSortTile srcTile(1, cols);
        PtoPackedSortTile tmpTile(1, cols);
        pto::TASSIGN(srcTile, packedAddr);
        pto::TASSIGN(tmpTile, tmpAddr);
        pto::TMRGSORT(tmpTile, srcTile, blockLen);
        AscendC::PipeBarrier<PIPE_V>();
        pto::TMOV(srcTile, tmpTile);
        AscendC::PipeBarrier<PIPE_V>();
    }

    if (blockLen < validCols) {
        PtoPackedSortTile tailTile(1, validCols);
        PtoPackedSortTile tailTmpTile(1, validCols);
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
    ASCENDC_ASSERT((dstCols <= MAX_SORT_ELEMS * 2),
                   { KERNEL_LOG(KERNEL_ERROR, "dstCols exceeds PTO merge capacity"); });

    PtoPackedSortTile dstTile(1, dstCols);
    PtoPackedSortTile tmpTile(1, dstCols);
    PtoPackedSortTile src0Tile(1, src0Cols);
    PtoPackedSortTile src1Tile(1, src1Cols);
    PtoPackedSortTile src2Tile(1, src2Cols);
    PtoPackedSortTile src3Tile(1, src3Cols);
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
        pto::TMRGSORT<PtoPackedSortTile, PtoPackedSortTile, PtoPackedSortTile, PtoPackedSortTile, true>(
            dstTile, executedNumList, tmpTile, src0Tile, src1Tile);
    } else if (remainListNum == MERGE_LIST_THREE) {
        pto::TMRGSORT<PtoPackedSortTile, PtoPackedSortTile, PtoPackedSortTile, PtoPackedSortTile, PtoPackedSortTile,
                      true>(dstTile, executedNumList, tmpTile, src0Tile, src1Tile, src2Tile);
    } else {
        pto::TMRGSORT<PtoPackedSortTile, PtoPackedSortTile, PtoPackedSortTile, PtoPackedSortTile, PtoPackedSortTile,
                      PtoPackedSortTile, true>(dstTile, executedNumList, tmpTile, src0Tile, src1Tile, src2Tile,
                                               src3Tile);
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

    const uint32_t alignedElemNum =
        ((elemNum + PTO_SORT_BLOCK_ELEMS - 1) / PTO_SORT_BLOCK_ELEMS) * PTO_SORT_BLOCK_ELEMS;
    ASCENDC_ASSERT((alignedElemNum <= MAX_SORT_ELEMS),
                   { KERNEL_LOG(KERNEL_ERROR, "alignedElemNum exceeds PTO sort capacity"); });

    const uint64_t sortKeyUb = mergeTmpUb;
    pto_detail::PtoCastVector<float, int32_t>(sortKeyUb, inputValueUb, elemNum, pto::RoundMode::CAST_CEIL);
    pto_detail::PtoMulVector<float>(sortKeyUb, sortKeyUb, elemNum, -1.0F);

    __ubuf__ float *sortKeyPtr = reinterpret_cast<__ubuf__ float *>(sortKeyUb);
    __ubuf__ uint32_t *payloadPtr = reinterpret_cast<__ubuf__ uint32_t *>(inputPayloadUb);
    for (uint32_t i = elemNum; i < alignedElemNum; ++i) {
        sortKeyPtr[i] = PTO_SORT_NEG_INF;
        payloadPtr[i] = 0;
    }

    PtoSortKeyTile srcTile(1, alignedElemNum);
    PtoSortPayloadTile payloadTile(1, alignedElemNum);
    PtoPackedSortTile packedTile(1, alignedElemNum * 2);
    PtoPackedSortTile mergeTmpTile(1, alignedElemNum * 2);
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
    PtoPackedPayloadTile packedPayloadTile(1, elemNum * 2);
    PtoSortPayloadTile sortedPayloadTile(1, elemNum);
    pto::TASSIGN(packedPayloadTile, packedSortUb);
    pto::TASSIGN(sortedPayloadTile, sortedPayloadUb);
    pto::TGATHER<PtoSortPayloadTile, PtoPackedPayloadTile, pto::MaskPattern::P1010>(sortedPayloadTile,
                                                                                    packedPayloadTile);
    AscendC::PipeBarrier<PIPE_V>();

    PtoSortKeyTile sortedKeyTile(1, elemNum);
    PtoPackedSortTile packedTile(1, elemNum * 2);
    pto::TASSIGN(sortedKeyTile, sortedValueScratchUb);
    pto::TASSIGN(packedTile, packedSortUb);
    pto::TGATHER<PtoSortKeyTile, PtoPackedSortTile, pto::MaskPattern::P0101>(sortedKeyTile, packedTile);
    AscendC::PipeBarrier<PIPE_V>();
    pto_detail::PtoMulVector<float>(sortedValueScratchUb, sortedValueScratchUb, elemNum, -1.0F);

    pto_detail::PtoCastVector<int32_t, float>(sortedValueUb, sortedValueScratchUb, elemNum, pto::RoundMode::CAST_CEIL);
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

} // namespace MoeInitRoutingQuant

#endif // INNER_MOE_PTO_SORT_H
