/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef HEADER_IMPL_HPP_KIRINX90
#define HEADER_IMPL_HPP_KIRINX90

namespace pto {
namespace kirinX90 {

// ============================================================================
// Wrapper functions for A5/kirin9030 _IMPL functions in kirinX90 namespace
// Each wrapper defines the function in kirin9030 namespace, forwarding to ::pto::a5
// This replaces the previous 'using ::pto::a5::XXX' declarations
// ============================================================================

template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL>
PTO_INTERNAL void SETFMATRIX_IMPL(ConvTileData &src)
{
    ::pto::a5::SETFMATRIX_IMPL<ConvTileData, FmatrixMode>(src);
}

template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL>
PTO_INTERNAL void SET_IMG2COL_PADDING_IMPL(ConvTileData &src)
{
    ::pto::a5::SET_IMG2COL_PADDING_IMPL<ConvTileData, FmatrixMode>(src);
}

template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL>
PTO_INTERNAL void SET_IMG2COL_RPT_IMPL(ConvTileData &src)
{
    ::pto::a5::SET_IMG2COL_RPT_IMPL<ConvTileData, FmatrixMode>(src);
}

template <typename OutType>
PTO_INTERNAL void SET_QUANT_SCALAR_IMPL(float preQuantScalar)
{
    ::pto::a5::SET_QUANT_SCALAR_IMPL<OutType>(preQuantScalar);
}

template <typename FpTileData>
PTO_INTERNAL void SET_QUANT_VECTOR_IMPL(FpTileData &fpTile)
{
    ::pto::a5::SET_QUANT_VECTOR_IMPL<FpTileData>(fpTile);
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TABS_IMPL(DstTile &dst, SrcTile &src)
{
    ::pto::a5::TABS_IMPL<DstTile, SrcTile>(dst, src);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TADDDEQRELU_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, float deqScale,
                                   TileDataTmp &tmp)
{
    ::pto::a5::TADDDEQRELU_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1, TileDataTmp>(dst, src0, src1, deqScale, tmp);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TADD_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TADD_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TADDRELUCONV_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TADDRELUCONV_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TADDS_IMPL(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType src1)
{
    ::pto::a5::TADDS_IMPL<TileDataDst, TileDataSrc>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TAND_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TAND_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TANDS_IMPL(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType src1)
{
    ::pto::a5::TANDS_IMPL<TileDataDst, TileDataSrc>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TAXPY_IMPL(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar)
{
    ::pto::a5::TAXPY_IMPL<TileDataDst, TileDataSrc>(dst, src0, scalar);
}

template <typename TileData, typename T, int descending>
PTO_INTERNAL void TCI_IMPL(TileData &dst, T start)
{
    ::pto::a5::TCI_IMPL<TileData, T, descending>(dst, start);
}

template <typename TileData, typename TileDataTmp, typename T, int descending>
PTO_INTERNAL void TCI_IMPL(TileData &dst, T start, TileDataTmp &tmp)
{
    ::pto::a5::TCI_IMPL<TileData, TileDataTmp, T, descending>(dst, start, tmp);
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
PTO_INTERNAL void TCMP_IMPL(DstTile &dst, SrcTile0 &src0, SrcTile1 &src1, CmpMode cmpMode)
{
    ::pto::a5::TCMP_IMPL<DstTile, SrcTile0, SrcTile1>(dst, src0, src1, cmpMode);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TCMPS_IMPL(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType src1, CmpMode mode)
{
    ::pto::a5::TCMPS_IMPL<TileDataDst, TileDataSrc>(dst, src0, src1, mode);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1,
          typename = std::void_t<typename TileDataSrc1::DType>>
PTO_INTERNAL void TCMPS_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, CmpMode mode)
{
    ::pto::a5::TCMPS_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1, mode);
}

template <typename TileData, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TCOLEXPANDADD_IMPL(TileData &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TCOLEXPANDADD_IMPL<TileData, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileData, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TCOLEXPANDDIV_IMPL(TileData &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TCOLEXPANDDIV_IMPL<PrecisionType, TileData, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileData, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TCOLEXPANDEXPDIF_IMPL(TileData &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TCOLEXPANDEXPDIF_IMPL<TileData, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TCOLEXPAND_IMPL(TileDataDst &dst, TileDataSrc &src)
{
    ::pto::a5::TCOLEXPAND_IMPL<TileDataDst, TileDataSrc>(dst, src);
}

template <typename TileData, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TCOLEXPANDMAX_IMPL(TileData &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TCOLEXPANDMAX_IMPL<TileData, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileData, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TCOLEXPANDMIN_IMPL(TileData &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TCOLEXPANDMIN_IMPL<TileData, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileData, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TCOLEXPANDMUL_IMPL(TileData &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TCOLEXPANDMUL_IMPL<TileData, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileData, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TCOLEXPANDSUB_IMPL(TileData &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TCOLEXPANDSUB_IMPL<TileData, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileDataOut, typename TileDataIn>
PTO_INTERNAL void TCOLMAX_IMPL(TileDataOut &dst, TileDataIn &src)
{
    ::pto::a5::TCOLMAX_IMPL<TileDataOut, TileDataIn>(dst, src);
}

template <typename TileDataOut, typename TileDataIn>
PTO_INTERNAL void TCOLMIN_IMPL(TileDataOut &dst, TileDataIn &src)
{
    ::pto::a5::TCOLMIN_IMPL<TileDataOut, TileDataIn>(dst, src);
}

template <typename TileDataOut, typename TileDataIn>
PTO_INTERNAL void TCOLPROD_IMPL(TileDataOut &dst, TileDataIn &src)
{
    ::pto::a5::TCOLPROD_IMPL<TileDataOut, TileDataIn>(dst, src);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TCOLSUM_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, bool isBinary)
{
    ::pto::a5::TCOLSUM_IMPL<TileDataOut, TileDataIn, TileDataTmp>(dst, src, tmp, isBinary);
}

template <typename TileDataOut, typename TileDataIn>
PTO_INTERNAL void TCOLSUM_IMPL(TileDataOut &dst, TileDataIn &src)
{
    ::pto::a5::TCOLSUM_IMPL<TileDataOut, TileDataIn>(dst, src);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TCOLARGMAX_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    ::pto::a5::TCOLARGMAX_IMPL<TileDataOut, TileDataIn, TileDataTmp>(dst, src, tmp);
}

template <typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TCOLARGMAX_IMPL(TileDataOutVal &dstVal, TileDataOutIdx &dstIdx, TileDataIn &src, TileDataTmp &tmp)
{
    ::pto::a5::TCOLARGMAX_IMPL<TileDataOutVal, TileDataOutIdx, TileDataIn, TileDataTmp>(dstVal, dstIdx, src, tmp);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TCOLARGMIN_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    ::pto::a5::TCOLARGMIN_IMPL<TileDataOut, TileDataIn, TileDataTmp>(dst, src, tmp);
}

template <typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TCOLARGMIN_IMPL(TileDataOutVal &dstVal, TileDataOutIdx &dstIdx, TileDataIn &src, TileDataTmp &tmp)
{
    ::pto::a5::TCOLARGMIN_IMPL<TileDataOutVal, TileDataOutIdx, TileDataIn, TileDataTmp>(dstVal, dstIdx, src, tmp);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TCONCAT_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TCONCAT_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename TileSrc0Idx, typename TileSrc1Idx>
PTO_INTERNAL void TCONCAT_IMPL(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1, TileSrc0Idx &src0Idx, TileSrc1Idx &src1Idx)
{
    ::pto::a5::TCONCAT_IMPL<TileDst, TileSrc0, TileSrc1, TileSrc0Idx, TileSrc1Idx>(dst, src0, src1, src0Idx, src1Idx);
}

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename TileDstIdx, typename TileSrc0Idx,
          typename TileSrc1Idx>
PTO_INTERNAL void TCONCAT_IMPL(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1, TileDstIdx &dstIdx, TileSrc0Idx &src0Idx,
                               TileSrc1Idx &src1Idx)
{
    ::pto::a5::TCONCAT_IMPL<TileDst, TileSrc0, TileSrc1, TileDstIdx, TileSrc0Idx, TileSrc1Idx>(
        dst, src0, src1, dstIdx, src0Idx, src1Idx);
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataPara>
PTO_INTERNAL void TDEQUANT_IMPL(TileDataDst &dst, TileDataSrc &src, TileDataPara &scale, TileDataPara &offset)
{
    ::pto::a5::TDEQUANT_IMPL<TileDataDst, TileDataSrc, TileDataPara>(dst, src, scale, offset);
}

template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0,
          typename TileDataSrc1>
PTO_INTERNAL void TDIV_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TDIV_IMPL<PrecisionType, TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TDIVS_IMPL(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar)
{
    ::pto::a5::TDIVS_IMPL<PrecisionType, TileDataDst, TileDataSrc>(dst, src0, scalar);
}

template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TDIVS_IMPL(TileDataDst &dst, typename TileDataSrc::DType scalar, TileDataSrc &src0)
{
    ::pto::a5::TDIVS_IMPL<PrecisionType, TileDataDst, TileDataSrc>(dst, scalar, src0);
}

template <typename TileData>
PTO_INTERNAL void TEXPANDS_IMPL(TileData &dst, typename TileData::DType scalar)
{
    ::pto::a5::TEXPANDS_IMPL<TileData>(dst, scalar);
}

template <auto PrecisionType = ExpAlgorithm::DEFAULT, typename DstTile, typename SrcTile>
PTO_INTERNAL void TEXP_IMPL(DstTile &dst, SrcTile &src)
{
    ::pto::a5::TEXP_IMPL<PrecisionType, DstTile, SrcTile>(dst, src);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TFILLPAD_EXPAND_IMPL(TileDataDst &dst, TileDataSrc &src)
{
    ::pto::a5::TFILLPAD_EXPAND_IMPL<TileDataDst, TileDataSrc>(dst, src);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TFILLPAD_IMPL(TileDataDst &dst, TileDataSrc &src)
{
    ::pto::a5::TFILLPAD_IMPL<TileDataDst, TileDataSrc>(dst, src);
}

template <typename TileData, PadValue PadVal = PadValue::Zero>
PTO_INTERNAL void TFILLPAD_IMPL(TileData &dst, TileData &src)
{
    ::pto::a5::TFILLPAD_IMPL<TileData, PadVal>(dst, src);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TFILLPAD_INPLACE_IMPL(TileDataDst &dst, TileDataSrc &src)
{
    ::pto::a5::TFILLPAD_INPLACE_IMPL<TileDataDst, TileDataSrc>(dst, src);
}

template <auto PrecisionType = FmodAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0,
          typename TileDataSrc1>
PTO_INTERNAL void TFMOD_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TFMOD_IMPL<PrecisionType, TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <auto PrecisionType = FmodSAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TFMODS_IMPL(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar)
{
    ::pto::a5::TFMODS_IMPL<PrecisionType, TileDataDst, TileDataSrc>(dst, src, scalar);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TFUSEDMULADD_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TFUSEDMULADD_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TFUSEDMULADDRELU_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TFUSEDMULADDRELU_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataOffset>
PTO_INTERNAL void TGATHERB_IMPL(TileDataDst &dst, TileDataSrc &src, TileDataOffset &offset)
{
    ::pto::a5::TGATHERB_IMPL<TileDataDst, TileDataSrc, TileDataOffset>(dst, src, offset);
}

template <typename TileDataOut, typename TileDataIn>
PTO_INTERNAL void TGET_SCALE_ADDR_IMPL(TileDataOut &dst, TileDataIn &src)
{
    ::pto::a5::TGET_SCALE_ADDR_IMPL<TileDataOut, TileDataIn>(dst, src);
}

// THISTOGRAM_IMPL: not supported on kirinX90 (THistogram.hpp uses A5-specific vsts)
template <HistByte byte, typename TileDst, typename TileSrc, typename TileIdx>
PTO_INTERNAL void THISTOGRAM_IMPL(TileDst &dst, TileSrc &src, TileIdx &idx)
{
    static_assert(sizeof(TileDst) == 0, "THISTOGRAM is not supported on kirinX90");
}

template <typename TileData, typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL>
PTO_INTERNAL void TIMG2COL_IMPL(TileData &dst, ConvTileData &src, uint16_t posM, uint16_t posK)
{
    ::pto::a5::TIMG2COL_IMPL<TileData, ConvTileData, FmatrixMode>(dst, src, posM, posK);
}

// TINSERT_IMPL: kirinX90 has its own implementation in TInsert.hpp, no wrapper needed

template <auto PrecisionType = LogAlgorithm::DEFAULT, typename DstTile, typename SrcTile>
PTO_INTERNAL void TLOG_IMPL(DstTile &dst, SrcTile &src)
{
    ::pto::a5::TLOG_IMPL<PrecisionType, DstTile, SrcTile>(dst, src);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TLRELU_IMPL(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar)
{
    ::pto::a5::TLRELU_IMPL<TileDataDst, TileDataSrc>(dst, src0, scalar);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TMAX_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TMAX_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TMAXS_IMPL(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar)
{
    ::pto::a5::TMAXS_IMPL<TileDataDst, TileDataSrc>(dst, src, scalar);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TINTERLEAVE_IMPL(TileDataDst &dst1, TileDataDst &dst0, TileDataSrc &src1, TileDataSrc &src0)
{
    ::pto::a5::TINTERLEAVE_IMPL<TileDataDst, TileDataSrc>(dst1, dst0, src1, src0);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TDEINTERLEAVE_IMPL(TileDataDst &dst1, TileDataDst &dst0, TileDataSrc &src1, TileDataSrc &src0)
{
    ::pto::a5::TDEINTERLEAVE_IMPL<TileDataDst, TileDataSrc>(dst1, dst0, src1, src0);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TDEINTERLEAVE_IMPL(TileDataDst &dst1, TileDataDst &dst0, TileDataSrc &src)
{
    ::pto::a5::TDEINTERLEAVE_IMPL<TileDataDst, TileDataSrc>(dst1, dst0, src);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TMIN_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TMIN_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TMINS_IMPL(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType src1)
{
    ::pto::a5::TMINS_IMPL<TileDataDst, TileDataSrc>(dst, src0, src1);
}

template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
          typename Src2TileData, typename Src3TileData, bool exhausted>
PTO_INTERNAL void TMRGSORT_IMPL(DstTileData &dst, MrgSortExecutedNumList &executedNumList, TmpTileData &tmp,
                                 Src0TileData &src0, Src1TileData &src1, Src2TileData &src2, Src3TileData &src3)
{
    ::pto::a5::TMRGSORT_IMPL<DstTileData, TmpTileData, Src0TileData, Src1TileData, Src2TileData, Src3TileData, exhausted>(
        dst, executedNumList, tmp, src0, src1, src2, src3);
}

template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
          typename Src2TileData, bool exhausted>
PTO_INTERNAL void TMRGSORT_IMPL(DstTileData &dst, MrgSortExecutedNumList &executedNumList, TmpTileData &tmp,
                                 Src0TileData &src0, Src1TileData &src1, Src2TileData &src2)
{
    ::pto::a5::TMRGSORT_IMPL<DstTileData, TmpTileData, Src0TileData, Src1TileData, Src2TileData, exhausted>(
        dst, executedNumList, tmp, src0, src1, src2);
}

template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData, bool exhausted>
PTO_INTERNAL void TMRGSORT_IMPL(DstTileData &dst, MrgSortExecutedNumList &executedNumList, TmpTileData &tmp,
                                 Src0TileData &src0, Src1TileData &src1)
{
    ::pto::a5::TMRGSORT_IMPL<DstTileData, TmpTileData, Src0TileData, Src1TileData, exhausted>(
        dst, executedNumList, tmp, src0, src1);
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TMRGSORT_IMPL(DstTileData &dst, SrcTileData &src, uint32_t blockLen)
{
    ::pto::a5::TMRGSORT_IMPL<DstTileData, SrcTileData>(dst, src, blockLen);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TMUL_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TMUL_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TMULADDDST_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TMULADDDST_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TNEG_IMPL(DstTile &dst, SrcTile &src)
{
    ::pto::a5::TNEG_IMPL<DstTile, SrcTile>(dst, src);
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TNOT_IMPL(DstTile &dst, SrcTile &src)
{
    ::pto::a5::TNOT_IMPL<DstTile, SrcTile>(dst, src);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TOR_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TOR_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TORS_IMPL(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType src1)
{
    ::pto::a5::TORS_IMPL<TileDataDst, TileDataSrc>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc0>
PTO_INTERNAL void TPAIRREDUCESUM_IMPL(TileDataDst &dst, TileDataSrc0 &src0)
{
    ::pto::a5::TPAIRREDUCESUM_IMPL<TileDataDst, TileDataSrc0>(dst, src0);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TPARTADD_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TPARTADD_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename DstValTileData, typename Src0ValTileData, typename Src1ValTileData, typename DstIdxTileData,
          typename Src0IdxTileData, typename Src1IdxTileData>
PTO_INTERNAL void TPARTARGMAX_IMPL(DstValTileData &dstVal, Src0ValTileData &src0Val, Src1ValTileData &src1Val,
                                   DstIdxTileData &dstIdx, Src0IdxTileData &src0Idx, Src1IdxTileData &src1Idx)
{
    ::pto::a5::TPARTARGMAX_IMPL<DstValTileData, Src0ValTileData, Src1ValTileData, DstIdxTileData,
                                Src0IdxTileData, Src1IdxTileData>(
        dstVal, src0Val, src1Val, dstIdx, src0Idx, src1Idx);
}

template <typename DstValTileData, typename Src0ValTileData, typename Src1ValTileData, typename DstIdxTileData,
          typename Src0IdxTileData, typename Src1IdxTileData>
PTO_INTERNAL void TPARTARGMIN_IMPL(DstValTileData &dstVal, Src0ValTileData &src0Val, Src1ValTileData &src1Val,
                                   DstIdxTileData &dstIdx, Src0IdxTileData &src0Idx, Src1IdxTileData &src1Idx)
{
    ::pto::a5::TPARTARGMIN_IMPL<DstValTileData, Src0ValTileData, Src1ValTileData, DstIdxTileData,
                                Src0IdxTileData, Src1IdxTileData>(
        dstVal, src0Val, src1Val, dstIdx, src0Idx, src1Idx);
}

template <typename DstTileData, typename Src0TileData, typename Src1TileData>
PTO_INTERNAL void TPARTMAX_IMPL(DstTileData &dst, Src0TileData &src0, Src1TileData &src1)
{
    ::pto::a5::TPARTMAX_IMPL<DstTileData, Src0TileData, Src1TileData>(dst, src0, src1);
}

template <typename DstTileData, typename Src0TileData, typename Src1TileData>
PTO_INTERNAL void TPARTMIN_IMPL(DstTileData &dst, Src0TileData &src0, Src1TileData &src1)
{
    ::pto::a5::TPARTMIN_IMPL<DstTileData, Src0TileData, Src1TileData>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TPARTMUL_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TPARTMUL_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <PowAlgorithm algo, typename DstTile, typename BaseTile, typename ExpTile, typename TmpTile>
PTO_INTERNAL void TPOW_IMPL(DstTile &dst, BaseTile &base, ExpTile &exp, TmpTile &tmp)
{
    ::pto::a5::TPOW_IMPL<algo, DstTile, BaseTile, ExpTile, TmpTile>(dst, base, exp, tmp);
}

template <PowAlgorithm algo, typename DstTile, typename BaseTile, typename TmpTile>
PTO_INTERNAL void TPOWS_IMPL(DstTile &dst, BaseTile &base, typename DstTile::DType exp, TmpTile &tmp)
{
    ::pto::a5::TPOWS_IMPL<algo, DstTile, BaseTile, TmpTile>(dst, base, exp, tmp);
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TPREFETCH_IMPL(TileData &dst, GlobalData &src)
{
    ::pto::a5::TPREFETCH_IMPL<TileData, GlobalData>(dst, src);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TPRELU_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp)
{
    ::pto::a5::TPRELU_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1, TileDataTmp>(dst, src0, src1, tmp);
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TRELU_IMPL(DstTile &dst, SrcTile &src)
{
    ::pto::a5::TRELU_IMPL<DstTile, SrcTile>(dst, src);
}

template <typename TileDataOut, typename TileDataIn>
PTO_INTERNAL void TRESHAPE_IMPL(TileDataOut &dst, TileDataIn &src)
{
    ::pto::a5::TRESHAPE_IMPL<TileDataOut, TileDataIn>(dst, src);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWARGMAX_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    ::pto::a5::TROWARGMAX_IMPL<TileDataOut, TileDataIn, TileDataTmp>(dst, src, tmp);
}

template <typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWARGMAX_IMPL(TileDataOutVal &dstVal, TileDataOutIdx &dstIdx, TileDataIn &src, TileDataTmp &tmp)
{
    ::pto::a5::TROWARGMAX_IMPL<TileDataOutVal, TileDataOutIdx, TileDataIn, TileDataTmp>(dstVal, dstIdx, src, tmp);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWARGMIN_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    ::pto::a5::TROWARGMIN_IMPL<TileDataOut, TileDataIn, TileDataTmp>(dst, src, tmp);
}

template <typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWARGMIN_IMPL(TileDataOutVal &dstVal, TileDataOutIdx &dstIdx, TileDataIn &src, TileDataTmp &tmp)
{
    ::pto::a5::TROWARGMIN_IMPL<TileDataOutVal, TileDataOutIdx, TileDataIn, TileDataTmp>(dstVal, dstIdx, src, tmp);
}

template <typename TileDataOut, typename TileDataIn>
PTO_INTERNAL void TROWEXPAND_IMPL(TileDataOut &dst, TileDataIn &src)
{
    ::pto::a5::TROWEXPAND_IMPL<TileDataOut, TileDataIn>(dst, src);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWMAX_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    ::pto::a5::TROWMAX_IMPL<TileDataOut, TileDataIn, TileDataTmp>(dst, src, tmp);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWMIN_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    ::pto::a5::TROWMIN_IMPL<TileDataOut, TileDataIn, TileDataTmp>(dst, src, tmp);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWPROD_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    ::pto::a5::TROWPROD_IMPL<TileDataOut, TileDataIn, TileDataTmp>(dst, src, tmp);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWSUM_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    ::pto::a5::TROWSUM_IMPL<TileDataOut, TileDataIn, TileDataTmp>(dst, src, tmp);
}

template <typename DstTile, typename SrcTile, typename IdxTile>
PTO_INTERNAL void TSCATTER_IMPL(DstTile &dst, SrcTile &src, IdxTile &idx)
{
    ::pto::a5::TSCATTER_IMPL<DstTile, SrcTile, IdxTile>(dst, src, idx);
}

template <MaskPattern mask, auto ScatterType = ScatterAxis::SCATTER_ROW, typename DstTile, typename SrcTile>
PTO_INTERNAL void TSCATTER_IMPL(DstTile &dst, SrcTile &src)
{
    ::pto::a5::TSCATTER_IMPL<mask, ScatterType, DstTile, SrcTile>(dst, src);
}

template <typename DstTile, typename MaskTile, typename Src0Tile, typename Src1Tile, typename TmpTile>
PTO_INTERNAL void TSEL_IMPL(DstTile &dst, MaskTile &selMask, Src0Tile &src0, Src1Tile &src1, TmpTile &tmp)
{
    ::pto::a5::TSEL_IMPL<DstTile, MaskTile, Src0Tile, Src1Tile, TmpTile>(dst, selMask, src0, src1, tmp);
}

template <typename TileDataDst, typename TileDataMask, typename TileDataSrc, typename TileDataTmp>
PTO_INTERNAL void TSELS_IMPL(TileDataDst &dst, TileDataMask &mask, TileDataSrc &src, TileDataTmp &tmp,
                             typename TileDataSrc::DType scalar)
{
    ::pto::a5::TSELS_IMPL<TileDataDst, TileDataMask, TileDataSrc, TileDataTmp>(dst, mask, src, tmp, scalar);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TSHL_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TSHL_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TSHLS_IMPL(TileDataDst &dst, TileDataSrc &src0, typename TileDataDst::DType src1)
{
    ::pto::a5::TSHLS_IMPL<TileDataDst, TileDataSrc>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TSHR_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TSHR_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TSHRS_IMPL(TileDataDst &dst, TileDataSrc &src0, typename TileDataDst::DType src1)
{
    ::pto::a5::TSHRS_IMPL<TileDataDst, TileDataSrc>(dst, src0, src1);
}

template <typename DstTileData, typename SrcTileData, typename IdxTileData>
PTO_INTERNAL void TSORT32_IMPL(DstTileData &dst, SrcTileData &src, IdxTileData &idx)
{
    ::pto::a5::TSORT32_IMPL<DstTileData, SrcTileData, IdxTileData>(dst, src, idx);
}

template <typename DstTileData, typename SrcTileData, typename IdxTileData, typename TmpTileData>
PTO_INTERNAL void TSORT32_IMPL(DstTileData &dst, SrcTileData &src, IdxTileData &idx, TmpTileData &tmp)
{
    ::pto::a5::TSORT32_IMPL<DstTileData, SrcTileData, IdxTileData, TmpTileData>(dst, src, idx, tmp);
}

template <auto PrecisionType = SqrtAlgorithm::DEFAULT, typename DstTile, typename SrcTile>
PTO_INTERNAL void TSQRT_IMPL(DstTile &dst, SrcTile &src)
{
    ::pto::a5::TSQRT_IMPL<PrecisionType, DstTile, SrcTile>(dst, src);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TSUB_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TSUB_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TSUBRELU_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TSUBRELU_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TSUBRELUCONV_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TSUBRELUCONV_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TSUBVIEW_IMPL(TileDataDst &dst, TileDataSrc &src, uint16_t rowIdx, uint16_t colIdx)
{
    ::pto::a2a3::TSUBVIEW_IMPL<TileDataDst, TileDataSrc>(dst, src, rowIdx, colIdx);
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INTERNAL void TTRANS_IMPL(TileDataDst &dst, TileDataSrc &src, TileDataTmp &tmp)
{
    ::pto::a5::TTRANS_IMPL<TileDataDst, TileDataSrc, TileDataTmp>(dst, src, tmp);
}

template <typename TileData, int upperOrLower>
PTO_INTERNAL void TTRI_IMPL(TileData &dst, int diagonal)
{
    ::pto::a5::TTRI_IMPL<TileData, upperOrLower>(dst, diagonal);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TXOR_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp)
{
    ::pto::a5::TXOR_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1, TileDataTmp>(dst, src0, src1, tmp);
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INTERNAL void TXORS_IMPL(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType src1, TileDataTmp &tmp)
{
    ::pto::a5::TXORS_IMPL<TileDataDst, TileDataSrc, TileDataTmp>(dst, src0, src1, tmp);
}


// ============================================================================
// kirin9030-specific wrappers (kirinX90 reuses kirin9030 implementations)
// ============================================================================

template <Op OpCode>
PTO_INTERNAL void TSYNC_IMPL()
{
    ::pto::kirin9030::TSYNC_IMPL<OpCode>();
}

template <typename T>
struct RemOp {
    using U = std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint16_t>;
    __tf__ U operator()(U src0, U src1) const { return src0 % src1; }
};

template <auto PrecisionType = RemAlgorithm::DEFAULT, typename DstTile, typename Src0Tile, typename Src1Tile,
          typename TileDataTmp>
PTO_INTERNAL void TREM_IMPL(DstTile &dst, Src0Tile &src0, Src1Tile &src1, TileDataTmp &tmp)
{
    ::pto::kirin9030::TREM_IMPL<PrecisionType, DstTile, Src0Tile, Src1Tile, TileDataTmp>(dst, src0, src1, tmp);
}

template <auto PrecisionType = RemSAlgorithm::DEFAULT, typename DstTile, typename SrcTile, typename TileDataTmp>
PTO_INTERNAL void TREMS_IMPL(DstTile &dst, SrcTile &src, typename SrcTile::DType scalar, TileDataTmp &tmp)
{
    ::pto::kirin9030::TREMS_IMPL<PrecisionType, DstTile, SrcTile, TileDataTmp>(dst, src, scalar, tmp);
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TSUBS_IMPL(DstTile &dst, SrcTile &src0, typename SrcTile::DType src1)
{
    ::pto::kirin9030::TSUBS_IMPL<DstTile, SrcTile>(dst, src0, src1);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    ::pto::kirin9030::TMATMUL_IMPL<Phase, TileRes, TileLeft, TileRight>(cMatrix, aMatrix, bMatrix);
}

template <typename TileDataD, typename TileDataS0, typename TileDataS1, typename TileDataTmp>
PTO_INTERNAL void TGATHER_IMPL(TileDataD &dst, TileDataS0 &src0, TileDataS1 &src1, TileDataTmp &tmp)
{
    ::pto::kirin9030::TGATHER_IMPL<TileDataD, TileDataS0, TileDataS1, TileDataTmp>(dst, src0, src1, tmp);
}

template <typename DstTileData, typename SrcTileData, MaskPattern maskPattern, auto gatherType = GatherAxis::GATHER_ROW>
PTO_INTERNAL void TGATHER_IMPL(DstTileData &dst, SrcTileData &src)
{
    ::pto::kirin9030::TGATHER_IMPL<DstTileData, SrcTileData, maskPattern, gatherType>(dst, src);
}

template <typename TileDataD, typename TileDataS, typename TileDataS1, typename TileDataC, typename TileDataTmp>
PTO_INTERNAL void TGATHER_IMPL(TileDataD &dst, TileDataS &src0, TileDataS1 &k_value, TileDataC &cdst, TileDataTmp &tmp,
                               uint32_t offset)
{
    ::pto::kirin9030::TGATHER_IMPL<TileDataD, TileDataS, TileDataS1, TileDataC, TileDataTmp>(
        dst, src0, k_value, cdst, tmp, offset);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_ACC_IMPL(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    ::pto::kirin9030::TMATMUL_ACC_IMPL<Phase, TileRes, TileLeft, TileRight>(cOutMatrix, cInMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_ACC_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    ::pto::kirin9030::TMATMUL_ACC_IMPL<Phase, TileRes, TileLeft, TileRight>(cMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight,
          typename TileBias>
PTO_INTERNAL void TMATMUL_BIAS_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData)
{
    ::pto::kirin9030::TMATMUL_BIAS_IMPL<Phase, TileRes, TileLeft, TileRight, TileBias>(cMatrix, aMatrix, bMatrix, biasData);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TGEMV_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    ::pto::kirin9030::TGEMV_IMPL<Phase, TileRes, TileLeft, TileRight>(cMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TGEMV_ACC_IMPL(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    ::pto::kirin9030::TGEMV_ACC_IMPL<Phase, TileRes, TileLeft, TileRight>(cOutMatrix, cInMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight,
          typename TileBias>
PTO_INTERNAL void TGEMV_BIAS_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData)
{
    ::pto::kirin9030::TGEMV_BIAS_IMPL<Phase, TileRes, TileLeft, TileRight, TileBias>(cMatrix, aMatrix, bMatrix, biasData);
}

template <typename Op, typename Op2, typename TileData, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TCOLEXPANDOP_IMPL(TileData &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TCOLEXPANDOP_IMPL<Op, Op2, TileData, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TROWEXPANDADD_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TROWEXPANDADD_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TROWEXPANDADD_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp)
{
    ::pto::a5::TROWEXPANDADD_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1, TileDataTmp>(dst, src0, src1, tmp);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TROWEXPANDSUB_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TROWEXPANDSUB_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TROWEXPANDSUB_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp)
{
    ::pto::a5::TROWEXPANDSUB_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1, TileDataTmp>(dst, src0, src1, tmp);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TROWEXPANDMUL_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TROWEXPANDMUL_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TROWEXPANDMUL_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp)
{
    ::pto::a5::TROWEXPANDMUL_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1, TileDataTmp>(dst, src0, src1, tmp);
}

template <auto PrecisionType, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TROWEXPANDDIV_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TROWEXPANDDIV_IMPL<PrecisionType, TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <auto PrecisionType, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TROWEXPANDDIV_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp)
{
    ::pto::a5::TROWEXPANDDIV_IMPL<PrecisionType, TileDataDst, TileDataSrc0, TileDataSrc1, TileDataTmp>(dst, src0, src1, tmp);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TROWEXPANDMAX_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TROWEXPANDMAX_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TROWEXPANDMAX_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp)
{
    ::pto::a5::TROWEXPANDMAX_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1, TileDataTmp>(dst, src0, src1, tmp);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TROWEXPANDMIN_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TROWEXPANDMIN_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TROWEXPANDMIN_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp)
{
    ::pto::a5::TROWEXPANDMIN_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1, TileDataTmp>(dst, src0, src1, tmp);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TROWEXPANDEXPDIF_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::a5::TROWEXPANDEXPDIF_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TROWEXPANDEXPDIF_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp)
{
    ::pto::a5::TROWEXPANDEXPDIF_IMPL<TileDataDst, TileDataSrc0, TileDataSrc1, TileDataTmp>(dst, src0, src1, tmp);
}

template <typename TileDataDst, typename TileDataSrc, bool inplace>
PTO_INTERNAL void TFILLPAD_GENERIC_IMPL(TileDataDst &dst, TileDataSrc &src)
{
    ::pto::a5::TFILLPAD_GENERIC_IMPL<TileDataDst, TileDataSrc, inplace>(dst, src);
}

template <auto PrecisionType, typename DstTile, typename SrcTile>
PTO_INTERNAL void TRSQRT_IMPL(DstTile &dst, SrcTile &src)
{
    ::pto::a5::TRSQRT_IMPL<PrecisionType, DstTile, SrcTile>(dst, src);
}

template <auto PrecisionType, typename DstTile, typename SrcTile, typename TmpTile>
PTO_INTERNAL void TRSQRT_IMPL(DstTile &dst, SrcTile &src, TmpTile &tmp)
{
    ::pto::a5::TRSQRT_IMPL<PrecisionType, DstTile, SrcTile, TmpTile>(dst, src, tmp);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TMULS_IMPL(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType src1)
{
    ::pto::a5::TMULS_IMPL<TileDataDst, TileDataSrc>(dst, src0, src1);
}

template <typename... Args>
PTO_INTERNAL void MGATHER_IMPL(Args&&...) { static_assert(sizeof...(Args) == 0, "MGATHER not supported on kirinX90"); }

template <typename... Args>
PTO_INTERNAL void MSCATTER_IMPL(Args&&...) { static_assert(sizeof...(Args) == 0, "MSCATTER not supported on kirinX90"); }

#ifdef __DAV_VEC__
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVT_IMPL(TileDataD &dst, TileDataS &src, RoundMode mode, bool needSetCtrl = true)
{
    ::TCVT_IMPL(dst, src, mode, needSetCtrl);
}

template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVT_IMPL(TileDataD &dst, TileDataS &src, RoundMode mode, SaturationMode satMode,
                            bool needSetCtrl = true)
{
    ::TCVT_IMPL(dst, src, mode, satMode, needSetCtrl);
}

template <typename TileDataD, typename TileDataS, typename TmpTileData>
PTO_INTERNAL void TCVT_IMPL(TileDataD &dst, TileDataS &src, TmpTileData &tmp, RoundMode mode,
                            bool needSetCtrl = true)
{
    ::TCVT_IMPL(dst, src, tmp, mode, needSetCtrl);
}

template <typename TileDataD, typename TileDataS, typename TmpTileData>
PTO_INTERNAL void TCVT_IMPL(TileDataD &dst, TileDataS &src, TmpTileData &tmp, RoundMode mode,
                            SaturationMode satMode, bool needSetCtrl = true)
{
    ::TCVT_IMPL(dst, src, tmp, mode, satMode, needSetCtrl);
}
#elif defined(__DAV_CUBE__)
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVT_IMPL(TileDataD &, TileDataS &, RoundMode, bool = true) {}
#endif

// TASSIGN_IMPL: kirinX90 has its own implementation in TAssign.hpp, no wrapper needed

} // namespace kirinX90
} // namespace pto

#endif // HEADER_IMPL_HPP_KIRINX90
