/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef HEADER_HPP
#define HEADER_HPP
#define bfloat16_t half
#define float8_e4m3_t int8_t
#define float8_e5m2_t int8_t
#define hifloat8_t int8_t
#define float8_e8m0_t int8_t
#define float4_e2m1x2_t int64_t
#define float4_e1m2x2_t int64_t
#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/kirin9030/common.hpp>
#include <pto/npu/kirin9030/datatype.hpp>
#include <pto/npu/kirin9030/utils.hpp>

// Kirin9030-specific includes (forwarding wrappers above are visible for MAP_INSTR_IMPL resolution)
#include "pto/npu/kirin9030/TAssign.hpp"
#include "pto/npu/kirin9030/TExtract.hpp"
#include "pto/npu/kirin9030/TMov.hpp"
#include "pto/npu/kirin9030/TCvt.hpp"
#include "pto/npu/kirin9030/TGather.hpp"
#include "pto/npu/kirin9030/TInsert.hpp"
#include "pto/npu/kirin9030/TLoad.hpp"
#include "pto/npu/kirin9030/TMatmul.hpp"
#include "pto/npu/kirin9030/TQuant.hpp"
#include "pto/npu/kirin9030/TStore.hpp"
#include "pto/npu/kirin9030/TSubS.hpp"
#include "pto/npu/kirin9030/TSync.hpp"
#include "pto/npu/kirin9030/TRem.hpp"
#include "pto/npu/kirin9030/TRemS.hpp"

// A5 includes
#include "pto/npu/a5/TAssign.hpp"
#include "pto/npu/a5/TAdd.hpp"
#include "pto/npu/a5/TAnd.hpp"
#include "pto/npu/a5/TAndS.hpp"
#include "pto/npu/a5/TOr.hpp"
#include "pto/npu/a5/TOrS.hpp"
#include "pto/npu/a5/TXor.hpp"
#include "pto/npu/a5/TXorS.hpp"
#include "pto/npu/a5/TFModS.hpp"
#include "pto/npu/a5/TShl.hpp"
#include "pto/npu/a5/TShlS.hpp"
#include "pto/npu/a5/TShr.hpp"
#include "pto/npu/a5/TShrS.hpp"
#include "pto/npu/a5/TPrelu.hpp"
#include "pto/npu/a5/TFMod.hpp"
#include "pto/npu/a5/TAddS.hpp"
#include "pto/npu/a5/TAxpy.hpp"
#include "pto/npu/a5/TDivS.hpp"
#include "pto/npu/a5/TMulS.hpp"
#include "pto/npu/a5/TSub.hpp"
#include "pto/npu/a5/TMin.hpp"
#include "pto/npu/a5/TMax.hpp"
#include "pto/npu/a5/TSubView.hpp"
#include "pto/npu/a5/TGetScaleAddr.hpp"
#include "pto/npu/a5/TMrgSort.hpp"
#include "pto/npu/a5/TCmps.hpp"
#include "pto/npu/a5/TCmp.hpp"
#include "pto/npu/a5/TColSum.hpp"
#include "pto/npu/a5/TColProd.hpp"
#include "pto/npu/a5/TColMax.hpp"
#include "pto/npu/a5/TColMin.hpp"
#include "pto/npu/a5/TColExpand.hpp"
#include "pto/npu/a5/TConcat.hpp"
#include "pto/npu/a5/TReshape.hpp"
#include "pto/npu/a5/TRowReduce.hpp"
#include "pto/npu/a5/TRowReduceIdx.hpp"
#include "pto/npu/a5/TRowProd.hpp"
#include "pto/npu/a5/TFillPad.hpp"
#include "pto/npu/a5/TTrans.hpp"
#include "pto/npu/a5/TLRelu.hpp"
#include "pto/npu/a5/Tci.hpp"
#include "pto/npu/a5/TSels.hpp"
#include "pto/npu/a5/TSel.hpp"
#include "pto/npu/a5/TExpandS.hpp"
#include "pto/npu/a5/TSort32.hpp"
#include "pto/npu/a5/TMins.hpp"
#include "pto/npu/a5/TMaxs.hpp"
#include "pto/npu/a5/TRowExpand.hpp"
#include "pto/npu/a5/TRowExpandAdd.hpp"
#include "pto/npu/a5/TRowExpandDiv.hpp"
#include "pto/npu/a5/TRowExpandMax.hpp"
#include "pto/npu/a5/TRowExpandMin.hpp"
#include "pto/npu/a5/TRowExpandMul.hpp"
#include "pto/npu/a5/TRowExpandSub.hpp"
#include "pto/npu/a5/TRowExpandExpdif.hpp"
#include "pto/npu/a5/TPartAdd.hpp"
#include "pto/npu/a5/TPartMul.hpp"
#include "pto/npu/a5/TPartMax.hpp"
#include "pto/npu/a5/TPartMin.hpp"
#include "pto/npu/a5/TPartArgMax.hpp"
#include "pto/npu/a5/TPartArgMin.hpp"
#include "pto/npu/a5/TPow.hpp"
#include "pto/npu/a5/TDeQuant.hpp"
#include "pto/npu/a5/TImg2col.hpp"
#include "pto/npu/a5/SetFmatrix.hpp"
#include "pto/npu/a5/SetImg2colRpt.hpp"
#include "pto/npu/a5/SetImg2colPadding.hpp"
#include "pto/npu/a5/THistogram.hpp"
// TRandom to be evaluated
#include "pto/npu/a5/TRsqrt.hpp"
#include "pto/npu/a5/TUnaryOp.hpp"
#include "pto/npu/a5/TGatherB.hpp"
#include "pto/npu/a5/TBinSOp.hpp"
#include "pto/npu/a5/TDiv.hpp"
#include "pto/npu/a5/TMul.hpp"
#include "pto/npu/a5/TScatter.hpp"
// MGather to be evaluated
// MScatter to be evaluated
#include "pto/npu/a5/TColExpandDiv.hpp"
#include "pto/npu/a5/TColExpandMul.hpp"
#include "pto/npu/a5/TColExpandSub.hpp"
#include "pto/npu/a5/TColExpandExpdif.hpp"
#include "pto/npu/a5/TColExpandAdd.hpp"
#include "pto/npu/a5/TColExpandMax.hpp"
#include "pto/npu/a5/TColExpandMin.hpp"
#include "pto/npu/a5/TTri.hpp"
// TPrefetch to be evaluated
// TPush to be evaluated
// TPop to be evaluated
// TAlloc to be evaluated
// TFree to be evaluated
#include "pto/npu/a5/TColReduceIdx.hpp"
#include "pto/npu/a5/SetQuantScalar.hpp"
#include "pto/npu/a5/SetQuantVector.hpp"
#include "pto/npu/a5/TAddReluConv.hpp"
#include "pto/npu/a5/TSubReluConv.hpp"
#include "pto/npu/a5/TFusedMulAdd.hpp"
#include "pto/npu/a5/TPairReduceSum.hpp"
#include "pto/npu/a5/TPrefetch.hpp"
#include "pto/npu/a5/TFusedMulAddRelu.hpp"
#include "pto/npu/a5/TMulAddDst.hpp"
#include "pto/npu/a5/TSubRelu.hpp"
#include "pto/npu/a5/TAddDeqRelu.hpp"
#include "pto/npu/a5/TInterleave.hpp"
#include "pto/npu/a5/TDeInterleave.hpp"


// Forwarding wrappers for A5 functions called via MAP_INSTR_IMPL from within A5 template definitions
// Must be defined BEFORE kirin9030 includes because kirin9030 files internally include A5 files
namespace pto {
namespace kirin9030 {

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

// Stub implementations for MGATHER/MSCATTER (not supported on kirin9030)
template <typename... Args>
PTO_INTERNAL void MGATHER_IMPL(Args&&...) { static_assert(sizeof...(Args) == 0, "MGATHER not supported on kirin9030"); }

template <typename... Args>
PTO_INTERNAL void MSCATTER_IMPL(Args&&...) { static_assert(sizeof...(Args) == 0, "MSCATTER not supported on kirin9030"); }

} // namespace kirin9030
} // namespace pto


#include "pto/npu/kirin9030/header_impl.hpp"

#undef bfloat16_t
#undef float8_e4m3_t
#undef float8_e5m2_t
#undef hifloat8_t
#undef float8_e8m0_t
#undef float4_e2m1x2_t
#undef float4_e1m2x2_t
#endif
