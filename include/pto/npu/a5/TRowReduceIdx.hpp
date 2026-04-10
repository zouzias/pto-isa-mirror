/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#ifndef __ROW_REDUCE_IDX__
#define __ROW_REDUCE_IDX__

#include "common.hpp"
#include "pto/common/pto_tile.hpp"
#include "TPartBinOps.hpp"
#include <math.h>
#include <type_traits>

namespace pto {

template <typename T>
struct ROWIDXMAX {
    static constexpr typename Padding<T>::Type InitVal = Padding<T>::Min;
    using PaddingType = typename Padding<T>::Type;
    using RegType = typename TypeGet<T>::T;
    static PTO_INTERNAL void Reduce(RegType &dst, RegType &src, MaskReg &preg)
    {
        vcmax(dst, src, preg, MODE_ZEROING);
    }
    static PTO_INTERNAL void Compare(MaskReg &dst, RegType &src0, RegType &src1, MaskReg &preg)
    {
        vcmp_lt(dst, src0, src1, preg);
    }
};

template <typename T>
struct ROWIDXMIN {
    static constexpr typename Padding<T>::Type InitVal = Padding<T>::Max;
    using PaddingType = typename Padding<T>::Type;
    using RegType = typename TypeGet<T>::T;
    static PTO_INTERNAL void Reduce(RegType &dst, RegType &src, MaskReg &preg)
    {
        vcmin(dst, src, preg, MODE_ZEROING);
    }
    static PTO_INTERNAL void Compare(MaskReg &dst, RegType &src0, RegType &src1, MaskReg &preg)
    {
        vcmp_gt(dst, src0, src1, preg);
    }
};

template <typename ReduceIdxOp, typename TDst, typename TSrc>
PTO_INTERNAL void UpdateIdxValue(RegTensor<TDst> &vregIdxOrig, RegTensor<TSrc> &vregValOrig, RegTensor<TSrc> &vregVal,
                                 RegTensor<TSrc> &vregZero, TDst currentOffset, MaskReg &pRegOneElem)
{
    MaskReg pregCmp;
    RegTensor<TDst> vregIdx;
    vdintlv(vregVal, (RegTensor<TSrc> &)vregIdx, vregVal, vregZero);
    vadds(vregIdx, vregIdx, currentOffset, pRegOneElem, MODE_ZEROING);
    ReduceIdxOp::Compare(pregCmp, vregValOrig, vregVal, pRegOneElem);
    vsel(vregValOrig, vregVal, vregValOrig, pregCmp);
    vsel(vregIdxOrig, vregIdx, vregIdxOrig, pregCmp);
}

template <typename T>
using DType = typename std::conditional_t<std::is_void<T>::value, std::nullptr_t, __ubuf__ typename T::DType>

template <typename T>
using TileDType = typename std::conditional_t<std::is_void<T>::value, std::nullptr_t, typename T::TileDType>

template <typename ReduceIdxOp, typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn>
PTO_INTERNAL void TRowReduceIdxImpl(DType<TileDataOutVal> *dstPtrVal,
                                    DType<TileDataOutIdx> *dstPtrIdx,
                                    __ubuf__ typename TileDataIn::DType *srcPtr, uint32_t rows, uint32_t cols,
                                    unsigned version)
{
    using TDstVal = DType<TileDataOutVal>;
    using TDstIdx = DType<TileDataOutIdx>;
    using TSrc = typename TileDataIn::DType;
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(TSrc);
    uint16_t repeatTimes = CeilDivision(cols, elementsPerRepeat);
    __VEC_SCOPE__
    {
        RegTensor<TSrc> vregZero, vregSrc, vregValOrig;
        RegTensor<TDstIdx> vregIdxOrig;
        vbr(vregZero, 0);

        uint32_t dstDup = 1;
        MaskReg pRegOneElem = CreatePredicate<TSrc>(dstDup);
        if (version == VFIMPL_2D_NO_POST_UPDATE) {
            for (uint16_t i = 0; i < (uint16_t)rows; ++i) {
                vdup((RegTensor<typename ReduceIdxOp::PaddingType> &)vregValOrig, ReduceIdxOp::InitVal, pRegOneElem,
                     MODE_ZEROING);
                vbr(vregIdxOrig, 0);
                uint32_t sregCol = cols;
                for (uint16_t j = 0; j < (uint16_t)repeatTimes; j++) {
                    MaskReg pRegSrc = CreatePredicate<TSrc>(sregCol);
                    vlds(vregSrc, srcPtr, i * TileDataIn::RowStride + j * elementsPerRepeat, NORM);
                    ReduceIdxOp::Reduce(vregSrc, vregSrc, pRegSrc);
                    UpdateIdxValue<ReduceIdxOp, TDst, TSrc>(vregIdxOrig, vregValOrig, vregSrc, vregZero,
                                                            j * elementsPerRepeat, pRegOneElem);
                }
                if constexpr (!std::is_void<TileDataOutIdx>::value) {
                    constexpr auto distValueIdx =
                        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<TDstIdx, DistVST::DIST_ONEPT>())>();
                    vsts(vregIdxOrig, dstPtrIdx, i * TileDataOutIdx::RowStride, distValueIdx, pRegOneElem);
                }
                if constexpr (!std::is_void<TileDataOutVal>::value) {
                    constexpr auto distValueVal =
                        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<TDstVal, DistVST::DIST_ONEPT>())>();
                    vsts(vregValOrig, dstPtrVal, i * TileDataOutVal::RowStride, distValueVal, pRegOneElem);
                }
            }
        } else {
            for (uint16_t i = 0; i < (uint16_t)rows; ++i) {
                vdup((RegTensor<typename ReduceIdxOp::PaddingType> &)vregValOrig, ReduceIdxOp::InitVal, pRegOneElem,
                     MODE_ZEROING);
                vbr(vregIdxOrig, 0);
                __ubuf__ TSrc *rowPtr = srcPtr + i * TileDataIn::RowStride;
                uint32_t sregCol = cols;
                for (uint16_t j = 0; j < (uint16_t)repeatTimes; j++) {
                    MaskReg pRegSrc = CreatePredicate<TSrc>(sregCol);
                    vlds(vregSrc, rowPtr, elementsPerRepeat, NORM, POST_UPDATE);
                    ReduceIdxOp::Reduce(vregSrc, vregSrc, pRegSrc);
                    UpdateIdxValue<ReduceIdxOp, TDst, TSrc>(vregIdxOrig, vregValOrig, vregSrc, vregZero,
                                                            j * elementsPerRepeat, pRegOneElem);
                }
                if constexpr (!std::is_void<TileDataOutIdx>::value) {
                    constexpr auto distValueIdx =
                        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<TDstIdx, DistVST::DIST_ONEPT>())>();
                    vsts(vregIdxOrig, dstPtr, TileDataOutIdx::RowStride, distValueIdx, pRegOneElem, POST_UPDATE);
                }
                if constexpr (!std::is_void<TileDataOutVal>::value) {
                    constexpr auto distValueVal =
                        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<TDstVal, DistVST::DIST_ONEPT>())>();
                    vsts(vregValOrig, dstPtrVal, i * TileDataOutVal::RowStride, distValueVal, pRegOneElem, POST_UPDATE);
                }
            }
        }
    }
}

template <typename TileDataOut, typename TileDataIn>
__tf__ PTO_INTERNAL OP_NAME(TROWARGMAX)
    OP_TYPE(reduce) void TRowArgMax(typename TileDataOut::TileDType __out__ dst,
                                    typename TileDataIn::TileDType __in__ src, uint32_t srcValidRows,
                                    uint32_t srcValidCols, uint32_t dstValidRow,
                                    unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    using TDst = typename TileDataOut::DType;
    using TSrc = typename TileDataIn::DType;
    TRowReduceCheck<TileDataOut, TileDataIn, true>(srcValidRows, srcValidCols, dstValidRow);
    __ubuf__ TDst *dstPtr = __cce_get_tile_ptr(dst);
    __ubuf__ TSrc *srcPtr = __cce_get_tile_ptr(src);
    TRowReduceIdxImpl<ROWIDXMAX<TSrc>, void, TileDataOut, TileDataIn>(nullptr, dstPtr, srcPtr, srcValidRows, srcValidCols, version);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWARGMAX_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    TRowArgMax<TileDataOut, TileDataIn>(dst.data(), src.data(), src.GetValidRow(), src.GetValidCol(),
                                        dst.GetValidRow());
}

template <typename TileDataOut, typename TileDataIn>
__tf__ PTO_INTERNAL OP_NAME(TROWARGMIN)
    OP_TYPE(reduce) void TRowArgMin(typename TileDataOut::TileDType __out__ dst,
                                    typename TileDataIn::TileDType __in__ src, uint32_t srcValidRows,
                                    uint32_t srcValidCols, uint32_t dstValidRow,
                                    unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    using TDst = typename TileDataOut::DType;
    using TSrc = typename TileDataIn::DType;
    TRowReduceCheck<TileDataOut, TileDataIn, true>(srcValidRows, srcValidCols, dstValidRow);
    __ubuf__ TDst *dstPtr = __cce_get_tile_ptr(dst);
    __ubuf__ TSrc *srcPtr = __cce_get_tile_ptr(src);
    TRowReduceIdxImpl<ROWIDXMIN<TSrc>, void, TileDataOut, TileDataIn>(nullptr, dstPtr, srcPtr, srcValidRows, srcValidCols, version);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWARGMIN_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    TRowArgMin<TileDataOut, TileDataIn>(dst.data(), src.data(), src.GetValidRow(), src.GetValidCol(),
                                        dst.GetValidRow());
}
} // namespace pto

#endif