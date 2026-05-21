#ifndef PTO_EXT_DISPATCH_COMBINE_MOE_PTO_VECTOR_OPS_HPP
#define PTO_EXT_DISPATCH_COMBINE_MOE_PTO_VECTOR_OPS_HPP

#include "kernel_operator.h"

#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

#include <type_traits>

namespace pto_ext::dispatch_combine_moe::pto_bridge {

using PtoShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using PtoStrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;

template <typename Element>
using PtoGlobalNd = pto::GlobalTensor<Element, PtoShapeDyn, PtoStrideDyn, pto::Layout::ND>;

template <typename Element>
PTO_INTERNAL PtoGlobalNd<Element> MakeContiguousGlobalFromPtr(__gm__ Element *ptr, uint32_t elemNum)
{
    PtoShapeDyn shape(1, 1, 1, 1, elemNum);
    PtoStrideDyn stride(elemNum, elemNum, elemNum, elemNum, 1);
    return PtoGlobalNd<Element>(ptr, shape, stride);
}

template <typename Element>
PTO_INTERNAL PtoGlobalNd<Element> MakeGlobalFromPtr(__gm__ Element *ptr, int64_t validRow, int64_t validCol,
                                                    int64_t leadingDim)
{
    PtoShapeDyn shape(1, 1, 1, validRow, validCol);
    PtoStrideDyn stride(validRow * leadingDim, validRow * leadingDim, validRow * leadingDim, leadingDim, 1);
    return PtoGlobalNd<Element>(ptr, shape, stride);
}

template <typename Element, int TileElems = 1024>
using PtoVecTile = pto::Tile<pto::TileType::Vec, Element, 1, TileElems, pto::BLayout::RowMajor, -1, -1>;

template <typename UbTensor>
PTO_DEVICE uint64_t PtoUbBaseAddr(UbTensor const &tensor)
{
    auto base = tensor[0];
    return reinterpret_cast<uint64_t>(base.GetPhyAddr());
}

template <typename Element>
PTO_DEVICE uint64_t PtoElemOffsetBytes(uint32_t offset)
{
    return static_cast<uint64_t>(offset) * sizeof(Element);
}

template <typename Tile, typename Element>
PTO_DEVICE void PtoAssignUbTile(Tile &tile, uint64_t baseOffsetBytes, uint32_t elemOffset)
{
    pto::TASSIGN(tile, baseOffsetBytes + PtoElemOffsetBytes<Element>(elemOffset));
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoLoadVector(uint64_t dstUbOffsetBytes, __gm__ Element *src, uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto srcGlobal = MakeContiguousGlobalFromPtr(src + offset, cur);
        Tile tile(1, cur);
        PtoAssignUbTile<Tile, Element>(tile, dstUbOffsetBytes, offset);
        pto::TLOAD(tile, srcGlobal);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoStoreVector(__gm__ Element *dst, uint64_t srcUbOffsetBytes, uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto dstGlobal = MakeContiguousGlobalFromPtr(dst + offset, cur);
        Tile tile(1, cur);
        PtoAssignUbTile<Tile, Element>(tile, srcUbOffsetBytes, offset);
        pto::TSTORE(dstGlobal, tile);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoStoreAtomicAddVector(__gm__ Element *dst, uint64_t srcUbOffsetBytes, uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto dstGlobal = MakeContiguousGlobalFromPtr(dst + offset, cur);
        Tile tile(1, cur);
        PtoAssignUbTile<Tile, Element>(tile, srcUbOffsetBytes, offset);
        pto::TSTORE<Tile, decltype(dstGlobal), pto::AtomicType::AtomicAdd>(dstGlobal, tile);
    }
}

template <typename DstElement, typename SrcElement, int TileElems = 1024>
PTO_DEVICE void PtoCastVector(uint64_t dstUbOffsetBytes, uint64_t srcUbOffsetBytes, uint32_t elemNum,
                              pto::RoundMode mode)
{
#if defined(__DAV_VEC__)
    if constexpr (std::is_same_v<DstElement, half> && std::is_same_v<SrcElement, int32_t>) {
        PtoCastVector<float, SrcElement, TileElems>(dstUbOffsetBytes, srcUbOffsetBytes, elemNum, mode);
        PtoCastVector<DstElement, float, TileElems>(dstUbOffsetBytes, dstUbOffsetBytes, elemNum,
                                                    pto::RoundMode::CAST_NONE);
    } else {
        using DstTile = PtoVecTile<DstElement, TileElems>;
        using SrcTile = PtoVecTile<SrcElement, TileElems>;
        for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
            const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
            DstTile dstTile(1, cur);
            SrcTile srcTile(1, cur);
            PtoAssignUbTile<DstTile, DstElement>(dstTile, dstUbOffsetBytes, offset);
            PtoAssignUbTile<SrcTile, SrcElement>(srcTile, srcUbOffsetBytes, offset);
            pto::TCVT(dstTile, srcTile, mode);
        }
    }
#else
    (void)dstUbOffsetBytes;
    (void)srcUbOffsetBytes;
    (void)elemNum;
    (void)mode;
#endif
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoMoveVector(uint64_t dstUbOffsetBytes, uint64_t srcUbOffsetBytes, uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        Tile dstTile(1, cur);
        Tile srcTile(1, cur);
        PtoAssignUbTile<Tile, Element>(dstTile, dstUbOffsetBytes, offset);
        PtoAssignUbTile<Tile, Element>(srcTile, srcUbOffsetBytes, offset);
        pto::TMOV(dstTile, srcTile);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoFillVector(uint64_t dstUbOffsetBytes, Element scalar, uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        Tile tile(1, cur);
        PtoAssignUbTile<Tile, Element>(tile, dstUbOffsetBytes, offset);
        pto::TEXPANDS(tile, scalar);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE Element PtoGetValue(uint64_t ubOffsetBytes, uint32_t elemOffset)
{
    using Tile = PtoVecTile<Element, TileElems>;
    uint64_t tileOffsetBytes =
        ubOffsetBytes + static_cast<uint64_t>(elemOffset / TileElems) * TileElems * sizeof(Element);
    Tile tile(1, TileElems);
    pto::TASSIGN(tile, tileOffsetBytes);
    return tile.GetValue(elemOffset % TileElems);
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoSetValue(uint64_t ubOffsetBytes, uint32_t elemOffset, Element value)
{
    using Tile = PtoVecTile<Element, TileElems>;
    uint64_t tileOffsetBytes =
        ubOffsetBytes + static_cast<uint64_t>(elemOffset / TileElems) * TileElems * sizeof(Element);
    Tile tile(1, TileElems);
    pto::TASSIGN(tile, tileOffsetBytes);
    tile.SetValue(elemOffset % TileElems, value);
}

template <typename Element, int TileElems = 1024, typename UbTensor>
PTO_DEVICE Element PtoGetValue(UbTensor const &src, uint32_t elemOffset)
{
    return PtoGetValue<Element, TileElems>(PtoUbBaseAddr(src), elemOffset);
}

template <typename Element, int TileElems = 1024, typename UbTensor>
PTO_DEVICE void PtoSetValue(UbTensor const &dst, uint32_t elemOffset, Element value)
{
    PtoSetValue<Element, TileElems>(PtoUbBaseAddr(dst), elemOffset, value);
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoMulVector(uint64_t dstUbOffsetBytes, uint64_t srcUbOffsetBytes, uint32_t elemNum, Element scalar)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        Tile dstTile(1, cur);
        Tile srcTile(1, cur);
        PtoAssignUbTile<Tile, Element>(dstTile, dstUbOffsetBytes, offset);
        PtoAssignUbTile<Tile, Element>(srcTile, srcUbOffsetBytes, offset);
        pto::TMULS(dstTile, srcTile, scalar);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoAddVector(uint64_t dstUbOffsetBytes, uint64_t src0UbOffsetBytes, uint64_t src1UbOffsetBytes,
                             uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        Tile dstTile(1, cur);
        Tile src0Tile(1, cur);
        Tile src1Tile(1, cur);
        PtoAssignUbTile<Tile, Element>(dstTile, dstUbOffsetBytes, offset);
        PtoAssignUbTile<Tile, Element>(src0Tile, src0UbOffsetBytes, offset);
        PtoAssignUbTile<Tile, Element>(src1Tile, src1UbOffsetBytes, offset);
        pto::TADD(dstTile, src0Tile, src1Tile);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoAddScalarVector(uint64_t dstUbOffsetBytes, uint64_t srcUbOffsetBytes, uint32_t elemNum,
                                   Element scalar)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        Tile dstTile(1, cur);
        Tile srcTile(1, cur);
        PtoAssignUbTile<Tile, Element>(dstTile, dstUbOffsetBytes, offset);
        PtoAssignUbTile<Tile, Element>(srcTile, srcUbOffsetBytes, offset);
        pto::TADDS(dstTile, srcTile, scalar);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoMulElementwiseVector(uint64_t dstUbOffsetBytes, uint64_t src0UbOffsetBytes,
                                        uint64_t src1UbOffsetBytes, uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        Tile dstTile(1, cur);
        Tile src0Tile(1, cur);
        Tile src1Tile(1, cur);
        PtoAssignUbTile<Tile, Element>(dstTile, dstUbOffsetBytes, offset);
        PtoAssignUbTile<Tile, Element>(src0Tile, src0UbOffsetBytes, offset);
        PtoAssignUbTile<Tile, Element>(src1Tile, src1UbOffsetBytes, offset);
        pto::TMUL(dstTile, src0Tile, src1Tile);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoDivVector(uint64_t dstUbOffsetBytes, uint64_t src0UbOffsetBytes, uint64_t src1UbOffsetBytes,
                             uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        Tile dstTile(1, cur);
        Tile src0Tile(1, cur);
        Tile src1Tile(1, cur);
        PtoAssignUbTile<Tile, Element>(dstTile, dstUbOffsetBytes, offset);
        PtoAssignUbTile<Tile, Element>(src0Tile, src0UbOffsetBytes, offset);
        PtoAssignUbTile<Tile, Element>(src1Tile, src1UbOffsetBytes, offset);
        pto::TDIV(dstTile, src0Tile, src1Tile);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoAbsVector(uint64_t dstUbOffsetBytes, uint64_t srcUbOffsetBytes, uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        Tile dstTile(1, cur);
        Tile srcTile(1, cur);
        PtoAssignUbTile<Tile, Element>(dstTile, dstUbOffsetBytes, offset);
        PtoAssignUbTile<Tile, Element>(srcTile, srcUbOffsetBytes, offset);
        pto::TABS(dstTile, srcTile);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoExpVector(uint64_t dstUbOffsetBytes, uint64_t srcUbOffsetBytes, uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        Tile dstTile(1, cur);
        Tile srcTile(1, cur);
        PtoAssignUbTile<Tile, Element>(dstTile, dstUbOffsetBytes, offset);
        PtoAssignUbTile<Tile, Element>(srcTile, srcUbOffsetBytes, offset);
        pto::TEXP(dstTile, srcTile);
    }
}

template <int TileElems = 1024>
PTO_DEVICE void PtoReduceMaxVector(uint64_t dstUbOffsetBytes, uint64_t srcUbOffsetBytes, uint64_t tmpUbOffsetBytes,
                                   uint32_t elemNum)
{
    using SrcTile = PtoVecTile<float, TileElems>;
    using TmpTile = PtoVecTile<float, TileElems>;
    using RowMaxTile = pto::Tile<pto::TileType::Vec, float, 8, 1, pto::BLayout::ColMajor, -1, 1>;
    using ScalarTile = pto::Tile<pto::TileType::Vec, float, 1, 8, pto::BLayout::RowMajor, -1, -1>;

    bool firstChunk = true;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        SrcTile srcTile(1, cur);
        TmpTile tmpTile(1, cur);
        RowMaxTile rowMaxTile(1);
        PtoAssignUbTile<SrcTile, float>(srcTile, srcUbOffsetBytes, offset);
        PtoAssignUbTile<TmpTile, float>(tmpTile, tmpUbOffsetBytes, offset);
        pto::TASSIGN(rowMaxTile, firstChunk ? dstUbOffsetBytes : tmpUbOffsetBytes);
        pto::TROWMAX(rowMaxTile, srcTile, tmpTile);
        AscendC::PipeBarrier<PIPE_V>();

        if (!firstChunk) {
            ScalarTile accTile(1, 1);
            ScalarTile newTile(1, 1);
            ScalarTile dstTile(1, 1);
            pto::TASSIGN(accTile, dstUbOffsetBytes);
            pto::TASSIGN(newTile, tmpUbOffsetBytes);
            pto::TASSIGN(dstTile, dstUbOffsetBytes);
            pto::TMAX(dstTile, accTile, newTile);
            AscendC::PipeBarrier<PIPE_V>();
        }
        firstChunk = false;
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoLoadMatrixRows(uint64_t dstUbOffsetBytes, __gm__ Element *src, uint32_t rowNum, uint32_t colNum,
                                  uint32_t dstStride, uint32_t srcStride)
{
    for (uint32_t rowIdx = 0; rowIdx < rowNum; ++rowIdx) {
        PtoLoadVector<Element, TileElems>(dstUbOffsetBytes + PtoElemOffsetBytes<Element>(rowIdx * dstStride),
                                          src + rowIdx * srcStride, colNum);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoStoreMatrixRows(__gm__ Element *dst, uint64_t srcUbOffsetBytes, uint32_t rowNum, uint32_t colNum,
                                   uint32_t dstStride, uint32_t srcStride)
{
    for (uint32_t rowIdx = 0; rowIdx < rowNum; ++rowIdx) {
        PtoStoreVector<Element, TileElems>(dst + rowIdx * dstStride,
                                           srcUbOffsetBytes + PtoElemOffsetBytes<Element>(rowIdx * srcStride), colNum);
    }
}

template <typename Element, int TileElems = 1024, typename UbTensor>
PTO_DEVICE void PtoLoadVector(UbTensor const &dst, __gm__ Element *src, uint32_t elemNum)
{
    PtoLoadVector<Element, TileElems>(PtoUbBaseAddr(dst), src, elemNum);
}

template <typename Element, int TileElems = 1024, typename UbTensor>
PTO_DEVICE void PtoStoreVector(__gm__ Element *dst, UbTensor const &src, uint32_t elemNum)
{
    PtoStoreVector<Element, TileElems>(dst, PtoUbBaseAddr(src), elemNum);
}

template <typename Element, int TileElems = 1024, typename UbTensor>
PTO_DEVICE void PtoStoreAtomicAddVector(__gm__ Element *dst, UbTensor const &src, uint32_t elemNum)
{
    PtoStoreAtomicAddVector<Element, TileElems>(dst, PtoUbBaseAddr(src), elemNum);
}

template <typename DstElement, typename SrcElement, int TileElems = 1024, typename DstUbTensor, typename SrcUbTensor>
PTO_DEVICE void PtoCastVector(DstUbTensor const &dst, SrcUbTensor const &src, uint32_t elemNum, pto::RoundMode mode)
{
    PtoCastVector<DstElement, SrcElement, TileElems>(PtoUbBaseAddr(dst), PtoUbBaseAddr(src), elemNum, mode);
}

template <typename Element, int TileElems = 1024, typename DstUbTensor, typename SrcUbTensor>
PTO_DEVICE void PtoMoveVector(DstUbTensor const &dst, SrcUbTensor const &src, uint32_t elemNum)
{
    PtoMoveVector<Element, TileElems>(PtoUbBaseAddr(dst), PtoUbBaseAddr(src), elemNum);
}

template <typename Element, int TileElems = 1024, typename UbTensor>
PTO_DEVICE void PtoFillVector(UbTensor const &dst, Element scalar, uint32_t elemNum)
{
    PtoFillVector<Element, TileElems>(PtoUbBaseAddr(dst), scalar, elemNum);
}

template <typename Element, int TileElems = 1024, typename DstUbTensor, typename SrcUbTensor>
PTO_DEVICE void PtoMulVector(DstUbTensor const &dst, SrcUbTensor const &src, uint32_t elemNum, Element scalar)
{
    PtoMulVector<Element, TileElems>(PtoUbBaseAddr(dst), PtoUbBaseAddr(src), elemNum, scalar);
}

template <typename Element, int TileElems = 1024, typename DstUbTensor, typename Src0UbTensor, typename Src1UbTensor>
PTO_DEVICE void PtoAddVector(DstUbTensor const &dst, Src0UbTensor const &src0, Src1UbTensor const &src1,
                             uint32_t elemNum)
{
    PtoAddVector<Element, TileElems>(PtoUbBaseAddr(dst), PtoUbBaseAddr(src0), PtoUbBaseAddr(src1), elemNum);
}

template <typename Element, int TileElems = 1024, typename DstUbTensor, typename SrcUbTensor>
PTO_DEVICE void PtoAddScalarVector(DstUbTensor const &dst, SrcUbTensor const &src, uint32_t elemNum, Element scalar)
{
    PtoAddScalarVector<Element, TileElems>(PtoUbBaseAddr(dst), PtoUbBaseAddr(src), elemNum, scalar);
}

template <typename Element, int TileElems = 1024, typename DstUbTensor, typename Src0UbTensor, typename Src1UbTensor>
PTO_DEVICE void PtoMulElementwiseVector(DstUbTensor const &dst, Src0UbTensor const &src0, Src1UbTensor const &src1,
                                        uint32_t elemNum)
{
    PtoMulElementwiseVector<Element, TileElems>(PtoUbBaseAddr(dst), PtoUbBaseAddr(src0), PtoUbBaseAddr(src1), elemNum);
}

template <typename Element, int TileElems = 1024, typename DstUbTensor, typename Src0UbTensor, typename Src1UbTensor>
PTO_DEVICE void PtoDivVector(DstUbTensor const &dst, Src0UbTensor const &src0, Src1UbTensor const &src1,
                             uint32_t elemNum)
{
    PtoDivVector<Element, TileElems>(PtoUbBaseAddr(dst), PtoUbBaseAddr(src0), PtoUbBaseAddr(src1), elemNum);
}

template <typename Element, int TileElems = 1024, typename DstUbTensor, typename SrcUbTensor>
PTO_DEVICE void PtoAbsVector(DstUbTensor const &dst, SrcUbTensor const &src, uint32_t elemNum)
{
    PtoAbsVector<Element, TileElems>(PtoUbBaseAddr(dst), PtoUbBaseAddr(src), elemNum);
}

template <typename Element, int TileElems = 1024, typename DstUbTensor, typename SrcUbTensor>
PTO_DEVICE void PtoExpVector(DstUbTensor const &dst, SrcUbTensor const &src, uint32_t elemNum)
{
    PtoExpVector<Element, TileElems>(PtoUbBaseAddr(dst), PtoUbBaseAddr(src), elemNum);
}

template <int TileElems = 1024, typename DstUbTensor, typename SrcUbTensor, typename TmpUbTensor>
PTO_DEVICE void PtoReduceMaxVector(DstUbTensor const &dst, SrcUbTensor const &src, TmpUbTensor const &tmp,
                                   uint32_t elemNum)
{
    PtoReduceMaxVector<TileElems>(PtoUbBaseAddr(dst), PtoUbBaseAddr(src), PtoUbBaseAddr(tmp), elemNum);
}

template <typename Element, int TileElems = 1024, typename UbTensor>
PTO_DEVICE void PtoLoadMatrixRows(UbTensor const &dst, __gm__ Element *src, uint32_t rowNum, uint32_t colNum,
                                  uint32_t dstStride, uint32_t srcStride)
{
    PtoLoadMatrixRows<Element, TileElems>(PtoUbBaseAddr(dst), src, rowNum, colNum, dstStride, srcStride);
}

template <typename Element, int TileElems = 1024, typename UbTensor>
PTO_DEVICE void PtoStoreMatrixRows(__gm__ Element *dst, UbTensor const &src, uint32_t rowNum, uint32_t colNum,
                                   uint32_t dstStride, uint32_t srcStride)
{
    PtoStoreMatrixRows<Element, TileElems>(dst, PtoUbBaseAddr(src), rowNum, colNum, dstStride, srcStride);
}

template <template <typename> class DstUbTensor, typename DstElement, template <typename> class SrcUbTensor,
          typename SrcElement, int TileElems = 1024>
PTO_DEVICE void PtoCastVector(DstUbTensor<DstElement> const &dst, SrcUbTensor<SrcElement> const &src, uint32_t elemNum,
                              pto::RoundMode mode)
{
    PtoCastVector<DstElement, SrcElement, TileElems>(PtoUbBaseAddr(dst), PtoUbBaseAddr(src), elemNum, mode);
}

template <template <typename> class DstUbTensor, template <typename> class SrcUbTensor, typename Element,
          int TileElems = 1024>
PTO_DEVICE void PtoMoveVector(DstUbTensor<Element> const &dst, SrcUbTensor<Element> const &src, uint32_t elemNum)
{
    PtoMoveVector<Element, TileElems>(PtoUbBaseAddr(dst), PtoUbBaseAddr(src), elemNum);
}

template <template <typename> class DstUbTensor, template <typename> class SrcUbTensor, typename Element,
          int TileElems = 1024>
PTO_DEVICE void PtoMulVector(DstUbTensor<Element> const &dst, SrcUbTensor<Element> const &src, uint32_t elemNum,
                             Element scalar)
{
    PtoMulVector<Element, TileElems>(PtoUbBaseAddr(dst), PtoUbBaseAddr(src), elemNum, scalar);
}

template <template <typename> class DstUbTensor, template <typename> class Src0UbTensor,
          template <typename> class Src1UbTensor, typename Element, int TileElems = 1024>
PTO_DEVICE void PtoAddVector(DstUbTensor<Element> const &dst, Src0UbTensor<Element> const &src0,
                             Src1UbTensor<Element> const &src1, uint32_t elemNum)
{
    PtoAddVector<Element, TileElems>(PtoUbBaseAddr(dst), PtoUbBaseAddr(src0), PtoUbBaseAddr(src1), elemNum);
}

template <template <typename> class DstUbTensor, template <typename> class SrcUbTensor, typename Element,
          int TileElems = 1024>
PTO_DEVICE void PtoAddScalarVector(DstUbTensor<Element> const &dst, SrcUbTensor<Element> const &src, uint32_t elemNum,
                                   Element scalar)
{
    PtoAddScalarVector<Element, TileElems>(PtoUbBaseAddr(dst), PtoUbBaseAddr(src), elemNum, scalar);
}

template <template <typename> class DstUbTensor, template <typename> class Src0UbTensor,
          template <typename> class Src1UbTensor, typename Element, int TileElems = 1024>
PTO_DEVICE void PtoMulElementwiseVector(DstUbTensor<Element> const &dst, Src0UbTensor<Element> const &src0,
                                        Src1UbTensor<Element> const &src1, uint32_t elemNum)
{
    PtoMulElementwiseVector<Element, TileElems>(PtoUbBaseAddr(dst), PtoUbBaseAddr(src0), PtoUbBaseAddr(src1), elemNum);
}

template <template <typename> class DstUbTensor, template <typename> class Src0UbTensor,
          template <typename> class Src1UbTensor, typename Element, int TileElems = 1024>
PTO_DEVICE void PtoDivVector(DstUbTensor<Element> const &dst, Src0UbTensor<Element> const &src0,
                             Src1UbTensor<Element> const &src1, uint32_t elemNum)
{
    PtoDivVector<Element, TileElems>(PtoUbBaseAddr(dst), PtoUbBaseAddr(src0), PtoUbBaseAddr(src1), elemNum);
}

template <template <typename> class DstUbTensor, template <typename> class SrcUbTensor, typename Element,
          int TileElems = 1024>
PTO_DEVICE void PtoAbsVector(DstUbTensor<Element> const &dst, SrcUbTensor<Element> const &src, uint32_t elemNum)
{
    PtoAbsVector<Element, TileElems>(PtoUbBaseAddr(dst), PtoUbBaseAddr(src), elemNum);
}

template <template <typename> class DstUbTensor, template <typename> class SrcUbTensor, typename Element,
          int TileElems = 1024>
PTO_DEVICE void PtoExpVector(DstUbTensor<Element> const &dst, SrcUbTensor<Element> const &src, uint32_t elemNum)
{
    PtoExpVector<Element, TileElems>(PtoUbBaseAddr(dst), PtoUbBaseAddr(src), elemNum);
}

template <template <typename> class DstUbTensor, template <typename> class SrcUbTensor,
          template <typename> class TmpUbTensor, int TileElems = 1024>
PTO_DEVICE void PtoReduceMaxVector(DstUbTensor<float> const &dst, SrcUbTensor<float> const &src,
                                   TmpUbTensor<float> const &tmp, uint32_t elemNum)
{
    PtoReduceMaxVector<TileElems>(PtoUbBaseAddr(dst), PtoUbBaseAddr(src), PtoUbBaseAddr(tmp), elemNum);
}

} // namespace pto_ext::dispatch_combine_moe::pto_bridge

#endif // PTO_EXT_DISPATCH_COMBINE_MOE_PTO_VECTOR_OPS_HPP
