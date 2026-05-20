#ifndef PTO_EXT_DISPATCH_FFN_COMBINE_V3_PTO_VECTOR_OPS_HPP
#define PTO_EXT_DISPATCH_FFN_COMBINE_V3_PTO_VECTOR_OPS_HPP

#include "pto_global_view.hpp"

namespace pto_ext::dispatch_ffn_combine_v3::pto_bridge {

template <typename Element, int TileElems = 1024>
using PtoVecTile = pto::Tile<pto::TileType::Vec, Element, 1, TileElems, pto::BLayout::RowMajor, -1, -1>;

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoLoadVector(AscendC::LocalTensor<Element> const &dst,
                              AscendC::GlobalTensor<Element> const &src,
                              uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto dstChunk = dst[offset];
        auto srcChunk = src[offset];
        auto srcGlobal = MakeContiguousGlobal(srcChunk, cur);
        Tile tile(1, cur);
        pto::TASSIGN(tile, reinterpret_cast<uint64_t>(dstChunk.GetPhyAddr()));
        pto::TLOAD(tile, srcGlobal);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoStoreVector(AscendC::GlobalTensor<Element> const &dst,
                               AscendC::LocalTensor<Element> const &src,
                               uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto dstChunk = dst[offset];
        auto srcChunk = src[offset];
        auto dstGlobal = MakeContiguousGlobal(dstChunk, cur);
        Tile tile(1, cur);
        pto::TASSIGN(tile, reinterpret_cast<uint64_t>(srcChunk.GetPhyAddr()));
        pto::TSTORE(dstGlobal, tile);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoStoreAtomicAddVector(AscendC::GlobalTensor<Element> const &dst,
                                        AscendC::LocalTensor<Element> const &src,
                                        uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto dstChunk = dst[offset];
        auto srcChunk = src[offset];
        auto dstGlobal = MakeContiguousGlobal(dstChunk, cur);
        Tile tile(1, cur);
        pto::TASSIGN(tile, reinterpret_cast<uint64_t>(srcChunk.GetPhyAddr()));
        pto::TSTORE<Tile, decltype(dstGlobal), pto::AtomicType::AtomicAdd>(dstGlobal, tile);
    }
}

template <typename DstElement, typename SrcElement, int TileElems = 1024>
PTO_DEVICE void PtoCastVector(AscendC::LocalTensor<DstElement> const &dst,
                              AscendC::LocalTensor<SrcElement> const &src,
                              uint32_t elemNum,
                              pto::RoundMode mode)
{
    AscendC::RoundMode ascendMode = AscendC::RoundMode::CAST_NONE;
    switch (mode) {
        case pto::RoundMode::CAST_RINT:
            ascendMode = AscendC::RoundMode::CAST_RINT;
            break;
        case pto::RoundMode::CAST_FLOOR:
            ascendMode = AscendC::RoundMode::CAST_FLOOR;
            break;
        case pto::RoundMode::CAST_CEIL:
            ascendMode = AscendC::RoundMode::CAST_CEIL;
            break;
        case pto::RoundMode::CAST_ROUND:
            ascendMode = AscendC::RoundMode::CAST_ROUND;
            break;
        case pto::RoundMode::CAST_TRUNC:
            ascendMode = AscendC::RoundMode::CAST_TRUNC;
            break;
        default:
            ascendMode = AscendC::RoundMode::CAST_NONE;
            break;
    }
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        AscendC::Cast(dst[offset], src[offset], ascendMode, cur);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoMoveVector(AscendC::LocalTensor<Element> const &dst,
                              AscendC::LocalTensor<Element> const &src,
                              uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto dstChunk = dst[offset];
        auto srcChunk = src[offset];
        Tile dstTile(1, cur);
        Tile srcTile(1, cur);
        pto::TASSIGN(dstTile, reinterpret_cast<uint64_t>(dstChunk.GetPhyAddr()));
        pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(srcChunk.GetPhyAddr()));
        pto::TMOV(dstTile, srcTile);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoFillVector(AscendC::LocalTensor<Element> const &dst,
                              Element scalar,
                              uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto dstChunk = dst[offset];
        Tile tile(1, cur);
        pto::TASSIGN(tile, reinterpret_cast<uint64_t>(dstChunk.GetPhyAddr()));
        pto::TEXPANDS(tile, scalar);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoMulVector(AscendC::LocalTensor<Element> const &dst,
                             AscendC::LocalTensor<Element> const &src,
                             uint32_t elemNum,
                             Element scalar)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto dstChunk = dst[offset];
        auto srcChunk = src[offset];
        Tile dstTile(1, cur);
        Tile srcTile(1, cur);
        pto::TASSIGN(dstTile, reinterpret_cast<uint64_t>(dstChunk.GetPhyAddr()));
        pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(srcChunk.GetPhyAddr()));
        pto::TMULS(dstTile, srcTile, scalar);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoAddVector(AscendC::LocalTensor<Element> const &dst,
                             AscendC::LocalTensor<Element> const &src0,
                             AscendC::LocalTensor<Element> const &src1,
                             uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto dstChunk = dst[offset];
        auto src0Chunk = src0[offset];
        auto src1Chunk = src1[offset];
        Tile dstTile(1, cur);
        Tile src0Tile(1, cur);
        Tile src1Tile(1, cur);
        pto::TASSIGN(dstTile, reinterpret_cast<uint64_t>(dstChunk.GetPhyAddr()));
        pto::TASSIGN(src0Tile, reinterpret_cast<uint64_t>(src0Chunk.GetPhyAddr()));
        pto::TASSIGN(src1Tile, reinterpret_cast<uint64_t>(src1Chunk.GetPhyAddr()));
        pto::TADD(dstTile, src0Tile, src1Tile);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoAddScalarVector(AscendC::LocalTensor<Element> const &dst,
                                   AscendC::LocalTensor<Element> const &src,
                                   uint32_t elemNum,
                                   Element scalar)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto dstChunk = dst[offset];
        auto srcChunk = src[offset];
        Tile dstTile(1, cur);
        Tile srcTile(1, cur);
        pto::TASSIGN(dstTile, reinterpret_cast<uint64_t>(dstChunk.GetPhyAddr()));
        pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(srcChunk.GetPhyAddr()));
        pto::TADDS(dstTile, srcTile, scalar);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoMulElementwiseVector(AscendC::LocalTensor<Element> const &dst,
                                        AscendC::LocalTensor<Element> const &src0,
                                        AscendC::LocalTensor<Element> const &src1,
                                        uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto dstChunk = dst[offset];
        auto src0Chunk = src0[offset];
        auto src1Chunk = src1[offset];
        Tile dstTile(1, cur);
        Tile src0Tile(1, cur);
        Tile src1Tile(1, cur);
        pto::TASSIGN(dstTile, reinterpret_cast<uint64_t>(dstChunk.GetPhyAddr()));
        pto::TASSIGN(src0Tile, reinterpret_cast<uint64_t>(src0Chunk.GetPhyAddr()));
        pto::TASSIGN(src1Tile, reinterpret_cast<uint64_t>(src1Chunk.GetPhyAddr()));
        pto::TMUL(dstTile, src0Tile, src1Tile);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoDivVector(AscendC::LocalTensor<Element> const &dst,
                             AscendC::LocalTensor<Element> const &src0,
                             AscendC::LocalTensor<Element> const &src1,
                             uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto dstChunk = dst[offset];
        auto src0Chunk = src0[offset];
        auto src1Chunk = src1[offset];
        Tile dstTile(1, cur);
        Tile src0Tile(1, cur);
        Tile src1Tile(1, cur);
        pto::TASSIGN(dstTile, reinterpret_cast<uint64_t>(dstChunk.GetPhyAddr()));
        pto::TASSIGN(src0Tile, reinterpret_cast<uint64_t>(src0Chunk.GetPhyAddr()));
        pto::TASSIGN(src1Tile, reinterpret_cast<uint64_t>(src1Chunk.GetPhyAddr()));
        pto::TDIV(dstTile, src0Tile, src1Tile);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoAbsVector(AscendC::LocalTensor<Element> const &dst,
                             AscendC::LocalTensor<Element> const &src,
                             uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto dstChunk = dst[offset];
        auto srcChunk = src[offset];
        Tile dstTile(1, cur);
        Tile srcTile(1, cur);
        pto::TASSIGN(dstTile, reinterpret_cast<uint64_t>(dstChunk.GetPhyAddr()));
        pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(srcChunk.GetPhyAddr()));
        pto::TABS(dstTile, srcTile);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoExpVector(AscendC::LocalTensor<Element> const &dst,
                             AscendC::LocalTensor<Element> const &src,
                             uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto dstChunk = dst[offset];
        auto srcChunk = src[offset];
        Tile dstTile(1, cur);
        Tile srcTile(1, cur);
        pto::TASSIGN(dstTile, reinterpret_cast<uint64_t>(dstChunk.GetPhyAddr()));
        pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(srcChunk.GetPhyAddr()));
        pto::TEXP(dstTile, srcTile);
    }
}

template <int TileElems = 1024>
PTO_DEVICE void PtoReduceMaxVector(AscendC::LocalTensor<float> const &dst,
                                   AscendC::LocalTensor<float> const &src,
                                   AscendC::LocalTensor<float> const &tmp,
                                   uint32_t elemNum)
{
    using SrcTile = PtoVecTile<float, TileElems>;
    using TmpTile = PtoVecTile<float, TileElems>;
    using RowMaxTile = pto::Tile<pto::TileType::Vec, float, 8, 1, pto::BLayout::ColMajor, -1, 1>;
    using ScalarTile = pto::Tile<pto::TileType::Vec, float, 1, 8, pto::BLayout::RowMajor, -1, -1>;

    bool firstChunk = true;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto srcChunk = src[offset];
        auto tmpChunk = tmp[offset];
        auto chunkMaxChunk = firstChunk ? dst[0] : tmp[0];

        SrcTile srcTile(1, cur);
        TmpTile tmpTile(1, cur);
        RowMaxTile rowMaxTile(1);
        pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(srcChunk.GetPhyAddr()));
        pto::TASSIGN(tmpTile, reinterpret_cast<uint64_t>(tmpChunk.GetPhyAddr()));
        pto::TASSIGN(rowMaxTile, reinterpret_cast<uint64_t>(chunkMaxChunk.GetPhyAddr()));
        pto::TROWMAX(rowMaxTile, srcTile, tmpTile);
        AscendC::PipeBarrier<PIPE_V>();

        if (!firstChunk) {
            auto accChunk = dst[0];
            auto newChunk = tmp[0];
            ScalarTile accTile(1, 1);
            ScalarTile newTile(1, 1);
            ScalarTile dstTile(1, 1);
            pto::TASSIGN(accTile, reinterpret_cast<uint64_t>(accChunk.GetPhyAddr()));
            pto::TASSIGN(newTile, reinterpret_cast<uint64_t>(newChunk.GetPhyAddr()));
            pto::TASSIGN(dstTile, reinterpret_cast<uint64_t>(accChunk.GetPhyAddr()));
            pto::TMAX(dstTile, accTile, newTile);
            AscendC::PipeBarrier<PIPE_V>();
        }
        firstChunk = false;
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoLoadMatrixRows(AscendC::LocalTensor<Element> const &dst,
                                  AscendC::GlobalTensor<Element> const &src,
                                  uint32_t rowNum,
                                  uint32_t colNum,
                                  uint32_t dstStride,
                                  uint32_t srcStride)
{
    for (uint32_t rowIdx = 0; rowIdx < rowNum; ++rowIdx) {
        PtoLoadVector<Element, TileElems>(dst[rowIdx * dstStride], src[rowIdx * srcStride], colNum);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoStoreMatrixRows(AscendC::GlobalTensor<Element> const &dst,
                                   AscendC::LocalTensor<Element> const &src,
                                   uint32_t rowNum,
                                   uint32_t colNum,
                                   uint32_t dstStride,
                                   uint32_t srcStride)
{
    for (uint32_t rowIdx = 0; rowIdx < rowNum; ++rowIdx) {
        PtoStoreVector<Element, TileElems>(dst[rowIdx * dstStride], src[rowIdx * srcStride], colNum);
    }
}

}  // namespace pto_ext::dispatch_ffn_combine_v3::pto_bridge

#endif  // PTO_EXT_DISPATCH_FFN_COMBINE_V3_PTO_VECTOR_OPS_HPP
