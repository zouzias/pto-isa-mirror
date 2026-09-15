/**
 * Costmodel-only lowering for TPOW/TPOWS on A2/A3.
 *
 * Keep the same vector instruction sequence as the hardware implementation in
 * pto/npu/a2a3/TPow.hpp.  The hardware-only scalar loop merely repairs special
 * floating-point values in UB; it emits no CCE instruction and must not run in
 * the CPU mock because mock UB pointers are not host-addressable.
 */
#pragma once

#include "pto/npu/a2a3/TBinOp.hpp"
#include "pto/npu/a2a3/TBinSOp.hpp"

namespace pto {

template <typename T, bool NeedAbs>
struct CostmodelPowOp {
    PTO_INTERNAL static void BinInstr(
        __ubuf__ T* dst, __ubuf__ T* base, __ubuf__ T* exp, uint8_t repeats, uint8_t dstStride = 8,
        uint8_t baseStride = 8, uint8_t expStride = 8)
    {
        if constexpr (NeedAbs) {
            vabs(dst, base, repeats, 1, 1, dstStride, baseStride);
            pipe_barrier(PIPE_V);
            vln(dst, dst, repeats, 1, 1, dstStride, dstStride);
        } else {
            vln(dst, base, repeats, 1, 1, dstStride, baseStride);
        }
        pipe_barrier(PIPE_V);
        vmul(dst, dst, exp, repeats, 1, 1, 1, dstStride, dstStride, expStride);
        pipe_barrier(PIPE_V);
        vexp(dst, dst, repeats, 1, 1, dstStride, dstStride);
    }
};

template <typename T, bool NeedAbs>
struct CostmodelPowSOp {
    PTO_INTERNAL static void BinSInstr(
        __ubuf__ T* dst, __ubuf__ T* base, T exp, uint8_t repeats, uint8_t dstStride = 8,
        uint8_t baseStride = 8)
    {
        if constexpr (NeedAbs) {
            vabs(dst, base, repeats, 1, 1, dstStride, baseStride);
            pipe_barrier(PIPE_V);
            vln(dst, dst, repeats, 1, 1, dstStride, dstStride);
        } else {
            vln(dst, base, repeats, 1, 1, dstStride, baseStride);
        }
        pipe_barrier(PIPE_V);
        vmuls(dst, dst, exp, repeats, 1, 1, dstStride, dstStride);
        pipe_barrier(PIPE_V);
        vexp(dst, dst, repeats, 1, 1, dstStride, dstStride);
    }
};

template <PowAlgorithm algo, typename DstTile, typename BaseTile, typename ExpTile, typename TmpTile>
PTO_INTERNAL void TPOW_IMPL(DstTile& dst, BaseTile& base, ExpTile& exp, TmpTile& tmp)
{
    using T = typename DstTile::DType;
    if constexpr (!std::is_integral_v<T>) {
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
        BinaryInstr<
            CostmodelPowOp<T, true>, T, elementsPerRepeat, blockSizeElem, TmpTile::RowStride, BaseTile::RowStride,
            ExpTile::RowStride>(tmp.data(), base.data(), exp.data(), dst.GetValidRow(), dst.GetValidCol());
        BinaryInstr<
            CostmodelPowOp<T, false>, T, elementsPerRepeat, blockSizeElem, DstTile::RowStride, BaseTile::RowStride,
            ExpTile::RowStride>(dst.data(), base.data(), exp.data(), dst.GetValidRow(), dst.GetValidCol());
    }
}

template <PowAlgorithm algo, typename DstTile, typename BaseTile, typename TmpTile>
PTO_INTERNAL void TPOWS_IMPL(DstTile& dst, BaseTile& base, typename DstTile::DType exp, TmpTile& tmp)
{
    using T = typename DstTile::DType;
    if constexpr (!std::is_integral_v<T>) {
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
        TBinSInstr<
            CostmodelPowSOp<T, true>, TmpTile, BaseTile, elementsPerRepeat, blockSizeElem, TmpTile::RowStride,
            BaseTile::RowStride>(tmp.data(), base.data(), exp, dst.GetValidRow(), dst.GetValidCol());
        TBinSInstr<
            CostmodelPowSOp<T, false>, DstTile, BaseTile, elementsPerRepeat, blockSizeElem, DstTile::RowStride,
            BaseTile::RowStride>(dst.data(), base.data(), exp, dst.GetValidRow(), dst.GetValidCol());
    }
}

} // namespace pto
