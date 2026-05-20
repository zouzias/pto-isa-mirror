/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

template <typename T, typename U, typename S, int M, int K, int N, uint16_t indexM, uint16_t indexK, uint16_t indexN,
          int baseM, int baseK, int baseN, bool isAtranspose, bool isBtranspose>
AICORE inline void runTEXTRACT_UNALIGN(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    constexpr int mValid = M - indexM;
    constexpr int kValid = K - indexK;
    constexpr int nValid = N - indexN;

    using GlobalDataSrc0 = std::conditional_t<
        isAtranspose,
        GlobalTensor<U, pto::Shape<1, 1, 1, baseM, baseK>,
                     pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, 1, baseM>, Layout::DN>,
        GlobalTensor<U, pto::Shape<1, 1, 1, baseM, baseK>,
                     pto::Stride<1 * baseM * baseK, 1 * baseM * baseK, baseM * baseK, baseK, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t<
        isBtranspose,
        GlobalTensor<S, pto::Shape<1, 1, 1, baseK, baseN>,
                     pto::Stride<1 * baseK * baseN, 1 * baseK * baseN, baseK * baseN, baseN, 1>, Layout::ND>,
        GlobalTensor<S, pto::Shape<1, 1, 1, baseK, baseN>,
                     pto::Stride<1 * baseK * baseN, 1 * baseK * baseN, baseK * baseN, 1, baseK>, Layout::DN>>;
    using GlobalDataOut =
        GlobalTensor<T, pto::Shape<1, 1, 1, mValid, nValid>,
                     pto::Stride<1 * mValid * nValid, 1 * mValid * nValid, mValid * nValid, nValid, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = std::conditional_t<
        isAtranspose, Tile<TileType::Mat, U, baseM, baseK, BLayout::RowMajor, baseM, baseK, SLayout::ColMajor, 512>,
        Tile<TileType::Mat, U, baseM, baseK, BLayout::ColMajor, baseM, baseK, SLayout::RowMajor, 512>>;
    using TileMatBData = std::conditional_t<
        isBtranspose, Tile<TileType::Mat, S, baseK, baseN, BLayout::ColMajor, baseK, baseN, SLayout::RowMajor, 512>,
        Tile<TileType::Mat, S, baseK, baseN, BLayout::RowMajor, baseK, baseN, SLayout::ColMajor, 512>>;

    using LeftTile = TileLeftCompact<U, baseM, baseK, mValid, kValid>;
    using RightTile = TileRightCompact<S, baseK, baseN, kValid, nValid>;
    using ResTile = TileAccCompact<T, baseM, baseN, mValid, nValid>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    LeftTile aTile;
    RightTile bTile;
    ResTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    /*********************************TEXTRACT**********************************/
    TEXTRACT(aTile, aMatTile, indexM, indexK);
    TEXTRACT(bTile, bMatTile, indexK, indexN);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    TMATMUL(cTile, aTile, bTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    /****************************************TSTORE*****************************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTEXTRACT_11(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, float, float, 32, 256, 32, 0, 0, 0, 32, 256, 32, false, false>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_12(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, float, float, 256, 32, 32, 0, 0, 0, 256, 32, 32, false, false>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_13(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runTEXTRACT_UNALIGN<float, float, float, 128, 128, 128, 0, 0, 0, 128, 128, 128, false, false>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}

template <int32_t tilingKey>
void launchTEXTRACT(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    } else if constexpr (tilingKey == 72) {
        launchTEXTRACT_72<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 73) {
        launchTEXTRACT_73<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 74) {
template void launchTEXTRACT<62>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<63>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<64>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<71>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<72>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<73>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<74>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);