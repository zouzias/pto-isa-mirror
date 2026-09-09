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

using namespace pto;

template <STPhase phase, bool kSplit = false, int fractalSize = 1024>
__global__ AICORE void RunTExtractAcc2Mat(__gm__ float* out, __gm__ half* src0, __gm__ half* src1)
{
    constexpr int M = 32;
    constexpr int K = 96;
    constexpr int N = 64;
    constexpr int outputElements = 16 * 48;
    constexpr int outputCount = phase == STPhase::Partial ? 2 : 1;
#if defined(__DAV_CUBE__)
    using GlobalDataSrc0 = GlobalTensor<half, Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<half, Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, 1, K>, Layout::DN>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);

    using TileMatAData = Tile<TileType::Mat, half, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor>;
    using TileMatBData = Tile<TileType::Mat, half, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor>;
    constexpr int kStep = kSplit ? K / 2 : K;
    using LeftTile = TileLeft<half, M, kStep>;
    using RightTile = TileRight<half, kStep, N>;
    using ResTile = TileAcc<float, M, N>;
    using DstMatTile = Tile<TileType::Mat, float, 16, 48, BLayout::ColMajor, 16, 48, SLayout::RowMajor, fractalSize>;

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

    DstMatTile dstMatTile;
    TASSIGN(dstMatTile, 0x20000);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    TEXTRACT(aTile, aMatTile, 0, 0);
    TEXTRACT(bTile, bMatTile, 0, 0);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    if constexpr (kSplit) {
        // 非末次累加用 AccPhase::Partial，末次用 Final，对应 issue 564 描述的 K 切分场景
        TMATMUL<AccPhase::Partial>(cTile, aTile, bTile);
#ifndef __PTO_AUTO__
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
#endif
        TEXTRACT(aTile, aMatTile, 0, static_cast<uint16_t>(kStep));
        TEXTRACT(bTile, bMatTile, static_cast<uint16_t>(kStep), 0);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
        TMATMUL_ACC<AccPhase::Final>(cTile, aTile, bTile);
    } else {
        TMATMUL<phase == STPhase::Unspecified ? AccPhase::Unspecified : AccPhase::Final>(cTile, aTile, bTile);
    }

    if constexpr (phase == STPhase::Unspecified) {
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    }

    if constexpr (phase == STPhase::Partial) {
        DstMatTile finalMatTile;
        TASSIGN(finalMatTile, 0x30000);
        TEXTRACT<STPhase::Partial>(dstMatTile, cTile, 16, 16);
        TEXTRACT<STPhase::Final>(finalMatTile, cTile, 0, 0);
    } else {
        TEXTRACT<phase>(dstMatTile, cTile, 16, 16);
    }

    set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    // Copy the complete NZ tile; the golden uses the destination fractal size.
    copy_cbuf_to_ubuf((__ubuf__ void*)0, (__cbuf__ void*)0x20000, 0, 1, outputElements * sizeof(float) / 32, 0, 0);
    if constexpr (phase == STPhase::Partial) {
        copy_cbuf_to_ubuf(
            (__ubuf__ void*)(outputElements * sizeof(float)), (__cbuf__ void*)0x30000, 0, 1,
            outputElements * sizeof(float) / 32, 0, 0);
    }
    set_intra_block(PIPE_MTE1, 0);
    set_intra_block(PIPE_MTE1, 16);
#endif
#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, 0);
    if (get_subblockid() == 0) {
        using VecTile = Tile<TileType::Vec, float, 1, outputElements * outputCount>;
        using Output = GlobalTensor<
            float, Shape<1, 1, 1, 1, outputElements * outputCount>,
            pto::Stride<
                outputElements * outputCount, outputElements * outputCount, outputElements * outputCount,
                outputElements * outputCount, 1>>;
        VecTile vecTile;
        TASSIGN(vecTile, 0);
        Output dst(out);
        TSTORE(dst, vecTile);
    }
#endif
}

template <int32_t key>
void launchTEXTRACTAcc2Mat(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream)
{
    constexpr STPhase phase =
        key == 26 ? STPhase::Unspecified : (key == 22 || key == 25 ? STPhase::Partial : STPhase::Final);
    RunTExtractAcc2Mat<phase, key == 23, key >= 24 ? 512 : 1024><<<1, nullptr, stream>>>(
        reinterpret_cast<float*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
}

template void launchTEXTRACTAcc2Mat<21>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTEXTRACTAcc2Mat<22>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTEXTRACTAcc2Mat<23>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTEXTRACTAcc2Mat<24>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTEXTRACTAcc2Mat<25>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTEXTRACTAcc2Mat<26>(uint8_t*, uint8_t*, uint8_t*, void*);
