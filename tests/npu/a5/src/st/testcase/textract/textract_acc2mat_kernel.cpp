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

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL void readbackTExtractNd2NzMat(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t vectorId,
    uint16_t blockCount, uint32_t srcByteOffset = 0)
{
    __cbuf__ T* srcMatAddr = (__cbuf__ T*)((__cbuf__ uint8_t*)__cce_get_tile_ptr(src) + srcByteOffset);
    __ubuf__ T* dstUbAddr = __cce_get_tile_ptr(dst);
    copy_cbuf_to_ubuf((__ubuf__ void*)dstUbAddr, (__cbuf__ void*)srcMatAddr, vectorId, 1, blockCount, 0, 0);
}

template <typename MatTile, typename UbTile>
__tf__ PTO_INTERNAL void initTExtractNd2NzMat(
    typename MatTile::TileDType __out__ dst, typename UbTile::TileDType __in__ src, uint32_t byteCount,
    uint32_t dstByteOffset)
{
    copy_ubuf_to_cbuf(
        (__cbuf__ uint8_t*)__cce_get_tile_ptr(dst) + dstByteOffset, (__ubuf__ void*)__cce_get_tile_ptr(src), 0, 1,
        byteCount / BLOCK_BYTE_SIZE, 0, 0);
}

template <
    typename T, int ElementBits, int SrcRows, int SrcCols, int DstRows, int DstCols, int ValidRows, int ValidCols,
    bool UseDefaultCopy, int IndexRow, int IndexCol, bool Dynamic, int SrcValidRows, int SrcValidCols>
__global__ AICORE void runTExtractUbToL1Nd2Nz(
    __gm__ uint8_t* out, __gm__ uint8_t* input, uint32_t validRows, uint32_t validCols, uint32_t srcValidRows,
    uint32_t srcValidCols)
{
    constexpr uint32_t L1_READY = 0;
    constexpr uint32_t UB_READY = 1;
    constexpr uint32_t DST_BYTES = DstRows * DstCols * ElementBits / 8;
    static_assert(SrcRows * SrcCols * ElementBits / 8 <= 256 * 1024, "ST source exceeds A5 UB capacity.");
    static_assert(DST_BYTES <= 512 * 1024, "ST destination exceeds A5 L1 capacity.");
    constexpr uint32_t MAX_READBACK_BYTES = 128 * 1024;
    constexpr uint32_t READBACK_BYTES = DST_BYTES > MAX_READBACK_BYTES ? MAX_READBACK_BYTES : DST_BYTES;
    constexpr uint32_t READBACK_COUNT = (DST_BYTES + READBACK_BYTES - 1) / READBACK_BYTES;
    using SrcTile = Tile<
        TileType::Vec, T, SrcRows, SrcCols, BLayout::RowMajor, Dynamic ? DYNAMIC : SrcValidRows,
        Dynamic ? DYNAMIC : SrcValidCols>;
    using DstTile = Tile<
        TileType::Mat, T, DstRows, DstCols, BLayout::ColMajor, Dynamic ? DYNAMIC : ValidRows,
        Dynamic ? DYNAMIC : ValidCols, SLayout::RowMajor>;
    using RawSrc = Tile<TileType::Vec, uint8_t, SrcRows, SrcCols * ElementBits / 8>;
    using RawDst = Tile<TileType::Vec, uint8_t, 1, READBACK_BYTES, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    using SrcGlobal = GlobalTensor<
        uint8_t, Shape<1, 1, 1, SrcRows, SrcCols * ElementBits / 8>,
        pto::Stride<1, 1, 1, SrcCols * ElementBits / 8, 1>>;
    using DstGlobal = GlobalTensor<uint8_t, Shape<1, 1, 1, 1, DYNAMIC>, pto::Stride<1, 1, 1, READBACK_BYTES, 1>>;
    SrcTile src;
    DstTile dst;
    if constexpr (Dynamic) {
        src.SetValidShape(srcValidRows, srcValidCols);
        dst.SetValidShape(validRows, validCols);
    }
    RawSrc rawSrc;
    RawDst rawDst(1, READBACK_BYTES);
    TASSIGN(src, 0);
    TASSIGN(dst, 0);
    TASSIGN(rawSrc, 0);
    TASSIGN(rawDst, 0);
    SrcGlobal srcGlobal(input);
#if defined(__DAV_VEC__)
    if (get_subblockid() == 0) {
        for (uint32_t chunk = 0; chunk < READBACK_COUNT; ++chunk) {
            const uint32_t offset = chunk * READBACK_BYTES;
            const uint32_t bytes = min(READBACK_BYTES, DST_BYTES - offset);
            rawDst.SetValidShape(1, bytes);
            DstGlobal dstGlobal(out + offset, Shape<1, 1, 1, 1, DYNAMIC>(bytes));
            TLOAD(rawDst, dstGlobal);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            initTExtractNd2NzMat<DstTile, RawDst>(dst.data(), rawDst.data(), bytes, offset);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
        TLOAD(rawSrc, srcGlobal);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        if constexpr (UseDefaultCopy) {
            TEXTRACT(dst, src, IndexRow, IndexCol);
        } else {
            TEXTRACT<TileCopyMode::ND2NZ>(dst, src, IndexRow, IndexCol);
        }
        set_intra_block(PIPE_MTE3, L1_READY);
        for (uint32_t chunk = 0; chunk < READBACK_COUNT; ++chunk) {
            const uint32_t offset = chunk * READBACK_BYTES;
            const uint32_t bytes = min(READBACK_BYTES, DST_BYTES - offset);
            rawDst.SetValidShape(1, bytes);
            DstGlobal dstGlobal(out + offset, Shape<1, 1, 1, 1, DYNAMIC>(bytes));
            wait_intra_block(PIPE_MTE3, UB_READY);
            TSTORE(dstGlobal, rawDst);
            if (chunk + 1 < READBACK_COUNT) {
                set_intra_block(PIPE_MTE3, L1_READY);
            }
        }
    }
#endif
#if defined(__DAV_CUBE__)
    wait_intra_block(PIPE_MTE1, L1_READY);
    // Wait for GM stores before reusing the bounded UB readback buffer.
    for (uint32_t chunk = 0; chunk < READBACK_COUNT; ++chunk) {
        if (chunk > 0) {
            wait_intra_block(PIPE_MTE1, L1_READY);
        }
        const uint32_t offset = chunk * READBACK_BYTES;
        const uint32_t bytes = min(READBACK_BYTES, DST_BYTES - offset);
        readbackTExtractNd2NzMat<uint8_t, RawDst, DstTile>(
            rawDst.data(), dst.data(), 0, bytes / BLOCK_BYTE_SIZE, offset);
        set_intra_block(PIPE_MTE1, UB_READY);
    }
#endif
}

template <int32_t TestKey>
void launchTExtractNd2Nz(uint64_t* out, uint64_t* src, void* stream);

template <>
void launchTExtractNd2Nz<21>(uint64_t* out, uint64_t* src, void* stream)
{
    runTExtractUbToL1Nd2Nz<half, 16, 48, 96, 32, 64, 17, 48, false, 3, 16, true, 48, 96>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 17, 48, 48, 96);
}

template <>
void launchTExtractNd2Nz<30>(uint64_t* out, uint64_t* src, void* stream)
{
    runTExtractUbToL1Nd2Nz<bfloat16_t, 16, 48, 96, 32, 64, 17, 48, false, 3, 16, true, 24, 80>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 17, 48, 24, 80);
}

template <>
void launchTExtractNd2Nz<31>(uint64_t* out, uint64_t* src, void* stream)
{
    runTExtractUbToL1Nd2Nz<float, 32, 48, 48, 32, 32, 17, 24, false, 3, 8, true, 24, 40>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 17, 24, 24, 40);
}

template <>
void launchTExtractNd2Nz<32>(uint64_t* out, uint64_t* src, void* stream)
{
    runTExtractUbToL1Nd2Nz<int8_t, 8, 48, 192, 32, 128, 17, 96, false, 3, 32, true, 24, 160>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 17, 96, 24, 160);
}

template <>
void launchTExtractNd2Nz<33>(uint64_t* out, uint64_t* src, void* stream)
{
    runTExtractUbToL1Nd2Nz<hifloat8_t, 8, 48, 192, 32, 128, 17, 96, false, 3, 32, true, 24, 160>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 17, 96, 24, 160);
}

template <>
void launchTExtractNd2Nz<34>(uint64_t* out, uint64_t* src, void* stream)
{
    runTExtractUbToL1Nd2Nz<float8_e4m3_t, 8, 48, 192, 32, 128, 17, 96, false, 3, 32, true, 24, 160>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 17, 96, 24, 160);
}

template <>
void launchTExtractNd2Nz<35>(uint64_t* out, uint64_t* src, void* stream)
{
    runTExtractUbToL1Nd2Nz<float8_e5m2_t, 8, 48, 192, 32, 128, 17, 96, false, 3, 32, true, 24, 160>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 17, 96, 24, 160);
}

template <>
void launchTExtractNd2Nz<36>(uint64_t* out, uint64_t* src, void* stream)
{
    runTExtractUbToL1Nd2Nz<float8_e8m0_t, 8, 48, 192, 32, 128, 17, 96, false, 3, 32, true, 24, 160>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 17, 96, 24, 160);
}

template <>
void launchTExtractNd2Nz<37>(uint64_t* out, uint64_t* src, void* stream)
{
    runTExtractUbToL1Nd2Nz<float4_e2m1x2_t, 4, 48, 384, 32, 256, 17, 192, false, 3, 64, true, 24, 320>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 17, 192, 24, 320);
}

template <>
void launchTExtractNd2Nz<38>(uint64_t* out, uint64_t* src, void* stream)
{
    runTExtractUbToL1Nd2Nz<float4_e1m2x2_t, 4, 48, 384, 32, 256, 17, 192, false, 3, 64, true, 24, 320>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 17, 192, 24, 320);
}

template <>
void launchTExtractNd2Nz<40>(uint64_t* out, uint64_t* src, void* stream)
{
    runTExtractUbToL1Nd2Nz<half, 16, 48, 96, 32, 64, 17, 48, false, 3, 16, false, 20, 64>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 17, 48, 20, 64);
}

template <>
void launchTExtractNd2Nz<41>(uint64_t* out, uint64_t* src, void* stream)
{
    runTExtractUbToL1Nd2Nz<half, 16, 16, 64, 16, 32, 0, 16, false, 3, 16, true, 8, 48>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 0, 16, 8, 48);
}

template <>
void launchTExtractNd2Nz<42>(uint64_t* out, uint64_t* src, void* stream)
{
    runTExtractUbToL1Nd2Nz<half, 16, 16, 64, 16, 32, 7, 0, false, 3, 16, true, 10, 48>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 7, 0, 10, 48);
}

template <>
void launchTExtractNd2Nz<43>(uint64_t* out, uint64_t* src, void* stream)
{
    runTExtractUbToL1Nd2Nz<float, 32, 48, 64, 32, 32, 17, 24, false, 3, 8, true, 20, 32>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 17, 24, 20, 32);
}

template <>
void launchTExtractNd2Nz<70>(uint64_t* out, uint64_t* src, void* stream)
{
    runTExtractUbToL1Nd2Nz<half, 16, 16, 64, 16, 64, 2, 32, true, 1, 16, false, 16, 64>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 2, 32, 16, 64);
}

template <>
void launchTExtractNd2Nz<71>(uint64_t* out, uint64_t* src, void* stream)
{
    runTExtractUbToL1Nd2Nz<half, 16, 16, 64, 16, 64, 7, 32, true, 3, 16, true, 16, 64>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 7, 32, 16, 64);
}
