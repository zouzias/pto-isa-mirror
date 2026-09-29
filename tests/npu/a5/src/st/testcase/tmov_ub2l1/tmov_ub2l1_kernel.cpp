/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <iostream>

#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>

using namespace std;
using namespace pto;

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL void tf_copy_cbuf_to_ubuf(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t vector,
    uint16_t blockLen)
{
    __cbuf__ T* srcMatAddr = (__cbuf__ T*)__cce_get_tile_ptr(src);
    __ubuf__ T* dstUbAddr = __cce_get_tile_ptr(dst);
    copy_cbuf_to_ubuf(
        (__ubuf__ void*)dstUbAddr, (__cbuf__ void*)srcMatAddr, vector, 1, blockLen, 0,
        0); // move to vector0 or vector1
}

template <
    typename T, uint32_t Rows, uint32_t Cols, uint32_t ExtraRows, uint32_t ValidRows = Rows, uint32_t ValidCols = Cols,
    uint32_t IndexRows = 0, uint32_t IndexCols = 0>
AICORE void runTmovUb2l1(__gm__ T* out, __gm__ T* src)
{
    using SrcShapeDim5 = pto::Shape<1, 1, 1, Rows, Cols>;
    using SrcStridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStridDim5>;

    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    using OutShapeDim5 = pto::Shape<1, ValidCols / c0Size, ValidRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using OutStridDim5 =
        pto::Stride<Cols / c0Size * c0Size * ValidRows, ValidRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;

    using SrcTileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, -1, -1>;
    using TmpTileData = std::conditional_t<
        (ExtraRows > Rows),
        Tile<
            TileType::Vec, T, ExtraRows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::RowPlusOne>,
        Tile<TileType::Vec, T, ExtraRows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>>;

    using DstTileData = Tile<TileType::Vec, T, ValidRows, ValidCols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    using MatTileData = Tile<TileType::Mat, T, ValidRows, ValidCols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    SrcTileData srcTile(Rows, Cols);
    TmpTileData tmpTile(Rows, Cols);
    DstTileData dstTile(ValidRows, ValidCols);
    MatTileData matTile(ValidRows, ValidCols);
    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, 0x10000);
    TASSIGN(dstTile, 0x20000);
    TASSIGN(matTile, 0x0);

    SrcGlobalData srcGlobal(src);
    OutGlobalData dstGlobal(out);
    uint8_t syncId = 0;
    uint8_t eventIdNum = 16;
    uint16_t blockLen = ValidRows * ValidCols * sizeof(T) / BLOCK_BYTE_SIZE;

#if defined(__DAV_VEC__)
    TLOAD(srcTile, srcGlobal); // gm->ub
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TMOV(tmpTile, srcTile); // ub2Ub
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    if constexpr (IndexRows != 0 || IndexCols != 0) {
        TEXTRACT(matTile, tmpTile, IndexRows, IndexCols);
    } else {
        TMOV(matTile, tmpTile); // ub2l1
    }
    set_intra_block(PIPE_MTE3, syncId);
#endif

#if defined(__DAV_CUBE__)
    wait_intra_block(PIPE_MTE1, syncId); // MTE1 等待V侧MTE3流水
    wait_intra_block(PIPE_MTE1, syncId + eventIdNum);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
#endif
    tf_copy_cbuf_to_ubuf<T, DstTileData, MatTileData>(dstTile.data(), matTile.data(), 0, blockLen);
    tf_copy_cbuf_to_ubuf<T, DstTileData, MatTileData>(dstTile.data(), matTile.data(), 1, blockLen);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
#endif

    set_intra_block(PIPE_MTE1, syncId); // ub2l1 告诉V侧已经搬完,C侧L12UB MTE1流水
    set_intra_block(PIPE_MTE1, syncId + eventIdNum);
#endif

#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncId);
    TSTORE(dstGlobal, dstTile); // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}

template <
    typename T, uint32_t Rows, uint32_t Cols, uint32_t ExtraRows, uint32_t ValidRows = Rows, uint32_t ValidCols = Cols,
    uint32_t IndexRows = 0, uint32_t IndexCols = 0>
__global__ AICORE void launchTmovUb2l1(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTmovUb2l1<T, Rows, Cols, ExtraRows, ValidRows, ValidCols, IndexRows, IndexCols>(
        reinterpret_cast<__gm__ T*>(out), reinterpret_cast<__gm__ T*>(src));
}

template <int32_t testKey>
void launchTmovUb2l1(uint64_t* out, uint64_t* src, void* stream)
{
    cout << "launchTmovUb2l1 start!" << endl;
    if constexpr (testKey == 1) {
        launchTmovUb2l1<half, 16, 32, 17><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 2) {
        launchTmovUb2l1<half, 64, 256, 65><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 3) {
        launchTmovUb2l1<float, 48, 72, 48><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 4) {
        launchTmovUb2l1<float, 96, 8, 97><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 5) {
        launchTmovUb2l1<int8_t, 32, 512, 32><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 6) {
        launchTmovUb2l1<int8_t, 64, 96, 64><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 7) {
        launchTmovUb2l1<half, 64, 64, 65, 48, 48, 16, 16><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 8) {
        launchTmovUb2l1<float, 128, 128, 128, 64, 64, 64, 64><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (testKey == 9) {
        launchTmovUb2l1<int8_t, 256, 256, 256, 32, 32, 224, 224><<<1, nullptr, stream>>>(out, src);
    }
    cout << "launchTmovUb2l1 end!" << endl;
}

template void launchTmovUb2l1<1>(uint64_t* out, uint64_t* src, void* stream);
template void launchTmovUb2l1<2>(uint64_t* out, uint64_t* src, void* stream);
template void launchTmovUb2l1<3>(uint64_t* out, uint64_t* src, void* stream);
template void launchTmovUb2l1<4>(uint64_t* out, uint64_t* src, void* stream);
template void launchTmovUb2l1<5>(uint64_t* out, uint64_t* src, void* stream);
template void launchTmovUb2l1<6>(uint64_t* out, uint64_t* src, void* stream);
template void launchTmovUb2l1<7>(uint64_t* out, uint64_t* src, void* stream);
template void launchTmovUb2l1<8>(uint64_t* out, uint64_t* src, void* stream);
template void launchTmovUb2l1<9>(uint64_t* out, uint64_t* src, void* stream);

#ifndef PTO_SKIP_UB2L1_ND2NZ_ST
template <typename T, typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL void readbackTMovNd2NzMat(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t vectorId,
    uint16_t blockCount, uint32_t srcByteOffset = 0)
{
    __cbuf__ T* srcMatAddr = (__cbuf__ T*)((__cbuf__ uint8_t*)__cce_get_tile_ptr(src) + srcByteOffset);
    __ubuf__ T* dstUbAddr = __cce_get_tile_ptr(dst);
    copy_cbuf_to_ubuf((__ubuf__ void*)dstUbAddr, (__cbuf__ void*)srcMatAddr, vectorId, 1, blockCount, 0, 0);
}

template <typename MatTile, typename UbTile>
__tf__ PTO_INTERNAL void initTMovNd2NzMat(
    typename MatTile::TileDType __out__ dst, typename UbTile::TileDType __in__ src, uint32_t byteCount,
    uint32_t dstByteOffset)
{
    copy_ubuf_to_cbuf(
        (__cbuf__ uint8_t*)__cce_get_tile_ptr(dst) + dstByteOffset, (__ubuf__ void*)__cce_get_tile_ptr(src), 0, 1,
        byteCount / BLOCK_BYTE_SIZE, 0, 0);
}

template <
    typename T, int ElementBits, int SrcRows, int SrcCols, int DstRows, int DstCols, int ValidRows, int ValidCols,
    bool UseDefaultCopy, bool Dynamic, int SrcValidRows, int SrcValidCols>
__global__ AICORE void runTMovUbToL1Nd2Nz(
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
            initTMovNd2NzMat<DstTile, RawDst>(dst.data(), rawDst.data(), bytes, offset);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
        TLOAD(rawSrc, srcGlobal);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        if constexpr (UseDefaultCopy) {
            TMOV(dst, src);
        } else {
            TMOV<TileCopyMode::ND2NZ>(dst, src);
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
        readbackTMovNd2NzMat<uint8_t, RawDst, DstTile>(rawDst.data(), dst.data(), 0, bytes / BLOCK_BYTE_SIZE, offset);
        set_intra_block(PIPE_MTE1, UB_READY);
    }
#endif
}

template <>
void launchTmovUb2l1<10>(uint64_t* out, uint64_t* src, void* stream)
{
    runTMovUbToL1Nd2Nz<half, 16, 16, 32, 16, 32, 16, 32, false, false, 16, 32>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 16, 32, 16, 32);
}

template <>
void launchTmovUb2l1<11>(uint64_t* out, uint64_t* src, void* stream)
{
    runTMovUbToL1Nd2Nz<half, 16, 16, 512, 16, 512, 3, 512, false, true, 3, 512>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 3, 512, 3, 512);
}

template <>
void launchTmovUb2l1<12>(uint64_t* out, uint64_t* src, void* stream)
{
    runTMovUbToL1Nd2Nz<float, 32, 32, 64, 32, 32, 19, 24, false, true, 19, 24>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 19, 24, 19, 24);
}

template <>
void launchTmovUb2l1<13>(uint64_t* out, uint64_t* src, void* stream)
{
    runTMovUbToL1Nd2Nz<bfloat16_t, 16, 32, 96, 32, 64, 17, 48, false, true, 17, 48>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 17, 48, 17, 48);
}

template <>
void launchTmovUb2l1<14>(uint64_t* out, uint64_t* src, void* stream)
{
    runTMovUbToL1Nd2Nz<int8_t, 8, 32, 160, 32, 128, 31, 96, false, true, 31, 96>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 31, 96, 31, 96);
}

template <>
void launchTmovUb2l1<15>(uint64_t* out, uint64_t* src, void* stream)
{
    runTMovUbToL1Nd2Nz<float8_e4m3_t, 8, 16, 64, 16, 64, 16, 64, false, false, 16, 64>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 16, 64, 16, 64);
}

template <>
void launchTmovUb2l1<16>(uint64_t* out, uint64_t* src, void* stream)
{
    runTMovUbToL1Nd2Nz<float8_e5m2_t, 8, 16, 64, 16, 64, 16, 64, false, false, 16, 64>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 16, 64, 16, 64);
}

template <>
void launchTmovUb2l1<17>(uint64_t* out, uint64_t* src, void* stream)
{
    runTMovUbToL1Nd2Nz<hifloat8_t, 8, 16, 64, 16, 64, 16, 64, false, false, 16, 64>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 16, 64, 16, 64);
}

template <>
void launchTmovUb2l1<18>(uint64_t* out, uint64_t* src, void* stream)
{
    runTMovUbToL1Nd2Nz<float8_e8m0_t, 8, 16, 64, 16, 64, 16, 64, false, false, 16, 64>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 16, 64, 16, 64);
}

template <>
void launchTmovUb2l1<19>(uint64_t* out, uint64_t* src, void* stream)
{
    runTMovUbToL1Nd2Nz<float4_e2m1x2_t, 4, 32, 256, 32, 192, 17, 128, false, true, 17, 128>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 17, 128, 17, 128);
}

template <>
void launchTmovUb2l1<20>(uint64_t* out, uint64_t* src, void* stream)
{
    runTMovUbToL1Nd2Nz<float4_e1m2x2_t, 4, 16, 128, 16, 128, 16, 128, false, false, 16, 128>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 16, 128, 16, 128);
}

template <>
void launchTmovUb2l1<24>(uint64_t* out, uint64_t* src, void* stream)
{
    runTMovUbToL1Nd2Nz<half, 16, 4112, 16, 4112, 16, 4101, 16, false, true, 4101, 16>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 4101, 16, 4101, 16);
}

template <>
void launchTmovUb2l1<25>(uint64_t* out, uint64_t* src, void* stream)
{
    runTMovUbToL1Nd2Nz<half, 16, 16, 32, 16, 32, 0, 32, false, true, 0, 32>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 0, 32, 0, 32);
}

template <>
void launchTmovUb2l1<27>(uint64_t* out, uint64_t* src, void* stream)
{
    runTMovUbToL1Nd2Nz<float, 32, 16, 32, 16, 32, 7, 0, false, true, 7, 0>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 7, 0, 7, 0);
}

template <>
void launchTmovUb2l1<70>(uint64_t* out, uint64_t* src, void* stream)
{
    runTMovUbToL1Nd2Nz<half, 16, 16, 64, 16, 64, 2, 32, true, false, 2, 32>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 2, 32, 2, 32);
}

template <>
void launchTmovUb2l1<71>(uint64_t* out, uint64_t* src, void* stream)
{
    runTMovUbToL1Nd2Nz<half, 16, 16, 64, 16, 64, 7, 32, true, true, 7, 32>
        <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(out), reinterpret_cast<uint8_t*>(src), 7, 32, 7, 32);
}

#endif
