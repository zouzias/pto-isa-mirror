/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>

#ifndef PIPE_FIX
#define PIPE_FIX static_cast<pipe_t>(10)
#endif

#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

#include "control_metadata.hpp"
#include "kernel_launchers.hpp"
#include "moe_new_dispatch_combine_a8w8_runtime_types.hpp"

using dispatch_combine_tile::DispatchCombineTileShape;
using dispatch_combine_tile::HcclDeviceContext;
using dispatch_combine_tile::PeerWindowLayout;
using dispatch_combine_tile::WorkspaceLayout;

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t *
#endif

// Kernel source contract for later tasks:
//   - Only PTO and C/C++ headers are allowed in this file.
//   - Device payload movement will use pto::GlobalTensor and pto::Tile views.
//   - Cross-rank movement/readiness will use PTO comm primitives.
//   - There are exactly two public device kernel names in this project.

namespace {

constexpr int kDefaultTileCols = 1024;
constexpr int kMetaTileCols = 16;
constexpr uint64_t kPingUbAddr = 0x0;
constexpr uint64_t kPongUbAddr = 0x1000;
constexpr uint64_t kSoftSyncUbAddr = 0x5000;
constexpr int kM2RouteQuantTileCols = 1024;
constexpr uint64_t kM2RouteHalfTileOffset = 0x0;
constexpr uint64_t kM2RouteFloatTileOffset = 0x1000;
constexpr uint64_t kM2RouteAbsTileOffset = 0x2400;
constexpr uint64_t kM2RouteQuantTileOffset = 0x3800;
constexpr uint64_t kM2RouteRowMaxTileOffset = 0x3C00;
constexpr uint64_t kM2RouteChunkMaxTileOffset = 0x3D00;
constexpr uint64_t kM2RouteScaleParamTileOffset = 0x3E00;
constexpr uint64_t kM2RouteInvScaleParamTileOffset = 0x3F00;
constexpr uint64_t kM2RouteScaleStoreTileOffset = 0x4000;
constexpr uint64_t kM2RoutePongHalfTileOffset = 0x7000;
constexpr uint64_t kM2RoutePongFloatTileOffset = 0x8000;
constexpr uint64_t kM2RoutePongAbsTileOffset = 0x9400;
constexpr uint64_t kM2RoutePongQuantTileOffset = 0xA800;
constexpr uint64_t kM2RouteQuantS32ScratchOffset = 0xB000;
constexpr uint64_t kM2RouteQuantF16ScratchOffset = 0xC000;
constexpr uint64_t kM2RoutePackedRowsUbAddr = 0xD000;
constexpr uint32_t kM2RoutePackedRowsUbCapacity = 256U;
constexpr uint32_t kM2RouteQuantReuseFloatCols = 512U;
constexpr int kM2EpilogueTileCols = 1024;
constexpr uint64_t kM2EpilogueAccTileOffset = 0x0;
constexpr uint64_t kM2EpilogueFloatTileOffset = 0x1000;
constexpr uint64_t kM2EpilogueScaleTileOffset = 0x2400;
constexpr uint64_t kM2EpilogueScaledTileOffset = 0x3800;
constexpr uint64_t kM2EpilogueHalfTileOffset = 0x4C00;
constexpr uint32_t kM2GmmBaseM = moe_new_dispatch_combine_a8w8::kGmmBaseM;
constexpr uint32_t kM2GmmBaseN = moe_new_dispatch_combine_a8w8::kGmmBaseN;
constexpr uint32_t kM2ReturnTileRows = moe_new_dispatch_combine_a8w8::kReturnTileRows;
constexpr uint32_t kM2SwigluGroupFields = 8;
constexpr uint32_t kM2GmmTileTaskFields = 8;
constexpr uint32_t kM2ReturnPlanFields = 8;
constexpr uint32_t kM2OwnerSegmentFields = 8;
constexpr uint32_t kM3CounterBase = 24U * 16U;
constexpr uint32_t kM3NDispatchCounterBase = kM3CounterBase + 16U;
constexpr uint32_t kM3NActivationCounterBase = kM3CounterBase + 32U;
constexpr uint32_t kM3NCombineCounterBase = kM3CounterBase + 48U;
constexpr uint32_t kM3NGmm2CounterBase = kM3CounterBase + 64U;
constexpr uint32_t kM3NDispatchScratchBase = 32U * 16U;
constexpr uint32_t kM3NDispatchScratchLimit = 40U * 16U;
constexpr uint32_t kM3NDispatchMaxWorkers = moe_new_dispatch_combine_a8w8::kInitQuantMaxDispatchWorkers;
constexpr uint32_t kM3NInitQuantFullLoadRouteThreshold = 128U;
constexpr uint32_t kM3NInitQuantFullLoadColsThreshold = 1024U;
constexpr uint32_t kM3NInitQuantFullLoadExpertThreshold = 16U;
constexpr uint64_t kM3NInitQuantFullLoadCountUbAddr = 0x6000;
constexpr uint64_t kM3NInitQuantFullLoadCursorUbAddr = 0x6100;
constexpr uint64_t kM3NRoutePackExpertBaseUbAddr = 0xE000;
constexpr uint64_t kM3NRoutePackLocalOrdinalUbAddr = 0xF000;
constexpr uint32_t kM3NRoutePackExpertUbCapacity = 1024U;
constexpr uint32_t kM3NCombineWorkerScratchBase = 72U * 16U;
constexpr uint32_t kM3NCombineWorkerScratchStride = 16U;
constexpr uint32_t kM3N8CombineCounterBase = kM3CounterBase + 80U;
constexpr uint32_t kM3N11SubtileRows = moe_new_dispatch_combine_a8w8::kM3N11SubtileRows;
constexpr uint32_t kM3N11SubtileCounterBase = moe_new_dispatch_combine_a8w8::kM3N11SubtileCounterBase;

using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;

template <typename T>
using GlobalNd = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

template <typename T, int kCols = kDefaultTileCols>
using VecTile = pto::Tile<pto::TileType::Vec, T, 1, kCols, pto::BLayout::RowMajor, -1, -1>;

using M2RouteHalfTile =
    pto::Tile<pto::TileType::Vec, half, 1, kM2RouteQuantTileCols, pto::BLayout::RowMajor, 1, pto::DYNAMIC>;
using M2RouteFloatTile =
    pto::Tile<pto::TileType::Vec, float, 1, kM2RouteQuantTileCols, pto::BLayout::RowMajor, 1, pto::DYNAMIC>;
using M2RouteQuantTile =
    pto::Tile<pto::TileType::Vec, int8_t, 1, kM2RouteQuantTileCols, pto::BLayout::RowMajor, 1, pto::DYNAMIC>;
using M2RouteRowStatTile = pto::Tile<pto::TileType::Vec, float, 1, 16, pto::BLayout::RowMajor, 1, 1>;
using M2RouteParamTile = pto::Tile<pto::TileType::Vec, float, 1, 8, pto::BLayout::RowMajor, 1, 8>;
using M2RouteScaleStoreTile = pto::Tile<pto::TileType::Vec, float, 1, 8, pto::BLayout::RowMajor, 1, 1>;
using M2EpilogueAccTile =
    pto::Tile<pto::TileType::Vec, int32_t, 1, kM2EpilogueTileCols, pto::BLayout::RowMajor, 1, pto::DYNAMIC>;
using M2EpilogueFloatTile =
    pto::Tile<pto::TileType::Vec, float, 1, kM2EpilogueTileCols, pto::BLayout::RowMajor, 1, pto::DYNAMIC>;
using M2EpilogueHalfTile =
    pto::Tile<pto::TileType::Vec, half, 1, kM2EpilogueTileCols, pto::BLayout::RowMajor, 1, pto::DYNAMIC>;

template <typename T>
AICORE inline void M2RawVecDup(uint64_t ubAddr, T value, uint32_t elemCount)
{
    if (elemCount == 0U) {
        return;
    }
    __ubuf__ T *ub = reinterpret_cast<__ubuf__ T *>(ubAddr);
    set_mask_count();
    pto::SetVectorCount(elemCount);
    vector_dup(ub, value, 0, 1, 1, 8, 8);
    set_mask_norm();
    pto::SetFullVecMaskByDType<T>();
    pipe_barrier(PIPE_V);
}

AICORE inline void M2RawZeroI8Tile(uint64_t ubAddr, uint32_t byteCount)
{
    M2RawVecDup<int16_t>(ubAddr, static_cast<int16_t>(0), (byteCount + 1U) / 2U);
}

template <typename T>
AICORE inline void M2RawVecLoad(uint64_t ubAddr, __gm__ T *src, uint32_t elemCount)
{
    if (elemCount == 0U) {
        return;
    }
    __ubuf__ T *dst = reinterpret_cast<__ubuf__ T *>(ubAddr);
    uint32_t byteCount = elemCount * static_cast<uint32_t>(sizeof(T));
    if constexpr (sizeof(T) == 1) {
        copy_gm_to_ubuf_align_b8(dst, src, 0, 1, byteCount, 0, 0, 0, 0);
    } else if constexpr (sizeof(T) == 2) {
        copy_gm_to_ubuf_align_b16(dst, src, 0, 1, byteCount, 0, 0, 0, 0);
    } else if constexpr (sizeof(T) == 4) {
        copy_gm_to_ubuf_align_b32(dst, src, 0, 1, byteCount, 0, 0, 0, 0);
    }
}

template <typename T>
AICORE inline void M2RawVecStore(__gm__ T *dst, uint64_t ubAddr, uint32_t elemCount)
{
    if (elemCount == 0U) {
        return;
    }
    __ubuf__ T *src = reinterpret_cast<__ubuf__ T *>(ubAddr);
    uint32_t byteCount = elemCount * static_cast<uint32_t>(sizeof(T));
    if constexpr (sizeof(T) == 1) {
        copy_ubuf_to_gm_align_b8(dst, src, 0, 1, byteCount, 0, 0, 0, 0);
    } else if constexpr (sizeof(T) == 2) {
        copy_ubuf_to_gm_align_b16(dst, src, 0, 1, byteCount, 0, 0, 0, 0);
    } else if constexpr (sizeof(T) == 4) {
        copy_ubuf_to_gm_align_b32(dst, src, 0, 1, byteCount, 0, 0, 0, 0);
    }
}

template <typename T>
AICORE inline T M2RawUbScalarLoad(uint64_t ubAddr)
{
    __ubuf__ T *src = reinterpret_cast<__ubuf__ T *>(ubAddr);
    return *src;
}

template <typename T>
AICORE inline void M2RawUbScalarStore(uint64_t ubAddr, T value)
{
    __ubuf__ T *dst = reinterpret_cast<__ubuf__ T *>(ubAddr);
    *dst = value;
}

AICORE inline int32_t M2RawUbLoadI32(uint64_t ubAddr, uint32_t index)
{
    __ubuf__ int32_t *src = reinterpret_cast<__ubuf__ int32_t *>(ubAddr);
    return src[index];
}

AICORE inline void M2RawUbStoreI32(uint64_t ubAddr, uint32_t index, int32_t value)
{
    __ubuf__ int32_t *dst = reinterpret_cast<__ubuf__ int32_t *>(ubAddr);
    dst[index] = value;
}

AICORE inline void M2RawHalfToFloat(uint64_t dstAddr, uint64_t srcAddr, uint32_t elemCount)
{
    if (elemCount == 0U) {
        return;
    }
    __ubuf__ float *dst = reinterpret_cast<__ubuf__ float *>(dstAddr);
    __ubuf__ half *src = reinterpret_cast<__ubuf__ half *>(srcAddr);
    constexpr uint32_t kElemsPerRepeat = 64U;
    uint32_t repeats = elemCount / kElemsPerRepeat;
    uint32_t remain = elemCount % kElemsPerRepeat;
    if (repeats > 0U) {
        vconv_f162f32(dst, src, static_cast<uint8_t>(repeats), 1, 1, 8, 4);
    }
    if (remain > 0U) {
        pto::SetContMaskByDType<float>(remain);
        vconv_f162f32(dst + repeats * kElemsPerRepeat, src + repeats * kElemsPerRepeat, 1, 1, 1, 8, 4);
        pto::SetFullVecMaskByDType<float>();
    }
}

AICORE inline void M2RawAbsFloat(uint64_t dstAddr, uint64_t srcAddr, uint32_t elemCount)
{
    if (elemCount == 0U) {
        return;
    }
    __ubuf__ float *dst = reinterpret_cast<__ubuf__ float *>(dstAddr);
    __ubuf__ float *src = reinterpret_cast<__ubuf__ float *>(srcAddr);
    set_mask_count();
    pto::SetVectorCount(elemCount);
    vabs(dst, src, 0, 1, 1, 8, 8);
    set_mask_norm();
    pto::SetFullVecMaskByDType<float>();
}

AICORE inline void M2RawRowMaxFloat(uint64_t dstAddr, uint64_t srcAddr, uint64_t tmpAddr, uint32_t elemCount)
{
    __ubuf__ float *dst = reinterpret_cast<__ubuf__ float *>(dstAddr);
    __ubuf__ float *src = reinterpret_cast<__ubuf__ float *>(srcAddr);
    __ubuf__ float *tmp = reinterpret_cast<__ubuf__ float *>(tmpAddr);
    pto::TRowReduceInstr<pto::TRowMaxOp<float>, float, M2RouteRowStatTile, M2RouteFloatTile, M2RouteFloatTile>(
        dst, src, tmp, static_cast<int>(elemCount), 1);
}

AICORE inline void M2RawUpdateRowMax(uint64_t rowMaxAddr, uint64_t chunkMaxAddr)
{
    pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
    float rowMax = M2RawUbScalarLoad<float>(rowMaxAddr);
    float chunkMax = M2RawUbScalarLoad<float>(chunkMaxAddr);
    M2RawUbScalarStore<float>(rowMaxAddr, rowMax > chunkMax ? rowMax : chunkMax);
    pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
}

AICORE inline void M2RawMulFloat(uint64_t ubAddr, float scale, uint32_t elemCount)
{
    if (elemCount == 0U) {
        return;
    }
    __ubuf__ float *data = reinterpret_cast<__ubuf__ float *>(ubAddr);
    set_mask_count();
    pto::SetVectorCount(elemCount);
    vmuls(data, data, scale, 0, 1, 1, 8, 8);
    set_mask_norm();
    pto::SetFullVecMaskByDType<float>();
}

AICORE inline void M2RawFloatToInt8(uint64_t dstAddr, uint64_t srcAddr, uint64_t s32ScratchAddr,
                                    uint64_t f16ScratchAddr, uint32_t elemCount)
{
    __ubuf__ int8_t *dst = reinterpret_cast<__ubuf__ int8_t *>(dstAddr);
    __ubuf__ float *src = reinterpret_cast<__ubuf__ float *>(srcAddr);
    __ubuf__ int32_t *tmpS32 = reinterpret_cast<__ubuf__ int32_t *>(s32ScratchAddr);
    __ubuf__ half *tmpF16 = reinterpret_cast<__ubuf__ half *>(f16ScratchAddr);
    constexpr uint32_t kFp32ElemsPerRepeat = 64U;
    uint32_t fp32Repeats = elemCount / kFp32ElemsPerRepeat;
    uint32_t fp32Remain = elemCount % kFp32ElemsPerRepeat;
    if (fp32Repeats > 0U) {
        vconv_f322s32r(tmpS32, src, static_cast<uint8_t>(fp32Repeats), 1, 1, 8, 8);
    }
    if (fp32Remain > 0U) {
        pto::SetContMaskByDType<float>(fp32Remain);
        vconv_f322s32r(tmpS32 + fp32Repeats * kFp32ElemsPerRepeat, src + fp32Repeats * kFp32ElemsPerRepeat, 1, 1, 1, 8,
                       8);
        pto::SetFullVecMaskByDType<float>();
    }
    pipe_barrier(PIPE_V);

    set_deqscale(static_cast<half>(1.0));
    set_deqscale(static_cast<half>(1.0));
    pipe_barrier(PIPE_V);
    if (fp32Repeats > 0U) {
        vconv_deq(tmpF16, tmpS32, static_cast<uint8_t>(fp32Repeats), 1, 1, 4, 8);
    }
    if (fp32Remain > 0U) {
        pto::SetContMaskByDType<half>(fp32Remain);
        vconv_deq(tmpF16 + fp32Repeats * kFp32ElemsPerRepeat, tmpS32 + fp32Repeats * kFp32ElemsPerRepeat, 1, 1, 1, 4,
                  8);
        pto::SetFullVecMaskByDType<half>();
    }
    pipe_barrier(PIPE_V);

    constexpr uint32_t kFp16ElemsPerRepeat = 128U;
    uint32_t fp16Repeats = elemCount / kFp16ElemsPerRepeat;
    uint32_t fp16Remain = elemCount % kFp16ElemsPerRepeat;
    if (fp16Repeats > 0U) {
        vconv_f162s8r(dst, tmpF16, static_cast<uint8_t>(fp16Repeats), 1, 1, 4, 8);
    }
    if (fp16Remain > 0U) {
        pto::SetContMaskByDType<half>(fp16Remain);
        vconv_f162s8r(dst + fp16Repeats * kFp16ElemsPerRepeat, tmpF16 + fp16Repeats * kFp16ElemsPerRepeat, 1, 1, 1, 4,
                      8);
        pto::SetFullVecMaskByDType<half>();
    }
}

AICORE inline void M2RawFloatToInt8(uint64_t dstAddr, uint64_t srcAddr, uint32_t elemCount)
{
    M2RawFloatToInt8(dstAddr, srcAddr, kM2RouteQuantS32ScratchOffset, kM2RouteQuantF16ScratchOffset, elemCount);
}

AICORE inline uint64_t Align64Device(uint64_t value)
{
    return ((value + 63) / 64) * 64;
}

AICORE inline uint64_t M2CeilDivDevice(uint64_t value, uint64_t divisor)
{
    if (divisor == 0) {
        return 0;
    }
    return (value + divisor - 1U) / divisor;
}

AICORE inline uint64_t AppendFieldDevice(uint64_t &offset, uint64_t bytes)
{
    offset = Align64Device(offset);
    uint64_t fieldOffset = offset;
    offset += bytes;
    return fieldOffset;
}

struct LocalWorkspaceView {
    GM_ADDR base;
    __gm__ int32_t *localTokenPerExpert;
    __gm__ int32_t *blockTokenPerExpert;
    __gm__ int32_t *blockPrefixPerExpert;
    __gm__ int32_t *cumsumPerExpert;
    __gm__ int32_t *dispatchOffset;
    __gm__ int32_t *prevSumBeforeRank;
    __gm__ int32_t *localSync;
    __gm__ float *floatScratch;
    __gm__ half *dispatchedA;
    __gm__ half *ptrDLocal;
};

struct LocalPeerWindowView {
    GM_ADDR base;
    __gm__ int32_t *peerTokenPerExpert;
    __gm__ int32_t *expandedRowIdx;
    __gm__ half *packedA;
    __gm__ half *ptrD;
    __gm__ int32_t *countReadySignal;
    __gm__ int32_t *combineDoneSignal;
};

AICORE inline WorkspaceLayout MakeWorkspaceLayout(DispatchCombineTileShape shape)
{
    const uint64_t i32 = 4;
    const uint64_t f32 = 4;
    const uint64_t f16 = 2;
    uint64_t expertNumPadded =
        ((static_cast<uint64_t>(shape.expertNum) + shape.metadataPad - 1) / shape.metadataPad) * shape.metadataPad;
    uint64_t aivBlocks = shape.aivBlocks == 0 ? 1 : shape.aivBlocks;
    uint64_t expandedRows = static_cast<uint64_t>(shape.m) * shape.topK;
    uint64_t offset = 0;

    WorkspaceLayout layout{};
    layout.localTokenPerExpert = AppendFieldDevice(offset, expertNumPadded * i32);
    layout.blockTokenPerExpert = AppendFieldDevice(offset, aivBlocks * expertNumPadded * i32);
    layout.blockPrefixPerExpert = AppendFieldDevice(offset, aivBlocks * expertNumPadded * i32);
    layout.cumsumPerExpert = AppendFieldDevice(offset, static_cast<uint64_t>(shape.ep) * expertNumPadded * i32);
    layout.dispatchOffset = AppendFieldDevice(offset, static_cast<uint64_t>(shape.expertPerRank) * i32);
    layout.prevSumBeforeRank = AppendFieldDevice(offset, static_cast<uint64_t>(shape.ep) * shape.expertPerRank * i32);
    uint64_t syncSlots = aivBlocks * (8 + expertNumPadded);
    syncSlots = syncSlots < 64 ? 64 : syncSlots;
    layout.localSync = AppendFieldDevice(offset, syncSlots * i32);
    layout.floatScratch = AppendFieldDevice(offset, aivBlocks * shape.tileCols * f32);
    layout.dispatchedA = AppendFieldDevice(offset, static_cast<uint64_t>(shape.maxOutputSize) * shape.k * f16);
    layout.ptrDLocal = AppendFieldDevice(offset, expandedRows * shape.k * f16);
    layout.totalBytes = Align64Device(offset);
    return layout;
}

AICORE inline PeerWindowLayout MakePeerWindowLayout(DispatchCombineTileShape shape)
{
    const uint64_t i32 = 4;
    const uint64_t f16 = 2;
    uint64_t expertNumPadded =
        ((static_cast<uint64_t>(shape.expertNum) + shape.metadataPad - 1) / shape.metadataPad) * shape.metadataPad;
    uint64_t expandedRows = static_cast<uint64_t>(shape.m) * shape.topK;
    uint64_t offset = 0;

    PeerWindowLayout layout{};
    layout.peerTokenPerExpert = AppendFieldDevice(offset, static_cast<uint64_t>(shape.ep) * expertNumPadded * i32);
    layout.expandedRowIdx = AppendFieldDevice(offset, expandedRows * i32);
    layout.packedA = AppendFieldDevice(offset, expandedRows * shape.k * f16);
    layout.ptrD = AppendFieldDevice(offset, expandedRows * shape.k * f16);
    layout.countReadySignal = AppendFieldDevice(offset, static_cast<uint64_t>(shape.ep) * i32);
    layout.combineDoneSignal = AppendFieldDevice(offset, static_cast<uint64_t>(shape.ep) * i32);
    layout.totalBytes = Align64Device(offset);
    return layout;
}

AICORE inline LocalWorkspaceView MakeLocalWorkspaceView(GM_ADDR workspaceBase, const WorkspaceLayout &layout)
{
    LocalWorkspaceView view{};
    view.base = workspaceBase;
    view.localTokenPerExpert = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.localTokenPerExpert);
    view.blockTokenPerExpert = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.blockTokenPerExpert);
    view.blockPrefixPerExpert = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.blockPrefixPerExpert);
    view.cumsumPerExpert = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.cumsumPerExpert);
    view.dispatchOffset = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.dispatchOffset);
    view.prevSumBeforeRank = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.prevSumBeforeRank);
    view.localSync = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.localSync);
    view.floatScratch = reinterpret_cast<__gm__ float *>(workspaceBase + layout.floatScratch);
    view.dispatchedA = reinterpret_cast<__gm__ half *>(workspaceBase + layout.dispatchedA);
    view.ptrDLocal = reinterpret_cast<__gm__ half *>(workspaceBase + layout.ptrDLocal);
    return view;
}

AICORE inline LocalPeerWindowView MakeLocalPeerWindowView(GM_ADDR peerWindowBase, const PeerWindowLayout &layout)
{
    LocalPeerWindowView view{};
    view.base = peerWindowBase;
    view.peerTokenPerExpert = reinterpret_cast<__gm__ int32_t *>(peerWindowBase + layout.peerTokenPerExpert);
    view.expandedRowIdx = reinterpret_cast<__gm__ int32_t *>(peerWindowBase + layout.expandedRowIdx);
    view.packedA = reinterpret_cast<__gm__ half *>(peerWindowBase + layout.packedA);
    view.ptrD = reinterpret_cast<__gm__ half *>(peerWindowBase + layout.ptrD);
    view.countReadySignal = reinterpret_cast<__gm__ int32_t *>(peerWindowBase + layout.countReadySignal);
    view.combineDoneSignal = reinterpret_cast<__gm__ int32_t *>(peerWindowBase + layout.combineDoneSignal);
    return view;
}

template <typename T>
AICORE inline __gm__ T *RemotePtr(__gm__ HcclDeviceContext *ctx, __gm__ T *localPtr, uint32_t peerRank)
{
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = reinterpret_cast<uint64_t>(localPtr) - localBase;
    return reinterpret_cast<__gm__ T *>(ctx->windowsIn[peerRank] + offset);
}

AICORE inline LocalPeerWindowView MakeRemotePeerWindowView(__gm__ HcclDeviceContext *ctx, GM_ADDR localPeerWindowBase,
                                                           uint32_t peerRank, const PeerWindowLayout &layout)
{
    GM_ADDR remoteBase = RemotePtr<uint8_t>(ctx, localPeerWindowBase, peerRank);
    return MakeLocalPeerWindowView(remoteBase, layout);
}

AICORE inline moe_new_dispatch_combine_a8w8::FieldLayout M2AppendFieldDevice(uint64_t &offset, uint64_t bytes)
{
    offset = Align64Device(offset);
    moe_new_dispatch_combine_a8w8::FieldLayout field{offset, bytes, 64};
    offset += bytes;
    return field;
}

AICORE inline uint64_t M2DTypeBytes(uint32_t dtype)
{
    if (dtype == 3U) {
        return 1;
    }
    if (dtype == 4U) {
        return 4;
    }
    return 2;
}

AICORE inline uint64_t M2ExpandedRows(moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    return static_cast<uint64_t>(shape.m) * shape.topK;
}

AICORE inline uint64_t M2LocalRows(moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    return static_cast<uint64_t>(shape.maxTokensPerExpert) * shape.expertPerRank;
}

AICORE inline uint64_t M2GlobalExpertNum(moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    return static_cast<uint64_t>(shape.rankNum) * shape.expertPerRank;
}

AICORE inline uint64_t M2TokenPerExpertMatrixRowStride(moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    constexpr uint64_t kI32PerCacheLine = 16;
    return ((M2GlobalExpertNum(shape) + kI32PerCacheLine - 1) / kI32PerCacheLine) * kI32PerCacheLine;
}

AICORE inline uint64_t M2DispatchPayloadRowBytes(moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    return Align64Device(static_cast<uint64_t>(shape.hiddenSize) * M2DTypeBytes(3U));
}

AICORE inline uint64_t M2ReturnPayloadRowBytes(moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    return Align64Device(static_cast<uint64_t>(shape.hiddenSize) * M2DTypeBytes(shape.dtypeOut));
}

AICORE inline uint64_t M2ReturnHiddenChunkCols(moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    return shape.gmmBlockN == 0 ? kM2GmmBaseN : shape.gmmBlockN;
}

AICORE inline uint64_t M2GmmTileTaskCapacity(moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    uint64_t rowTiles = M2CeilDivDevice(M2LocalRows(shape), kM2GmmBaseM);
    uint64_t w1Cols = static_cast<uint64_t>(shape.intermediateSize) * 2U;
    uint64_t gmm1NTiles = M2CeilDivDevice(w1Cols, kM2GmmBaseN);
    uint64_t gmm2NTiles = M2CeilDivDevice(shape.hiddenSize, kM2GmmBaseN);
    uint64_t nTiles = gmm1NTiles > gmm2NTiles ? gmm1NTiles : gmm2NTiles;
    return static_cast<uint64_t>(shape.expertPerRank) * rowTiles * nTiles;
}

AICORE inline uint64_t M2ReturnSegmentCapacity(moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    uint64_t rowTiles = M2CeilDivDevice(M2LocalRows(shape), kM2ReturnTileRows);
    uint64_t hiddenChunks = M2CeilDivDevice(shape.hiddenSize, M2ReturnHiddenChunkCols(shape));
    return static_cast<uint64_t>(shape.expertPerRank) * rowTiles * hiddenChunks * shape.rankNum;
}

AICORE inline moe_new_dispatch_combine_a8w8::WorkspaceLayout MakeM2WorkspaceLayoutDevice(
    moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    moe_new_dispatch_combine_a8w8::WorkspaceLayout layout{};
    uint64_t offset = 0;
    uint64_t expandedRows = M2ExpandedRows(shape);
    uint64_t localRows = M2LocalRows(shape);
    uint64_t globalExpertNum = M2GlobalExpertNum(shape);
    uint64_t tokenMatrixRowStride = M2TokenPerExpertMatrixRowStride(shape);
    uint64_t matrixCount = static_cast<uint64_t>(shape.rankNum) * tokenMatrixRowStride;
    uint64_t initQuantWorkerElems =
        static_cast<uint64_t>(moe_new_dispatch_combine_a8w8::kInitQuantMaxDispatchWorkers) * tokenMatrixRowStride;
    uint64_t rankExpertCount = static_cast<uint64_t>(shape.rankNum) * shape.expertPerRank;
    uint64_t dispatchRowBytes = M2DispatchPayloadRowBytes(shape);
    uint64_t returnRowBytes = M2ReturnPayloadRowBytes(shape);
    uint64_t w1Cols = static_cast<uint64_t>(shape.intermediateSize) * 2U;
    uint64_t syncGroupCap = static_cast<uint64_t>(shape.expertPerRank) + 1U;
    uint64_t gmmTaskCap = M2GmmTileTaskCapacity(shape);
    uint64_t subTileCap = M2ReturnSegmentCapacity(shape);

    layout.tokenPerExpertMatrix = M2AppendFieldDevice(offset, matrixCount * sizeof(int32_t));
    layout.blockTokenPerExpert = M2AppendFieldDevice(offset, globalExpertNum * sizeof(int32_t));
    layout.blockPrefixPerExpert = M2AppendFieldDevice(offset, globalExpertNum * sizeof(int32_t));
    layout.initQuantWorkerTokenPerExpert = M2AppendFieldDevice(offset, initQuantWorkerElems * sizeof(int32_t));
    layout.initQuantWorkerPrefixPerExpert = M2AppendFieldDevice(offset, initQuantWorkerElems * sizeof(int32_t));
    layout.expandedRowIdx = M2AppendFieldDevice(offset, expandedRows * sizeof(int32_t));
    layout.packedRowToRouteIndex = M2AppendFieldDevice(offset, expandedRows * sizeof(int32_t));
    layout.dispatchOffset = M2AppendFieldDevice(offset, shape.expertPerRank * sizeof(int32_t));
    layout.cumsumMM = M2AppendFieldDevice(offset, rankExpertCount * sizeof(int32_t));
    layout.preSumBeforeRank = M2AppendFieldDevice(offset, rankExpertCount * sizeof(int32_t));
    layout.expertTokenNums = M2AppendFieldDevice(offset, shape.expertPerRank * sizeof(int32_t));
    layout.tokenOwnerRankOffsets = M2AppendFieldDevice(offset, rankExpertCount * sizeof(int32_t));
    layout.dispatchedA = M2AppendFieldDevice(offset, expandedRows * returnRowBytes);
    layout.dispatchedScale = M2AppendFieldDevice(offset, expandedRows * sizeof(float));
    layout.gmm1InputInt8 = M2AppendFieldDevice(offset, localRows * dispatchRowBytes);
    layout.routingPerTokenScale = M2AppendFieldDevice(offset, localRows * sizeof(float));
    layout.gmm1WeightInt8 = M2AppendFieldDevice(offset, globalExpertNum * shape.hiddenSize * w1Cols);
    layout.scale1Uint64 = M2AppendFieldDevice(offset, w1Cols * sizeof(uint64_t));
    layout.gmm1AccInt32 = M2AppendFieldDevice(offset, localRows * w1Cols * sizeof(int32_t));
    layout.gmm1Out = M2AppendFieldDevice(offset, localRows * w1Cols * sizeof(float));
    layout.swigluOut = M2AppendFieldDevice(offset, localRows * shape.intermediateSize * sizeof(float));
    layout.gmm2InputInt8 = M2AppendFieldDevice(offset, localRows * Align64Device(shape.intermediateSize));
    layout.gmm2PerTokenScale = M2AppendFieldDevice(offset, localRows * sizeof(float));
    layout.gmm2WeightInt8 = M2AppendFieldDevice(offset, globalExpertNum * shape.intermediateSize * shape.hiddenSize);
    layout.scale2Uint64 = M2AppendFieldDevice(offset, shape.hiddenSize * sizeof(uint64_t));
    layout.gmm2AccInt32 = M2AppendFieldDevice(offset, localRows * shape.hiddenSize * sizeof(int32_t));
    layout.gmm2Out = M2AppendFieldDevice(offset, localRows * returnRowBytes);
    layout.returnSegmentStaging =
        M2AppendFieldDevice(offset, kM2ReturnTileRows * M2ReturnHiddenChunkCols(shape) * M2DTypeBytes(shape.dtypeOut));
    layout.readyCounters = M2AppendFieldDevice(offset, 16U * 64U);
    layout.dispatchGroupReady = M2AppendFieldDevice(offset, shape.expertPerRank * 64U);
    layout.gmm1SyncGroupReady = M2AppendFieldDevice(offset, syncGroupCap * 64U);
    layout.activationSyncGroupReady = M2AppendFieldDevice(offset, syncGroupCap * 64U);
    layout.gmm2GroupReady = M2AppendFieldDevice(offset, shape.expertPerRank * 64U);
    layout.stageStatus = M2AppendFieldDevice(offset, 16U * 64U);
    layout.swigluSyncGroups = M2AppendFieldDevice(offset, syncGroupCap * sizeof(int32_t));
    layout.dequantSum = M2AppendFieldDevice(offset, (syncGroupCap + 1U) * sizeof(int32_t));
    layout.swigluGroupDesc = M2AppendFieldDevice(offset, syncGroupCap * kM2SwigluGroupFields * sizeof(int32_t));
    layout.gmm1TileTaskPlan = M2AppendFieldDevice(offset, gmmTaskCap * kM2GmmTileTaskFields * sizeof(int32_t));
    layout.gmm2TileTaskPlan = M2AppendFieldDevice(offset, gmmTaskCap * kM2GmmTileTaskFields * sizeof(int32_t));
    layout.timeoutDump = M2AppendFieldDevice(offset, shape.expertPerRank * 64U);
    layout.subTileReturnPlan = M2AppendFieldDevice(offset, subTileCap * kM2ReturnPlanFields * sizeof(int32_t));
    layout.subTileOwnerSegments = M2AppendFieldDevice(offset, subTileCap * kM2OwnerSegmentFields * sizeof(int32_t));
    layout.subTileReady = M2AppendFieldDevice(offset, subTileCap * 64U);
    layout.timelineScratch = M2AppendFieldDevice(offset, 64U * 4U * sizeof(uint64_t));
    layout.totalBytes = Align64Device(offset);
    return layout;
}

AICORE inline moe_new_dispatch_combine_a8w8::PeerWindowLayout MakeM2PeerWindowLayoutDevice(
    moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    moe_new_dispatch_combine_a8w8::PeerWindowLayout layout{};
    uint64_t offset = 0;
    uint64_t expandedRows = M2ExpandedRows(shape);
    uint64_t tokenMatrixRowStride = M2TokenPerExpertMatrixRowStride(shape);
    uint64_t matrixCount = static_cast<uint64_t>(shape.rankNum) * tokenMatrixRowStride;
    layout.dispatchPayloadRowBytes = M2DispatchPayloadRowBytes(shape);
    layout.returnPayloadRowBytes = M2ReturnPayloadRowBytes(shape);
    layout.header = M2AppendFieldDevice(offset, moe_new_dispatch_combine_a8w8::kPeerWindowHeaderBytes);
    layout.tokenPerExpertMatrix = M2AppendFieldDevice(offset, matrixCount * sizeof(int32_t));
    layout.countReadySignal = M2AppendFieldDevice(offset, shape.rankNum * 64U);
    layout.dispatchPayload = M2AppendFieldDevice(offset, expandedRows * layout.dispatchPayloadRowBytes);
    layout.dispatchScale = M2AppendFieldDevice(offset, expandedRows * sizeof(float));
    layout.returnPayload = M2AppendFieldDevice(offset, expandedRows * layout.returnPayloadRowBytes);
    layout.combineDoneSignal = M2AppendFieldDevice(offset, shape.rankNum * 64U);
    layout.returnSegmentCounters =
        M2AppendFieldDevice(offset, static_cast<uint64_t>(shape.rankNum) * shape.expertPerRank * 64U);
    layout.debugCounters = M2AppendFieldDevice(offset, 80U * 64U);
    layout.timeline = M2AppendFieldDevice(offset, 64U * 4U * sizeof(uint64_t));
    layout.totalBytes = Align64Device(offset);
    return layout;
}

struct M2WorkspaceViewDevice {
    GM_ADDR base;
    __gm__ int32_t *tokenPerExpertMatrix;
    __gm__ int32_t *blockTokenPerExpert;
    __gm__ int32_t *blockPrefixPerExpert;
    __gm__ int32_t *initQuantWorkerTokenPerExpert;
    __gm__ int32_t *initQuantWorkerPrefixPerExpert;
    __gm__ int32_t *expandedRowIdx;
    __gm__ int32_t *packedRowToRouteIndex;
    __gm__ int32_t *dispatchOffset;
    __gm__ int32_t *cumsumMM;
    __gm__ int32_t *preSumBeforeRank;
    __gm__ int32_t *expertTokenNums;
    __gm__ int32_t *tokenOwnerRankOffsets;
    __gm__ int8_t *gmm1InputInt8;
    __gm__ float *routingPerTokenScale;
    __gm__ uint64_t *scale1Uint64;
    __gm__ int32_t *gmm1AccInt32;
    __gm__ float *gmm1Out;
    __gm__ float *swigluOut;
    __gm__ int8_t *gmm2InputInt8;
    __gm__ float *gmm2PerTokenScale;
    __gm__ uint64_t *scale2Uint64;
    __gm__ int32_t *gmm2AccInt32;
    __gm__ half *gmm2Out;
    __gm__ half *returnSegmentStaging;
    __gm__ int32_t *dispatchGroupReady;
    __gm__ int32_t *gmm1SyncGroupReady;
    __gm__ int32_t *activationSyncGroupReady;
    __gm__ int32_t *gmm2GroupReady;
    __gm__ int32_t *stageStatus;
    __gm__ int32_t *swigluSyncGroups;
    __gm__ int32_t *dequantSum;
    __gm__ int32_t *swigluGroupDesc;
    __gm__ int32_t *gmm1TileTaskPlan;
    __gm__ int32_t *gmm2TileTaskPlan;
    __gm__ int32_t *timeoutDump;
    __gm__ int32_t *subTileReturnPlan;
    __gm__ int32_t *subTileOwnerSegments;
    __gm__ int32_t *subTileReady;
    __gm__ uint64_t *timelineScratch;
};

struct M2PeerWindowViewDevice {
    GM_ADDR base;
    __gm__ int32_t *tokenPerExpertMatrix;
    __gm__ int32_t *countReadySignal;
    __gm__ int8_t *dispatchPayload;
    __gm__ float *dispatchScale;
    __gm__ half *returnPayload;
    __gm__ int32_t *combineDoneSignal;
    __gm__ int32_t *returnSegmentCounters;
    __gm__ int32_t *debugCounters;
    __gm__ uint64_t *timeline;
};

AICORE inline M2WorkspaceViewDevice MakeM2WorkspaceViewDevice(
    GM_ADDR workspaceBase, const moe_new_dispatch_combine_a8w8::WorkspaceLayout &layout)
{
    M2WorkspaceViewDevice view{};
    view.base = workspaceBase;
    view.tokenPerExpertMatrix = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.tokenPerExpertMatrix.offset);
    view.blockTokenPerExpert = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.blockTokenPerExpert.offset);
    view.blockPrefixPerExpert = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.blockPrefixPerExpert.offset);
    view.initQuantWorkerTokenPerExpert =
        reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.initQuantWorkerTokenPerExpert.offset);
    view.initQuantWorkerPrefixPerExpert =
        reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.initQuantWorkerPrefixPerExpert.offset);
    view.expandedRowIdx = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.expandedRowIdx.offset);
    view.packedRowToRouteIndex =
        reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.packedRowToRouteIndex.offset);
    view.dispatchOffset = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.dispatchOffset.offset);
    view.cumsumMM = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.cumsumMM.offset);
    view.preSumBeforeRank = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.preSumBeforeRank.offset);
    view.expertTokenNums = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.expertTokenNums.offset);
    view.tokenOwnerRankOffsets =
        reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.tokenOwnerRankOffsets.offset);
    view.gmm1InputInt8 = reinterpret_cast<__gm__ int8_t *>(workspaceBase + layout.gmm1InputInt8.offset);
    view.routingPerTokenScale = reinterpret_cast<__gm__ float *>(workspaceBase + layout.routingPerTokenScale.offset);
    view.scale1Uint64 = reinterpret_cast<__gm__ uint64_t *>(workspaceBase + layout.scale1Uint64.offset);
    view.gmm1AccInt32 = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.gmm1AccInt32.offset);
    view.gmm1Out = reinterpret_cast<__gm__ float *>(workspaceBase + layout.gmm1Out.offset);
    view.swigluOut = reinterpret_cast<__gm__ float *>(workspaceBase + layout.swigluOut.offset);
    view.gmm2InputInt8 = reinterpret_cast<__gm__ int8_t *>(workspaceBase + layout.gmm2InputInt8.offset);
    view.gmm2PerTokenScale = reinterpret_cast<__gm__ float *>(workspaceBase + layout.gmm2PerTokenScale.offset);
    view.scale2Uint64 = reinterpret_cast<__gm__ uint64_t *>(workspaceBase + layout.scale2Uint64.offset);
    view.gmm2AccInt32 = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.gmm2AccInt32.offset);
    view.gmm2Out = reinterpret_cast<__gm__ half *>(workspaceBase + layout.gmm2Out.offset);
    view.returnSegmentStaging = reinterpret_cast<__gm__ half *>(workspaceBase + layout.returnSegmentStaging.offset);
    view.dispatchGroupReady = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.dispatchGroupReady.offset);
    view.gmm1SyncGroupReady = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.gmm1SyncGroupReady.offset);
    view.activationSyncGroupReady =
        reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.activationSyncGroupReady.offset);
    view.gmm2GroupReady = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.gmm2GroupReady.offset);
    view.stageStatus = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.stageStatus.offset);
    view.swigluSyncGroups = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.swigluSyncGroups.offset);
    view.dequantSum = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.dequantSum.offset);
    view.swigluGroupDesc = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.swigluGroupDesc.offset);
    view.gmm1TileTaskPlan = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.gmm1TileTaskPlan.offset);
    view.gmm2TileTaskPlan = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.gmm2TileTaskPlan.offset);
    view.timeoutDump = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.timeoutDump.offset);
    view.subTileReturnPlan = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.subTileReturnPlan.offset);
    view.subTileOwnerSegments = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.subTileOwnerSegments.offset);
    view.subTileReady = reinterpret_cast<__gm__ int32_t *>(workspaceBase + layout.subTileReady.offset);
    view.timelineScratch = reinterpret_cast<__gm__ uint64_t *>(workspaceBase + layout.timelineScratch.offset);
    return view;
}

AICORE inline M2PeerWindowViewDevice MakeM2PeerWindowViewDevice(
    GM_ADDR peerWindowBase, const moe_new_dispatch_combine_a8w8::PeerWindowLayout &layout)
{
    M2PeerWindowViewDevice view{};
    view.base = peerWindowBase;
    view.tokenPerExpertMatrix = reinterpret_cast<__gm__ int32_t *>(peerWindowBase + layout.tokenPerExpertMatrix.offset);
    view.countReadySignal = reinterpret_cast<__gm__ int32_t *>(peerWindowBase + layout.countReadySignal.offset);
    view.dispatchPayload = reinterpret_cast<__gm__ int8_t *>(peerWindowBase + layout.dispatchPayload.offset);
    view.dispatchScale = reinterpret_cast<__gm__ float *>(peerWindowBase + layout.dispatchScale.offset);
    view.returnPayload = reinterpret_cast<__gm__ half *>(peerWindowBase + layout.returnPayload.offset);
    view.combineDoneSignal = reinterpret_cast<__gm__ int32_t *>(peerWindowBase + layout.combineDoneSignal.offset);
    view.returnSegmentCounters =
        reinterpret_cast<__gm__ int32_t *>(peerWindowBase + layout.returnSegmentCounters.offset);
    view.debugCounters = reinterpret_cast<__gm__ int32_t *>(peerWindowBase + layout.debugCounters.offset);
    view.timeline = reinterpret_cast<__gm__ uint64_t *>(peerWindowBase + layout.timeline.offset);
    return view;
}

AICORE inline M2PeerWindowViewDevice MakeM2RemotePeerWindowViewDevice(
    __gm__ HcclDeviceContext *ctx, GM_ADDR localPeerWindowBase, uint32_t peerRank,
    const moe_new_dispatch_combine_a8w8::PeerWindowLayout &layout)
{
    GM_ADDR remoteBase = RemotePtr<uint8_t>(ctx, localPeerWindowBase, peerRank);
    return MakeM2PeerWindowViewDevice(remoteBase, layout);
}

AICORE inline uint64_t M2TokenPerExpertIndex(moe_new_dispatch_combine_a8w8::ShapeConfig shape, uint32_t tokenOwnerRank,
                                             uint32_t expertOwnerRank, uint32_t localExpert)
{
    return static_cast<uint64_t>(tokenOwnerRank) * M2TokenPerExpertMatrixRowStride(shape) +
           static_cast<uint64_t>(expertOwnerRank) * shape.expertPerRank + localExpert;
}

AICORE inline int32_t M2LoadTokenPerExpert(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                           M2PeerWindowViewDevice localPeer, uint32_t tokenOwnerRank,
                                           uint32_t expertOwnerRank, uint32_t localExpert)
{
    return *(localPeer.tokenPerExpertMatrix +
             M2TokenPerExpertIndex(shape, tokenOwnerRank, expertOwnerRank, localExpert));
}

AICORE inline bool M2TokenIsActive(GM_ADDR xActiveMask, uint32_t token)
{
    if (xActiveMask == nullptr) {
        return true;
    }
    __gm__ uint8_t *mask = reinterpret_cast<__gm__ uint8_t *>(xActiveMask);
    return mask[token] != 0U;
}

AICORE inline int32_t M2EffectiveTokenOwnerRows(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                M2PeerWindowViewDevice localPeer, uint32_t tokenOwnerRank,
                                                uint32_t expertOwnerRank, uint32_t localExpert)
{
    int32_t cursor = 0;
    int32_t cap = static_cast<int32_t>(M2LocalRows(shape));
    for (uint32_t prevLocalExpert = 0; prevLocalExpert < shape.expertPerRank; ++prevLocalExpert) {
        for (uint32_t src = 0; src < shape.rankNum; ++src) {
            int32_t rows = M2LoadTokenPerExpert(shape, localPeer, src, expertOwnerRank, prevLocalExpert);
            int32_t effective = 0;
            if (cursor < cap && rows > 0) {
                int32_t available = cap - cursor;
                effective = rows < available ? rows : available;
            }
            if (prevLocalExpert == localExpert && src == tokenOwnerRank) {
                return effective;
            }
            cursor += effective;
        }
    }
    return 0;
}

AICORE inline uint32_t M2GlobalExpert(uint32_t expertOwnerRank, uint32_t localExpert,
                                      moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    return expertOwnerRank * shape.expertPerRank + localExpert;
}

AICORE inline uint32_t M2ExpertOwnerRank(uint32_t globalExpert, moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    return globalExpert / shape.expertPerRank;
}

AICORE inline uint32_t M2LocalExpert(uint32_t globalExpert, moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    return globalExpert % shape.expertPerRank;
}

AICORE inline float M2Abs(float value)
{
    return value < 0.0f ? -value : value;
}

AICORE inline float M2Sigmoid(float value)
{
    float clipped = value;
    if (clipped > 16.0f) {
        clipped = 16.0f;
    }
    if (clipped < -16.0f) {
        clipped = -16.0f;
    }
    return 0.5f + clipped / (2.0f * (1.0f + M2Abs(clipped)));
}

AICORE inline int8_t M2QuantizeToInt8(float value, float scale)
{
    if (scale <= 0.0f) {
        return 0;
    }
    float scaled = value / scale;
    int32_t quant = scaled >= 0.0f ? static_cast<int32_t>(scaled + 0.5f) : static_cast<int32_t>(scaled - 0.5f);
    if (quant > 127) {
        quant = 127;
    }
    if (quant < -127) {
        quant = -127;
    }
    return static_cast<int8_t>(quant);
}

AICORE inline float M2DecodeUint64Scale(uint64_t value)
{
    union {
        uint32_t u;
        float f;
    } bits{static_cast<uint32_t>(value & 0xFFFFFFFFULL)};
    return bits.f;
}

template <typename T>
AICORE inline GlobalNd<T> MakeGlobal1D(__gm__ T *ptr, int32_t elems)
{
    ShapeDyn shape(1, 1, 1, 1, elems);
    StrideDyn stride(elems, elems, elems, elems, 1);
    return GlobalNd<T>(ptr, shape, stride);
}

template <typename T>
AICORE inline GlobalNd<T> MakeGlobal2D(__gm__ T *ptr, int32_t rows, int32_t cols, int32_t rowStride)
{
    ShapeDyn shape(1, 1, 1, rows, cols);
    StrideDyn stride(rowStride * rows, rowStride * rows, rowStride * rows, rowStride, 1);
    return GlobalNd<T>(ptr, shape, stride);
}

AICORE inline pto::comm::Signal MakeSignal(__gm__ int32_t *ptr)
{
    return pto::comm::Signal(ptr);
}

AICORE inline void WaitStoreTileReusable()
{
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
}

AICORE inline void M2WaitRouteQuantStoreReusable()
{
    pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_MTE2>(EVENT_ID2, EVENT_ID2);
    pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_V>(EVENT_ID3, EVENT_ID3);
}

template <int kCols = kDefaultTileCols>
AICORE inline void CopyRowHalf(__gm__ half *dstBase, int32_t dstRowStride, int32_t dstRow, __gm__ half *srcBase,
                               int32_t srcRowStride, int32_t srcRow, int32_t rowLen)
{
    for (int32_t col = 0; col < rowLen; col += kCols) {
        int32_t cols = rowLen - col < kCols ? rowLen - col : kCols;
        VecTile<half, kCols> ping(1, cols);
        VecTile<half, kCols> pong(1, cols);
        TASSIGN(ping, kPingUbAddr);
        TASSIGN(pong, kPongUbAddr);
        VecTile<half, kCols> &tile = ((col / kCols) & 1) == 0 ? ping : pong;
        event_t event = ((col / kCols) & 1) == 0 ? EVENT_ID0 : EVENT_ID1;
        GlobalNd<half> src =
            MakeGlobal2D(srcBase + static_cast<int64_t>(srcRow) * srcRowStride + col, 1, cols, srcRowStride);
        GlobalNd<half> dst =
            MakeGlobal2D(dstBase + static_cast<int64_t>(dstRow) * dstRowStride + col, 1, cols, dstRowStride);
        TLOAD(tile, src);
        set_flag(PIPE_MTE2, PIPE_MTE3, event);
        wait_flag(PIPE_MTE2, PIPE_MTE3, event);
        TSTORE(dst, tile);
        WaitStoreTileReusable();
    }
}

template <typename T, int kCols = kDefaultTileCols>
AICORE inline void TGetRows(GlobalNd<T> &dst, GlobalNd<T> &remoteSrc)
{
    VecTile<T, kCols> ping(1, kCols);
    VecTile<T, kCols> pong(1, kCols);
    TASSIGN(ping, kPingUbAddr);
    TASSIGN(pong, kPongUbAddr);
    pto::comm::TGET(dst, remoteSrc, ping, pong);
}

template <typename T, int kCols = kDefaultTileCols>
AICORE inline void TPutRows(GlobalNd<T> &remoteDst, GlobalNd<T> &src)
{
    VecTile<T, kCols> ping(1, kCols);
    VecTile<T, kCols> pong(1, kCols);
    TASSIGN(ping, kPingUbAddr);
    TASSIGN(pong, kPongUbAddr);
    pto::comm::TPUT(remoteDst, src, ping, pong);
}

AICORE inline void NotifySignal(__gm__ int32_t *signal, int32_t value)
{
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    pto::comm::Signal sig = MakeSignal(signal);
    (void)value;
    pto::comm::TNOTIFY(sig, 1, pto::comm::NotifyOp::AtomicAdd);
}

AICORE inline void WaitSignal(__gm__ int32_t *signal, int32_t value)
{
    pto::comm::Signal sig = MakeSignal(signal);
    if (!pto::comm::TTEST(sig, value, pto::comm::WaitCmp::GE)) {
        pto::comm::TWAIT(sig, value, pto::comm::WaitCmp::GE);
    }
}

AICORE inline int32_t LoadScalarI32(__gm__ int32_t *ptr)
{
    return *ptr;
}

AICORE inline void StoreScalarI32(__gm__ int32_t *ptr, int32_t value)
{
    *ptr = value;
}

AICORE inline void PtoGmStoreDrain()
{
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

AICORE inline void PtoGmPublishRangeNoFinalDsb(__gm__ void *ptr, uint32_t bytes)
{
    uint64_t start = reinterpret_cast<uint64_t>(ptr) & ~static_cast<uint64_t>(63);
    uint64_t end = (reinterpret_cast<uint64_t>(ptr) + bytes + 63) & ~static_cast<uint64_t>(63);
    for (uint64_t addr = start; addr < end; addr += 64) {
        dcci(reinterpret_cast<__gm__ void *>(addr), SINGLE_CACHE_LINE);
    }
}

AICORE inline void PtoGmPublishBatchFinish()
{
    dsb(DSB_DDR);
}

AICORE inline void InvalidateGmCacheLines(__gm__ void *ptr, uint32_t bytes)
{
    PtoGmStoreDrain();
    PtoGmPublishRangeNoFinalDsb(ptr, bytes);
    PtoGmPublishBatchFinish();
}

AICORE inline uint32_t ExpertNumPaddedDevice(DispatchCombineTileShape shape)
{
    return ((shape.expertNum + shape.metadataPad - 1) / shape.metadataPad) * shape.metadataPad;
}

AICORE inline uint32_t TokenShardBegin(uint32_t totalTokens, uint32_t blockId, uint32_t blockNum)
{
    uint32_t base = totalTokens / blockNum;
    uint32_t rem = totalTokens % blockNum;
    return blockId * base + (blockId < rem ? blockId : rem);
}

AICORE inline uint32_t TokenShardEnd(uint32_t totalTokens, uint32_t blockId, uint32_t blockNum)
{
    return TokenShardBegin(totalTokens, blockId + 1, blockNum);
}

AICORE inline uint32_t TokenShardBlockForToken(uint32_t totalTokens, uint32_t token, uint32_t blockNum)
{
    uint32_t base = totalTokens / blockNum;
    uint32_t rem = totalTokens % blockNum;
    uint32_t largeShardTokens = (base + 1) * rem;
    if (token < largeShardTokens) {
        return token / (base + 1);
    }
    if (base == 0) {
        return token;
    }
    return rem + (token - largeShardTokens) / base;
}

AICORE inline bool M3NDispatchScratchFits(moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    return globalExpertNum > 0U;
}

AICORE inline bool M3NInitQuantFullLoadFits(moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    uint64_t routeCount = static_cast<uint64_t>(shape.m) * shape.topK;
    return globalExpertNum > 0U && routeCount > 0U && routeCount <= kM3NInitQuantFullLoadRouteThreshold &&
           shape.hiddenSize <= kM3NInitQuantFullLoadColsThreshold &&
           globalExpertNum <= kM3NInitQuantFullLoadExpertThreshold && M2LocalRows(shape) >= routeCount;
}

AICORE inline uint32_t M3NDispatchWorkerScratchStride(uint32_t globalExpertNum)
{
    constexpr uint32_t kI32PerCacheLine = 16U;
    return ((globalExpertNum + kI32PerCacheLine - 1U) / kI32PerCacheLine) * kI32PerCacheLine;
}

AICORE inline uint32_t M3NDispatchMinU32(uint32_t lhs, uint32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

AICORE inline uint32_t M3NDispatchMaxU32(uint32_t lhs, uint32_t rhs)
{
    return lhs > rhs ? lhs : rhs;
}

AICORE inline uint32_t M3NDispatchCeilDivU64(uint64_t value, uint32_t divisor)
{
    if (divisor == 0U || value == 0U) {
        return 0U;
    }
    uint64_t result = ((value - 1U) / divisor) + 1U;
    constexpr uint64_t kMaxU32 = static_cast<uint64_t>(0xffffffffU);
    return result > kMaxU32 ? 0xffffffffU : static_cast<uint32_t>(result);
}

AICORE inline uint32_t M3NDispatchMinTokensPerWorker(moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    uint32_t minTokens = 64U;
    if (shape.hiddenSize >= 4096U) {
        minTokens = 8U;
    } else if (shape.hiddenSize >= 1024U) {
        minTokens = 16U;
    } else if (shape.hiddenSize >= 512U) {
        minTokens = 32U;
    }

    uint32_t fanout = shape.topK == 0U ? 1U : shape.topK;
    if (fanout > 8U) {
        fanout = 8U;
    }
    if (fanout >= 4U) {
        uint32_t divisor = fanout / 2U;
        minTokens = M3NDispatchMaxU32(8U, minTokens / divisor);
    }
    return minTokens == 0U ? 1U : minTokens;
}

AICORE inline uint32_t M3NDispatchPrefixWorkerCap(uint32_t globalExpertNum)
{
    constexpr uint32_t kMaxWorkerExpertPrefixOps = 4096U;
    if (globalExpertNum == 0U) {
        return 1U;
    }
    uint32_t cap = kMaxWorkerExpertPrefixOps / globalExpertNum;
    if (cap == 0U) {
        cap = 1U;
    }
    return cap > kM3NDispatchMaxWorkers ? kM3NDispatchMaxWorkers : cap;
}

AICORE inline uint32_t M3NDispatchTargetWorkerCount(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                    uint32_t logicalAivCount, uint32_t laneSlotLimit)
{
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    if (logicalAivCount == 0U || globalExpertNum == 0U || shape.m == 0U || shape.topK == 0U ||
        !M3NDispatchScratchFits(shape)) {
        return 1U;
    }

    uint32_t availableWorkers = M3NDispatchMinU32(logicalAivCount, laneSlotLimit);
    availableWorkers = M3NDispatchMinU32(availableWorkers, kM3NDispatchMaxWorkers);
    availableWorkers = M3NDispatchMinU32(availableWorkers, shape.m);
    if (availableWorkers == 0U) {
        return 1U;
    }

    constexpr uint32_t kMinRoutesPerWorker = 64U;
    uint64_t routeCount = static_cast<uint64_t>(shape.m) * shape.topK;
    uint32_t tokenUsefulWorkers = M3NDispatchCeilDivU64(shape.m, M3NDispatchMinTokensPerWorker(shape));
    uint32_t routeUsefulWorkers = M3NDispatchCeilDivU64(routeCount, kMinRoutesPerWorker);
    uint32_t usefulWorkers = M3NDispatchMaxU32(tokenUsefulWorkers, routeUsefulWorkers);
    usefulWorkers = M3NDispatchMinU32(usefulWorkers, M3NDispatchPrefixWorkerCap(globalExpertNum));
    usefulWorkers = M3NDispatchMinU32(usefulWorkers, availableWorkers);
    return usefulWorkers == 0U ? 1U : usefulWorkers;
}

AICORE inline uint32_t M3NDispatchWorkerCount(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                              uint32_t logicalAivCount)
{
    return M3NDispatchTargetWorkerCount(shape, logicalAivCount, kM3NDispatchMaxWorkers);
}

AICORE inline void SoftSyncAiv(__gm__ int32_t *gmWorkspace, uint32_t blockNum)
{
    pto::Tile<pto::TileType::Vec, int32_t, 1, pto::SYNCALL_SOFT_SLOT_INT32, pto::BLayout::RowMajor, -1, -1> syncTile(
        1, pto::SYNCALL_SOFT_SLOT_INT32);
#ifndef __PTO_AUTO__
    syncTile.data() = reinterpret_cast<__ubuf__ int32_t *>(kSoftSyncUbAddr);
#endif
    GlobalNd<int32_t> syncGlobal =
        MakeGlobal1D(gmWorkspace, static_cast<int32_t>(blockNum * pto::SYNCALL_SOFT_SLOT_INT32));
    pto::SYNCALL<pto::SyncAllMode::Soft>(syncGlobal, syncTile, static_cast<int32_t>(blockNum));
}

AICORE inline __gm__ int32_t *PackCursorBase(LocalWorkspaceView workspaceView, uint32_t blockNum)
{
    return workspaceView.localSync + blockNum * pto::SYNCALL_SOFT_SLOT_INT32;
}

AICORE inline void ClearDispatchState(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                      LocalPeerWindowView localPeer, uint32_t blockId, uint32_t blockNum)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    uint32_t expandedRows = shape.m * shape.topK;
    for (uint32_t idx = blockId; idx < expertNumPadded; idx += blockNum) {
        StoreScalarI32(workspaceView.localTokenPerExpert + idx, 0);
    }
    for (uint32_t idx = blockId; idx < blockNum * expertNumPadded; idx += blockNum) {
        StoreScalarI32(workspaceView.blockTokenPerExpert + idx, 0);
        StoreScalarI32(workspaceView.blockPrefixPerExpert + idx, 0);
    }
    for (uint32_t idx = blockId; idx < shape.ep * expertNumPadded; idx += blockNum) {
        StoreScalarI32(workspaceView.cumsumPerExpert + idx, 0);
    }
    for (uint32_t idx = blockId; idx < shape.expertPerRank; idx += blockNum) {
        StoreScalarI32(workspaceView.dispatchOffset + idx, 0);
    }
    for (uint32_t idx = blockId; idx < shape.ep * shape.expertPerRank; idx += blockNum) {
        StoreScalarI32(workspaceView.prevSumBeforeRank + idx, 0);
    }
    (void)expandedRows;
    (void)localPeer;
}

AICORE inline void InitPackCursors(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView, uint32_t blockId,
                                   uint32_t blockNum)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    InvalidateGmCacheLines(workspaceView.blockPrefixPerExpert,
                           static_cast<uint32_t>(blockNum * expertNumPadded * sizeof(int32_t)));
    __gm__ int32_t *cursorBase = PackCursorBase(workspaceView, blockNum) + blockId * expertNumPadded;
    for (uint32_t expert = 0; expert < expertNumPadded; ++expert) {
        int32_t prefix = LoadScalarI32(workspaceView.blockPrefixPerExpert +
                                       static_cast<uint64_t>(blockId) * expertNumPadded + expert);
        StoreScalarI32(cursorBase + expert, prefix);
    }
}

AICORE inline int32_t PackedExpertOffset(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                         uint32_t myRank, uint32_t expert)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    return LoadScalarI32(workspaceView.cumsumPerExpert + static_cast<uint64_t>(myRank) * expertNumPadded + expert);
}

AICORE inline void PackLocalRowsToWindow(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                         LocalPeerWindowView localPeer, GM_ADDR inputA, GM_ADDR expertIdx,
                                         uint32_t myRank, uint32_t blockId, uint32_t blockNum)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    __gm__ int32_t *cursorBase = PackCursorBase(workspaceView, blockNum) + blockId * expertNumPadded;
    __gm__ int32_t *expertIds = reinterpret_cast<__gm__ int32_t *>(expertIdx);
    __gm__ half *input = reinterpret_cast<__gm__ half *>(inputA);
    InvalidateGmCacheLines(workspaceView.cumsumPerExpert + static_cast<uint64_t>(myRank) * expertNumPadded,
                           static_cast<uint32_t>(expertNumPadded * sizeof(int32_t)));
    uint32_t tokenBegin = TokenShardBegin(shape.m, blockId, blockNum);
    uint32_t tokenEnd = TokenShardEnd(shape.m, blockId, blockNum);
    for (uint32_t token = tokenBegin; token < tokenEnd; ++token) {
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            uint32_t routeIndex = token * shape.topK + slot;
            int32_t expert = LoadScalarI32(expertIds + routeIndex);
            if (expert < 0 || static_cast<uint32_t>(expert) >= shape.expertNum) {
                continue;
            }
            uint32_t expertId = static_cast<uint32_t>(expert);
            int32_t cursor = LoadScalarI32(cursorBase + expertId);
            StoreScalarI32(cursorBase + expertId, cursor + 1);
            int32_t packedRow = PackedExpertOffset(shape, workspaceView, myRank, expertId) + cursor;
            CopyRowHalf(localPeer.packedA, static_cast<int32_t>(shape.k), packedRow, input,
                        static_cast<int32_t>(shape.k), static_cast<int32_t>(token), static_cast<int32_t>(shape.k));
        }
    }
}

AICORE inline void CountLocalRoutes(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                    LocalPeerWindowView localPeer, GM_ADDR expertIdx, uint32_t blockId,
                                    uint32_t blockNum)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    __gm__ int32_t *expertIds = reinterpret_cast<__gm__ int32_t *>(expertIdx);
    uint32_t tokenBegin = TokenShardBegin(shape.m, blockId, blockNum);
    uint32_t tokenEnd = TokenShardEnd(shape.m, blockId, blockNum);
    __gm__ int32_t *blockCounts = workspaceView.blockTokenPerExpert + blockId * expertNumPadded;
    for (uint32_t token = tokenBegin; token < tokenEnd; ++token) {
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            uint32_t routeIndex = token * shape.topK + slot;
            int32_t expert = LoadScalarI32(expertIds + routeIndex);
            if (expert < 0 || static_cast<uint32_t>(expert) >= shape.expertNum) {
                continue;
            }
            __gm__ int32_t *count = blockCounts + static_cast<uint32_t>(expert);
            StoreScalarI32(count, LoadScalarI32(count) + 1);
        }
    }
    InvalidateGmCacheLines(blockCounts, static_cast<uint32_t>(expertNumPadded * sizeof(int32_t)));
}

AICORE inline void RebuildExpandedRowIdx(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                         LocalPeerWindowView localPeer, GM_ADDR expertIdx, uint32_t myRank,
                                         uint32_t blockId, uint32_t blockNum)
{
    constexpr uint32_t kI32PerCacheLine = 16;
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    uint32_t routeCount = shape.m * shape.topK;
    uint32_t lineCount = (routeCount + kI32PerCacheLine - 1) / kI32PerCacheLine;
    __gm__ int32_t *expertIds = reinterpret_cast<__gm__ int32_t *>(expertIdx);
    InvalidateGmCacheLines(workspaceView.blockPrefixPerExpert,
                           static_cast<uint32_t>(blockNum * expertNumPadded * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.cumsumPerExpert + static_cast<uint64_t>(myRank) * expertNumPadded,
                           static_cast<uint32_t>(expertNumPadded * sizeof(int32_t)));
    for (uint32_t line = blockId; line < lineCount; line += blockNum) {
        uint32_t begin = line * kI32PerCacheLine;
        uint32_t end = begin + kI32PerCacheLine;
        if (end > routeCount) {
            end = routeCount;
        }
        for (uint32_t routeIndex = begin; routeIndex < end; ++routeIndex) {
            int32_t expert = LoadScalarI32(expertIds + routeIndex);
            if (expert < 0 || static_cast<uint32_t>(expert) >= shape.expertNum) {
                StoreScalarI32(localPeer.expandedRowIdx + routeIndex, -1);
                continue;
            }
            uint32_t expertId = static_cast<uint32_t>(expert);
            uint32_t token = routeIndex / shape.topK;
            uint32_t sourceBlock = TokenShardBlockForToken(shape.m, token, blockNum);
            uint32_t localBegin = TokenShardBegin(shape.m, sourceBlock, blockNum) * shape.topK;
            int32_t localOrdinal = 0;
            for (uint32_t prev = localBegin; prev < routeIndex; ++prev) {
                int32_t prevExpert = LoadScalarI32(expertIds + prev);
                if (prevExpert == expert) {
                    ++localOrdinal;
                }
            }
            int32_t blockPrefix = LoadScalarI32(workspaceView.blockPrefixPerExpert +
                                                static_cast<uint64_t>(sourceBlock) * expertNumPadded + expertId);
            int32_t packedRow = PackedExpertOffset(shape, workspaceView, myRank, expertId) + blockPrefix + localOrdinal;
            StoreScalarI32(localPeer.expandedRowIdx + routeIndex, packedRow);
        }
        InvalidateGmCacheLines(localPeer.expandedRowIdx + begin,
                               static_cast<uint32_t>((end - begin) * sizeof(int32_t)));
    }
}

AICORE inline void BuildBlockPrefixAndLocalCounts(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                                  uint32_t blockId, uint32_t blockNum)
{
    if (blockId != 0) {
        return;
    }
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    InvalidateGmCacheLines(workspaceView.blockTokenPerExpert,
                           static_cast<uint32_t>(blockNum * expertNumPadded * sizeof(int32_t)));
    for (uint32_t expert = 0; expert < expertNumPadded; ++expert) {
        int32_t sum = 0;
        for (uint32_t block = 0; block < blockNum; ++block) {
            uint32_t idx = block * expertNumPadded + expert;
            StoreScalarI32(workspaceView.blockPrefixPerExpert + idx, sum);
            if (expert < shape.expertNum) {
                sum += LoadScalarI32(workspaceView.blockTokenPerExpert + idx);
            }
        }
        StoreScalarI32(workspaceView.localTokenPerExpert + expert, expert < shape.expertNum ? sum : 0);
    }
    InvalidateGmCacheLines(workspaceView.blockPrefixPerExpert,
                           static_cast<uint32_t>(blockNum * expertNumPadded * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.localTokenPerExpert, static_cast<uint32_t>(expertNumPadded * sizeof(int32_t)));
}

AICORE inline void BuildPackedExpertOffset(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                           uint32_t myRank, uint32_t blockId)
{
    if (blockId != 0) {
        return;
    }
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    InvalidateGmCacheLines(workspaceView.localTokenPerExpert, static_cast<uint32_t>(expertNumPadded * sizeof(int32_t)));
    int32_t sum = 0;
    for (uint32_t expert = 0; expert < expertNumPadded; ++expert) {
        int32_t count = LoadScalarI32(workspaceView.localTokenPerExpert + expert);
        StoreScalarI32(workspaceView.cumsumPerExpert + static_cast<uint64_t>(myRank) * expertNumPadded + expert, sum);
        if (expert < shape.expertNum) {
            sum += count;
        }
    }
    InvalidateGmCacheLines(workspaceView.cumsumPerExpert + static_cast<uint64_t>(myRank) * expertNumPadded,
                           static_cast<uint32_t>(expertNumPadded * sizeof(int32_t)));
}

AICORE inline void PublishCountRows(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                    LocalPeerWindowView localPeer, __gm__ HcclDeviceContext *ctx, GM_ADDR peerWindow,
                                    uint32_t myRank, uint32_t blockId, uint32_t blockNum,
                                    const PeerWindowLayout &peerWindowLayout)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    for (uint32_t dst = blockId; dst < shape.ep; dst += blockNum) {
        LocalPeerWindowView remotePeer = MakeRemotePeerWindowView(ctx, peerWindow, dst, peerWindowLayout);
        InvalidateGmCacheLines(workspaceView.localTokenPerExpert,
                               static_cast<uint32_t>(expertNumPadded * sizeof(int32_t)));
        GlobalNd<int32_t> localCount =
            MakeGlobal2D(workspaceView.localTokenPerExpert, 1, static_cast<int32_t>(expertNumPadded),
                         static_cast<int32_t>(expertNumPadded));
        GlobalNd<int32_t> remoteCount =
            MakeGlobal2D(remotePeer.peerTokenPerExpert + static_cast<uint64_t>(myRank) * expertNumPadded, 1,
                         static_cast<int32_t>(expertNumPadded), static_cast<int32_t>(expertNumPadded));
        TPutRows<int32_t, kMetaTileCols>(remoteCount, localCount);
        NotifySignal(remotePeer.countReadySignal + myRank, static_cast<int32_t>(shape.signalValue));
    }
}

AICORE inline void WaitCountRows(DispatchCombineTileShape shape, LocalPeerWindowView localPeer, uint32_t blockId,
                                 uint32_t blockNum)
{
    for (uint32_t src = blockId; src < shape.ep; src += blockNum) {
        WaitSignal(localPeer.countReadySignal + src, static_cast<int32_t>(shape.signalValue));
    }
}

AICORE inline void BuildPrefixMetadata(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                       LocalPeerWindowView localPeer, uint32_t myRank, uint32_t blockId,
                                       uint32_t blockNum)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    InvalidateGmCacheLines(localPeer.peerTokenPerExpert,
                           static_cast<uint32_t>(shape.ep * expertNumPadded * sizeof(int32_t)));
    for (uint32_t src = blockId; src < shape.ep; src += blockNum) {
        int32_t sum = 0;
        for (uint32_t expert = 0; expert < expertNumPadded; ++expert) {
            sum += LoadScalarI32(localPeer.peerTokenPerExpert + static_cast<uint64_t>(src) * expertNumPadded + expert);
            StoreScalarI32(workspaceView.cumsumPerExpert + static_cast<uint64_t>(src) * expertNumPadded + expert, sum);
        }
        InvalidateGmCacheLines(workspaceView.cumsumPerExpert + static_cast<uint64_t>(src) * expertNumPadded,
                               static_cast<uint32_t>(expertNumPadded * sizeof(int32_t)));
    }

    if (blockId == 0) {
        int32_t dispatchCursor = 0;
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            uint32_t globalExpert = myRank * shape.expertPerRank + localExpert;
            StoreScalarI32(workspaceView.dispatchOffset + localExpert, dispatchCursor);
            int32_t beforeRank = 0;
            for (uint32_t src = 0; src < shape.ep; ++src) {
                StoreScalarI32(
                    workspaceView.prevSumBeforeRank + static_cast<uint64_t>(src) * shape.expertPerRank + localExpert,
                    beforeRank);
                int32_t rows = LoadScalarI32(localPeer.peerTokenPerExpert +
                                             static_cast<uint64_t>(src) * expertNumPadded + globalExpert);
                beforeRank += rows;
                dispatchCursor += rows;
            }
        }
        InvalidateGmCacheLines(workspaceView.dispatchOffset,
                               static_cast<uint32_t>(shape.expertPerRank * sizeof(int32_t)));
        InvalidateGmCacheLines(workspaceView.prevSumBeforeRank,
                               static_cast<uint32_t>(shape.ep * shape.expertPerRank * sizeof(int32_t)));
    }
}

AICORE inline void TGetRowsHalf(__gm__ half *dstBase, int32_t dstRowStride, int32_t dstRow, __gm__ half *remoteSrcBase,
                                int32_t srcRowStride, int32_t srcRow, int32_t rows, int32_t cols)
{
    if (rows <= 0 || cols <= 0) {
        return;
    }
    GlobalNd<half> dst = MakeGlobal2D(dstBase + static_cast<int64_t>(dstRow) * dstRowStride, rows, cols, dstRowStride);
    GlobalNd<half> src =
        MakeGlobal2D(remoteSrcBase + static_cast<int64_t>(srcRow) * srcRowStride, rows, cols, srcRowStride);
    TGetRows<half, kDefaultTileCols>(dst, src);
}

AICORE inline void TPutRowsHalf(__gm__ half *remoteDstBase, int32_t dstRowStride, int32_t dstRow, __gm__ half *srcBase,
                                int32_t srcRowStride, int32_t srcRow, int32_t rows, int32_t cols)
{
    if (rows <= 0 || cols <= 0) {
        return;
    }
    GlobalNd<half> dst =
        MakeGlobal2D(remoteDstBase + static_cast<int64_t>(dstRow) * dstRowStride, rows, cols, dstRowStride);
    GlobalNd<half> src = MakeGlobal2D(srcBase + static_cast<int64_t>(srcRow) * srcRowStride, rows, cols, srcRowStride);
    TPutRows<half, kDefaultTileCols>(dst, src);
}

AICORE inline void TPutContiguousHalf(__gm__ half *remoteDst, __gm__ half *src, int32_t elems)
{
    if (elems <= 0) {
        return;
    }
    for (int32_t offset = 0; offset < elems; offset += kDefaultTileCols) {
        int32_t cols = elems - offset;
        if (cols > kDefaultTileCols) {
            cols = kDefaultTileCols;
        }
        ShapeDyn shape(1, 1, 1, 1, cols);
        StrideDyn stride(cols, cols, cols, cols, 1);
        GlobalNd<half> dst(remoteDst + offset, shape, stride);
        GlobalNd<half> srcGlobal(src + offset, shape, stride);
        VecTile<half, kDefaultTileCols> tile(1, cols);
        TASSIGN(tile, kPingUbAddr);
        pto::comm::TPUT(dst, srcGlobal, tile);
        pipe_barrier(PIPE_ALL);
    }
    dsb(DSB_DDR);
}

AICORE inline void TPutRowsHalfContiguous(__gm__ half *remoteDstBase, int32_t dstRowStride, int32_t dstRow,
                                          __gm__ half *srcBase, int32_t srcRowStride, int32_t srcRow, int32_t rows,
                                          int32_t cols)
{
    if (rows <= 0 || cols <= 0) {
        return;
    }
    for (int32_t row = 0; row < rows; ++row) {
        __gm__ half *dst = remoteDstBase + static_cast<int64_t>(dstRow + row) * dstRowStride;
        __gm__ half *src = srcBase + static_cast<int64_t>(srcRow + row) * srcRowStride;
        TPutContiguousHalf(dst, src, cols);
    }
}

AICORE inline void GatherLocalExpertPayload(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                            LocalPeerWindowView localPeer, __gm__ HcclDeviceContext *ctx,
                                            GM_ADDR peerWindow, uint32_t myRank, uint32_t blockId, uint32_t blockNum,
                                            const PeerWindowLayout &peerWindowLayout)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    InvalidateGmCacheLines(localPeer.peerTokenPerExpert,
                           static_cast<uint32_t>(shape.ep * expertNumPadded * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.cumsumPerExpert,
                           static_cast<uint32_t>(shape.ep * expertNumPadded * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.dispatchOffset, static_cast<uint32_t>(shape.expertPerRank * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.prevSumBeforeRank,
                           static_cast<uint32_t>(shape.ep * shape.expertPerRank * sizeof(int32_t)));
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        uint32_t globalExpert = myRank * shape.expertPerRank + localExpert;
        for (uint32_t src = blockId; src < shape.ep; src += blockNum) {
            int32_t rows = LoadScalarI32(localPeer.peerTokenPerExpert + static_cast<uint64_t>(src) * expertNumPadded +
                                         globalExpert);
            if (rows <= 0) {
                continue;
            }
            int32_t srcStart = globalExpert == 0 ?
                                   0 :
                                   LoadScalarI32(workspaceView.cumsumPerExpert +
                                                 static_cast<uint64_t>(src) * expertNumPadded + globalExpert - 1);
            int32_t dstStart = LoadScalarI32(workspaceView.dispatchOffset + localExpert) +
                               LoadScalarI32(workspaceView.prevSumBeforeRank +
                                             static_cast<uint64_t>(src) * shape.expertPerRank + localExpert);
            LocalPeerWindowView remotePeer = MakeRemotePeerWindowView(ctx, peerWindow, src, peerWindowLayout);
            TGetRowsHalf(workspaceView.dispatchedA, static_cast<int32_t>(shape.k), dstStart, remotePeer.packedA,
                         static_cast<int32_t>(shape.k), srcStart, rows, static_cast<int32_t>(shape.k));
        }
    }
}

AICORE inline void M2ClearDispatchState(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                        M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer)
{
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    uint32_t expandedRows = shape.m * shape.topK;
    uint32_t rankExpertCount = shape.rankNum * shape.expertPerRank;
    uint32_t tokenMatrixStorageCount = static_cast<uint32_t>(shape.rankNum * M2TokenPerExpertMatrixRowStride(shape));
    uint32_t localRows = static_cast<uint32_t>(M2LocalRows(shape));
    uint32_t syncGroupCap = shape.expertPerRank + 1U;
    uint32_t gmmTaskCap = static_cast<uint32_t>(M2GmmTileTaskCapacity(shape));
    uint32_t subTileCap = static_cast<uint32_t>(M2ReturnSegmentCapacity(shape));
    for (uint32_t idx = 0; idx < globalExpertNum; ++idx) {
        StoreScalarI32(workspaceView.blockTokenPerExpert + idx, 0);
        StoreScalarI32(workspaceView.blockPrefixPerExpert + idx, 0);
        StoreScalarI32(workspaceView.tokenOwnerRankOffsets + idx, 0);
    }
    for (uint32_t idx = 0; idx < tokenMatrixStorageCount; ++idx) {
        StoreScalarI32(workspaceView.tokenPerExpertMatrix + idx, 0);
    }
    for (uint32_t idx = 0; idx < expandedRows; ++idx) {
        StoreScalarI32(workspaceView.expandedRowIdx + idx, -1);
        StoreScalarI32(workspaceView.packedRowToRouteIndex + idx, -1);
    }
    for (uint32_t idx = 0; idx < rankExpertCount; ++idx) {
        StoreScalarI32(workspaceView.cumsumMM + idx, 0);
        StoreScalarI32(workspaceView.preSumBeforeRank + idx, 0);
    }
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        StoreScalarI32(workspaceView.expertTokenNums + localExpert, 0);
        StoreScalarI32(workspaceView.dispatchGroupReady + localExpert * 16U, 0);
        StoreScalarI32(workspaceView.gmm2GroupReady + localExpert * 16U, 0);
        for (uint32_t slot = 0; slot < 16U; ++slot) {
            StoreScalarI32(workspaceView.timeoutDump + localExpert * 16U + slot, 0);
        }
    }
    for (uint32_t idx = 0; idx < syncGroupCap; ++idx) {
        StoreScalarI32(workspaceView.swigluSyncGroups + idx, 0);
        StoreScalarI32(workspaceView.dequantSum + idx, 0);
        StoreScalarI32(workspaceView.gmm1SyncGroupReady + idx * 16U, 0);
        StoreScalarI32(workspaceView.activationSyncGroupReady + idx * 16U, 0);
        for (uint32_t field = 0; field < kM2SwigluGroupFields; ++field) {
            StoreScalarI32(workspaceView.swigluGroupDesc + idx * kM2SwigluGroupFields + field, 0);
        }
    }
    StoreScalarI32(workspaceView.dequantSum + syncGroupCap, 0);
    for (uint32_t idx = 0; idx < gmmTaskCap; ++idx) {
        for (uint32_t field = 0; field < kM2GmmTileTaskFields; ++field) {
            StoreScalarI32(workspaceView.gmm1TileTaskPlan + idx * kM2GmmTileTaskFields + field, 0);
            StoreScalarI32(workspaceView.gmm2TileTaskPlan + idx * kM2GmmTileTaskFields + field, 0);
        }
    }
    for (uint32_t row = 0; row < localRows; ++row) {
        workspaceView.routingPerTokenScale[row] = 1.0f;
    }
    // gmm1InputInt8 is produced by dispatch gather. Clearing the full row buffer here would serialize a large GM memset
    // into the front-reorder prepare path.
    // Peer-visible count matrix and ready signals are cleared by the host before
    // the cross-rank launch. Clearing them inside the kernel races with peer
    // ranks that may already have published their count rows.
    for (uint32_t idx = 0; idx < subTileCap; ++idx) {
        for (uint32_t field = 0; field < kM2ReturnPlanFields; ++field) {
            StoreScalarI32(workspaceView.subTileReturnPlan + idx * kM2ReturnPlanFields + field, 0);
        }
        for (uint32_t field = 0; field < kM2OwnerSegmentFields; ++field) {
            StoreScalarI32(workspaceView.subTileOwnerSegments + idx * kM2OwnerSegmentFields + field, 0);
        }
        StoreScalarI32(workspaceView.subTileReady + idx * 16U, 0);
    }
}

AICORE inline void M2ClearInitQuantState(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                         M2WorkspaceViewDevice workspaceView)
{
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    uint32_t rankExpertCount = shape.rankNum * shape.expertPerRank;
    uint32_t tokenMatrixStorageCount = static_cast<uint32_t>(shape.rankNum * M2TokenPerExpertMatrixRowStride(shape));
    uint32_t localRows = static_cast<uint32_t>(M2LocalRows(shape));
    for (uint32_t idx = 0; idx < globalExpertNum; ++idx) {
        StoreScalarI32(workspaceView.blockTokenPerExpert + idx, 0);
        StoreScalarI32(workspaceView.blockPrefixPerExpert + idx, 0);
        StoreScalarI32(workspaceView.tokenOwnerRankOffsets + idx, 0);
    }
    for (uint32_t idx = 0; idx < tokenMatrixStorageCount; ++idx) {
        StoreScalarI32(workspaceView.tokenPerExpertMatrix + idx, 0);
    }
    for (uint32_t idx = 0; idx < rankExpertCount; ++idx) {
        StoreScalarI32(workspaceView.cumsumMM + idx, 0);
        StoreScalarI32(workspaceView.preSumBeforeRank + idx, 0);
    }
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        StoreScalarI32(workspaceView.expertTokenNums + localExpert, 0);
        StoreScalarI32(workspaceView.dispatchGroupReady + localExpert * 16U, 0);
    }
    for (uint32_t row = 0; row < localRows; ++row) {
        workspaceView.routingPerTokenScale[row] = 1.0f;
    }
}

AICORE inline void M2CountLocalRoutes(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                      M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                      GM_ADDR expertIdx, GM_ADDR xActiveMask, uint32_t myRank)
{
    __gm__ int32_t *expertIds = reinterpret_cast<__gm__ int32_t *>(expertIdx);
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    for (uint32_t routeIndex = 0; routeIndex < shape.m * shape.topK; ++routeIndex) {
        uint32_t token = routeIndex / shape.topK;
        if (!M2TokenIsActive(xActiveMask, token)) {
            continue;
        }
        int32_t expert = LoadScalarI32(expertIds + routeIndex);
        if (expert < 0 || static_cast<uint32_t>(expert) >= globalExpertNum) {
            continue;
        }
        uint32_t globalExpert = static_cast<uint32_t>(expert);
        StoreScalarI32(workspaceView.blockTokenPerExpert + globalExpert,
                       LoadScalarI32(workspaceView.blockTokenPerExpert + globalExpert) + 1);
    }
    int32_t running = 0;
    for (uint32_t globalExpert = 0; globalExpert < globalExpertNum; ++globalExpert) {
        StoreScalarI32(workspaceView.blockPrefixPerExpert + globalExpert, running);
        uint32_t expertOwner = M2ExpertOwnerRank(globalExpert, shape);
        uint32_t localExpert = M2LocalExpert(globalExpert, shape);
        int32_t rows = LoadScalarI32(workspaceView.blockTokenPerExpert + globalExpert);
        StoreScalarI32(localPeer.tokenPerExpertMatrix + M2TokenPerExpertIndex(shape, myRank, expertOwner, localExpert),
                       rows);
        StoreScalarI32(
            workspaceView.tokenPerExpertMatrix + M2TokenPerExpertIndex(shape, myRank, expertOwner, localExpert), rows);
        running += rows;
    }
    InvalidateGmCacheLines(workspaceView.blockTokenPerExpert, static_cast<uint32_t>(globalExpertNum * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.blockPrefixPerExpert,
                           static_cast<uint32_t>(globalExpertNum * sizeof(int32_t)));
    InvalidateGmCacheLines(
        workspaceView.tokenPerExpertMatrix,
        static_cast<uint32_t>(shape.rankNum * M2TokenPerExpertMatrixRowStride(shape) * sizeof(int32_t)));
    InvalidateGmCacheLines(
        localPeer.tokenPerExpertMatrix,
        static_cast<uint32_t>(shape.rankNum * M2TokenPerExpertMatrixRowStride(shape) * sizeof(int32_t)));
}

AICORE inline __gm__ int32_t *M3NDispatchWorkerCountScratch(M2WorkspaceViewDevice workspaceView)
{
    return workspaceView.initQuantWorkerTokenPerExpert;
}

AICORE inline __gm__ int32_t *M3NDispatchWorkerPrefixScratch(M2WorkspaceViewDevice workspaceView)
{
    return workspaceView.initQuantWorkerPrefixPerExpert;
}

AICORE inline int32_t M3NDispatchLoadWorkerExpertCount(M2WorkspaceViewDevice workspaceView, uint32_t workerId,
                                                       uint32_t globalExpertNum, uint32_t globalExpert)
{
    uint32_t workerStride = M3NDispatchWorkerScratchStride(globalExpertNum);
    return LoadScalarI32(M3NDispatchWorkerCountScratch(workspaceView) + static_cast<uint64_t>(workerId) * workerStride +
                         globalExpert);
}

AICORE inline int32_t M3NDispatchLoadWorkerExpertPrefix(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                        M2WorkspaceViewDevice workspaceView, uint32_t workerId,
                                                        uint32_t globalExpert)
{
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    uint32_t workerStride = M3NDispatchWorkerScratchStride(globalExpertNum);
    return LoadScalarI32(M3NDispatchWorkerPrefixScratch(workspaceView) +
                         static_cast<uint64_t>(workerId) * workerStride + globalExpert);
}

AICORE inline bool M3NRoutePackExpertCacheFits(uint32_t globalExpertNum)
{
    return globalExpertNum <= kM3NRoutePackExpertUbCapacity;
}

AICORE inline void M3NPrepareRoutePackExpertCache(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                  M2WorkspaceViewDevice workspaceView, uint32_t workerId,
                                                  uint32_t globalExpertNum)
{
    for (uint32_t globalExpert = 0; globalExpert < globalExpertNum; ++globalExpert) {
        int32_t expertBase = LoadScalarI32(workspaceView.blockPrefixPerExpert + globalExpert) +
                             M3NDispatchLoadWorkerExpertPrefix(shape, workspaceView, workerId, globalExpert);
        M2RawUbStoreI32(kM3NRoutePackExpertBaseUbAddr, globalExpert, expertBase);
        M2RawUbStoreI32(kM3NRoutePackLocalOrdinalUbAddr, globalExpert, 0);
    }
}

AICORE inline int32_t M3NNextPackedRowFromExpertCache(uint32_t globalExpert)
{
    int32_t localOrdinal = M2RawUbLoadI32(kM3NRoutePackLocalOrdinalUbAddr, globalExpert);
    M2RawUbStoreI32(kM3NRoutePackLocalOrdinalUbAddr, globalExpert, localOrdinal + 1);
    return M2RawUbLoadI32(kM3NRoutePackExpertBaseUbAddr, globalExpert) + localOrdinal;
}

AICORE inline int32_t M3NNextPackedRowFromGmCursor(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                   M2WorkspaceViewDevice workspaceView,
                                                   __gm__ int32_t *localOrdinalCursor, uint32_t workerId,
                                                   uint32_t globalExpert)
{
    int32_t localOrdinal = LoadScalarI32(localOrdinalCursor + globalExpert);
    StoreScalarI32(localOrdinalCursor + globalExpert, localOrdinal + 1);
    return LoadScalarI32(workspaceView.blockPrefixPerExpert + globalExpert) +
           M3NDispatchLoadWorkerExpertPrefix(shape, workspaceView, workerId, globalExpert) + localOrdinal;
}

AICORE inline int32_t M3NSumWorkerExpertCount(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                              M2WorkspaceViewDevice workspaceView, uint32_t workerCount,
                                              uint32_t globalExpert)
{
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    int32_t rows = 0;
    for (uint32_t worker = 0; worker < workerCount; ++worker) {
        rows += M3NDispatchLoadWorkerExpertCount(workspaceView, worker, globalExpertNum, globalExpert);
    }
    return rows;
}

AICORE inline int32_t M3NGlobalExpertBaseRows(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                              M2WorkspaceViewDevice workspaceView, uint32_t workerCount,
                                              uint32_t targetGlobalExpert)
{
    int32_t running = 0;
    for (uint32_t globalExpert = 0; globalExpert < targetGlobalExpert; ++globalExpert) {
        running += M3NSumWorkerExpertCount(shape, workspaceView, workerCount, globalExpert);
    }
    return running;
}

AICORE inline void M3NClearDispatchWorkerScratch(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                 M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                                 uint32_t workerId, uint32_t workerCount)
{
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    uint32_t workerStride = M3NDispatchWorkerScratchStride(globalExpertNum);
    __gm__ int32_t *workerCounts =
        M3NDispatchWorkerCountScratch(workspaceView) + static_cast<uint64_t>(workerId) * workerStride;
    for (uint32_t idx = 0; idx < workerStride; ++idx) {
        StoreScalarI32(workerCounts + idx, 0);
    }
    __gm__ int32_t *workerPrefixes =
        M3NDispatchWorkerPrefixScratch(workspaceView) + static_cast<uint64_t>(workerId) * workerStride;
    for (uint32_t idx = 0; idx < workerStride; ++idx) {
        StoreScalarI32(workerPrefixes + idx, 0);
    }
    if (workerId == 0U) {
        for (uint32_t idx = 0; idx < 16U; ++idx) {
            StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + idx, 0);
        }
        StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 3U, static_cast<int32_t>(workerCount));
        StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 7U, 1);
        StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 11U, static_cast<int32_t>(workerCount));
    }
}

AICORE inline void M3NCountLocalRoutesShard(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                            M2WorkspaceViewDevice workspaceView, GM_ADDR expertIdx, GM_ADDR xActiveMask,
                                            uint32_t workerId, uint32_t workerCount)
{
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    uint32_t workerStride = M3NDispatchWorkerScratchStride(globalExpertNum);
    __gm__ int32_t *expertIds = reinterpret_cast<__gm__ int32_t *>(expertIdx);
    __gm__ int32_t *workerCounts =
        M3NDispatchWorkerCountScratch(workspaceView) + static_cast<uint64_t>(workerId) * workerStride;
    uint32_t tokenBegin = TokenShardBegin(shape.m, workerId, workerCount);
    uint32_t tokenEnd = TokenShardEnd(shape.m, workerId, workerCount);
    for (uint32_t token = tokenBegin; token < tokenEnd; ++token) {
        if (!M2TokenIsActive(xActiveMask, token)) {
            continue;
        }
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            uint32_t routeIndex = token * shape.topK + slot;
            int32_t expert = LoadScalarI32(expertIds + routeIndex);
            if (expert < 0 || static_cast<uint32_t>(expert) >= globalExpertNum) {
                continue;
            }
            __gm__ int32_t *count = workerCounts + static_cast<uint32_t>(expert);
            StoreScalarI32(count, LoadScalarI32(count) + 1);
        }
    }
    InvalidateGmCacheLines(workerCounts, static_cast<uint32_t>(workerStride * sizeof(int32_t)));
}

AICORE inline int32_t M3NCountActiveDispatchWorkers(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                    M2WorkspaceViewDevice workspaceView, uint32_t workerCount,
                                                    __gm__ int32_t *activeWorkerMaskOut)
{
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    int32_t activeWorkers = 0;
    int32_t activeWorkerMask = 0;
    for (uint32_t worker = 0; worker < workerCount; ++worker) {
        int32_t workerRows = 0;
        for (uint32_t globalExpert = 0; globalExpert < globalExpertNum; ++globalExpert) {
            int32_t rows = M3NDispatchLoadWorkerExpertCount(workspaceView, worker, globalExpertNum, globalExpert);
            workerRows += rows > 0 ? rows : 0;
        }
        if (workerRows > 0) {
            ++activeWorkers;
            if (worker < 31U && activeWorkerMask >= 0) {
                activeWorkerMask |= static_cast<int32_t>(1U << worker);
            } else {
                activeWorkerMask = -1;
            }
        }
    }
    StoreScalarI32(activeWorkerMaskOut, activeWorkerMask);
    return activeWorkers;
}

AICORE inline void M3NMergeLocalRouteCounts(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                            M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                            uint32_t myRank, uint32_t workerCount)
{
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    uint32_t workerStride = M3NDispatchWorkerScratchStride(globalExpertNum);
    InvalidateGmCacheLines(M3NDispatchWorkerCountScratch(workspaceView),
                           static_cast<uint32_t>(workerCount * workerStride * sizeof(int32_t)));
    __gm__ int32_t *workerPrefixes = M3NDispatchWorkerPrefixScratch(workspaceView);
    int32_t running = 0;
    for (uint32_t globalExpert = 0; globalExpert < globalExpertNum; ++globalExpert) {
        int32_t rows = 0;
        int32_t workerPrefix = 0;
        for (uint32_t worker = 0; worker < workerCount; ++worker) {
            StoreScalarI32(workerPrefixes + static_cast<uint64_t>(worker) * workerStride + globalExpert, workerPrefix);
            int32_t workerRows = M3NDispatchLoadWorkerExpertCount(workspaceView, worker, globalExpertNum, globalExpert);
            workerPrefix += workerRows;
            rows += workerRows;
        }
        StoreScalarI32(workspaceView.blockTokenPerExpert + globalExpert, rows);
        StoreScalarI32(workspaceView.blockPrefixPerExpert + globalExpert, running);
        uint32_t expertOwner = M2ExpertOwnerRank(globalExpert, shape);
        uint32_t localExpert = M2LocalExpert(globalExpert, shape);
        StoreScalarI32(localPeer.tokenPerExpertMatrix + M2TokenPerExpertIndex(shape, myRank, expertOwner, localExpert),
                       rows);
        StoreScalarI32(
            workspaceView.tokenPerExpertMatrix + M2TokenPerExpertIndex(shape, myRank, expertOwner, localExpert), rows);
        running += rows;
    }
    int32_t activeWorkers = M3NCountActiveDispatchWorkers(shape, workspaceView, workerCount,
                                                          localPeer.debugCounters + kM3NDispatchCounterBase + 10U);
    StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 0U, activeWorkers);
    StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 1U, static_cast<int32_t>(workerCount));
    StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 3U, static_cast<int32_t>(workerCount));
    StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 4U, running);
    StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 7U, 1);
    StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 8U, 1);
    StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 11U, static_cast<int32_t>(workerCount));
    InvalidateGmCacheLines(workspaceView.blockTokenPerExpert, static_cast<uint32_t>(globalExpertNum * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.blockPrefixPerExpert,
                           static_cast<uint32_t>(globalExpertNum * sizeof(int32_t)));
    InvalidateGmCacheLines(workerPrefixes, static_cast<uint32_t>(workerCount * workerStride * sizeof(int32_t)));
    InvalidateGmCacheLines(
        workspaceView.tokenPerExpertMatrix,
        static_cast<uint32_t>(shape.rankNum * M2TokenPerExpertMatrixRowStride(shape) * sizeof(int32_t)));
    InvalidateGmCacheLines(
        localPeer.tokenPerExpertMatrix,
        static_cast<uint32_t>(shape.rankNum * M2TokenPerExpertMatrixRowStride(shape) * sizeof(int32_t)));
}

AICORE inline void M3NFinalizeLocalRouteCountCounters(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                      M2WorkspaceViewDevice workspaceView,
                                                      M2PeerWindowViewDevice localPeer, uint32_t workerCount)
{
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    uint32_t workerStride = M3NDispatchWorkerScratchStride(globalExpertNum);
    InvalidateGmCacheLines(M3NDispatchWorkerCountScratch(workspaceView),
                           static_cast<uint32_t>(workerCount * workerStride * sizeof(int32_t)));
    int32_t totalRows = M3NGlobalExpertBaseRows(shape, workspaceView, workerCount, globalExpertNum);
    int32_t activeWorkers = M3NCountActiveDispatchWorkers(shape, workspaceView, workerCount,
                                                          localPeer.debugCounters + kM3NDispatchCounterBase + 10U);
    StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 0U, activeWorkers);
    StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 1U, static_cast<int32_t>(workerCount));
    StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 3U, static_cast<int32_t>(workerCount));
    StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 4U, totalRows);
    StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 7U, 1);
    StoreScalarI32(localPeer.debugCounters + kM3NDispatchCounterBase + 8U, 1);
}

AICORE inline void M2QuantizeRowToPeerPayload(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                              M2PeerWindowViewDevice localPeer, GM_ADDR inputA, uint32_t token,
                                              uint32_t packedRow, uint32_t rowBytes)
{
    __gm__ half *input = reinterpret_cast<__gm__ half *>(inputA);
    M2RawVecDup<float>(kM2RouteRowMaxTileOffset, 0.0f, 1U);

    for (uint32_t colBegin = 0; colBegin < shape.hiddenSize; colBegin += kM2RouteQuantTileCols) {
        uint32_t cols = shape.hiddenSize - colBegin;
        if (cols > kM2RouteQuantTileCols) {
            cols = kM2RouteQuantTileCols;
        }
        M2RawVecLoad<half>(kM2RouteHalfTileOffset, input + static_cast<uint64_t>(token) * shape.hiddenSize + colBegin,
                           cols);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        M2RawHalfToFloat(kM2RouteFloatTileOffset, kM2RouteHalfTileOffset, cols);
        pipe_barrier(PIPE_V);
        M2RawAbsFloat(kM2RouteAbsTileOffset, kM2RouteFloatTileOffset, cols);
        pipe_barrier(PIPE_V);
        M2RawRowMaxFloat(kM2RouteChunkMaxTileOffset, kM2RouteAbsTileOffset, kM2RouteFloatTileOffset, cols);
        pipe_barrier(PIPE_V);
        M2RawUpdateRowMax(kM2RouteRowMaxTileOffset, kM2RouteChunkMaxTileOffset);
    }

    pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
    float maxAbs = M2RawUbScalarLoad<float>(kM2RouteRowMaxTileOffset);
    pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
    float scale = maxAbs == 0.0f ? 1.0f : maxAbs / 127.0f;
    float invScale = maxAbs == 0.0f ? 1.0f : 1.0f / scale;

    M2RawVecDup<float>(kM2RouteScaleStoreTileOffset, scale, 1U);
    M2RawVecDup<float>(kM2RouteInvScaleParamTileOffset, invScale, 8U);

    pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>(EVENT_ID2, EVENT_ID2);
    M2RawVecStore<float>(localPeer.dispatchScale + packedRow, kM2RouteScaleStoreTileOffset, 1U);
    M2WaitRouteQuantStoreReusable();

    __gm__ int8_t *dst = localPeer.dispatchPayload + static_cast<uint64_t>(packedRow) * rowBytes;
    for (uint32_t colBegin = 0; colBegin < shape.hiddenSize; colBegin += kM2RouteQuantTileCols) {
        uint32_t cols = shape.hiddenSize - colBegin;
        if (cols > kM2RouteQuantTileCols) {
            cols = kM2RouteQuantTileCols;
        }
        M2RawVecLoad<half>(kM2RouteHalfTileOffset, input + static_cast<uint64_t>(token) * shape.hiddenSize + colBegin,
                           cols);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        M2RawHalfToFloat(kM2RouteFloatTileOffset, kM2RouteHalfTileOffset, cols);
        pipe_barrier(PIPE_V);
        M2RawMulFloat(kM2RouteFloatTileOffset, invScale, cols);
        pipe_barrier(PIPE_V);
        M2RawFloatToInt8(kM2RouteQuantTileOffset, kM2RouteFloatTileOffset, cols);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        M2RawVecStore<int8_t>(dst + colBegin, kM2RouteQuantTileOffset, cols);
        WaitStoreTileReusable();
    }

    for (uint32_t colBegin = shape.hiddenSize; colBegin < rowBytes; colBegin += kM2RouteQuantTileCols) {
        uint32_t cols = rowBytes - colBegin;
        if (cols > kM2RouteQuantTileCols) {
            cols = kM2RouteQuantTileCols;
        }
        M2RawZeroI8Tile(kM2RouteQuantTileOffset, cols);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        M2RawVecStore<int8_t>(dst + colBegin, kM2RouteQuantTileOffset, cols);
        WaitStoreTileReusable();
    }
    InvalidateGmCacheLines(dst, rowBytes);
    InvalidateGmCacheLines(localPeer.dispatchScale + packedRow, sizeof(float));
}

struct M2RoutePackedRowsCache {
    uint32_t validRows;
    uint32_t cachedRows;
    bool overflow;
};

AICORE inline M2RoutePackedRowsCache M2RoutePackedRowsCacheEmpty()
{
    return M2RoutePackedRowsCache{0U, 0U, false};
}

AICORE inline void M2RoutePackedRowsCachePush(M2RoutePackedRowsCache &cache, int32_t packedRow, uint32_t localRows)
{
    if (packedRow < 0 || static_cast<uint32_t>(packedRow) >= localRows) {
        return;
    }
    if (cache.cachedRows < kM2RoutePackedRowsUbCapacity) {
        M2RawUbStoreI32(kM2RoutePackedRowsUbAddr, cache.cachedRows, packedRow);
        ++cache.cachedRows;
    } else {
        cache.overflow = true;
    }
    ++cache.validRows;
}

struct M2RouteQuantBuffer {
    uint64_t halfAddr;
    uint64_t floatAddr;
    uint64_t absAddr;
    uint64_t quantAddr;
    uint32_t lane;
};

AICORE inline M2RouteQuantBuffer M2RouteQuantBufferByLane(uint32_t lane)
{
    if ((lane & 1U) == 0U) {
        return M2RouteQuantBuffer{kM2RouteHalfTileOffset, kM2RouteFloatTileOffset, kM2RouteAbsTileOffset,
                                  kM2RouteQuantTileOffset, 0U};
    }
    return M2RouteQuantBuffer{kM2RoutePongHalfTileOffset, kM2RoutePongFloatTileOffset, kM2RoutePongAbsTileOffset,
                              kM2RoutePongQuantTileOffset, 1U};
}

AICORE inline uint32_t M2RouteQuantChunkCols(moe_new_dispatch_combine_a8w8::ShapeConfig shape, uint32_t colBegin)
{
    uint32_t cols = shape.hiddenSize - colBegin;
    return cols > kM2RouteQuantTileCols ? kM2RouteQuantTileCols : cols;
}

AICORE inline void M2RouteQuantLoadChunk(const M2RouteQuantBuffer &buffer, __gm__ half *input, uint32_t token,
                                         uint32_t hiddenSize, uint32_t colBegin, uint32_t cols)
{
    M2RawVecLoad<half>(buffer.halfAddr, input + static_cast<uint64_t>(token) * hiddenSize + colBegin, cols);
    if ((buffer.lane & 1U) == 0U) {
        pto::Event<pto::Op::TLOAD, pto::Op::VECTOR, false, EVENT_ID0> loadReady;
        loadReady.Record();
    } else {
        pto::Event<pto::Op::TLOAD, pto::Op::VECTOR, false, EVENT_ID1> loadReady;
        loadReady.Record();
    }
}

AICORE inline void M2RouteQuantWaitLoadReady(const M2RouteQuantBuffer &buffer)
{
    if ((buffer.lane & 1U) == 0U) {
        pto::Event<pto::Op::TLOAD, pto::Op::VECTOR, false, EVENT_ID0> loadReady;
        loadReady.Wait();
    } else {
        pto::Event<pto::Op::TLOAD, pto::Op::VECTOR, false, EVENT_ID1> loadReady;
        loadReady.Wait();
    }
}

AICORE inline M2RoutePackedRowsCache M2CollectRoutePackedRows(M2WorkspaceViewDevice workspaceView, uint32_t routeBase,
                                                              uint32_t topK, uint32_t localRows)
{
    M2RoutePackedRowsCache cache = M2RoutePackedRowsCacheEmpty();
    for (uint32_t slot = 0; slot < topK; ++slot) {
        int32_t packedRow = LoadScalarI32(workspaceView.expandedRowIdx + routeBase + slot);
        M2RoutePackedRowsCachePush(cache, packedRow, localRows);
    }
    return cache;
}

AICORE inline void M2StoreScaleToPackedRows(M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                            uint32_t routeBase, uint32_t topK, uint32_t localRows,
                                            const M2RoutePackedRowsCache &cache)
{
    if (cache.validRows == 0U) {
        return;
    }
    pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>(EVENT_ID2, EVENT_ID2);
    if (!cache.overflow) {
        for (uint32_t idx = 0; idx < cache.cachedRows; ++idx) {
            uint32_t packedRow = static_cast<uint32_t>(M2RawUbLoadI32(kM2RoutePackedRowsUbAddr, idx));
            M2RawVecStore<float>(localPeer.dispatchScale + packedRow, kM2RouteScaleStoreTileOffset, 1U);
        }
    } else {
        for (uint32_t slot = 0; slot < topK; ++slot) {
            int32_t packedRow = LoadScalarI32(workspaceView.expandedRowIdx + routeBase + slot);
            if (packedRow < 0 || static_cast<uint32_t>(packedRow) >= localRows) {
                continue;
            }
            M2RawVecStore<float>(localPeer.dispatchScale + static_cast<uint32_t>(packedRow),
                                 kM2RouteScaleStoreTileOffset, 1U);
        }
    }
    M2WaitRouteQuantStoreReusable();
}

AICORE inline void M2StoreQuantChunkToPackedRows(M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                                 uint32_t routeBase, uint32_t topK, uint32_t localRows,
                                                 const M2RoutePackedRowsCache &cache, const M2RouteQuantBuffer &buffer,
                                                 uint32_t rowBytes, uint32_t colBegin, uint32_t cols)
{
    if (cache.validRows == 0U) {
        return;
    }
    pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>(EVENT_ID2, EVENT_ID2);
    if (!cache.overflow) {
        for (uint32_t idx = 0; idx < cache.cachedRows; ++idx) {
            uint32_t packedRow = static_cast<uint32_t>(M2RawUbLoadI32(kM2RoutePackedRowsUbAddr, idx));
            __gm__ int8_t *dst = localPeer.dispatchPayload + static_cast<uint64_t>(packedRow) * rowBytes;
            M2RawVecStore<int8_t>(dst + colBegin, buffer.quantAddr, cols);
        }
    } else {
        for (uint32_t slot = 0; slot < topK; ++slot) {
            int32_t packedRow = LoadScalarI32(workspaceView.expandedRowIdx + routeBase + slot);
            if (packedRow < 0 || static_cast<uint32_t>(packedRow) >= localRows) {
                continue;
            }
            __gm__ int8_t *dst =
                localPeer.dispatchPayload + static_cast<uint64_t>(static_cast<uint32_t>(packedRow)) * rowBytes;
            M2RawVecStore<int8_t>(dst + colBegin, buffer.quantAddr, cols);
        }
    }
    M2WaitRouteQuantStoreReusable();
}

AICORE inline void M2StoreZeroPadToPackedRows(M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                              uint32_t routeBase, uint32_t topK, uint32_t localRows,
                                              const M2RoutePackedRowsCache &cache, uint32_t rowBytes, uint32_t colBegin,
                                              uint32_t cols)
{
    M2RawZeroI8Tile(kM2RouteQuantTileOffset, cols);
    M2RouteQuantBuffer buffer = M2RouteQuantBufferByLane(0U);
    M2StoreQuantChunkToPackedRows(workspaceView, localPeer, routeBase, topK, localRows, cache, buffer, rowBytes,
                                  colBegin, cols);
}

AICORE inline void M2QuantizeTokenToPackedRowsReuseFloat(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                         M2WorkspaceViewDevice workspaceView,
                                                         M2PeerWindowViewDevice localPeer, GM_ADDR inputA,
                                                         uint32_t token, uint32_t rowBytes,
                                                         const M2RoutePackedRowsCache &packedRows)
{
    uint32_t localRows = static_cast<uint32_t>(M2LocalRows(shape));
    uint32_t routeBase = token * shape.topK;
    uint32_t cols = shape.hiddenSize;
    __gm__ half *input = reinterpret_cast<__gm__ half *>(inputA);
    M2RouteQuantBuffer buffer = M2RouteQuantBufferByLane(0U);

    M2RawVecDup<float>(kM2RouteRowMaxTileOffset, 0.0f, 1U);
    M2RouteQuantLoadChunk(buffer, input, token, shape.hiddenSize, 0U, cols);
    M2RouteQuantWaitLoadReady(buffer);
    M2RawHalfToFloat(buffer.floatAddr, buffer.halfAddr, cols);
    pipe_barrier(PIPE_V);
    M2RawAbsFloat(buffer.absAddr, buffer.floatAddr, cols);
    pipe_barrier(PIPE_V);
    M2RawRowMaxFloat(kM2RouteChunkMaxTileOffset, buffer.absAddr, buffer.halfAddr, cols);
    pipe_barrier(PIPE_V);
    M2RawUpdateRowMax(kM2RouteRowMaxTileOffset, kM2RouteChunkMaxTileOffset);

    pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
    float maxAbs = M2RawUbScalarLoad<float>(kM2RouteRowMaxTileOffset);
    pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
    float scale = maxAbs == 0.0f ? 1.0f : maxAbs / 127.0f;
    float invScale = maxAbs == 0.0f ? 1.0f : 1.0f / scale;
    M2RawVecDup<float>(kM2RouteScaleStoreTileOffset, scale, 1U);
    M2StoreScaleToPackedRows(workspaceView, localPeer, routeBase, shape.topK, localRows, packedRows);

    M2RawMulFloat(buffer.floatAddr, invScale, cols);
    pipe_barrier(PIPE_V);
    M2RawFloatToInt8(buffer.quantAddr, buffer.floatAddr, cols);
    pipe_barrier(PIPE_V);
    M2StoreQuantChunkToPackedRows(workspaceView, localPeer, routeBase, shape.topK, localRows, packedRows, buffer,
                                  rowBytes, 0U, cols);

    for (uint32_t colBegin = shape.hiddenSize; colBegin < rowBytes; colBegin += kM2RouteQuantTileCols) {
        uint32_t tailCols = rowBytes - colBegin;
        if (tailCols > kM2RouteQuantTileCols) {
            tailCols = kM2RouteQuantTileCols;
        }
        M2StoreZeroPadToPackedRows(workspaceView, localPeer, routeBase, shape.topK, localRows, packedRows, rowBytes,
                                   colBegin, tailCols);
    }
}

AICORE inline void M2QuantizeTokenToPackedRowsWithCache(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                        M2WorkspaceViewDevice workspaceView,
                                                        M2PeerWindowViewDevice localPeer, GM_ADDR inputA,
                                                        uint32_t token, uint32_t rowBytes,
                                                        const M2RoutePackedRowsCache &packedRows)
{
    uint32_t localRows = static_cast<uint32_t>(M2LocalRows(shape));
    uint32_t routeBase = token * shape.topK;
    if (packedRows.validRows == 0U) {
        return;
    }
    if (shape.hiddenSize > 0U && shape.hiddenSize <= kM2RouteQuantReuseFloatCols) {
        M2QuantizeTokenToPackedRowsReuseFloat(shape, workspaceView, localPeer, inputA, token, rowBytes, packedRows);
        return;
    }

    __gm__ half *input = reinterpret_cast<__gm__ half *>(inputA);
    M2RawVecDup<float>(kM2RouteRowMaxTileOffset, 0.0f, 1U);

    if (shape.hiddenSize > 0U) {
        M2RouteQuantBuffer firstBuffer = M2RouteQuantBufferByLane(0U);
        uint32_t firstCols = M2RouteQuantChunkCols(shape, 0U);
        M2RouteQuantLoadChunk(firstBuffer, input, token, shape.hiddenSize, 0U, firstCols);
    }
    for (uint32_t colBegin = 0, chunk = 0; colBegin < shape.hiddenSize; colBegin += kM2RouteQuantTileCols, ++chunk) {
        M2RouteQuantBuffer buffer = M2RouteQuantBufferByLane(chunk);
        uint32_t cols = M2RouteQuantChunkCols(shape, colBegin);
        M2RouteQuantWaitLoadReady(buffer);
        uint32_t nextColBegin = colBegin + kM2RouteQuantTileCols;
        if (nextColBegin < shape.hiddenSize) {
            M2RouteQuantBuffer nextBuffer = M2RouteQuantBufferByLane(chunk + 1U);
            uint32_t nextCols = M2RouteQuantChunkCols(shape, nextColBegin);
            M2RouteQuantLoadChunk(nextBuffer, input, token, shape.hiddenSize, nextColBegin, nextCols);
        }
        M2RawHalfToFloat(buffer.floatAddr, buffer.halfAddr, cols);
        pipe_barrier(PIPE_V);
        M2RawAbsFloat(buffer.absAddr, buffer.floatAddr, cols);
        pipe_barrier(PIPE_V);
        M2RawRowMaxFloat(kM2RouteChunkMaxTileOffset, buffer.absAddr, buffer.floatAddr, cols);
        pipe_barrier(PIPE_V);
        M2RawUpdateRowMax(kM2RouteRowMaxTileOffset, kM2RouteChunkMaxTileOffset);
    }

    pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
    float maxAbs = M2RawUbScalarLoad<float>(kM2RouteRowMaxTileOffset);
    pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
    float scale = maxAbs == 0.0f ? 1.0f : maxAbs / 127.0f;
    float invScale = maxAbs == 0.0f ? 1.0f : 1.0f / scale;
    M2RawVecDup<float>(kM2RouteScaleStoreTileOffset, scale, 1U);
    M2RawVecDup<float>(kM2RouteInvScaleParamTileOffset, invScale, 8U);
    M2StoreScaleToPackedRows(workspaceView, localPeer, routeBase, shape.topK, localRows, packedRows);

    if (shape.hiddenSize > 0U) {
        M2RouteQuantBuffer firstBuffer = M2RouteQuantBufferByLane(0U);
        uint32_t firstCols = M2RouteQuantChunkCols(shape, 0U);
        M2RouteQuantLoadChunk(firstBuffer, input, token, shape.hiddenSize, 0U, firstCols);
    }
    for (uint32_t colBegin = 0, chunk = 0; colBegin < shape.hiddenSize; colBegin += kM2RouteQuantTileCols, ++chunk) {
        M2RouteQuantBuffer buffer = M2RouteQuantBufferByLane(chunk);
        uint32_t cols = M2RouteQuantChunkCols(shape, colBegin);
        M2RouteQuantWaitLoadReady(buffer);
        uint32_t nextColBegin = colBegin + kM2RouteQuantTileCols;
        if (nextColBegin < shape.hiddenSize) {
            M2RouteQuantBuffer nextBuffer = M2RouteQuantBufferByLane(chunk + 1U);
            uint32_t nextCols = M2RouteQuantChunkCols(shape, nextColBegin);
            M2RouteQuantLoadChunk(nextBuffer, input, token, shape.hiddenSize, nextColBegin, nextCols);
        }
        M2RawHalfToFloat(buffer.floatAddr, buffer.halfAddr, cols);
        pipe_barrier(PIPE_V);
        M2RawMulFloat(buffer.floatAddr, invScale, cols);
        pipe_barrier(PIPE_V);
        M2RawFloatToInt8(buffer.quantAddr, buffer.floatAddr, cols);
        pipe_barrier(PIPE_V);
        M2StoreQuantChunkToPackedRows(workspaceView, localPeer, routeBase, shape.topK, localRows, packedRows, buffer,
                                      rowBytes, colBegin, cols);
    }

    for (uint32_t colBegin = shape.hiddenSize; colBegin < rowBytes; colBegin += kM2RouteQuantTileCols) {
        uint32_t cols = rowBytes - colBegin;
        if (cols > kM2RouteQuantTileCols) {
            cols = kM2RouteQuantTileCols;
        }
        M2StoreZeroPadToPackedRows(workspaceView, localPeer, routeBase, shape.topK, localRows, packedRows, rowBytes,
                                   colBegin, cols);
    }
}

AICORE inline void M2QuantizeTokenToPackedRows(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                               M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                               GM_ADDR inputA, uint32_t token, uint32_t rowBytes)
{
    uint32_t localRows = static_cast<uint32_t>(M2LocalRows(shape));
    uint32_t routeBase = token * shape.topK;
    M2RoutePackedRowsCache packedRows = M2CollectRoutePackedRows(workspaceView, routeBase, shape.topK, localRows);
    M2QuantizeTokenToPackedRowsWithCache(shape, workspaceView, localPeer, inputA, token, rowBytes, packedRows);
}

AICORE inline void M2PublishPackedRowToRouteIndexRangeNoFinalDsb(__gm__ int32_t *basePtr, int32_t base, int32_t rows,
                                                                 uint32_t localRows)
{
    if (base < 0 || rows <= 0) {
        return;
    }
    uint32_t begin = static_cast<uint32_t>(base);
    if (begin >= localRows) {
        return;
    }
    uint32_t available = localRows - begin;
    uint32_t effectiveRows = static_cast<uint32_t>(rows) < available ? static_cast<uint32_t>(rows) : available;
    if (effectiveRows == 0U) {
        return;
    }
    PtoGmPublishRangeNoFinalDsb(basePtr + begin, effectiveRows * sizeof(int32_t));
}

AICORE inline void M2PublishPackedRowToRouteIndexByExpertNoFinalDsb(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                                    M2WorkspaceViewDevice workspaceView)
{
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    uint32_t localRows = static_cast<uint32_t>(M2LocalRows(shape));
    for (uint32_t globalExpert = 0; globalExpert < globalExpertNum; ++globalExpert) {
        int32_t base = LoadScalarI32(workspaceView.blockPrefixPerExpert + globalExpert);
        int32_t rows = LoadScalarI32(workspaceView.blockTokenPerExpert + globalExpert);
        M2PublishPackedRowToRouteIndexRangeNoFinalDsb(workspaceView.packedRowToRouteIndex, base, rows, localRows);
    }
}

AICORE inline void M2PublishPackedRowToRouteIndexByExpert(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                          M2WorkspaceViewDevice workspaceView)
{
    PtoGmStoreDrain();
    M2PublishPackedRowToRouteIndexByExpertNoFinalDsb(shape, workspaceView);
    PtoGmPublishBatchFinish();
}

AICORE inline void M3NPublishPackedRowToRouteIndexShardNoFinalDsb(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                                  M2WorkspaceViewDevice workspaceView,
                                                                  uint32_t workerId)
{
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    uint32_t workerStride = M3NDispatchWorkerScratchStride(globalExpertNum);
    uint32_t localRows = static_cast<uint32_t>(M2LocalRows(shape));
    __gm__ int32_t *workerCounts =
        M3NDispatchWorkerCountScratch(workspaceView) + static_cast<uint64_t>(workerId) * workerStride;
    __gm__ int32_t *workerPrefixes =
        M3NDispatchWorkerPrefixScratch(workspaceView) + static_cast<uint64_t>(workerId) * workerStride;
    for (uint32_t globalExpert = 0; globalExpert < globalExpertNum; ++globalExpert) {
        int32_t expertBase = LoadScalarI32(workspaceView.blockPrefixPerExpert + globalExpert);
        int32_t workerBase = LoadScalarI32(workerPrefixes + globalExpert);
        int32_t rows = LoadScalarI32(workerCounts + globalExpert);
        M2PublishPackedRowToRouteIndexRangeNoFinalDsb(workspaceView.packedRowToRouteIndex, expertBase + workerBase,
                                                      rows, localRows);
    }
}

AICORE inline void M3NInvalidateExpandedRowIdxShard(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                    M2WorkspaceViewDevice workspaceView, uint32_t workerId,
                                                    uint32_t workerCount)
{
    uint32_t tokenBegin = TokenShardBegin(shape.m, workerId, workerCount);
    uint32_t tokenEnd = TokenShardEnd(shape.m, workerId, workerCount);
    InvalidateGmCacheLines(workspaceView.expandedRowIdx + tokenBegin * shape.topK,
                           static_cast<uint32_t>((tokenEnd - tokenBegin) * shape.topK * sizeof(int32_t)));
}

constexpr uint32_t kM2ExpandedRowIdxPublishBatchTokens = 64U;

AICORE inline void M2PublishExpandedRowIdxTokenRange(M2WorkspaceViewDevice workspaceView, uint32_t tokenBegin,
                                                     uint32_t tokenEnd, uint32_t topK)
{
    if (tokenEnd <= tokenBegin || topK == 0U) {
        return;
    }
    InvalidateGmCacheLines(workspaceView.expandedRowIdx + static_cast<uint64_t>(tokenBegin) * topK,
                           static_cast<uint32_t>((tokenEnd - tokenBegin) * topK * sizeof(int32_t)));
}

AICORE inline uint32_t M2MaybePublishExpandedRowIdxBatch(M2WorkspaceViewDevice workspaceView,
                                                         uint32_t publishBeginToken, uint32_t currentToken,
                                                         uint32_t topK)
{
    uint32_t publishEndToken = currentToken + 1U;
    if (publishEndToken - publishBeginToken < kM2ExpandedRowIdxPublishBatchTokens) {
        return publishBeginToken;
    }
    M2PublishExpandedRowIdxTokenRange(workspaceView, publishBeginToken, publishEndToken, topK);
    return publishEndToken;
}

AICORE inline void M2RoutePackStoreFence()
{
    PtoGmStoreDrain();
}

AICORE inline void M2RoutePackQuantLocal(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                         M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                         GM_ADDR inputA, GM_ADDR expertIdx, GM_ADDR xActiveMask, uint32_t rowBytes)
{
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    uint32_t localRows = static_cast<uint32_t>(M2LocalRows(shape));
    __gm__ int32_t *expertIds = reinterpret_cast<__gm__ int32_t *>(expertIdx);
    for (uint32_t globalExpert = 0; globalExpert < globalExpertNum; ++globalExpert) {
        StoreScalarI32(workspaceView.tokenOwnerRankOffsets + globalExpert,
                       LoadScalarI32(workspaceView.blockPrefixPerExpert + globalExpert));
    }
    uint32_t expandedPublishBegin = 0U;
    for (uint32_t token = 0; token < shape.m; ++token) {
        M2RoutePackedRowsCache packedRows = M2RoutePackedRowsCacheEmpty();
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            uint32_t routeIndex = token * shape.topK + slot;
            if (!M2TokenIsActive(xActiveMask, token)) {
                StoreScalarI32(workspaceView.expandedRowIdx + routeIndex, static_cast<int32_t>(localRows));
                continue;
            }
            int32_t expert = LoadScalarI32(expertIds + routeIndex);
            if (expert < 0 || static_cast<uint32_t>(expert) >= globalExpertNum) {
                StoreScalarI32(workspaceView.expandedRowIdx + routeIndex, -1);
                continue;
            }
            uint32_t globalExpert = static_cast<uint32_t>(expert);
            int32_t packedRow = LoadScalarI32(workspaceView.tokenOwnerRankOffsets + globalExpert);
            StoreScalarI32(workspaceView.tokenOwnerRankOffsets + globalExpert, packedRow + 1);
            if (packedRow < 0 || static_cast<uint32_t>(packedRow) >= localRows) {
                StoreScalarI32(workspaceView.expandedRowIdx + routeIndex, static_cast<int32_t>(localRows));
                continue;
            }
            StoreScalarI32(workspaceView.expandedRowIdx + routeIndex, packedRow);
            StoreScalarI32(workspaceView.packedRowToRouteIndex + static_cast<uint32_t>(packedRow),
                           static_cast<int32_t>(routeIndex));
            M2RoutePackedRowsCachePush(packedRows, packedRow, localRows);
        }
        M2QuantizeTokenToPackedRowsWithCache(shape, workspaceView, localPeer, inputA, token, rowBytes, packedRows);
        expandedPublishBegin =
            M2MaybePublishExpandedRowIdxBatch(workspaceView, expandedPublishBegin, token, shape.topK);
    }
    M2RoutePackStoreFence();
    M2PublishExpandedRowIdxTokenRange(workspaceView, expandedPublishBegin, shape.m, shape.topK);
    M2PublishPackedRowToRouteIndexByExpertNoFinalDsb(shape, workspaceView);
    PtoGmPublishBatchFinish();
}

AICORE inline void M2RunInitQuantFullLoadPtoVec(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                                GM_ADDR inputA, GM_ADDR expertIdx, GM_ADDR xActiveMask, uint32_t myRank,
                                                uint32_t rowBytes)
{
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    uint32_t localRows = static_cast<uint32_t>(M2LocalRows(shape));
    __gm__ int32_t *expertIds = reinterpret_cast<__gm__ int32_t *>(expertIdx);
    for (uint32_t expert = 0; expert < globalExpertNum; ++expert) {
        M2RawUbStoreI32(kM3NInitQuantFullLoadCountUbAddr, expert, 0);
        M2RawUbStoreI32(kM3NInitQuantFullLoadCursorUbAddr, expert, 0);
    }

    for (uint32_t token = 0; token < shape.m; ++token) {
        if (!M2TokenIsActive(xActiveMask, token)) {
            continue;
        }
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            uint32_t routeIndex = token * shape.topK + slot;
            int32_t expert = LoadScalarI32(expertIds + routeIndex);
            if (expert < 0 || static_cast<uint32_t>(expert) >= globalExpertNum) {
                continue;
            }
            uint32_t globalExpert = static_cast<uint32_t>(expert);
            M2RawUbStoreI32(kM3NInitQuantFullLoadCountUbAddr, globalExpert,
                            M2RawUbLoadI32(kM3NInitQuantFullLoadCountUbAddr, globalExpert) + 1);
        }
    }

    int32_t running = 0;
    for (uint32_t globalExpert = 0; globalExpert < globalExpertNum; ++globalExpert) {
        int32_t rows = M2RawUbLoadI32(kM3NInitQuantFullLoadCountUbAddr, globalExpert);
        StoreScalarI32(workspaceView.blockTokenPerExpert + globalExpert, rows);
        StoreScalarI32(workspaceView.blockPrefixPerExpert + globalExpert, running);
        StoreScalarI32(workspaceView.tokenOwnerRankOffsets + globalExpert, running);
        uint32_t expertOwner = M2ExpertOwnerRank(globalExpert, shape);
        uint32_t localExpert = M2LocalExpert(globalExpert, shape);
        StoreScalarI32(localPeer.tokenPerExpertMatrix + M2TokenPerExpertIndex(shape, myRank, expertOwner, localExpert),
                       rows);
        StoreScalarI32(
            workspaceView.tokenPerExpertMatrix + M2TokenPerExpertIndex(shape, myRank, expertOwner, localExpert), rows);
        running += rows;
    }

    uint32_t expandedPublishBegin = 0U;
    for (uint32_t token = 0; token < shape.m; ++token) {
        M2RoutePackedRowsCache packedRows = M2RoutePackedRowsCacheEmpty();
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            uint32_t routeIndex = token * shape.topK + slot;
            if (!M2TokenIsActive(xActiveMask, token)) {
                StoreScalarI32(workspaceView.expandedRowIdx + routeIndex, static_cast<int32_t>(localRows));
                continue;
            }
            int32_t expert = LoadScalarI32(expertIds + routeIndex);
            if (expert < 0 || static_cast<uint32_t>(expert) >= globalExpertNum) {
                StoreScalarI32(workspaceView.expandedRowIdx + routeIndex, -1);
                continue;
            }
            uint32_t globalExpert = static_cast<uint32_t>(expert);
            int32_t packedRow = LoadScalarI32(workspaceView.blockPrefixPerExpert + globalExpert) +
                                M2RawUbLoadI32(kM3NInitQuantFullLoadCursorUbAddr, globalExpert);
            M2RawUbStoreI32(kM3NInitQuantFullLoadCursorUbAddr, globalExpert,
                            M2RawUbLoadI32(kM3NInitQuantFullLoadCursorUbAddr, globalExpert) + 1);
            if (packedRow < 0 || static_cast<uint32_t>(packedRow) >= localRows) {
                StoreScalarI32(workspaceView.expandedRowIdx + routeIndex, static_cast<int32_t>(localRows));
                continue;
            }
            StoreScalarI32(workspaceView.expandedRowIdx + routeIndex, packedRow);
            StoreScalarI32(workspaceView.packedRowToRouteIndex + static_cast<uint32_t>(packedRow),
                           static_cast<int32_t>(routeIndex));
            M2RoutePackedRowsCachePush(packedRows, packedRow, localRows);
        }
        M2QuantizeTokenToPackedRowsWithCache(shape, workspaceView, localPeer, inputA, token, rowBytes, packedRows);
        expandedPublishBegin =
            M2MaybePublishExpandedRowIdxBatch(workspaceView, expandedPublishBegin, token, shape.topK);
    }

    M2RoutePackStoreFence();
    PtoGmPublishRangeNoFinalDsb(workspaceView.blockTokenPerExpert,
                                static_cast<uint32_t>(globalExpertNum * sizeof(int32_t)));
    PtoGmPublishRangeNoFinalDsb(workspaceView.blockPrefixPerExpert,
                                static_cast<uint32_t>(globalExpertNum * sizeof(int32_t)));
    PtoGmPublishRangeNoFinalDsb(workspaceView.tokenOwnerRankOffsets,
                                static_cast<uint32_t>(globalExpertNum * sizeof(int32_t)));
    M2PublishExpandedRowIdxTokenRange(workspaceView, expandedPublishBegin, shape.m, shape.topK);
    M2PublishPackedRowToRouteIndexByExpertNoFinalDsb(shape, workspaceView);
    PtoGmPublishRangeNoFinalDsb(
        workspaceView.tokenPerExpertMatrix,
        static_cast<uint32_t>(shape.rankNum * M2TokenPerExpertMatrixRowStride(shape) * sizeof(int32_t)));
    PtoGmPublishRangeNoFinalDsb(
        localPeer.tokenPerExpertMatrix,
        static_cast<uint32_t>(shape.rankNum * M2TokenPerExpertMatrixRowStride(shape) * sizeof(int32_t)));
    PtoGmPublishBatchFinish();
}

AICORE inline void M3NRoutePackQuantLocalShard(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                               M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                               GM_ADDR inputA, GM_ADDR expertIdx, GM_ADDR xActiveMask,
                                               uint32_t rowBytes, uint32_t workerId, uint32_t workerCount)
{
    uint32_t globalExpertNum = shape.rankNum * shape.expertPerRank;
    uint32_t workerStride = M3NDispatchWorkerScratchStride(globalExpertNum);
    uint32_t localRows = static_cast<uint32_t>(M2LocalRows(shape));
    __gm__ int32_t *expertIds = reinterpret_cast<__gm__ int32_t *>(expertIdx);
    InvalidateGmCacheLines(M3NDispatchWorkerCountScratch(workspaceView),
                           static_cast<uint32_t>(workerCount * workerStride * sizeof(int32_t)));
    InvalidateGmCacheLines(M3NDispatchWorkerPrefixScratch(workspaceView),
                           static_cast<uint32_t>(workerCount * workerStride * sizeof(int32_t)));
    __gm__ int32_t *localOrdinalCursor =
        M3NDispatchWorkerCountScratch(workspaceView) + static_cast<uint64_t>(workerId) * workerStride;
    bool useExpertCache = M3NRoutePackExpertCacheFits(globalExpertNum);
    if (useExpertCache) {
        M3NPrepareRoutePackExpertCache(shape, workspaceView, workerId, globalExpertNum);
    } else {
        for (uint32_t idx = 0; idx < workerStride; ++idx) {
            StoreScalarI32(localOrdinalCursor + idx, 0);
        }
    }
    uint32_t tokenBegin = TokenShardBegin(shape.m, workerId, workerCount);
    uint32_t tokenEnd = TokenShardEnd(shape.m, workerId, workerCount);
    uint32_t expandedPublishBegin = tokenBegin;
    for (uint32_t token = tokenBegin; token < tokenEnd; ++token) {
        M2RoutePackedRowsCache packedRows = M2RoutePackedRowsCacheEmpty();
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            uint32_t routeIndex = token * shape.topK + slot;
            if (!M2TokenIsActive(xActiveMask, token)) {
                StoreScalarI32(workspaceView.expandedRowIdx + routeIndex, static_cast<int32_t>(localRows));
                continue;
            }
            int32_t expert = LoadScalarI32(expertIds + routeIndex);
            if (expert < 0 || static_cast<uint32_t>(expert) >= globalExpertNum) {
                StoreScalarI32(workspaceView.expandedRowIdx + routeIndex, -1);
                continue;
            }
            uint32_t globalExpert = static_cast<uint32_t>(expert);
            int32_t packedRow = useExpertCache ? M3NNextPackedRowFromExpertCache(globalExpert) :
                                                 M3NNextPackedRowFromGmCursor(shape, workspaceView, localOrdinalCursor,
                                                                              workerId, globalExpert);
            if (packedRow < 0 || static_cast<uint32_t>(packedRow) >= localRows) {
                StoreScalarI32(workspaceView.expandedRowIdx + routeIndex, static_cast<int32_t>(localRows));
                continue;
            }
            StoreScalarI32(workspaceView.expandedRowIdx + routeIndex, packedRow);
            StoreScalarI32(workspaceView.packedRowToRouteIndex + static_cast<uint32_t>(packedRow),
                           static_cast<int32_t>(routeIndex));
            M2RoutePackedRowsCachePush(packedRows, packedRow, localRows);
        }
        M2QuantizeTokenToPackedRowsWithCache(shape, workspaceView, localPeer, inputA, token, rowBytes, packedRows);
        expandedPublishBegin =
            M2MaybePublishExpandedRowIdxBatch(workspaceView, expandedPublishBegin, token, shape.topK);
    }
    M2RoutePackStoreFence();
    M2PublishExpandedRowIdxTokenRange(workspaceView, expandedPublishBegin, tokenEnd, shape.topK);
    M3NPublishPackedRowToRouteIndexShardNoFinalDsb(shape, workspaceView, workerId);
    PtoGmPublishBatchFinish();
}

AICORE inline void M2PublishCountRows(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                      M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                      __gm__ HcclDeviceContext *ctx, GM_ADDR peerWindow, uint32_t myRank,
                                      const moe_new_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout)
{
    uint32_t rowElems = static_cast<uint32_t>(M2TokenPerExpertMatrixRowStride(shape));
    __gm__ int32_t *localRow =
        workspaceView.tokenPerExpertMatrix + static_cast<uint64_t>(myRank) * M2TokenPerExpertMatrixRowStride(shape);
    InvalidateGmCacheLines(localRow, rowElems * sizeof(int32_t));
    for (uint32_t dst = 0; dst < shape.rankNum; ++dst) {
        M2PeerWindowViewDevice remotePeer = MakeM2RemotePeerWindowViewDevice(ctx, peerWindow, dst, peerWindowLayout);
        __gm__ int32_t *remoteRow =
            remotePeer.tokenPerExpertMatrix + static_cast<uint64_t>(myRank) * M2TokenPerExpertMatrixRowStride(shape);
        if (dst == myRank) {
            for (uint32_t col = 0; col < rowElems; ++col) {
                StoreScalarI32(remoteRow + col, LoadScalarI32(localRow + col));
            }
            InvalidateGmCacheLines(remoteRow, rowElems * sizeof(int32_t));
            StoreScalarI32(remotePeer.countReadySignal + myRank * 16U, 1);
            InvalidateGmCacheLines(remotePeer.countReadySignal + myRank * 16U, sizeof(int32_t));
            continue;
        }
        GlobalNd<int32_t> localCount =
            MakeGlobal2D(localRow, 1, static_cast<int32_t>(rowElems), static_cast<int32_t>(rowElems));
        GlobalNd<int32_t> remoteCount =
            MakeGlobal2D(remoteRow, 1, static_cast<int32_t>(rowElems), static_cast<int32_t>(rowElems));
        TPutRows<int32_t, kMetaTileCols>(remoteCount, localCount);
        NotifySignal(remotePeer.countReadySignal + myRank * 16U, 1);
    }
}

AICORE inline void M3NPublishCountRowsShard(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                            M2WorkspaceViewDevice workspaceView, __gm__ HcclDeviceContext *ctx,
                                            GM_ADDR peerWindow, uint32_t myRank,
                                            const moe_new_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout,
                                            uint32_t workerId, uint32_t workerCount, uint32_t debugPublishMode = 0U)
{
    uint32_t rowElems = static_cast<uint32_t>(M2TokenPerExpertMatrixRowStride(shape));
    __gm__ int32_t *localRow =
        workspaceView.tokenPerExpertMatrix + static_cast<uint64_t>(myRank) * M2TokenPerExpertMatrixRowStride(shape);
    InvalidateGmCacheLines(localRow, rowElems * sizeof(int32_t));
    for (uint32_t dst = workerId; dst < shape.rankNum; dst += workerCount) {
        M2PeerWindowViewDevice remotePeer = MakeM2RemotePeerWindowViewDevice(ctx, peerWindow, dst, peerWindowLayout);
        __gm__ int32_t *remoteRow =
            remotePeer.tokenPerExpertMatrix + static_cast<uint64_t>(myRank) * M2TokenPerExpertMatrixRowStride(shape);
        if (dst == myRank) {
            if (debugPublishMode == 2U) {
                continue;
            }
            for (uint32_t col = 0; col < rowElems; ++col) {
                StoreScalarI32(remoteRow + col, LoadScalarI32(localRow + col));
            }
            InvalidateGmCacheLines(remoteRow, rowElems * sizeof(int32_t));
            StoreScalarI32(remotePeer.countReadySignal + myRank * 16U, 1);
            InvalidateGmCacheLines(remotePeer.countReadySignal + myRank * 16U, sizeof(int32_t));
            continue;
        }
        if (debugPublishMode == 3U) {
            for (uint32_t col = 0; col < rowElems; ++col) {
                StoreScalarI32(remoteRow + col, LoadScalarI32(localRow + col));
            }
            InvalidateGmCacheLines(remoteRow, rowElems * sizeof(int32_t));
            dsb(DSB_DDR);
            StoreScalarI32(remotePeer.countReadySignal + myRank * 16U, 1);
            InvalidateGmCacheLines(remotePeer.countReadySignal + myRank * 16U, sizeof(int32_t));
            dsb(DSB_DDR);
            continue;
        }
        if (debugPublishMode == 1U) {
            continue;
        }
        GlobalNd<int32_t> localCount =
            MakeGlobal2D(localRow, 1, static_cast<int32_t>(rowElems), static_cast<int32_t>(rowElems));
        GlobalNd<int32_t> remoteCount =
            MakeGlobal2D(remoteRow, 1, static_cast<int32_t>(rowElems), static_cast<int32_t>(rowElems));
        TPutRows<int32_t, kMetaTileCols>(remoteCount, localCount);
        if (debugPublishMode == 2U) {
            continue;
        }
        NotifySignal(remotePeer.countReadySignal + myRank * 16U, 1);
    }
}

AICORE inline void M3NPublishCountReadyShard(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                             M2PeerWindowViewDevice localPeer, __gm__ HcclDeviceContext *ctx,
                                             GM_ADDR peerWindow, uint32_t myRank,
                                             const moe_new_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout,
                                             uint32_t workerId, uint32_t workerCount)
{
    for (uint32_t dst = workerId; dst < shape.rankNum; dst += workerCount) {
        M2PeerWindowViewDevice remotePeer = MakeM2RemotePeerWindowViewDevice(ctx, peerWindow, dst, peerWindowLayout);
        if (dst == myRank) {
            StoreScalarI32(localPeer.countReadySignal + myRank * 16U, 1);
            InvalidateGmCacheLines(localPeer.countReadySignal + myRank * 16U, sizeof(int32_t));
            continue;
        }
        NotifySignal(remotePeer.countReadySignal + myRank * 16U, 1);
    }
}

AICORE inline void M2WaitCountRows(moe_new_dispatch_combine_a8w8::ShapeConfig shape, M2PeerWindowViewDevice localPeer,
                                   uint32_t myRank)
{
    for (uint32_t src = 0; src < shape.rankNum; ++src) {
        if (src == myRank) {
            continue;
        }
        WaitSignal(localPeer.countReadySignal + src * 16U, 1);
    }
}

AICORE inline void M3NWaitCountRowsShard(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                         M2PeerWindowViewDevice localPeer, uint32_t myRank, uint32_t workerId,
                                         uint32_t workerCount)
{
    for (uint32_t src = workerId; src < shape.rankNum; src += workerCount) {
        if (src == myRank) {
            continue;
        }
        WaitSignal(localPeer.countReadySignal + src * 16U, 1);
    }
}

AICORE inline void M3NWaitCountRowsShardScalar(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                               M2PeerWindowViewDevice localPeer, uint32_t myRank, uint32_t workerId,
                                               uint32_t workerCount)
{
    uint32_t rowElems = static_cast<uint32_t>(M2TokenPerExpertMatrixRowStride(shape));
    for (uint32_t src = workerId; src < shape.rankNum; src += workerCount) {
        if (src == myRank) {
            continue;
        }
        __gm__ int32_t *signal = localPeer.countReadySignal + src * 16U;
        while (LoadScalarI32(signal) < 1) {
            InvalidateGmCacheLines(signal, sizeof(int32_t));
        }
        __gm__ int32_t *localRow =
            localPeer.tokenPerExpertMatrix + static_cast<uint64_t>(src) * M2TokenPerExpertMatrixRowStride(shape);
        InvalidateGmCacheLines(localRow, rowElems * sizeof(int32_t));
    }
}

AICORE inline uint32_t M3NClampRowsToCapacity(uint32_t cursor, uint32_t rows, uint32_t cap)
{
    if (cursor >= cap || rows == 0U) {
        return 0U;
    }
    uint32_t available = cap - cursor;
    return rows < available ? rows : available;
}

AICORE inline void M3NApplyDispatchCapacityClip(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                                uint32_t myRank)
{
    uint32_t localRows = static_cast<uint32_t>(M2LocalRows(shape));
    uint32_t rowStride = static_cast<uint32_t>(M2TokenPerExpertMatrixRowStride(shape));
    InvalidateGmCacheLines(localPeer.tokenPerExpertMatrix,
                           static_cast<uint32_t>(shape.rankNum * rowStride * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.blockPrefixPerExpert,
                           static_cast<uint32_t>(M2GlobalExpertNum(shape) * sizeof(int32_t)));
    M2PublishPackedRowToRouteIndexByExpert(shape, workspaceView);
    for (uint32_t expertOwner = 0; expertOwner < shape.rankNum; ++expertOwner) {
        uint32_t dispatchCursor = 0U;
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            uint32_t globalExpert = expertOwner * shape.expertPerRank + localExpert;
            int32_t blockPrefix = LoadScalarI32(workspaceView.blockPrefixPerExpert + globalExpert);
            for (uint32_t src = 0; src < shape.rankNum; ++src) {
                int32_t rowsI32 = LoadScalarI32(localPeer.tokenPerExpertMatrix +
                                                M2TokenPerExpertIndex(shape, src, expertOwner, localExpert));
                uint32_t rows = rowsI32 > 0 ? static_cast<uint32_t>(rowsI32) : 0U;
                uint32_t effectiveRows = M3NClampRowsToCapacity(dispatchCursor, rows, localRows);
                if (src == myRank && effectiveRows < rows && blockPrefix >= 0) {
                    for (uint32_t row = effectiveRows; row < rows; ++row) {
                        uint32_t packedRow = static_cast<uint32_t>(blockPrefix) + row;
                        if (packedRow >= localRows) {
                            continue;
                        }
                        int32_t routeIndex = LoadScalarI32(workspaceView.packedRowToRouteIndex + packedRow);
                        if (routeIndex >= 0 && static_cast<uint32_t>(routeIndex) < shape.m * shape.topK) {
                            StoreScalarI32(workspaceView.expandedRowIdx + static_cast<uint32_t>(routeIndex),
                                           static_cast<int32_t>(localRows));
                        }
                    }
                }
                dispatchCursor += effectiveRows;
            }
        }
    }
    InvalidateGmCacheLines(workspaceView.expandedRowIdx, static_cast<uint32_t>(shape.m * shape.topK * sizeof(int32_t)));
}

AICORE inline void M3NFetchCountRowsShard(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                          M2PeerWindowViewDevice localPeer, __gm__ HcclDeviceContext *ctx,
                                          GM_ADDR peerWindow, uint32_t myRank,
                                          const moe_new_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout,
                                          uint32_t workerId, uint32_t workerCount, bool waitReady = true)
{
    uint32_t rowElems = static_cast<uint32_t>(M2TokenPerExpertMatrixRowStride(shape));
    for (uint32_t src = workerId; src < shape.rankNum; src += workerCount) {
        if (src == myRank) {
            continue;
        }
        if (waitReady) {
            WaitSignal(localPeer.countReadySignal + src * 16U, 1);
        }
        M2PeerWindowViewDevice remotePeer = MakeM2RemotePeerWindowViewDevice(ctx, peerWindow, src, peerWindowLayout);
        __gm__ int32_t *localRow =
            localPeer.tokenPerExpertMatrix + static_cast<uint64_t>(src) * M2TokenPerExpertMatrixRowStride(shape);
        __gm__ int32_t *remoteRow =
            remotePeer.tokenPerExpertMatrix + static_cast<uint64_t>(src) * M2TokenPerExpertMatrixRowStride(shape);
        GlobalNd<int32_t> localCount =
            MakeGlobal2D(localRow, 1, static_cast<int32_t>(rowElems), static_cast<int32_t>(rowElems));
        GlobalNd<int32_t> remoteCount =
            MakeGlobal2D(remoteRow, 1, static_cast<int32_t>(rowElems), static_cast<int32_t>(rowElems));
        TGetRows<int32_t, kMetaTileCols>(localCount, remoteCount);
        InvalidateGmCacheLines(localRow, rowElems * sizeof(int32_t));
    }
}

AICORE inline void M2BuildPrefixMetadata(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                         M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                         uint32_t myRank)
{
    int32_t dispatchCursor = 0;
    int32_t localRowCap = static_cast<int32_t>(M2LocalRows(shape));
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        StoreScalarI32(workspaceView.dispatchOffset + localExpert, dispatchCursor);
        int32_t before = 0;
        for (uint32_t tokenOwner = 0; tokenOwner < shape.rankNum; ++tokenOwner) {
            int32_t rows = M2LoadTokenPerExpert(shape, localPeer, tokenOwner, myRank, localExpert);
            if (dispatchCursor >= localRowCap || rows <= 0) {
                rows = 0;
            } else {
                int32_t available = localRowCap - dispatchCursor;
                if (rows > available) {
                    rows = available;
                }
            }
            StoreScalarI32(workspaceView.preSumBeforeRank + tokenOwner * shape.expertPerRank + localExpert, before);
            before += rows;
            StoreScalarI32(workspaceView.cumsumMM + tokenOwner * shape.expertPerRank + localExpert, before);
            dispatchCursor += rows;
        }
        StoreScalarI32(workspaceView.expertTokenNums + localExpert, before);
    }
    InvalidateGmCacheLines(workspaceView.dispatchOffset, static_cast<uint32_t>(shape.expertPerRank * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.preSumBeforeRank,
                           static_cast<uint32_t>(shape.rankNum * shape.expertPerRank * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.cumsumMM,
                           static_cast<uint32_t>(shape.rankNum * shape.expertPerRank * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.expertTokenNums, static_cast<uint32_t>(shape.expertPerRank * sizeof(int32_t)));
}

AICORE inline int32_t M2SourceGlobalExpertRowBase(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                  M2PeerWindowViewDevice localPeer, uint32_t tokenOwnerRank,
                                                  uint32_t globalExpert)
{
    int32_t base = 0;
    for (uint32_t expert = 0; expert < globalExpert; ++expert) {
        base += M2LoadTokenPerExpert(shape, localPeer, tokenOwnerRank, M2ExpertOwnerRank(expert, shape),
                                     M2LocalExpert(expert, shape));
    }
    return base;
}

AICORE inline void M2TGetRowsInt8(__gm__ int8_t *dstBase, int32_t dstRowStride, int32_t dstRow,
                                  __gm__ int8_t *remoteSrcBase, int32_t srcRowStride, int32_t srcRow, int32_t rows,
                                  int32_t cols)
{
    if (rows <= 0 || cols <= 0) {
        return;
    }
    GlobalNd<int8_t> dst =
        MakeGlobal2D(dstBase + static_cast<int64_t>(dstRow) * dstRowStride, rows, cols, dstRowStride);
    GlobalNd<int8_t> src =
        MakeGlobal2D(remoteSrcBase + static_cast<int64_t>(srcRow) * srcRowStride, rows, cols, srcRowStride);
    TGetRows<int8_t, kDefaultTileCols>(dst, src);
}

AICORE inline void M2TGetRowsFloat(__gm__ float *dstBase, int32_t dstRow, __gm__ float *remoteSrcBase, int32_t srcRow,
                                   int32_t rows)
{
    if (rows <= 0) {
        return;
    }
    GlobalNd<float> dst = MakeGlobal2D(dstBase + dstRow, rows, 1, 1);
    GlobalNd<float> src = MakeGlobal2D(remoteSrcBase + srcRow, rows, 1, 1);
    TGetRows<float, kMetaTileCols>(dst, src);
}

AICORE inline void M2CopyLocalDispatchRowsToGmm1(__gm__ int8_t *dstPayload, uint32_t dstRowBytes,
                                                 __gm__ float *dstScale, M2PeerWindowViewDevice localPeer,
                                                 uint32_t srcRowBytes, int32_t dstStart, int32_t srcStart, int32_t rows)
{
    for (int32_t row = 0; row < rows; ++row) {
        __gm__ int8_t *dst = dstPayload + static_cast<int64_t>(dstStart + row) * dstRowBytes;
        __gm__ int8_t *src = localPeer.dispatchPayload + static_cast<int64_t>(srcStart + row) * srcRowBytes;
        for (uint32_t col = 0; col < srcRowBytes; ++col) {
            dst[col] = src[col];
        }
        __gm__ int32_t *dstScaleBits = reinterpret_cast<__gm__ int32_t *>(dstScale + dstStart + row);
        __gm__ int32_t *srcScaleBits = reinterpret_cast<__gm__ int32_t *>(localPeer.dispatchScale + srcStart + row);
        StoreScalarI32(dstScaleBits, LoadScalarI32(srcScaleBits));
    }
}

AICORE inline void M2CopyRemoteDispatchRowsToGmm1Scalar(__gm__ int8_t *dstPayload, uint32_t dstRowBytes,
                                                        __gm__ float *dstScale, M2PeerWindowViewDevice remotePeer,
                                                        uint32_t srcRowBytes, int32_t dstStart, int32_t srcStart,
                                                        int32_t rows)
{
    for (int32_t row = 0; row < rows; ++row) {
        __gm__ int8_t *dst = dstPayload + static_cast<int64_t>(dstStart + row) * dstRowBytes;
        __gm__ int8_t *src = remotePeer.dispatchPayload + static_cast<int64_t>(srcStart + row) * srcRowBytes;
        InvalidateGmCacheLines(src, srcRowBytes);
        for (uint32_t col = 0; col < srcRowBytes; ++col) {
            dst[col] = src[col];
        }
        __gm__ int32_t *dstScaleBits = reinterpret_cast<__gm__ int32_t *>(dstScale + dstStart + row);
        __gm__ int32_t *srcScaleBits = reinterpret_cast<__gm__ int32_t *>(remotePeer.dispatchScale + srcStart + row);
        InvalidateGmCacheLines(srcScaleBits, sizeof(int32_t));
        StoreScalarI32(dstScaleBits, LoadScalarI32(srcScaleBits));
        InvalidateGmCacheLines(dst, dstRowBytes);
    }
}

AICORE inline void M2GatherDispatchToGmm1Input(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                               M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                               __gm__ HcclDeviceContext *ctx, GM_ADDR peerWindow, uint32_t myRank,
                                               uint32_t rowBytes,
                                               const moe_new_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout)
{
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        uint32_t globalExpert = M2GlobalExpert(myRank, localExpert, shape);
        for (uint32_t tokenOwner = 0; tokenOwner < shape.rankNum; ++tokenOwner) {
            int32_t current = LoadScalarI32(workspaceView.cumsumMM + tokenOwner * shape.expertPerRank + localExpert);
            int32_t previous =
                tokenOwner == 0U ?
                    0 :
                    LoadScalarI32(workspaceView.cumsumMM + (tokenOwner - 1U) * shape.expertPerRank + localExpert);
            int32_t rows = current - previous;
            int32_t dstStart =
                LoadScalarI32(workspaceView.dispatchOffset + localExpert) +
                LoadScalarI32(workspaceView.preSumBeforeRank + tokenOwner * shape.expertPerRank + localExpert);
            if (rows <= 0) {
                continue;
            }
            int32_t srcStart = M2SourceGlobalExpertRowBase(shape, localPeer, tokenOwner, globalExpert);
            int32_t localRowCap = static_cast<int32_t>(M2LocalRows(shape));
            if (srcStart >= localRowCap) {
                continue;
            }
            if (srcStart + rows > localRowCap) {
                rows = localRowCap - srcStart;
            }
            if (rows <= 0) {
                continue;
            }
            if (tokenOwner == myRank) {
                M2CopyLocalDispatchRowsToGmm1(workspaceView.gmm1InputInt8, rowBytes, workspaceView.routingPerTokenScale,
                                              localPeer, rowBytes, dstStart, srcStart, rows);
            } else {
                M2PeerWindowViewDevice remotePeer =
                    MakeM2RemotePeerWindowViewDevice(ctx, peerWindow, tokenOwner, peerWindowLayout);
                M2CopyRemoteDispatchRowsToGmm1Scalar(workspaceView.gmm1InputInt8, rowBytes,
                                                     workspaceView.routingPerTokenScale, remotePeer, rowBytes, dstStart,
                                                     srcStart, rows);
            }
        }
        StoreScalarI32(workspaceView.dispatchGroupReady + localExpert * 16U, 1);
    }
    InvalidateGmCacheLines(workspaceView.gmm1InputInt8, static_cast<uint32_t>(M2LocalRows(shape) * rowBytes));
}

AICORE inline void M3NGatherDispatchToGmm1InputShard(
    moe_new_dispatch_combine_a8w8::ShapeConfig shape, M2WorkspaceViewDevice workspaceView,
    M2PeerWindowViewDevice localPeer, __gm__ HcclDeviceContext *ctx, GM_ADDR peerWindow, uint32_t myRank,
    uint32_t rowBytes, const moe_new_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout, uint32_t workerId,
    uint32_t workerCount)
{
    for (uint32_t localExpert = workerId; localExpert < shape.expertPerRank; localExpert += workerCount) {
        uint32_t globalExpert = M2GlobalExpert(myRank, localExpert, shape);
        for (uint32_t tokenOwner = 0; tokenOwner < shape.rankNum; ++tokenOwner) {
            int32_t current = LoadScalarI32(workspaceView.cumsumMM + tokenOwner * shape.expertPerRank + localExpert);
            int32_t previous =
                tokenOwner == 0U ?
                    0 :
                    LoadScalarI32(workspaceView.cumsumMM + (tokenOwner - 1U) * shape.expertPerRank + localExpert);
            int32_t rows = current - previous;
            int32_t dstStart =
                LoadScalarI32(workspaceView.dispatchOffset + localExpert) +
                LoadScalarI32(workspaceView.preSumBeforeRank + tokenOwner * shape.expertPerRank + localExpert);
            if (rows <= 0) {
                continue;
            }
            int32_t srcStart = M2SourceGlobalExpertRowBase(shape, localPeer, tokenOwner, globalExpert);
            int32_t localRowCap = static_cast<int32_t>(M2LocalRows(shape));
            if (srcStart >= localRowCap) {
                continue;
            }
            if (srcStart + rows > localRowCap) {
                rows = localRowCap - srcStart;
            }
            if (rows <= 0) {
                continue;
            }
            if (tokenOwner == myRank) {
                M2CopyLocalDispatchRowsToGmm1(workspaceView.gmm1InputInt8, rowBytes, workspaceView.routingPerTokenScale,
                                              localPeer, rowBytes, dstStart, srcStart, rows);
            } else {
                M2PeerWindowViewDevice remotePeer =
                    MakeM2RemotePeerWindowViewDevice(ctx, peerWindow, tokenOwner, peerWindowLayout);
                M2CopyRemoteDispatchRowsToGmm1Scalar(workspaceView.gmm1InputInt8, rowBytes,
                                                     workspaceView.routingPerTokenScale, remotePeer, rowBytes, dstStart,
                                                     srcStart, rows);
            }
        }
    }
    InvalidateGmCacheLines(workspaceView.gmm1InputInt8, static_cast<uint32_t>(M2LocalRows(shape) * rowBytes));
}

AICORE inline void M3NPublishAllDispatchExpertsReadyAfterGather(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                                M2WorkspaceViewDevice workspaceView)
{
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        StoreScalarI32(workspaceView.dispatchGroupReady + localExpert * 16U, 1);
    }
}

AICORE inline void M2RunGmm1Epilogue(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                     M2WorkspaceViewDevice workspaceView, uint32_t myRank)
{
    uint32_t w1Cols = shape.intermediateSize * 2U;
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        int32_t rowBegin = LoadScalarI32(workspaceView.dispatchOffset + localExpert);
        int32_t rowCount = LoadScalarI32(workspaceView.expertTokenNums + localExpert);
        if (rowCount <= 0) {
            continue;
        }
        for (uint32_t row = 0; row < static_cast<uint32_t>(rowCount); ++row) {
            uint32_t globalRow = static_cast<uint32_t>(rowBegin) + row;
            for (uint32_t colBegin = 0; colBegin < w1Cols; colBegin += kM2EpilogueTileCols) {
                uint32_t cols = w1Cols - colBegin;
                if (cols > kM2EpilogueTileCols) {
                    cols = kM2EpilogueTileCols;
                }
                M2EpilogueAccTile accTile(cols);
                M2EpilogueFloatTile fpTile(cols);
                M2EpilogueFloatTile scaleTile(cols);
                M2EpilogueFloatTile outTile(cols);
                TASSIGN(accTile, kM2EpilogueAccTileOffset);
                TASSIGN(fpTile, kM2EpilogueFloatTileOffset);
                TASSIGN(scaleTile, kM2EpilogueScaleTileOffset);
                TASSIGN(outTile, kM2EpilogueScaledTileOffset);

                GlobalNd<int32_t> accGlobal =
                    MakeGlobal2D(workspaceView.gmm1AccInt32 + static_cast<uint64_t>(globalRow) * w1Cols + colBegin, 1,
                                 static_cast<int32_t>(cols), static_cast<int32_t>(w1Cols));
                GlobalNd<float> outGlobal =
                    MakeGlobal2D(workspaceView.gmm1Out + static_cast<uint64_t>(globalRow) * w1Cols + colBegin, 1,
                                 static_cast<int32_t>(cols), static_cast<int32_t>(w1Cols));
                for (uint32_t col = 0; col < cols; ++col) {
                    scaleTile.SetValue(col, M2DecodeUint64Scale(workspaceView.scale1Uint64[colBegin + col]));
                }
                pipe_barrier(PIPE_ALL);
                TLOAD(accTile, accGlobal);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                TCVT(fpTile, accTile, pto::RoundMode::CAST_RINT);
                pipe_barrier(PIPE_V);
                TMUL(outTile, fpTile, scaleTile);
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                TSTORE(outGlobal, outTile);
                WaitStoreTileReusable();
            }
        }
    }
    uint32_t syncGroupCount = static_cast<uint32_t>(LoadScalarI32(workspaceView.swigluSyncGroups));
    if (syncGroupCount == 0U || syncGroupCount > shape.expertPerRank) {
        syncGroupCount = shape.expertPerRank;
    }
    for (uint32_t syncIdx = 0; syncIdx < syncGroupCount; ++syncIdx) {
        StoreScalarI32(workspaceView.gmm1SyncGroupReady + syncIdx * 16U, 1);
    }
    InvalidateGmCacheLines(workspaceView.gmm1Out, static_cast<uint32_t>(M2LocalRows(shape) * w1Cols * sizeof(float)));
    (void)myRank;
}

AICORE inline void M3N6RunGmm1EpilogueSyncGroup(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                M2WorkspaceViewDevice workspaceView, uint32_t syncIdx)
{
    uint32_t syncGroupCount = static_cast<uint32_t>(LoadScalarI32(workspaceView.swigluSyncGroups));
    if (syncGroupCount == 0U || syncGroupCount > shape.expertPerRank || syncIdx >= syncGroupCount) {
        return;
    }
    __gm__ int32_t *group = workspaceView.swigluGroupDesc + syncIdx * kM2SwigluGroupFields;
    int32_t expertBeginI32 = LoadScalarI32(group + 1U);
    int32_t expertEndI32 = LoadScalarI32(group + 2U);
    if (expertBeginI32 < 0 || expertEndI32 <= expertBeginI32) {
        return;
    }
    uint32_t expertBegin = static_cast<uint32_t>(expertBeginI32);
    uint32_t expertEnd = static_cast<uint32_t>(expertEndI32);
    if (expertEnd > shape.expertPerRank) {
        expertEnd = shape.expertPerRank;
    }
    uint32_t w1Cols = shape.intermediateSize * 2U;
    for (uint32_t localExpert = expertBegin; localExpert < expertEnd; ++localExpert) {
        int32_t rowBegin = LoadScalarI32(workspaceView.dispatchOffset + localExpert);
        int32_t rowCount = LoadScalarI32(workspaceView.expertTokenNums + localExpert);
        if (rowCount <= 0) {
            continue;
        }
        for (uint32_t row = 0; row < static_cast<uint32_t>(rowCount); ++row) {
            uint32_t globalRow = static_cast<uint32_t>(rowBegin) + row;
            for (uint32_t colBegin = 0; colBegin < w1Cols; colBegin += kM2EpilogueTileCols) {
                uint32_t cols = w1Cols - colBegin;
                if (cols > kM2EpilogueTileCols) {
                    cols = kM2EpilogueTileCols;
                }
                M2EpilogueAccTile accTile(cols);
                M2EpilogueFloatTile fpTile(cols);
                M2EpilogueFloatTile scaleTile(cols);
                M2EpilogueFloatTile outTile(cols);
                TASSIGN(accTile, kM2EpilogueAccTileOffset);
                TASSIGN(fpTile, kM2EpilogueFloatTileOffset);
                TASSIGN(scaleTile, kM2EpilogueScaleTileOffset);
                TASSIGN(outTile, kM2EpilogueScaledTileOffset);

                GlobalNd<int32_t> accGlobal =
                    MakeGlobal2D(workspaceView.gmm1AccInt32 + static_cast<uint64_t>(globalRow) * w1Cols + colBegin, 1,
                                 static_cast<int32_t>(cols), static_cast<int32_t>(w1Cols));
                GlobalNd<float> outGlobal =
                    MakeGlobal2D(workspaceView.gmm1Out + static_cast<uint64_t>(globalRow) * w1Cols + colBegin, 1,
                                 static_cast<int32_t>(cols), static_cast<int32_t>(w1Cols));
                for (uint32_t col = 0; col < cols; ++col) {
                    scaleTile.SetValue(col, M2DecodeUint64Scale(workspaceView.scale1Uint64[colBegin + col]));
                }
                pipe_barrier(PIPE_ALL);
                TLOAD(accTile, accGlobal);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                TCVT(fpTile, accTile, pto::RoundMode::CAST_RINT);
                pipe_barrier(PIPE_V);
                TMUL(outTile, fpTile, scaleTile);
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                TSTORE(outGlobal, outTile);
                WaitStoreTileReusable();
            }
        }
    }
    InvalidateGmCacheLines(workspaceView.gmm1Out, static_cast<uint32_t>(M2LocalRows(shape) * w1Cols * sizeof(float)));
}

AICORE inline uint32_t M2NextSwigluGroupSize(uint32_t remainingExperts)
{
    if (remainingExperts <= 1U) {
        return 1U;
    }
    return remainingExperts / 2U;
}

AICORE inline uint32_t M2BuildGmmTileTaskPlan(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                              M2WorkspaceViewDevice workspaceView, __gm__ int32_t *taskPlan,
                                              uint32_t nCols, uint32_t kSize, uint32_t stageId)
{
    uint32_t taskId = 0;
    uint32_t maxTasks = static_cast<uint32_t>(M2GmmTileTaskCapacity(shape));
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        int32_t rowBegin = LoadScalarI32(workspaceView.dispatchOffset + localExpert);
        int32_t rowCount = LoadScalarI32(workspaceView.expertTokenNums + localExpert);
        if (rowCount <= 0) {
            continue;
        }
        for (uint32_t rowOffset = 0; rowOffset < static_cast<uint32_t>(rowCount); rowOffset += kM2GmmBaseM) {
            uint32_t rows = static_cast<uint32_t>(rowCount) - rowOffset;
            if (rows > kM2GmmBaseM) {
                rows = kM2GmmBaseM;
            }
            for (uint32_t nBase = 0; nBase < nCols; nBase += kM2GmmBaseN) {
                uint32_t cols = nCols - nBase;
                if (cols > kM2GmmBaseN) {
                    cols = kM2GmmBaseN;
                }
                if (taskId < maxTasks) {
                    __gm__ int32_t *task = taskPlan + taskId * kM2GmmTileTaskFields;
                    StoreScalarI32(task + 0U, static_cast<int32_t>(taskId));
                    StoreScalarI32(task + 1U, static_cast<int32_t>(stageId));
                    StoreScalarI32(task + 2U, static_cast<int32_t>(localExpert));
                    StoreScalarI32(task + 3U, rowBegin + static_cast<int32_t>(rowOffset));
                    StoreScalarI32(task + 4U, static_cast<int32_t>(rows));
                    StoreScalarI32(task + 5U, static_cast<int32_t>(nBase));
                    StoreScalarI32(task + 6U, static_cast<int32_t>(cols));
                    StoreScalarI32(task + 7U, static_cast<int32_t>(kSize));
                }
                ++taskId;
            }
        }
    }
    return taskId;
}

AICORE inline void M2BuildSwigluSyncMetadata(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                             M2WorkspaceViewDevice workspaceView)
{
    uint32_t groupCount = 0;
    uint32_t expert = 0;
    int32_t rowPrefix = 0;
    uint32_t tilePrefix = 0;
    StoreScalarI32(workspaceView.dequantSum, 0);
    while (expert < shape.expertPerRank) {
        uint32_t remaining = shape.expertPerRank - expert;
        uint32_t groupSize = M2NextSwigluGroupSize(remaining);
        int32_t rowBegin = rowPrefix;
        StoreScalarI32(workspaceView.swigluSyncGroups + groupCount + 1U, static_cast<int32_t>(groupSize));
        for (uint32_t idx = 0; idx < groupSize; ++idx) {
            rowPrefix += LoadScalarI32(workspaceView.expertTokenNums + expert + idx);
        }
        uint32_t tileBegin = tilePrefix;
        uint32_t tileEnd = tileBegin + static_cast<uint32_t>(
                                           M2CeilDivDevice(static_cast<uint64_t>(rowPrefix - rowBegin), kM2GmmBaseM));
        __gm__ int32_t *group = workspaceView.swigluGroupDesc + groupCount * kM2SwigluGroupFields;
        StoreScalarI32(group + 0U, static_cast<int32_t>(groupCount));
        StoreScalarI32(group + 1U, static_cast<int32_t>(expert));
        StoreScalarI32(group + 2U, static_cast<int32_t>(expert + groupSize));
        StoreScalarI32(group + 3U, rowBegin);
        StoreScalarI32(group + 4U, rowPrefix);
        StoreScalarI32(group + 5U, static_cast<int32_t>(tileBegin));
        StoreScalarI32(group + 6U, static_cast<int32_t>(tileEnd));
        StoreScalarI32(group + 7U, rowPrefix == rowBegin ? 1 : 0);
        StoreScalarI32(workspaceView.dequantSum + groupCount + 1U, rowPrefix);
        ++groupCount;
        tilePrefix = tileEnd;
        expert += groupSize;
    }
    StoreScalarI32(workspaceView.swigluSyncGroups, static_cast<int32_t>(groupCount));
    uint32_t gmm2Tasks = M2BuildGmmTileTaskPlan(shape, workspaceView, workspaceView.gmm2TileTaskPlan, shape.hiddenSize,
                                                shape.intermediateSize, 2U);
    StoreScalarI32(workspaceView.stageStatus + 2U * 16U, static_cast<int32_t>(groupCount));
    StoreScalarI32(workspaceView.stageStatus + 3U * 16U, static_cast<int32_t>(gmm2Tasks));
}

AICORE inline void M2ComputeSwigluRowPto(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                         M2WorkspaceViewDevice workspaceView, uint32_t globalRow, float routingScale)
{
    constexpr int kSwiGluCols = 64;
    uint32_t w1Cols = shape.intermediateSize * 2U;
    for (uint32_t colBegin = 0; colBegin < shape.intermediateSize; colBegin += kSwiGluCols) {
        uint32_t cols = shape.intermediateSize - colBegin;
        if (cols > kSwiGluCols) {
            cols = kSwiGluCols;
        }
        VecTile<float, kSwiGluCols> gateTile(1, cols);
        VecTile<float, kSwiGluCols> upTile(1, cols);
        VecTile<float, kSwiGluCols> negGateTile(1, cols);
        VecTile<float, kSwiGluCols> expTile(1, cols);
        VecTile<float, kSwiGluCols> denomTile(1, cols);
        VecTile<float, kSwiGluCols> sigmoidTile(1, cols);
        VecTile<float, kSwiGluCols> scaledTile(1, cols);
        TASSIGN(gateTile, 0x0);
        TASSIGN(upTile, 0x400);
        TASSIGN(negGateTile, 0x800);
        TASSIGN(expTile, 0xC00);
        TASSIGN(denomTile, 0x1000);
        TASSIGN(sigmoidTile, 0x1400);
        TASSIGN(scaledTile, 0x1800);

        GlobalNd<float> gateGlobal =
            MakeGlobal2D(workspaceView.gmm1Out + static_cast<uint64_t>(globalRow) * w1Cols + colBegin, 1,
                         static_cast<int32_t>(cols), static_cast<int32_t>(w1Cols));
        GlobalNd<float> upGlobal = MakeGlobal2D(
            workspaceView.gmm1Out + static_cast<uint64_t>(globalRow) * w1Cols + shape.intermediateSize + colBegin, 1,
            static_cast<int32_t>(cols), static_cast<int32_t>(w1Cols));
        GlobalNd<float> dstGlobal =
            MakeGlobal2D(workspaceView.swigluOut + static_cast<uint64_t>(globalRow) * shape.intermediateSize + colBegin,
                         1, static_cast<int32_t>(cols), static_cast<int32_t>(shape.intermediateSize));
        TLOAD(gateTile, gateGlobal);
        TLOAD(upTile, upGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TMULS(gateTile, gateTile, routingScale);
        pipe_barrier(PIPE_V);
        TMULS(upTile, upTile, routingScale);
        pipe_barrier(PIPE_V);
        TMULS(negGateTile, gateTile, -1.0f);
        pipe_barrier(PIPE_V);
        TEXP(expTile, negGateTile);
        pipe_barrier(PIPE_V);
        TADDS(denomTile, expTile, 1.0f);
        pipe_barrier(PIPE_V);
        TDIVS(sigmoidTile, 1.0f, denomTile);
        pipe_barrier(PIPE_V);
        TMUL(scaledTile, sigmoidTile, upTile);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstGlobal, scaledTile);
        WaitStoreTileReusable();
    }
}

AICORE inline void M2RequantizeSwigluRowPto(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                            M2WorkspaceViewDevice workspaceView, uint32_t globalRow,
                                            uint32_t gmm2RowStride)
{
    M2RouteRowStatTile rowMaxTile;
    M2RouteRowStatTile chunkMaxTile;
    TASSIGN(rowMaxTile, kM2RouteRowMaxTileOffset);
    TASSIGN(chunkMaxTile, kM2RouteChunkMaxTileOffset);
    M2RawVecDup<float>(kM2RouteRowMaxTileOffset, 0.0f, 1U);

    for (uint32_t colBegin = 0; colBegin < shape.intermediateSize; colBegin += kM2RouteQuantTileCols) {
        uint32_t cols = shape.intermediateSize - colBegin;
        if (cols > kM2RouteQuantTileCols) {
            cols = kM2RouteQuantTileCols;
        }
        M2RouteFloatTile fpTile(cols);
        M2RouteFloatTile absTile(cols);
        TASSIGN(fpTile, kM2RouteFloatTileOffset);
        TASSIGN(absTile, kM2RouteAbsTileOffset);

        M2RawVecLoad<float>(
            kM2RouteFloatTileOffset,
            workspaceView.swigluOut + static_cast<uint64_t>(globalRow) * shape.intermediateSize + colBegin, cols);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TABS(absTile, fpTile);
        pipe_barrier(PIPE_V);
        TROWMAX(chunkMaxTile, absTile, fpTile);
        pipe_barrier(PIPE_V);
        TMAX(rowMaxTile, rowMaxTile, chunkMaxTile);
        pipe_barrier(PIPE_V);
    }

    pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
    float maxAbs = M2RawUbScalarLoad<float>(kM2RouteRowMaxTileOffset);
    pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
    float scale = maxAbs == 0.0f ? 1.0f : maxAbs / 127.0f;
    float invScale = maxAbs == 0.0f ? 1.0f : 1.0f / scale;

    M2RouteScaleStoreTile scaleStoreTile;
    M2RouteParamTile invScaleTile;
    TASSIGN(scaleStoreTile, kM2RouteScaleStoreTileOffset);
    TASSIGN(invScaleTile, kM2RouteInvScaleParamTileOffset);
    M2RawVecDup<float>(kM2RouteScaleStoreTileOffset, scale, 1U);
    M2RawVecDup<float>(kM2RouteInvScaleParamTileOffset, invScale, 8U);

    GlobalNd<float> scaleGlobal = MakeGlobal2D(workspaceView.gmm2PerTokenScale + globalRow, 1, 1, 1);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(scaleGlobal, scaleStoreTile);
    WaitStoreTileReusable();

    __gm__ int8_t *dst = workspaceView.gmm2InputInt8 + static_cast<uint64_t>(globalRow) * gmm2RowStride;
    for (uint32_t colBegin = 0; colBegin < shape.intermediateSize; colBegin += kM2RouteQuantTileCols) {
        uint32_t cols = shape.intermediateSize - colBegin;
        if (cols > kM2RouteQuantTileCols) {
            cols = kM2RouteQuantTileCols;
        }
        M2RouteFloatTile fpTile(cols);
        M2RouteQuantTile quantTile(cols);
        TASSIGN(fpTile, kM2RouteFloatTileOffset);
        TASSIGN(quantTile, kM2RouteQuantTileOffset);

        M2RawVecLoad<float>(
            kM2RouteFloatTileOffset,
            workspaceView.swigluOut + static_cast<uint64_t>(globalRow) * shape.intermediateSize + colBegin, cols);
        GlobalNd<int8_t> quantDst =
            MakeGlobal2D(dst + colBegin, 1, static_cast<int32_t>(cols), static_cast<int32_t>(gmm2RowStride));
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pto::TQUANT<pto::QuantType::INT8_SYM>(quantTile, fpTile, invScaleTile);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(quantDst, quantTile);
        WaitStoreTileReusable();
    }

    for (uint32_t colBegin = shape.intermediateSize; colBegin < gmm2RowStride; colBegin += kM2RouteQuantTileCols) {
        uint32_t cols = gmm2RowStride - colBegin;
        if (cols > kM2RouteQuantTileCols) {
            cols = kM2RouteQuantTileCols;
        }
        M2RouteQuantTile zeroTile(cols);
        TASSIGN(zeroTile, kM2RouteQuantTileOffset);
        M2RawZeroI8Tile(kM2RouteQuantTileOffset, cols);
        GlobalNd<int8_t> padDst =
            MakeGlobal2D(dst + colBegin, 1, static_cast<int32_t>(cols), static_cast<int32_t>(gmm2RowStride));
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(padDst, zeroTile);
        WaitStoreTileReusable();
    }
}

AICORE inline void M3NInitActivationScaleShard(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                               M2WorkspaceViewDevice workspaceView, uint32_t workerId,
                                               uint32_t workerCount)
{
    uint32_t localRows = static_cast<uint32_t>(M2LocalRows(shape));
    uint32_t rowBegin = TokenShardBegin(localRows, workerId, workerCount);
    uint32_t rowEnd = TokenShardEnd(localRows, workerId, workerCount);
    for (uint32_t row = rowBegin; row < rowEnd; ++row) {
        workspaceView.gmm2PerTokenScale[row] = 1.0f;
    }
}

AICORE inline uint32_t M3NRunActivationQuantShard(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                  M2WorkspaceViewDevice workspaceView, uint32_t workerId,
                                                  uint32_t workerCount)
{
    uint32_t gmm2RowStride = static_cast<uint32_t>(Align64Device(shape.intermediateSize));
    uint32_t syncGroupCount = static_cast<uint32_t>(LoadScalarI32(workspaceView.swigluSyncGroups));
    if (syncGroupCount == 0U || syncGroupCount > shape.expertPerRank) {
        syncGroupCount = shape.expertPerRank;
    }
    uint32_t rowsProcessed = 0;
    for (uint32_t syncIdx = 0; syncIdx < syncGroupCount; ++syncIdx) {
        __gm__ int32_t *group = workspaceView.swigluGroupDesc + syncIdx * kM2SwigluGroupFields;
        int32_t rowBeginI32 = LoadScalarI32(group + 3U);
        int32_t rowEndI32 = LoadScalarI32(group + 4U);
        if (rowBeginI32 < 0 || rowEndI32 <= rowBeginI32) {
            continue;
        }
        uint32_t rowBegin = static_cast<uint32_t>(rowBeginI32);
        uint32_t rowEnd = static_cast<uint32_t>(rowEndI32);
        for (uint32_t globalRow = rowBegin + workerId; globalRow < rowEnd; globalRow += workerCount) {
            float routingScale = workspaceView.routingPerTokenScale[globalRow];
            M2ComputeSwigluRowPto(shape, workspaceView, globalRow, routingScale);
            InvalidateGmCacheLines(workspaceView.swigluOut + static_cast<uint64_t>(globalRow) * shape.intermediateSize,
                                   static_cast<uint32_t>(shape.intermediateSize * sizeof(float)));
            M2RequantizeSwigluRowPto(shape, workspaceView, globalRow, gmm2RowStride);
            ++rowsProcessed;
        }
    }
    return rowsProcessed;
}

AICORE inline uint32_t M3NRunActivationQuantSyncGroupShard(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                           M2WorkspaceViewDevice workspaceView, uint32_t syncIdx,
                                                           uint32_t workerId, uint32_t workerCount)
{
    if (workerCount == 0U) {
        return 0U;
    }
    uint32_t gmm2RowStride = static_cast<uint32_t>(Align64Device(shape.intermediateSize));
    uint32_t syncGroupCount = static_cast<uint32_t>(LoadScalarI32(workspaceView.swigluSyncGroups));
    if (syncGroupCount == 0U || syncGroupCount > shape.expertPerRank || syncIdx >= syncGroupCount) {
        return 0U;
    }
    __gm__ int32_t *group = workspaceView.swigluGroupDesc + syncIdx * kM2SwigluGroupFields;
    int32_t rowBeginI32 = LoadScalarI32(group + 3U);
    int32_t rowEndI32 = LoadScalarI32(group + 4U);
    if (rowBeginI32 < 0 || rowEndI32 <= rowBeginI32) {
        return 0U;
    }
    uint32_t rowBegin = static_cast<uint32_t>(rowBeginI32);
    uint32_t rowEnd = static_cast<uint32_t>(rowEndI32);
    uint32_t rowsProcessed = 0;
    for (uint32_t globalRow = rowBegin + workerId; globalRow < rowEnd; globalRow += workerCount) {
        float routingScale = workspaceView.routingPerTokenScale[globalRow];
        M2ComputeSwigluRowPto(shape, workspaceView, globalRow, routingScale);
        InvalidateGmCacheLines(workspaceView.swigluOut + static_cast<uint64_t>(globalRow) * shape.intermediateSize,
                               static_cast<uint32_t>(shape.intermediateSize * sizeof(float)));
        M2RequantizeSwigluRowPto(shape, workspaceView, globalRow, gmm2RowStride);
        ++rowsProcessed;
    }
    return rowsProcessed;
}

AICORE inline void M3NFinalizeActivationQuant(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                              M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                              uint32_t workerCount, uint32_t activeWorkerMask, uint32_t rowsProcessed)
{
    uint32_t gmm2RowStride = static_cast<uint32_t>(Align64Device(shape.intermediateSize));
    uint32_t localRows = static_cast<uint32_t>(M2LocalRows(shape));
    uint32_t syncGroupCount = static_cast<uint32_t>(LoadScalarI32(workspaceView.swigluSyncGroups));
    if (syncGroupCount == 0U || syncGroupCount > shape.expertPerRank) {
        syncGroupCount = shape.expertPerRank;
    }
    uint32_t tileCount = 0;
    uint32_t skippedGroups = 0;
    uint32_t firstTileBegin = 0;
    uint32_t lastTileEnd = 0;
    uint32_t firstRowBegin = 0;
    uint32_t lastRowEnd = 0;
    bool haveGroup = false;
    for (uint32_t syncIdx = 0; syncIdx < syncGroupCount; ++syncIdx) {
        __gm__ int32_t *group = workspaceView.swigluGroupDesc + syncIdx * kM2SwigluGroupFields;
        int32_t rowBegin = LoadScalarI32(group + 3U);
        int32_t rowEnd = LoadScalarI32(group + 4U);
        int32_t tileBegin = LoadScalarI32(group + 5U);
        int32_t tileEnd = LoadScalarI32(group + 6U);
        if (rowEnd <= rowBegin) {
            ++skippedGroups;
        }
        if (tileEnd > tileBegin) {
            tileCount += static_cast<uint32_t>(tileEnd - tileBegin);
        }
        if (!haveGroup) {
            firstTileBegin = tileBegin > 0 ? static_cast<uint32_t>(tileBegin) : 0U;
            firstRowBegin = rowBegin > 0 ? static_cast<uint32_t>(rowBegin) : 0U;
            haveGroup = true;
        }
        lastTileEnd = tileEnd > 0 ? static_cast<uint32_t>(tileEnd) : 0U;
        lastRowEnd = rowEnd > 0 ? static_cast<uint32_t>(rowEnd) : 0U;
        StoreScalarI32(workspaceView.activationSyncGroupReady + syncIdx * 16U, 1);
    }
    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 0U, static_cast<int32_t>(workerCount));
    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 1U, static_cast<int32_t>(rowsProcessed));
    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 2U, static_cast<int32_t>(tileCount));
    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 3U, static_cast<int32_t>(skippedGroups));
    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 4U, 1);
    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 5U, 0);
    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 6U, 0);
    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 7U, 1);
    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 8U, 0);
    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 9U, static_cast<int32_t>(syncGroupCount));
    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 10U, static_cast<int32_t>(activeWorkerMask));
    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 11U, static_cast<int32_t>(lastRowEnd));
    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 12U, static_cast<int32_t>(firstTileBegin));
    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 13U, static_cast<int32_t>(lastTileEnd));
    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 14U, static_cast<int32_t>(firstRowBegin));
    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 15U, static_cast<int32_t>(workerCount));
    InvalidateGmCacheLines(workspaceView.activationSyncGroupReady,
                           static_cast<uint32_t>(syncGroupCount * 16U * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.swigluOut,
                           static_cast<uint32_t>(localRows * shape.intermediateSize * sizeof(float)));
    InvalidateGmCacheLines(workspaceView.gmm2InputInt8, static_cast<uint32_t>(localRows * gmm2RowStride));
    InvalidateGmCacheLines(workspaceView.gmm2PerTokenScale, static_cast<uint32_t>(localRows * sizeof(float)));
}

AICORE inline void M3N6FinalizeActivationQuantSyncGroup(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                        M2WorkspaceViewDevice workspaceView,
                                                        M2PeerWindowViewDevice localPeer, uint32_t syncIdx,
                                                        uint32_t workerCount, uint32_t activeWorkerMask,
                                                        uint32_t rowsProcessed, bool firstGroup, bool lastGroup)
{
    uint32_t syncGroupCount = static_cast<uint32_t>(LoadScalarI32(workspaceView.swigluSyncGroups));
    if (syncGroupCount == 0U || syncGroupCount > shape.expertPerRank || syncIdx >= syncGroupCount) {
        return;
    }
    __gm__ int32_t *group = workspaceView.swigluGroupDesc + syncIdx * kM2SwigluGroupFields;
    int32_t rowBegin = LoadScalarI32(group + 3U);
    int32_t rowEnd = LoadScalarI32(group + 4U);
    int32_t tileBegin = LoadScalarI32(group + 5U);
    int32_t tileEnd = LoadScalarI32(group + 6U);
    uint32_t tileCount = 0;
    if (tileEnd > tileBegin) {
        tileCount = static_cast<uint32_t>(tileEnd - tileBegin);
    }
    if (firstGroup) {
        StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 0U, static_cast<int32_t>(workerCount));
        StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 1U, 0);
        StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 2U, 0);
        StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 3U, 0);
        StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 4U, 1);
        StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 5U, 0);
        StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 6U, 0);
        StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 7U, 1);
        StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 9U, static_cast<int32_t>(syncGroupCount));
        StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 10U,
                       static_cast<int32_t>(activeWorkerMask));
        StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 12U,
                       tileBegin > 0 ? static_cast<int32_t>(tileBegin) : 0);
        StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 14U,
                       rowBegin > 0 ? static_cast<int32_t>(rowBegin) : 0);
    }
    StoreScalarI32(
        localPeer.debugCounters + kM3NActivationCounterBase + 1U,
        LoadScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 1U) + static_cast<int32_t>(rowsProcessed));
    StoreScalarI32(
        localPeer.debugCounters + kM3NActivationCounterBase + 2U,
        LoadScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 2U) + static_cast<int32_t>(tileCount));
    if (rowEnd <= rowBegin) {
        StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 3U,
                       LoadScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 3U) + 1);
    }
    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 11U,
                   rowEnd > 0 ? static_cast<int32_t>(rowEnd) : 0);
    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 13U,
                   tileEnd > 0 ? static_cast<int32_t>(tileEnd) : 0);
    StoreScalarI32(localPeer.debugCounters + kM3NActivationCounterBase + 15U, static_cast<int32_t>(workerCount));
    StoreScalarI32(workspaceView.activationSyncGroupReady + syncIdx * 16U, 1);
    if (lastGroup) {
        uint32_t gmm2RowStride = static_cast<uint32_t>(Align64Device(shape.intermediateSize));
        uint32_t localRows = static_cast<uint32_t>(M2LocalRows(shape));
        InvalidateGmCacheLines(workspaceView.activationSyncGroupReady,
                               static_cast<uint32_t>(syncGroupCount * 16U * sizeof(int32_t)));
        InvalidateGmCacheLines(workspaceView.swigluOut,
                               static_cast<uint32_t>(localRows * shape.intermediateSize * sizeof(float)));
        InvalidateGmCacheLines(workspaceView.gmm2InputInt8, static_cast<uint32_t>(localRows * gmm2RowStride));
        InvalidateGmCacheLines(workspaceView.gmm2PerTokenScale, static_cast<uint32_t>(localRows * sizeof(float)));
    }
}

AICORE inline void M2RunActivationQuant(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                        M2WorkspaceViewDevice workspaceView, uint32_t myRank)
{
    uint32_t gmm2RowStride = static_cast<uint32_t>(Align64Device(shape.intermediateSize));
    uint32_t localRows = static_cast<uint32_t>(M2LocalRows(shape));
    for (uint32_t row = 0; row < localRows; ++row) {
        workspaceView.gmm2PerTokenScale[row] = 1.0f;
    }
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        int32_t rowBegin = LoadScalarI32(workspaceView.dispatchOffset + localExpert);
        int32_t rowCount = LoadScalarI32(workspaceView.expertTokenNums + localExpert);
        if (rowCount <= 0) {
            continue;
        }
        for (uint32_t row = 0; row < static_cast<uint32_t>(rowCount); ++row) {
            uint32_t globalRow = static_cast<uint32_t>(rowBegin) + row;
            float routingScale = workspaceView.routingPerTokenScale[globalRow];
            M2ComputeSwigluRowPto(shape, workspaceView, globalRow, routingScale);
            InvalidateGmCacheLines(workspaceView.swigluOut + static_cast<uint64_t>(globalRow) * shape.intermediateSize,
                                   static_cast<uint32_t>(shape.intermediateSize * sizeof(float)));
            M2RequantizeSwigluRowPto(shape, workspaceView, globalRow, gmm2RowStride);
        }
    }
    M2BuildSwigluSyncMetadata(shape, workspaceView);
    uint32_t syncGroupCount = static_cast<uint32_t>(LoadScalarI32(workspaceView.swigluSyncGroups));
    if (syncGroupCount == 0U || syncGroupCount > shape.expertPerRank) {
        syncGroupCount = shape.expertPerRank;
    }
    for (uint32_t syncIdx = 0; syncIdx < syncGroupCount; ++syncIdx) {
        StoreScalarI32(workspaceView.activationSyncGroupReady + syncIdx * 16U, 1);
    }
    InvalidateGmCacheLines(workspaceView.swigluOut,
                           static_cast<uint32_t>(localRows * shape.intermediateSize * sizeof(float)));
    InvalidateGmCacheLines(workspaceView.gmm2InputInt8, static_cast<uint32_t>(localRows * gmm2RowStride));
    InvalidateGmCacheLines(workspaceView.gmm2PerTokenScale, static_cast<uint32_t>(localRows * sizeof(float)));
    (void)myRank;
}

__global__ AICORE void M2Int8Dispatch(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                      moe_new_dispatch_combine_a8w8::RankConfig rank, GM_ADDR inputA, GM_ADDR expertIdx,
                                      GM_ADDR xActiveMask, GM_ADDR peerWindow, GM_ADDR hcclCtx, GM_ADDR workspace)
{
    uint32_t blockId = static_cast<uint32_t>(get_block_idx());
    if (blockId != 0 || shape.rankNum == 0 || shape.m == 0 || shape.hiddenSize == 0 || shape.topK == 0 ||
        shape.expertPerRank == 0) {
        return;
    }
    auto workspaceLayout = MakeM2WorkspaceLayoutDevice(shape);
    auto peerWindowLayout = MakeM2PeerWindowLayoutDevice(shape);
    M2WorkspaceViewDevice workspaceView = MakeM2WorkspaceViewDevice(workspace, workspaceLayout);
    M2PeerWindowViewDevice localPeer = MakeM2PeerWindowViewDevice(peerWindow, peerWindowLayout);
    __gm__ HcclDeviceContext *ctx = reinterpret_cast<__gm__ HcclDeviceContext *>(hcclCtx);
    uint32_t rowBytes = static_cast<uint32_t>(peerWindowLayout.dispatchPayloadRowBytes);
    uint32_t myRank = rank.rankId;

    M2ClearDispatchState(shape, workspaceView, localPeer);
    M2CountLocalRoutes(shape, workspaceView, localPeer, expertIdx, xActiveMask, myRank);
    M2RoutePackQuantLocal(shape, workspaceView, localPeer, inputA, expertIdx, xActiveMask, rowBytes);
    M2PublishCountRows(shape, workspaceView, localPeer, ctx, peerWindow, myRank, peerWindowLayout);
    M2WaitCountRows(shape, localPeer, myRank);
    M2BuildPrefixMetadata(shape, workspaceView, localPeer, myRank);
    M2GatherDispatchToGmm1Input(shape, workspaceView, localPeer, ctx, peerWindow, myRank, rowBytes, peerWindowLayout);
    M2BuildSwigluSyncMetadata(shape, workspaceView);
    StoreScalarI32(localPeer.debugCounters, 1);
}

__global__ AICORE void M2Gmm1Epilogue(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                      moe_new_dispatch_combine_a8w8::RankConfig rank, GM_ADDR workspace)
{
    uint32_t blockId = static_cast<uint32_t>(get_block_idx());
    if (blockId != 0 || shape.rankNum == 0 || shape.intermediateSize == 0 || shape.expertPerRank == 0) {
        return;
    }
    auto workspaceLayout = MakeM2WorkspaceLayoutDevice(shape);
    M2WorkspaceViewDevice workspaceView = MakeM2WorkspaceViewDevice(workspace, workspaceLayout);
    M2RunGmm1Epilogue(shape, workspaceView, rank.rankId);
}

__global__ AICORE void M2ActivationQuant(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                         moe_new_dispatch_combine_a8w8::RankConfig rank, GM_ADDR workspace)
{
    uint32_t blockId = static_cast<uint32_t>(get_block_idx());
    if (blockId != 0 || shape.rankNum == 0 || shape.intermediateSize == 0 || shape.expertPerRank == 0) {
        return;
    }
    auto workspaceLayout = MakeM2WorkspaceLayoutDevice(shape);
    M2WorkspaceViewDevice workspaceView = MakeM2WorkspaceViewDevice(workspace, workspaceLayout);
    M2RunActivationQuant(shape, workspaceView, rank.rankId);
}

AICORE inline void M2MaterializeGmm2EpilogueRows(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                 M2WorkspaceViewDevice workspaceView, int32_t srcStart, int32_t rows,
                                                 int32_t returnRowStride)
{
    for (int32_t row = 0; row < rows; ++row) {
        int32_t srcRow = srcStart + row;
        float tokenScale = workspaceView.gmm2PerTokenScale[srcRow];
        __gm__ half *dst = workspaceView.gmm2Out + static_cast<int64_t>(srcRow) * returnRowStride;
        for (uint32_t colBegin = 0; colBegin < shape.hiddenSize; colBegin += kM2EpilogueTileCols) {
            uint32_t cols = shape.hiddenSize - colBegin;
            if (cols > kM2EpilogueTileCols) {
                cols = kM2EpilogueTileCols;
            }
            M2EpilogueAccTile accTile(cols);
            M2EpilogueFloatTile fpTile(cols);
            M2EpilogueFloatTile scaleTile(cols);
            M2EpilogueFloatTile scaledTile(cols);
            M2EpilogueHalfTile halfTile(cols);
            TASSIGN(accTile, kM2EpilogueAccTileOffset);
            TASSIGN(fpTile, kM2EpilogueFloatTileOffset);
            TASSIGN(scaleTile, kM2EpilogueScaleTileOffset);
            TASSIGN(scaledTile, kM2EpilogueScaledTileOffset);
            TASSIGN(halfTile, kM2EpilogueHalfTileOffset);

            GlobalNd<int32_t> accGlobal =
                MakeGlobal2D(workspaceView.gmm2AccInt32 + static_cast<uint64_t>(srcRow) * shape.hiddenSize + colBegin,
                             1, static_cast<int32_t>(cols), static_cast<int32_t>(shape.hiddenSize));
            GlobalNd<half> dstGlobal = MakeGlobal2D(dst + colBegin, 1, static_cast<int32_t>(cols), returnRowStride);
            for (uint32_t col = 0; col < cols; ++col) {
                float scale = M2DecodeUint64Scale(workspaceView.scale2Uint64[colBegin + col]) * tokenScale;
                scaleTile.SetValue(col, scale);
            }
            pipe_barrier(PIPE_ALL);
            TLOAD(accTile, accGlobal);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TCVT(fpTile, accTile, pto::RoundMode::CAST_RINT);
            pipe_barrier(PIPE_V);
            TMUL(scaledTile, fpTile, scaleTile);
            pipe_barrier(PIPE_V);
            TCVT(halfTile, scaledTile, pto::RoundMode::CAST_RINT);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstGlobal, halfTile);
            WaitStoreTileReusable();
        }
        for (int32_t col = static_cast<int32_t>(shape.hiddenSize); col < returnRowStride; ++col) {
            dst[col] = static_cast<half>(0.0);
        }
    }
    InvalidateGmCacheLines(workspaceView.gmm2Out + static_cast<int64_t>(srcStart) * returnRowStride,
                           static_cast<uint32_t>(rows * returnRowStride * sizeof(half)));
}

AICORE inline void M2MaterializeGmm2EpilogueSegment(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                    M2WorkspaceViewDevice workspaceView, int32_t srcStart, int32_t rows,
                                                    int32_t hiddenBegin, int32_t hiddenCount, int32_t returnRowStride)
{
    for (int32_t row = 0; row < rows; ++row) {
        int32_t srcRow = srcStart + row;
        float tokenScale = workspaceView.gmm2PerTokenScale[srcRow];
        M2EpilogueAccTile accTile(static_cast<uint32_t>(hiddenCount));
        M2EpilogueFloatTile fpTile(static_cast<uint32_t>(hiddenCount));
        M2EpilogueFloatTile scaleTile(static_cast<uint32_t>(hiddenCount));
        M2EpilogueFloatTile scaledTile(static_cast<uint32_t>(hiddenCount));
        M2EpilogueHalfTile halfTile(static_cast<uint32_t>(hiddenCount));
        TASSIGN(accTile, kM2EpilogueAccTileOffset);
        TASSIGN(fpTile, kM2EpilogueFloatTileOffset);
        TASSIGN(scaleTile, kM2EpilogueScaleTileOffset);
        TASSIGN(scaledTile, kM2EpilogueScaledTileOffset);
        TASSIGN(halfTile, kM2EpilogueHalfTileOffset);

        GlobalNd<int32_t> accGlobal =
            MakeGlobal2D(workspaceView.gmm2AccInt32 + static_cast<uint64_t>(srcRow) * shape.hiddenSize + hiddenBegin, 1,
                         hiddenCount, static_cast<int32_t>(shape.hiddenSize));
        GlobalNd<half> stagingGlobal = MakeGlobal2D(
            workspaceView.returnSegmentStaging + static_cast<int64_t>(row) * hiddenCount, 1, hiddenCount, hiddenCount);
        GlobalNd<half> gmm2OutGlobal =
            MakeGlobal2D(workspaceView.gmm2Out + static_cast<int64_t>(srcRow) * returnRowStride + hiddenBegin, 1,
                         hiddenCount, returnRowStride);
        pipe_barrier(PIPE_ALL);
        for (int32_t col = 0; col < hiddenCount; ++col) {
            int32_t hiddenCol = hiddenBegin + col;
            float scale = M2DecodeUint64Scale(workspaceView.scale2Uint64[hiddenCol]) * tokenScale;
            scaleTile.SetValue(static_cast<uint32_t>(col), scale);
        }
        pipe_barrier(PIPE_ALL);
        TLOAD(accTile, accGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TCVT(fpTile, accTile, pto::RoundMode::CAST_RINT);
        pipe_barrier(PIPE_V);
        TMUL(scaledTile, fpTile, scaleTile);
        pipe_barrier(PIPE_V);
        TCVT(halfTile, scaledTile, pto::RoundMode::CAST_RINT);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(stagingGlobal, halfTile);
        WaitStoreTileReusable();
        pipe_barrier(PIPE_ALL);
        TSTORE(gmm2OutGlobal, halfTile);
        WaitStoreTileReusable();
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    InvalidateGmCacheLines(workspaceView.returnSegmentStaging,
                           static_cast<uint32_t>(rows * hiddenCount * sizeof(half)));
}

AICORE inline void M3NMaterializeGmm2EpilogueSegmentToGmm2Out(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                              M2WorkspaceViewDevice workspaceView, int32_t srcStart,
                                                              int32_t rows, int32_t hiddenBegin, int32_t hiddenCount,
                                                              int32_t returnRowStride)
{
    if (rows <= 0 || hiddenCount <= 0) {
        return;
    }
    InvalidateGmCacheLines(workspaceView.gmm2PerTokenScale + srcStart, static_cast<uint32_t>(rows * sizeof(float)));
    InvalidateGmCacheLines(workspaceView.scale2Uint64 + hiddenBegin,
                           static_cast<uint32_t>(hiddenCount * sizeof(uint64_t)));
    __gm__ int32_t *accBase =
        workspaceView.gmm2AccInt32 + static_cast<uint64_t>(srcStart) * shape.hiddenSize + hiddenBegin;
    uint32_t accBytes =
        static_cast<uint32_t>(((static_cast<uint64_t>(rows - 1) * shape.hiddenSize) + hiddenCount) * sizeof(int32_t));
    InvalidateGmCacheLines(accBase, accBytes);
    for (int32_t row = 0; row < rows; ++row) {
        int32_t srcRow = srcStart + row;
        float tokenScale = workspaceView.gmm2PerTokenScale[srcRow];
        for (int32_t col = 0; col < hiddenCount; ++col) {
            int32_t hiddenCol = hiddenBegin + col;
            int32_t acc = LoadScalarI32(workspaceView.gmm2AccInt32 + static_cast<uint64_t>(srcRow) * shape.hiddenSize +
                                        hiddenCol);
            float scale = M2DecodeUint64Scale(workspaceView.scale2Uint64[hiddenCol]) * tokenScale;
            workspaceView.gmm2Out[static_cast<int64_t>(srcRow) * returnRowStride + hiddenCol] =
                static_cast<half>(static_cast<float>(acc) * scale);
        }
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

AICORE inline void M2WaitGmm2GroupReadyGm(M2WorkspaceViewDevice workspaceView, uint32_t localExpert)
{
    __gm__ int32_t *ready = workspaceView.gmm2GroupReady + localExpert * 16U;
    while (true) {
        pipe_barrier(PIPE_ALL);
        dcci(static_cast<__gm__ void *>(ready), SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
        if (LoadScalarI32(ready) != 0) {
            return;
        }
    }
}

AICORE inline uint32_t M2BuildReturnSegmentMap(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                               M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                               uint32_t myRank)
{
    uint32_t tileRows = shape.gmmBlockM == 0 ? kM2ReturnTileRows : shape.gmmBlockM;
    uint32_t hiddenChunk = static_cast<uint32_t>(M2ReturnHiddenChunkCols(shape));
    uint32_t capacity = static_cast<uint32_t>(M2ReturnSegmentCapacity(shape));
    uint32_t tileId = 0;
    uint32_t segmentId = 0;
    uint32_t maxSegmentsPerTile = 0;
    uint32_t overflow = 0;
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        uint32_t globalExpert = M2GlobalExpert(myRank, localExpert, shape);
        int32_t rowBegin = LoadScalarI32(workspaceView.dispatchOffset + localExpert);
        int32_t rowCount = LoadScalarI32(workspaceView.expertTokenNums + localExpert);
        if (rowCount <= 0) {
            continue;
        }
        for (int32_t tileOffset = 0; tileOffset < rowCount; tileOffset += static_cast<int32_t>(tileRows)) {
            int32_t tileStart = rowBegin + tileOffset;
            int32_t tileCount = rowCount - tileOffset;
            if (tileCount > static_cast<int32_t>(tileRows)) {
                tileCount = static_cast<int32_t>(tileRows);
            }
            for (uint32_t hiddenBegin = 0; hiddenBegin < shape.hiddenSize; hiddenBegin += hiddenChunk) {
                uint32_t hiddenCount = shape.hiddenSize - hiddenBegin;
                if (hiddenCount > hiddenChunk) {
                    hiddenCount = hiddenChunk;
                }
                uint32_t firstSegment = segmentId;
                uint32_t tileSegmentCount = 0;
                for (uint32_t tokenOwner = 0; tokenOwner < shape.rankNum; ++tokenOwner) {
                    int32_t current =
                        LoadScalarI32(workspaceView.cumsumMM + tokenOwner * shape.expertPerRank + localExpert);
                    int32_t previous = tokenOwner == 0U ?
                                           0 :
                                           LoadScalarI32(workspaceView.cumsumMM +
                                                         (tokenOwner - 1U) * shape.expertPerRank + localExpert);
                    int32_t ownerRows = current - previous;
                    if (ownerRows <= 0) {
                        continue;
                    }
                    int32_t ownerStart =
                        LoadScalarI32(workspaceView.dispatchOffset + localExpert) +
                        LoadScalarI32(workspaceView.preSumBeforeRank + tokenOwner * shape.expertPerRank + localExpert);
                    int32_t ownerEnd = ownerStart + ownerRows;
                    int32_t tileEnd = tileStart + tileCount;
                    int32_t segmentStart = tileStart > ownerStart ? tileStart : ownerStart;
                    int32_t segmentEnd = tileEnd < ownerEnd ? tileEnd : ownerEnd;
                    if (segmentEnd <= segmentStart) {
                        continue;
                    }
                    int32_t dstStart = M2SourceGlobalExpertRowBase(shape, localPeer, tokenOwner, globalExpert) +
                                       segmentStart - ownerStart;
                    if (segmentId < capacity) {
                        __gm__ int32_t *segment =
                            workspaceView.subTileOwnerSegments + segmentId * kM2OwnerSegmentFields;
                        StoreScalarI32(segment + 0U, static_cast<int32_t>(tokenOwner));
                        StoreScalarI32(segment + 1U, segmentStart);
                        StoreScalarI32(segment + 2U, segmentEnd - segmentStart);
                        StoreScalarI32(segment + 3U, dstStart);
                        StoreScalarI32(segment + 4U, static_cast<int32_t>(hiddenBegin));
                        StoreScalarI32(segment + 5U, static_cast<int32_t>(hiddenCount));
                        StoreScalarI32(segment + 6U, static_cast<int32_t>(localExpert));
                        StoreScalarI32(segment + 7U, static_cast<int32_t>(tileId));
                    } else {
                        ++overflow;
                    }
                    ++segmentId;
                    ++tileSegmentCount;
                }
                if (tileId < capacity) {
                    __gm__ int32_t *plan = workspaceView.subTileReturnPlan + tileId * kM2ReturnPlanFields;
                    StoreScalarI32(plan + 0U, static_cast<int32_t>(tileId));
                    StoreScalarI32(plan + 1U, static_cast<int32_t>(localExpert));
                    StoreScalarI32(plan + 2U, tileStart);
                    StoreScalarI32(plan + 3U, tileCount);
                    StoreScalarI32(plan + 4U, static_cast<int32_t>(hiddenBegin));
                    StoreScalarI32(plan + 5U, static_cast<int32_t>(hiddenCount));
                    StoreScalarI32(plan + 6U, static_cast<int32_t>(firstSegment));
                    StoreScalarI32(plan + 7U, static_cast<int32_t>(tileSegmentCount));
                    StoreScalarI32(workspaceView.subTileReady + tileId * 16U, 1);
                } else {
                    ++overflow;
                }
                if (tileSegmentCount > maxSegmentsPerTile) {
                    maxSegmentsPerTile = tileSegmentCount;
                }
                ++tileId;
            }
        }
    }
    StoreScalarI32(localPeer.debugCounters + 2U * 16U, static_cast<int32_t>(tileId));
    StoreScalarI32(localPeer.debugCounters + 3U * 16U, static_cast<int32_t>(segmentId));
    StoreScalarI32(localPeer.debugCounters + 4U * 16U, static_cast<int32_t>(maxSegmentsPerTile));
    StoreScalarI32(localPeer.debugCounters + 5U * 16U, static_cast<int32_t>(overflow));
    return segmentId;
}

AICORE inline void M2DebugProbeReturnSegmentMap(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                                uint32_t myRank, uint32_t debugStopStage)
{
    if (shape.expertPerRank == 0U) {
        return;
    }
    constexpr uint32_t kProbeBase = moe_new_dispatch_combine_a8w8::kM3OCombineProbeCounterBase;
    uint32_t localExpert = 0U;
    StoreScalarI32(localPeer.debugCounters + kProbeBase + 0U, static_cast<int32_t>(debugStopStage));
    if (debugStopStage == 59U) {
        M2WaitGmm2GroupReadyGm(workspaceView, localExpert);
        StoreScalarI32(localPeer.debugCounters + kProbeBase + 1U,
                       LoadScalarI32(workspaceView.gmm2GroupReady + localExpert * 16U));
        return;
    }

    uint32_t globalExpert = M2GlobalExpert(myRank, localExpert, shape);
    int32_t rowBegin = LoadScalarI32(workspaceView.dispatchOffset + localExpert);
    int32_t rowCount = LoadScalarI32(workspaceView.expertTokenNums + localExpert);
    int32_t current = LoadScalarI32(workspaceView.cumsumMM + localExpert);
    int32_t preSum = LoadScalarI32(workspaceView.preSumBeforeRank + localExpert);
    StoreScalarI32(localPeer.debugCounters + kProbeBase + 1U, rowBegin);
    StoreScalarI32(localPeer.debugCounters + kProbeBase + 2U, rowCount);
    StoreScalarI32(localPeer.debugCounters + kProbeBase + 3U, current);
    StoreScalarI32(localPeer.debugCounters + kProbeBase + 4U, preSum);
    if (debugStopStage == 60U || rowCount <= 0) {
        return;
    }

    uint32_t hiddenChunk = static_cast<uint32_t>(M2ReturnHiddenChunkCols(shape));
    uint32_t hiddenCount = shape.hiddenSize < hiddenChunk ? shape.hiddenSize : hiddenChunk;
    int32_t segmentStart = rowBegin;
    int32_t segmentRows = rowCount;
    uint32_t tileRows = shape.gmmBlockM == 0 ? kM2ReturnTileRows : shape.gmmBlockM;
    if (segmentRows > static_cast<int32_t>(tileRows)) {
        segmentRows = static_cast<int32_t>(tileRows);
    }
    int32_t dstStart = 0;
    if (debugStopStage == 62U) {
        dstStart = M2SourceGlobalExpertRowBase(shape, localPeer, 0U, globalExpert);
    }
    __gm__ int32_t *segment = workspaceView.subTileOwnerSegments;
    if (debugStopStage == 66U) {
        StoreScalarI32(segment + 0U, 0);
        return;
    }
    StoreScalarI32(segment + 0U, 0);
    StoreScalarI32(segment + 1U, segmentStart);
    StoreScalarI32(segment + 2U, segmentRows);
    StoreScalarI32(segment + 3U, dstStart);
    StoreScalarI32(segment + 4U, 0);
    StoreScalarI32(segment + 5U, static_cast<int32_t>(hiddenCount));
    StoreScalarI32(segment + 6U, static_cast<int32_t>(localExpert));
    StoreScalarI32(segment + 7U, 0);
    if (debugStopStage == 63U) {
        return;
    }
    __gm__ int32_t *plan = workspaceView.subTileReturnPlan;
    if (debugStopStage == 67U) {
        StoreScalarI32(plan + 0U, 0);
        return;
    }
    StoreScalarI32(plan + 0U, 0);
    StoreScalarI32(plan + 1U, static_cast<int32_t>(localExpert));
    StoreScalarI32(plan + 2U, rowBegin);
    StoreScalarI32(plan + 3U, segmentRows);
    StoreScalarI32(plan + 4U, 0);
    StoreScalarI32(plan + 5U, static_cast<int32_t>(hiddenCount));
    StoreScalarI32(plan + 6U, 0);
    StoreScalarI32(plan + 7U, 1);
    if (debugStopStage == 64U) {
        return;
    }
    StoreScalarI32(workspaceView.subTileReady, 1);
    if (debugStopStage == 65U) {
        return;
    }
    StoreScalarI32(localPeer.debugCounters + kProbeBase + 5U, dstStart);
}

AICORE inline void M3N8WaitGmm2ExpertReadyGm(M2WorkspaceViewDevice workspaceView, uint32_t localExpert)
{
    M2WaitGmm2GroupReadyGm(workspaceView, localExpert);
}

AICORE inline uint32_t M3N8BuildReturnSegmentMapForExpert(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                          M2WorkspaceViewDevice workspaceView,
                                                          M2PeerWindowViewDevice localPeer, uint32_t myRank,
                                                          uint32_t localExpert, uint32_t *tileId, uint32_t *segmentId,
                                                          uint32_t *maxSegmentsPerTile, uint32_t *overflow)
{
    uint32_t tileRows = shape.gmmBlockM == 0 ? kM2ReturnTileRows : shape.gmmBlockM;
    uint32_t hiddenChunk = static_cast<uint32_t>(M2ReturnHiddenChunkCols(shape));
    uint32_t capacity = static_cast<uint32_t>(M2ReturnSegmentCapacity(shape));
    uint32_t firstExpertSegment = *segmentId;
    uint32_t globalExpert = M2GlobalExpert(myRank, localExpert, shape);
    int32_t rowBegin = LoadScalarI32(workspaceView.dispatchOffset + localExpert);
    int32_t rowCount = LoadScalarI32(workspaceView.expertTokenNums + localExpert);
    if (rowCount <= 0) {
        return 0;
    }
    for (int32_t tileOffset = 0; tileOffset < rowCount; tileOffset += static_cast<int32_t>(tileRows)) {
        int32_t tileStart = rowBegin + tileOffset;
        int32_t tileCount = rowCount - tileOffset;
        if (tileCount > static_cast<int32_t>(tileRows)) {
            tileCount = static_cast<int32_t>(tileRows);
        }
        for (uint32_t hiddenBegin = 0; hiddenBegin < shape.hiddenSize; hiddenBegin += hiddenChunk) {
            uint32_t hiddenCount = shape.hiddenSize - hiddenBegin;
            if (hiddenCount > hiddenChunk) {
                hiddenCount = hiddenChunk;
            }
            uint32_t firstSegment = *segmentId;
            uint32_t tileSegmentCount = 0;
            for (uint32_t tokenOwner = 0; tokenOwner < shape.rankNum; ++tokenOwner) {
                int32_t current =
                    LoadScalarI32(workspaceView.cumsumMM + tokenOwner * shape.expertPerRank + localExpert);
                int32_t previous =
                    tokenOwner == 0U ?
                        0 :
                        LoadScalarI32(workspaceView.cumsumMM + (tokenOwner - 1U) * shape.expertPerRank + localExpert);
                int32_t ownerRows = current - previous;
                if (ownerRows <= 0) {
                    continue;
                }
                int32_t ownerStart =
                    LoadScalarI32(workspaceView.dispatchOffset + localExpert) +
                    LoadScalarI32(workspaceView.preSumBeforeRank + tokenOwner * shape.expertPerRank + localExpert);
                int32_t ownerEnd = ownerStart + ownerRows;
                int32_t tileEnd = tileStart + tileCount;
                int32_t segmentStart = tileStart > ownerStart ? tileStart : ownerStart;
                int32_t segmentEnd = tileEnd < ownerEnd ? tileEnd : ownerEnd;
                if (segmentEnd <= segmentStart) {
                    continue;
                }
                int32_t dstStart =
                    M2SourceGlobalExpertRowBase(shape, localPeer, tokenOwner, globalExpert) + segmentStart - ownerStart;
                if (*segmentId < capacity) {
                    __gm__ int32_t *segment = workspaceView.subTileOwnerSegments + *segmentId * kM2OwnerSegmentFields;
                    StoreScalarI32(segment + 0U, static_cast<int32_t>(tokenOwner));
                    StoreScalarI32(segment + 1U, segmentStart);
                    StoreScalarI32(segment + 2U, segmentEnd - segmentStart);
                    StoreScalarI32(segment + 3U, dstStart);
                    StoreScalarI32(segment + 4U, static_cast<int32_t>(hiddenBegin));
                    StoreScalarI32(segment + 5U, static_cast<int32_t>(hiddenCount));
                    StoreScalarI32(segment + 6U, static_cast<int32_t>(localExpert));
                    StoreScalarI32(segment + 7U, static_cast<int32_t>(*tileId));
                } else {
                    ++(*overflow);
                }
                ++(*segmentId);
                ++tileSegmentCount;
            }
            if (*tileId < capacity) {
                __gm__ int32_t *plan = workspaceView.subTileReturnPlan + *tileId * kM2ReturnPlanFields;
                StoreScalarI32(plan + 0U, static_cast<int32_t>(*tileId));
                StoreScalarI32(plan + 1U, static_cast<int32_t>(localExpert));
                StoreScalarI32(plan + 2U, tileStart);
                StoreScalarI32(plan + 3U, tileCount);
                StoreScalarI32(plan + 4U, static_cast<int32_t>(hiddenBegin));
                StoreScalarI32(plan + 5U, static_cast<int32_t>(hiddenCount));
                StoreScalarI32(plan + 6U, static_cast<int32_t>(firstSegment));
                StoreScalarI32(plan + 7U, static_cast<int32_t>(tileSegmentCount));
                StoreScalarI32(workspaceView.subTileReady + *tileId * 16U, 1);
            } else {
                ++(*overflow);
            }
            if (tileSegmentCount > *maxSegmentsPerTile) {
                *maxSegmentsPerTile = tileSegmentCount;
            }
            ++(*tileId);
        }
    }
    return *segmentId - firstExpertSegment;
}

AICORE inline int32_t M2ExpectedReturnSegmentCount(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                   M2PeerWindowViewDevice localPeer, uint32_t myRank,
                                                   uint32_t expertOwner, uint32_t localExpert)
{
    uint32_t tileRows = shape.gmmBlockM == 0 ? kM2ReturnTileRows : shape.gmmBlockM;
    uint32_t hiddenChunks = static_cast<uint32_t>(M2CeilDivDevice(shape.hiddenSize, M2ReturnHiddenChunkCols(shape)));
    int32_t ownerRows = M2EffectiveTokenOwnerRows(shape, localPeer, myRank, expertOwner, localExpert);
    if (ownerRows <= 0) {
        return 0;
    }
    int32_t rowBegin = 0;
    for (uint32_t prevLocalExpert = 0; prevLocalExpert < localExpert; ++prevLocalExpert) {
        for (uint32_t tokenOwner = 0; tokenOwner < shape.rankNum; ++tokenOwner) {
            rowBegin += M2EffectiveTokenOwnerRows(shape, localPeer, tokenOwner, expertOwner, prevLocalExpert);
        }
    }
    int32_t ownerPrefix = 0;
    for (uint32_t tokenOwner = 0; tokenOwner < myRank; ++tokenOwner) {
        ownerPrefix += M2EffectiveTokenOwnerRows(shape, localPeer, tokenOwner, expertOwner, localExpert);
    }
    int32_t rowCount = 0;
    for (uint32_t tokenOwner = 0; tokenOwner < shape.rankNum; ++tokenOwner) {
        rowCount += M2EffectiveTokenOwnerRows(shape, localPeer, tokenOwner, expertOwner, localExpert);
    }
    int32_t ownerStart = rowBegin + ownerPrefix;
    int32_t ownerEnd = ownerStart + ownerRows;
    int32_t expected = 0;
    for (int32_t tileOffset = 0; tileOffset < rowCount; tileOffset += static_cast<int32_t>(tileRows)) {
        int32_t tileStart = rowBegin + tileOffset;
        int32_t tileCount = rowCount - tileOffset;
        if (tileCount > static_cast<int32_t>(tileRows)) {
            tileCount = static_cast<int32_t>(tileRows);
        }
        int32_t tileEnd = tileStart + tileCount;
        int32_t segmentStart = tileStart > ownerStart ? tileStart : ownerStart;
        int32_t segmentEnd = tileEnd < ownerEnd ? tileEnd : ownerEnd;
        if (segmentEnd > segmentStart) {
            expected += static_cast<int32_t>(hiddenChunks);
        }
    }
    return expected;
}

AICORE inline int32_t M2WaitReturnSegmentCounters(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                  M2PeerWindowViewDevice localPeer, uint32_t myRank)
{
    int32_t expectedTotal = 0;
    for (uint32_t expertOwner = 0; expertOwner < shape.rankNum; ++expertOwner) {
        if (expertOwner == myRank) {
            continue;
        }
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            int32_t expected = M2ExpectedReturnSegmentCount(shape, localPeer, myRank, expertOwner, localExpert);
            if (expected <= 0) {
                continue;
            }
            __gm__ int32_t *counter = localPeer.returnSegmentCounters +
                                      (static_cast<uint64_t>(expertOwner) * shape.expertPerRank + localExpert) * 16U;
            while (true) {
                InvalidateGmCacheLines(counter, sizeof(int32_t));
                if (LoadScalarI32(counter) >= expected) {
                    break;
                }
            }
            expectedTotal += expected;
        }
    }
    return expectedTotal;
}

AICORE inline void M2PublishReturnSegmentCountersScalar(
    moe_new_dispatch_combine_a8w8::ShapeConfig shape, M2PeerWindowViewDevice localPeer, __gm__ HcclDeviceContext *ctx,
    GM_ADDR peerWindow, uint32_t myRank, const moe_new_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout)
{
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    for (uint32_t tokenOwner = 0; tokenOwner < shape.rankNum; ++tokenOwner) {
        if (tokenOwner == myRank) {
            continue;
        }
        M2PeerWindowViewDevice remotePeer =
            MakeM2RemotePeerWindowViewDevice(ctx, peerWindow, tokenOwner, peerWindowLayout);
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            int32_t expected = M2ExpectedReturnSegmentCount(shape, localPeer, tokenOwner, myRank, localExpert);
            if (expected <= 0) {
                continue;
            }
            __gm__ int32_t *counter = remotePeer.returnSegmentCounters +
                                      (static_cast<uint64_t>(myRank) * shape.expertPerRank + localExpert) * 16U;
            StoreScalarI32(counter, expected);
            InvalidateGmCacheLines(counter, sizeof(int32_t));
        }
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

AICORE inline int32_t LoadHalfBitsI32(__gm__ half *ptr)
{
    return static_cast<int32_t>(*reinterpret_cast<__gm__ uint16_t *>(ptr));
}

AICORE inline void M2RecordReturnSendTrace(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                           M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                           uint32_t tokenOwner, uint32_t localExpert, uint32_t segmentId,
                                           uint32_t tileId, int32_t srcStart, int32_t rows, int32_t dstStart,
                                           int32_t hiddenBegin, int32_t hiddenCount, __gm__ half *firstValue)
{
    constexpr uint32_t kTraceBase = 8U * 16U;
    uint32_t traceSlot = kTraceBase + (tokenOwner * shape.expertPerRank + localExpert) * 16U;
    __gm__ int32_t *trace = localPeer.debugCounters + traceSlot;
    int32_t count = LoadScalarI32(trace);
    (void)workspaceView;
    int32_t firstBits = (rows > 0 && hiddenCount > 0 && firstValue != nullptr) ? LoadHalfBitsI32(firstValue) : 0;
    if (count == 0) {
        StoreScalarI32(trace + 1U, static_cast<int32_t>(segmentId));
        StoreScalarI32(trace + 2U, static_cast<int32_t>(tileId));
        StoreScalarI32(trace + 3U, srcStart);
        StoreScalarI32(trace + 4U, rows);
        StoreScalarI32(trace + 5U, dstStart);
        StoreScalarI32(trace + 6U, hiddenBegin);
        StoreScalarI32(trace + 7U, hiddenCount);
        StoreScalarI32(trace + 8U, firstBits);
    }
    StoreScalarI32(trace + 9U, static_cast<int32_t>(segmentId));
    StoreScalarI32(trace + 10U, srcStart);
    StoreScalarI32(trace + 11U, rows);
    StoreScalarI32(trace + 12U, dstStart);
    StoreScalarI32(trace + 13U, hiddenBegin);
    StoreScalarI32(trace + 14U, hiddenCount);
    StoreScalarI32(trace + 15U, firstBits);
    StoreScalarI32(trace, count + 1);
}

AICORE inline void M2NotifyReturnSegmentCounter(M2PeerWindowViewDevice remotePeer, uint32_t myRank, uint32_t tokenOwner,
                                                uint32_t localExpert, moe_new_dispatch_combine_a8w8::ShapeConfig shape)
{
    (void)remotePeer;
    (void)myRank;
    (void)tokenOwner;
    (void)localExpert;
    (void)shape;
}

AICORE inline void M2NotifyCombineDone(M2PeerWindowViewDevice remotePeer, M2PeerWindowViewDevice localPeer,
                                       uint32_t myRank, uint32_t tokenOwner)
{
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    if (tokenOwner == myRank) {
        __gm__ int32_t *signal = localPeer.combineDoneSignal + myRank * 16U;
        StoreScalarI32(signal, 1);
        InvalidateGmCacheLines(signal, sizeof(int32_t));
        dsb(DSB_DDR);
        return;
    }
    __gm__ int32_t *signal = remotePeer.combineDoneSignal + myRank * 16U;
    StoreScalarI32(signal, 1);
    InvalidateGmCacheLines(signal, sizeof(int32_t));
    dsb(DSB_DDR);
}

AICORE inline void M2WaitCombineDonePeers(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                          M2PeerWindowViewDevice localPeer, uint32_t myRank)
{
    for (uint32_t peer = 0; peer < shape.rankNum; ++peer) {
        if (peer == myRank) {
            continue;
        }
        __gm__ int32_t *signal = localPeer.combineDoneSignal + peer * 16U;
        while (true) {
            InvalidateGmCacheLines(signal, sizeof(int32_t));
            if (LoadScalarI32(signal) >= 1) {
                break;
            }
        }
    }
}

AICORE inline void M2RunGmm2EpilogueAndReturn(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                              M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                              __gm__ HcclDeviceContext *ctx, GM_ADDR peerWindow, uint32_t myRank,
                                              const moe_new_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout)
{
    int32_t returnRowStride = static_cast<int32_t>(peerWindowLayout.returnPayloadRowBytes / sizeof(half));
    uint32_t mappedSegments = M2BuildReturnSegmentMap(shape, workspaceView, localPeer, myRank);
    uint32_t tileCount = static_cast<uint32_t>(LoadScalarI32(localPeer.debugCounters + 2U * 16U));
    int32_t segmentCount = 0;
    for (uint32_t tileId = 0; tileId < tileCount; ++tileId) {
        __gm__ int32_t *plan = workspaceView.subTileReturnPlan + tileId * kM2ReturnPlanFields;
        uint32_t firstSegment = static_cast<uint32_t>(LoadScalarI32(plan + 6U));
        uint32_t tileSegmentCount = static_cast<uint32_t>(LoadScalarI32(plan + 7U));
        for (uint32_t localSegment = 0; localSegment < tileSegmentCount; ++localSegment) {
            uint32_t segmentId = firstSegment + localSegment;
            if (segmentId >= mappedSegments) {
                continue;
            }
            __gm__ int32_t *segment = workspaceView.subTileOwnerSegments + segmentId * kM2OwnerSegmentFields;
            uint32_t tokenOwner = static_cast<uint32_t>(LoadScalarI32(segment + 0U));
            int32_t srcStart = LoadScalarI32(segment + 1U);
            int32_t rows = LoadScalarI32(segment + 2U);
            int32_t dstStart = LoadScalarI32(segment + 3U);
            int32_t hiddenBegin = LoadScalarI32(segment + 4U);
            int32_t hiddenCount = LoadScalarI32(segment + 5U);
            uint32_t localExpert = static_cast<uint32_t>(LoadScalarI32(segment + 6U));
            if (rows <= 0 || hiddenCount <= 0 || tokenOwner >= shape.rankNum) {
                continue;
            }
            M2MaterializeGmm2EpilogueSegment(shape, workspaceView, srcStart, rows, hiddenBegin, hiddenCount,
                                             returnRowStride);
            M2RecordReturnSendTrace(shape, workspaceView, localPeer, tokenOwner, localExpert, segmentId, tileId,
                                    srcStart, rows, dstStart, hiddenBegin, hiddenCount,
                                    workspaceView.returnSegmentStaging);
            M2PeerWindowViewDevice remotePeer =
                MakeM2RemotePeerWindowViewDevice(ctx, peerWindow, tokenOwner, peerWindowLayout);
            if (tokenOwner == myRank) {
                for (int32_t row = 0; row < rows; ++row) {
                    CopyRowHalf(localPeer.returnPayload + hiddenBegin, returnRowStride, dstStart + row,
                                workspaceView.returnSegmentStaging, hiddenCount, row, hiddenCount);
                }
            } else {
                TPutRowsHalfContiguous(remotePeer.returnPayload + hiddenBegin, returnRowStride, dstStart,
                                       workspaceView.returnSegmentStaging, hiddenCount, 0, rows, hiddenCount);
            }
            if (tokenOwner != myRank) {
                M2NotifyReturnSegmentCounter(remotePeer, myRank, tokenOwner, localExpert, shape);
            }
            ++segmentCount;
        }
    }
    for (uint32_t tokenOwner = 0; tokenOwner < shape.rankNum; ++tokenOwner) {
        M2PeerWindowViewDevice remotePeer =
            MakeM2RemotePeerWindowViewDevice(ctx, peerWindow, tokenOwner, peerWindowLayout);
        M2NotifyCombineDone(remotePeer, localPeer, myRank, tokenOwner);
    }
    M2PublishReturnSegmentCountersScalar(shape, localPeer, ctx, peerWindow, myRank, peerWindowLayout);
    M2WaitCombineDonePeers(shape, localPeer, myRank);
    int32_t expectedReturnSegments = M2WaitReturnSegmentCounters(shape, localPeer, myRank);
    InvalidateGmCacheLines(localPeer.returnPayload,
                           static_cast<uint32_t>(shape.m * shape.topK * returnRowStride * sizeof(half)));
    StoreScalarI32(localPeer.debugCounters, segmentCount);
    StoreScalarI32(localPeer.debugCounters + 16U, 1);
    StoreScalarI32(localPeer.debugCounters + 6U * 16U, expectedReturnSegments);
}

struct M3NCombineShardStats {
    int32_t segmentCount;
    int32_t skippedSegmentCount;
    int32_t subtileCount;
    int32_t remoteSubtileCount;
    int32_t localSubtileCount;
    int32_t readyWaitCount;
};

AICORE inline __gm__ int32_t *M3NCombineWorkerScratch(M2PeerWindowViewDevice localPeer, uint32_t workerId)
{
    return localPeer.debugCounters + kM3NCombineWorkerScratchBase + workerId * kM3NCombineWorkerScratchStride;
}

AICORE inline void M3NInitCombineCounters(M2PeerWindowViewDevice localPeer, uint32_t workerCount)
{
    for (uint32_t idx = 0; idx < 16U; ++idx) {
        StoreScalarI32(localPeer.debugCounters + kM3NCombineCounterBase + idx, 0);
        StoreScalarI32(localPeer.debugCounters + kM3N11SubtileCounterBase + idx, 0);
    }
    for (uint32_t worker = 0; worker < kM3NDispatchMaxWorkers; ++worker) {
        __gm__ int32_t *scratch = M3NCombineWorkerScratch(localPeer, worker);
        for (uint32_t slot = 0; slot < kM3NCombineWorkerScratchStride; ++slot) {
            StoreScalarI32(scratch + slot, 0);
        }
        StoreScalarI32(scratch + 2U, static_cast<int32_t>(worker));
        InvalidateGmCacheLines(scratch, 64U);
    }
    StoreScalarI32(localPeer.debugCounters + kM3NCombineCounterBase + 0U, static_cast<int32_t>(workerCount));
    StoreScalarI32(localPeer.debugCounters + kM3NCombineCounterBase + 7U, 1);
    StoreScalarI32(localPeer.debugCounters + kM3N11SubtileCounterBase + 0U, 1);
    StoreScalarI32(localPeer.debugCounters + kM3N11SubtileCounterBase + 1U, static_cast<int32_t>(kM3N11SubtileRows));
}

AICORE inline void M3N8AivOnlyPhaseSync()
{
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
}

AICORE inline void M3N11WaitSubTileReadyGm(M2WorkspaceViewDevice workspaceView, uint32_t tileId)
{
    __gm__ int32_t *ready = workspaceView.subTileReady + tileId * 16U;
    while (true) {
        pipe_barrier(PIPE_ALL);
        dcci(static_cast<__gm__ void *>(ready), SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
        if (LoadScalarI32(ready) != 0) {
            return;
        }
    }
}

AICORE inline int32_t M3N11TransferHalfSubtileStride(__gm__ half *dstBase, int32_t dstRowStride, int32_t dstRow,
                                                     __gm__ half *srcBase, int32_t srcRowStride, int32_t srcRow,
                                                     int32_t rows, int32_t cols, bool remote)
{
    int32_t subtileCount = 0;
    for (int32_t rowOffset = 0; rowOffset < rows; rowOffset += static_cast<int32_t>(kM3N11SubtileRows)) {
        int32_t subtileRows = rows - rowOffset;
        if (subtileRows > static_cast<int32_t>(kM3N11SubtileRows)) {
            subtileRows = static_cast<int32_t>(kM3N11SubtileRows);
        }
        if (remote) {
            for (int32_t row = 0; row < subtileRows; ++row) {
                __gm__ half *dst = dstBase + static_cast<int64_t>(dstRow + rowOffset + row) * dstRowStride;
                __gm__ half *src = srcBase + static_cast<int64_t>(srcRow + rowOffset + row) * srcRowStride;
                for (int32_t col = 0; col < cols; ++col) {
                    dst[col] = src[col];
                }
                InvalidateGmCacheLines(dst, static_cast<uint32_t>(cols * sizeof(half)));
            }
        } else {
            for (int32_t row = 0; row < subtileRows; ++row) {
                __gm__ half *dst = dstBase + static_cast<int64_t>(dstRow + rowOffset + row) * dstRowStride;
                __gm__ half *src = srcBase + static_cast<int64_t>(srcRow + rowOffset + row) * srcRowStride;
                for (int32_t col = 0; col < cols; ++col) {
                    dst[col] = src[col];
                }
            }
        }
        ++subtileCount;
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    return subtileCount;
}

AICORE inline bool M3NDebugProbeGmm2Materialize(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                                M2WorkspaceViewDevice workspaceView, M2PeerWindowViewDevice localPeer,
                                                int32_t srcStart, int32_t rows, int32_t hiddenBegin,
                                                int32_t hiddenCount, int32_t returnRowStride, uint32_t debugStopStage)
{
    if (debugStopStage < 68U || debugStopStage > 79U) {
        return false;
    }
    constexpr uint32_t kProbeBase = moe_new_dispatch_combine_a8w8::kM3OCombineProbeCounterBase;
    int32_t srcRow = srcStart;
    StoreScalarI32(localPeer.debugCounters + kProbeBase + 0U, static_cast<int32_t>(debugStopStage));
    StoreScalarI32(localPeer.debugCounters + kProbeBase + 1U, srcStart);
    StoreScalarI32(localPeer.debugCounters + kProbeBase + 2U, rows);
    StoreScalarI32(localPeer.debugCounters + kProbeBase + 3U, hiddenCount);
    if (debugStopStage == 68U) {
        return true;
    }
    float tokenScale = workspaceView.gmm2PerTokenScale[srcRow];
    StoreScalarI32(localPeer.debugCounters + kProbeBase + 4U, tokenScale == tokenScale ? 1 : 0);
    if (debugStopStage == 69U) {
        return true;
    }
    int32_t acc =
        LoadScalarI32(workspaceView.gmm2AccInt32 + static_cast<uint64_t>(srcRow) * shape.hiddenSize + hiddenBegin);
    StoreScalarI32(localPeer.debugCounters + kProbeBase + 5U, acc);
    if (debugStopStage == 70U) {
        return true;
    }
    float scale = M2DecodeUint64Scale(workspaceView.scale2Uint64[hiddenBegin]) * tokenScale;
    StoreScalarI32(localPeer.debugCounters + kProbeBase + 6U, scale == scale ? 1 : 0);
    if (debugStopStage == 71U) {
        return true;
    }
    __gm__ half *dst = workspaceView.gmm2Out + static_cast<int64_t>(srcRow) * returnRowStride + hiddenBegin;
    dst[0] = static_cast<half>(0.0);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    if (debugStopStage == 72U) {
        return true;
    }
    dst[0] = static_cast<half>(static_cast<float>(acc) * scale);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    if (debugStopStage == 73U) {
        return true;
    }
    for (int32_t col = 0; col < hiddenCount; ++col) {
        dst[col] = static_cast<half>(0.0);
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    if (debugStopStage == 74U) {
        return true;
    }
    for (int32_t col = 0; col < hiddenCount; ++col) {
        int32_t hiddenCol = hiddenBegin + col;
        int32_t colAcc =
            LoadScalarI32(workspaceView.gmm2AccInt32 + static_cast<uint64_t>(srcRow) * shape.hiddenSize + hiddenCol);
        float colScale = M2DecodeUint64Scale(workspaceView.scale2Uint64[hiddenCol]) * tokenScale;
        dst[col] = static_cast<half>(static_cast<float>(colAcc) * colScale);
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    if (debugStopStage == 75U) {
        return true;
    }
    for (int32_t row = 0; row < rows; ++row) {
        __gm__ half *rowDst =
            workspaceView.gmm2Out + static_cast<int64_t>(srcStart + row) * returnRowStride + hiddenBegin;
        for (int32_t col = 0; col < hiddenCount; ++col) {
            rowDst[col] = static_cast<half>(0.0);
        }
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    if (debugStopStage == 76U) {
        return true;
    }
    for (int32_t row = 0; row < rows; ++row) {
        int32_t rowIdx = srcStart + row;
        float rowTokenScale = workspaceView.gmm2PerTokenScale[rowIdx];
        int32_t rowAcc =
            LoadScalarI32(workspaceView.gmm2AccInt32 + static_cast<uint64_t>(rowIdx) * shape.hiddenSize + hiddenBegin);
        float rowScale = M2DecodeUint64Scale(workspaceView.scale2Uint64[hiddenBegin]) * rowTokenScale;
        workspaceView.gmm2Out[static_cast<int64_t>(rowIdx) * returnRowStride + hiddenBegin] =
            static_cast<half>(static_cast<float>(rowAcc) * rowScale);
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    if (debugStopStage == 77U) {
        return true;
    }
    for (int32_t row = 0; row < rows; ++row) {
        int32_t rowIdx = srcStart + row;
        float rowTokenScale = workspaceView.gmm2PerTokenScale[rowIdx];
        __gm__ half *rowDst = workspaceView.gmm2Out + static_cast<int64_t>(rowIdx) * returnRowStride + hiddenBegin;
        for (int32_t col = 0; col < hiddenCount; ++col) {
            int32_t hiddenCol = hiddenBegin + col;
            int32_t rowAcc = LoadScalarI32(workspaceView.gmm2AccInt32 +
                                           static_cast<uint64_t>(rowIdx) * shape.hiddenSize + hiddenCol);
            float rowScale = M2DecodeUint64Scale(workspaceView.scale2Uint64[hiddenCol]) * rowTokenScale;
            rowDst[col] = static_cast<half>(static_cast<float>(rowAcc) * rowScale);
        }
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    return true;
}

AICORE inline M3NCombineShardStats M3NRunGmm2EpilogueAndReturnShard(
    moe_new_dispatch_combine_a8w8::ShapeConfig shape, M2WorkspaceViewDevice workspaceView,
    M2PeerWindowViewDevice localPeer, __gm__ HcclDeviceContext *ctx, GM_ADDR peerWindow, uint32_t myRank,
    const moe_new_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout, uint32_t workerId, uint32_t workerCount,
    bool materializeEnabled, bool transferEnabled, bool notifyEnabled, uint32_t debugStopStage)
{
    M3NCombineShardStats stats{0, 0, 0, 0, 0, 0};
    if (workerCount == 0U || workerId >= workerCount) {
        return stats;
    }
    int32_t returnRowStride = static_cast<int32_t>(peerWindowLayout.returnPayloadRowBytes / sizeof(half));
    uint32_t mappedSegments = static_cast<uint32_t>(LoadScalarI32(localPeer.debugCounters + 3U * 16U));
    uint32_t tileCount = static_cast<uint32_t>(LoadScalarI32(localPeer.debugCounters + 2U * 16U));
    uint32_t capacity = static_cast<uint32_t>(M2ReturnSegmentCapacity(shape));
    for (uint32_t tileId = 0; tileId < tileCount; ++tileId) {
        __gm__ int32_t *plan = workspaceView.subTileReturnPlan + tileId * kM2ReturnPlanFields;
        uint32_t firstSegment = static_cast<uint32_t>(LoadScalarI32(plan + 6U));
        uint32_t tileSegmentCount = static_cast<uint32_t>(LoadScalarI32(plan + 7U));
        for (uint32_t localSegment = 0; localSegment < tileSegmentCount; ++localSegment) {
            uint32_t segmentId = firstSegment + localSegment;
            if (segmentId % workerCount != workerId) {
                continue;
            }
            if (segmentId >= mappedSegments || segmentId >= capacity) {
                ++stats.skippedSegmentCount;
                continue;
            }
            __gm__ int32_t *segment = workspaceView.subTileOwnerSegments + segmentId * kM2OwnerSegmentFields;
            uint32_t tokenOwner = static_cast<uint32_t>(LoadScalarI32(segment + 0U));
            int32_t srcStart = LoadScalarI32(segment + 1U);
            int32_t rows = LoadScalarI32(segment + 2U);
            int32_t dstStart = LoadScalarI32(segment + 3U);
            int32_t hiddenBegin = LoadScalarI32(segment + 4U);
            int32_t hiddenCount = LoadScalarI32(segment + 5U);
            uint32_t localExpert = static_cast<uint32_t>(LoadScalarI32(segment + 6U));
            uint32_t tileId = static_cast<uint32_t>(LoadScalarI32(segment + 7U));
            if (rows <= 0 || hiddenCount <= 0 || tokenOwner >= shape.rankNum) {
                ++stats.skippedSegmentCount;
                continue;
            }
            M3N11WaitSubTileReadyGm(workspaceView, tileId);
            ++stats.readyWaitCount;
            if (!materializeEnabled) {
                ++stats.segmentCount;
                continue;
            }
            if (M3NDebugProbeGmm2Materialize(shape, workspaceView, localPeer, srcStart, rows, hiddenBegin, hiddenCount,
                                             returnRowStride, debugStopStage)) {
                ++stats.segmentCount;
                continue;
            }
            M3NMaterializeGmm2EpilogueSegmentToGmm2Out(shape, workspaceView, srcStart, rows, hiddenBegin, hiddenCount,
                                                       returnRowStride);
            if (!transferEnabled) {
                ++stats.segmentCount;
                continue;
            }
            __gm__ half *segmentSource =
                workspaceView.gmm2Out + static_cast<int64_t>(srcStart) * returnRowStride + hiddenBegin;
            if (debugStopStage == 80U) {
                ++stats.segmentCount;
                continue;
            }
            M2RecordReturnSendTrace(shape, workspaceView, localPeer, tokenOwner, localExpert, segmentId, tileId,
                                    srcStart, rows, dstStart, hiddenBegin, hiddenCount, segmentSource);
            if (debugStopStage == 81U) {
                ++stats.segmentCount;
                continue;
            }
            M2PeerWindowViewDevice remotePeer =
                MakeM2RemotePeerWindowViewDevice(ctx, peerWindow, tokenOwner, peerWindowLayout);
            if (debugStopStage == 82U || debugStopStage == 83U) {
                if (tokenOwner == myRank && rows > 0 && hiddenCount > 0) {
                    __gm__ half *dst =
                        localPeer.returnPayload + static_cast<int64_t>(dstStart) * returnRowStride + hiddenBegin;
                    dst[0] = segmentSource[0];
                    if (debugStopStage == 83U) {
                        for (int32_t row = 0; row < rows; ++row) {
                            __gm__ half *rowDst = localPeer.returnPayload +
                                                  static_cast<int64_t>(dstStart + row) * returnRowStride + hiddenBegin;
                            __gm__ half *rowSrc = workspaceView.gmm2Out +
                                                  static_cast<int64_t>(srcStart + row) * returnRowStride + hiddenBegin;
                            for (int32_t col = 0; col < hiddenCount; ++col) {
                                rowDst[col] = rowSrc[col];
                            }
                        }
                    }
                    pipe_barrier(PIPE_ALL);
                    dsb(DSB_DDR);
                }
                ++stats.segmentCount;
                continue;
            }
            if (tokenOwner == myRank) {
                int32_t subtileCount = M3N11TransferHalfSubtileStride(
                    localPeer.returnPayload + hiddenBegin, returnRowStride, dstStart,
                    workspaceView.gmm2Out + hiddenBegin, returnRowStride, srcStart, rows, hiddenCount, false);
                stats.subtileCount += subtileCount;
                stats.localSubtileCount += subtileCount;
            } else {
                int32_t subtileCount = M3N11TransferHalfSubtileStride(
                    remotePeer.returnPayload + hiddenBegin, returnRowStride, dstStart,
                    workspaceView.gmm2Out + hiddenBegin, returnRowStride, srcStart, rows, hiddenCount, true);
                stats.subtileCount += subtileCount;
                stats.remoteSubtileCount += subtileCount;
            }
            if (debugStopStage == 84U) {
                ++stats.segmentCount;
                continue;
            }
            bool doNotify = notifyEnabled && tokenOwner != myRank;
            if (workerId < kM3NDispatchMaxWorkers) {
                __gm__ int32_t *scratch = M3NCombineWorkerScratch(localPeer, workerId);
                StoreScalarI32(scratch + 9U, doNotify ? 1 : 0);
                StoreScalarI32(scratch + 10U, static_cast<int32_t>(tokenOwner));
                StoreScalarI32(scratch + 11U, static_cast<int32_t>(myRank));
            }
            if (debugStopStage == 85U) {
                ++stats.segmentCount;
                continue;
            }
            if (doNotify) {
                M2NotifyReturnSegmentCounter(remotePeer, myRank, tokenOwner, localExpert, shape);
            }
            if (debugStopStage == 86U) {
                ++stats.segmentCount;
                continue;
            }
            ++stats.segmentCount;
        }
    }
    if (workerId < kM3NDispatchMaxWorkers) {
        __gm__ int32_t *scratch = M3NCombineWorkerScratch(localPeer, workerId);
        StoreScalarI32(scratch + 0U, stats.segmentCount);
        StoreScalarI32(scratch + 1U, stats.skippedSegmentCount);
        StoreScalarI32(scratch + 2U, static_cast<int32_t>(workerId));
        StoreScalarI32(scratch + 3U, static_cast<int32_t>(mappedSegments));
        StoreScalarI32(scratch + 4U, stats.subtileCount);
        StoreScalarI32(scratch + 5U, stats.segmentCount);
        StoreScalarI32(scratch + 6U, stats.remoteSubtileCount);
        StoreScalarI32(scratch + 7U, stats.localSubtileCount);
        StoreScalarI32(scratch + 8U, stats.readyWaitCount);
        InvalidateGmCacheLines(scratch, 64U);
    }
    return stats;
}

AICORE inline M3NCombineShardStats M3N8RunGmm2EpilogueAndReturnExpertShard(
    moe_new_dispatch_combine_a8w8::ShapeConfig shape, M2WorkspaceViewDevice workspaceView,
    M2PeerWindowViewDevice localPeer, __gm__ HcclDeviceContext *ctx, GM_ADDR peerWindow, uint32_t myRank,
    const moe_new_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout, uint32_t localExpert,
    uint32_t firstSegment, uint32_t segmentCount, uint32_t workerId, uint32_t workerCount, bool materializeEnabled,
    bool transferEnabled, bool notifyEnabled)
{
    M3NCombineShardStats stats{0, 0, 0, 0, 0, 0};
    if (workerCount == 0U || workerId >= workerCount) {
        return stats;
    }
    int32_t returnRowStride = static_cast<int32_t>(peerWindowLayout.returnPayloadRowBytes / sizeof(half));
    uint32_t capacity = static_cast<uint32_t>(M2ReturnSegmentCapacity(shape));
    uint32_t segmentEnd = firstSegment + segmentCount;
    for (uint32_t segmentId = firstSegment + workerId; segmentId < segmentEnd; segmentId += workerCount) {
        if (segmentId >= capacity) {
            ++stats.skippedSegmentCount;
            continue;
        }
        __gm__ int32_t *segment = workspaceView.subTileOwnerSegments + segmentId * kM2OwnerSegmentFields;
        uint32_t tokenOwner = static_cast<uint32_t>(LoadScalarI32(segment + 0U));
        int32_t srcStart = LoadScalarI32(segment + 1U);
        int32_t rows = LoadScalarI32(segment + 2U);
        int32_t dstStart = LoadScalarI32(segment + 3U);
        int32_t hiddenBegin = LoadScalarI32(segment + 4U);
        int32_t hiddenCount = LoadScalarI32(segment + 5U);
        uint32_t segmentLocalExpert = static_cast<uint32_t>(LoadScalarI32(segment + 6U));
        uint32_t tileId = static_cast<uint32_t>(LoadScalarI32(segment + 7U));
        if (rows <= 0 || hiddenCount <= 0 || tokenOwner >= shape.rankNum || segmentLocalExpert != localExpert) {
            ++stats.skippedSegmentCount;
            continue;
        }
        M3N11WaitSubTileReadyGm(workspaceView, tileId);
        ++stats.readyWaitCount;
        if (!materializeEnabled) {
            ++stats.segmentCount;
            continue;
        }
        M3NMaterializeGmm2EpilogueSegmentToGmm2Out(shape, workspaceView, srcStart, rows, hiddenBegin, hiddenCount,
                                                   returnRowStride);
        if (!transferEnabled) {
            ++stats.segmentCount;
            continue;
        }
        __gm__ half *segmentSource =
            workspaceView.gmm2Out + static_cast<int64_t>(srcStart) * returnRowStride + hiddenBegin;
        M2RecordReturnSendTrace(shape, workspaceView, localPeer, tokenOwner, localExpert, segmentId, tileId, srcStart,
                                rows, dstStart, hiddenBegin, hiddenCount, segmentSource);
        M2PeerWindowViewDevice remotePeer =
            MakeM2RemotePeerWindowViewDevice(ctx, peerWindow, tokenOwner, peerWindowLayout);
        if (tokenOwner == myRank) {
            int32_t subtileCount = M3N11TransferHalfSubtileStride(
                localPeer.returnPayload + hiddenBegin, returnRowStride, dstStart, workspaceView.gmm2Out + hiddenBegin,
                returnRowStride, srcStart, rows, hiddenCount, false);
            stats.subtileCount += subtileCount;
            stats.localSubtileCount += subtileCount;
        } else {
            int32_t subtileCount = M3N11TransferHalfSubtileStride(
                remotePeer.returnPayload + hiddenBegin, returnRowStride, dstStart, workspaceView.gmm2Out + hiddenBegin,
                returnRowStride, srcStart, rows, hiddenCount, true);
            stats.subtileCount += subtileCount;
            stats.remoteSubtileCount += subtileCount;
        }
        if (notifyEnabled && tokenOwner != myRank) {
            M2NotifyReturnSegmentCounter(remotePeer, myRank, tokenOwner, localExpert, shape);
        }
        ++stats.segmentCount;
    }
    return stats;
}

AICORE inline void M3N8AccumulateCombineWorkerScratch(M2PeerWindowViewDevice localPeer, uint32_t workerId,
                                                      M3NCombineShardStats stats, uint32_t mappedSegments)
{
    if (workerId >= kM3NDispatchMaxWorkers) {
        return;
    }
    __gm__ int32_t *scratch = M3NCombineWorkerScratch(localPeer, workerId);
    int32_t oldSegments = LoadScalarI32(scratch + 0U);
    int32_t oldSkipped = LoadScalarI32(scratch + 1U);
    StoreScalarI32(scratch + 0U, oldSegments + stats.segmentCount);
    StoreScalarI32(scratch + 1U, oldSkipped + stats.skippedSegmentCount);
    StoreScalarI32(scratch + 2U, static_cast<int32_t>(workerId));
    StoreScalarI32(scratch + 3U, static_cast<int32_t>(mappedSegments));
    StoreScalarI32(scratch + 4U, LoadScalarI32(scratch + 4U) + stats.subtileCount);
    StoreScalarI32(scratch + 5U, LoadScalarI32(scratch + 5U) + stats.segmentCount);
    StoreScalarI32(scratch + 6U, LoadScalarI32(scratch + 6U) + stats.remoteSubtileCount);
    StoreScalarI32(scratch + 7U, LoadScalarI32(scratch + 7U) + stats.localSubtileCount);
    StoreScalarI32(scratch + 8U, LoadScalarI32(scratch + 8U) + stats.readyWaitCount);
    InvalidateGmCacheLines(scratch, 64U);
}

AICORE inline void M3N8InitCombineCounters(M2PeerWindowViewDevice localPeer, uint32_t workerCount,
                                           uint32_t expertPerRank)
{
    M3NInitCombineCounters(localPeer, workerCount);
    StoreScalarI32(localPeer.debugCounters + 2U * 16U, 0);
    StoreScalarI32(localPeer.debugCounters + 3U * 16U, 0);
    StoreScalarI32(localPeer.debugCounters + 4U * 16U, 0);
    StoreScalarI32(localPeer.debugCounters + 5U * 16U, 0);
    for (uint32_t idx = 0; idx < 16U; ++idx) {
        StoreScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + idx, 0);
    }
    StoreScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 0U, 1);
    StoreScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 1U, static_cast<int32_t>(expertPerRank));
    StoreScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 4U, -1);
    StoreScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 6U, -1);
    StoreScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 7U, static_cast<int32_t>(workerCount));
    StoreScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 11U, 0);
}

AICORE inline bool M3N8HasLaterUnreadyGmm2Expert(M2WorkspaceViewDevice workspaceView, uint32_t nextExpert,
                                                 uint32_t expertPerRank)
{
    for (uint32_t expert = nextExpert; expert < expertPerRank; ++expert) {
        __gm__ int32_t *ready = workspaceView.gmm2GroupReady + expert * 16U;
        pipe_barrier(PIPE_ALL);
        dcci(static_cast<__gm__ void *>(ready), SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
        if (LoadScalarI32(ready) == 0) {
            return true;
        }
    }
    return false;
}

AICORE inline void M3N8RecordExpertConsumed(M2PeerWindowViewDevice localPeer, uint32_t localExpert,
                                            uint32_t expertSegmentCount, bool laterExpertWasUnready)
{
    int32_t consumed = LoadScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 2U) + 1;
    StoreScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 2U, consumed);
    StoreScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 3U,
                   LoadScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 3U) +
                       static_cast<int32_t>(expertSegmentCount));
    StoreScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 4U, static_cast<int32_t>(localExpert));
    StoreScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 10U,
                   LoadScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 10U) + 1);
    if (expertSegmentCount > 0U && laterExpertWasUnready) {
        StoreScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 5U, 1);
        if (LoadScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 6U) < 0) {
            StoreScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 6U, static_cast<int32_t>(localExpert));
        }
    }
}

AICORE inline void M3N8RunGmm2EpilogueAndReturnByExpert(
    moe_new_dispatch_combine_a8w8::ShapeConfig shape, M2WorkspaceViewDevice workspaceView,
    M2PeerWindowViewDevice localPeer, __gm__ HcclDeviceContext *ctx, GM_ADDR peerWindow, uint32_t myRank,
    const moe_new_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout, uint32_t workerId, uint32_t workerCount,
    bool activeWorker, bool mainAiv, bool materializeEnabled, bool transferEnabled, bool notifyEnabled)
{
    uint32_t tileId = 0;
    uint32_t segmentId = 0;
    uint32_t maxSegmentsPerTile = 0;
    uint32_t overflow = 0;
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        bool laterExpertWasUnready = false;
        if (mainAiv && localExpert + 1U < shape.expertPerRank) {
            laterExpertWasUnready = M3N8HasLaterUnreadyGmm2Expert(workspaceView, localExpert + 1U, shape.expertPerRank);
        }
        M3N8WaitGmm2ExpertReadyGm(workspaceView, localExpert);
        if (mainAiv) {
            uint32_t firstExpertSegment = static_cast<uint32_t>(LoadScalarI32(localPeer.debugCounters + 3U * 16U));
            uint32_t expertSegmentCount = 0;
            tileId = static_cast<uint32_t>(LoadScalarI32(localPeer.debugCounters + 2U * 16U));
            segmentId = firstExpertSegment;
            expertSegmentCount =
                M3N8BuildReturnSegmentMapForExpert(shape, workspaceView, localPeer, myRank, localExpert, &tileId,
                                                   &segmentId, &maxSegmentsPerTile, &overflow);
            StoreScalarI32(localPeer.debugCounters + 2U * 16U, static_cast<int32_t>(tileId));
            StoreScalarI32(localPeer.debugCounters + 3U * 16U, static_cast<int32_t>(segmentId));
            StoreScalarI32(localPeer.debugCounters + 4U * 16U, static_cast<int32_t>(maxSegmentsPerTile));
            StoreScalarI32(localPeer.debugCounters + 5U * 16U, static_cast<int32_t>(overflow));
            M3N8RecordExpertConsumed(localPeer, localExpert, expertSegmentCount, laterExpertWasUnready);
            StoreScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 8U,
                           static_cast<int32_t>(firstExpertSegment));
            StoreScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 9U,
                           static_cast<int32_t>(expertSegmentCount));
        }
        M3N8AivOnlyPhaseSync();
        InvalidateGmCacheLines(localPeer.debugCounters + kM3N8CombineCounterBase, 64U);
        uint32_t firstExpertSegment =
            static_cast<uint32_t>(LoadScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 8U));
        uint32_t publishedExpertSegments =
            static_cast<uint32_t>(LoadScalarI32(localPeer.debugCounters + kM3N8CombineCounterBase + 9U));
        uint32_t publishedSegmentId = firstExpertSegment + publishedExpertSegments;
        if (activeWorker) {
            M3NCombineShardStats stats = M3N8RunGmm2EpilogueAndReturnExpertShard(
                shape, workspaceView, localPeer, ctx, peerWindow, myRank, peerWindowLayout, localExpert,
                firstExpertSegment, publishedExpertSegments, workerId, workerCount, materializeEnabled, transferEnabled,
                notifyEnabled);
            M3N8AccumulateCombineWorkerScratch(localPeer, workerId, stats, publishedSegmentId);
        }
        M3N8AivOnlyPhaseSync();
    }
}

AICORE inline void M3NFinalizeGmm2EpilogueAndReturn(
    moe_new_dispatch_combine_a8w8::ShapeConfig shape, M2PeerWindowViewDevice localPeer, __gm__ HcclDeviceContext *ctx,
    GM_ADDR peerWindow, uint32_t myRank, const moe_new_dispatch_combine_a8w8::PeerWindowLayout &peerWindowLayout,
    uint32_t workerCount)
{
    int32_t totalSegments = 0;
    int32_t activeWorkerMask = 0;
    int32_t totalSubtiles = 0;
    int32_t totalSubtileSegments = 0;
    int32_t remoteSubtiles = 0;
    int32_t localSubtiles = 0;
    int32_t subTileReadyWaits = 0;
    for (uint32_t worker = 0; worker < workerCount && worker < kM3NDispatchMaxWorkers; ++worker) {
        __gm__ int32_t *scratch = M3NCombineWorkerScratch(localPeer, worker);
        InvalidateGmCacheLines(scratch, 64U);
        int32_t workerSegments = LoadScalarI32(scratch);
        StoreScalarI32(localPeer.debugCounters + kM3NCombineCounterBase + 8U + worker, workerSegments);
        totalSegments += workerSegments;
        totalSubtiles += LoadScalarI32(scratch + 4U);
        totalSubtileSegments += LoadScalarI32(scratch + 5U);
        remoteSubtiles += LoadScalarI32(scratch + 6U);
        localSubtiles += LoadScalarI32(scratch + 7U);
        subTileReadyWaits += LoadScalarI32(scratch + 8U);
        if (workerSegments > 0 && worker < 31U) {
            activeWorkerMask |= static_cast<int32_t>(1U << worker);
        }
    }
    int32_t mappedSegments = LoadScalarI32(localPeer.debugCounters + 3U * 16U);
    int32_t skippedSegments = mappedSegments > totalSegments ? mappedSegments - totalSegments : 0;
    StoreScalarI32(localPeer.debugCounters + kM3NCombineCounterBase + 1U, mappedSegments);
    StoreScalarI32(localPeer.debugCounters + kM3NCombineCounterBase + 2U, totalSegments);
    StoreScalarI32(localPeer.debugCounters + kM3NCombineCounterBase + 3U, skippedSegments);
    StoreScalarI32(localPeer.debugCounters + kM3NCombineCounterBase + 4U, activeWorkerMask);
    StoreScalarI32(localPeer.debugCounters + kM3NCombineCounterBase + 5U,
                   LoadScalarI32(localPeer.debugCounters + 2U * 16U));
    StoreScalarI32(localPeer.debugCounters + kM3NCombineCounterBase + 6U, 1);
    int32_t returnRowStride = static_cast<int32_t>(peerWindowLayout.returnPayloadRowBytes / sizeof(half));
    StoreScalarI32(localPeer.debugCounters + kM3N11SubtileCounterBase + 0U, 1);
    StoreScalarI32(localPeer.debugCounters + kM3N11SubtileCounterBase + 1U, static_cast<int32_t>(kM3N11SubtileRows));
    StoreScalarI32(localPeer.debugCounters + kM3N11SubtileCounterBase + 2U, returnRowStride);
    StoreScalarI32(localPeer.debugCounters + kM3N11SubtileCounterBase + 3U, totalSubtiles);
    StoreScalarI32(localPeer.debugCounters + kM3N11SubtileCounterBase + 4U, totalSubtileSegments);
    StoreScalarI32(localPeer.debugCounters + kM3N11SubtileCounterBase + 5U, remoteSubtiles);
    StoreScalarI32(localPeer.debugCounters + kM3N11SubtileCounterBase + 6U, localSubtiles);
    StoreScalarI32(localPeer.debugCounters + kM3N11SubtileCounterBase + 7U, subTileReadyWaits);
    StoreScalarI32(localPeer.debugCounters + kM3N11SubtileCounterBase + 8U,
                   LoadScalarI32(localPeer.debugCounters + 2U * 16U));
    StoreScalarI32(localPeer.debugCounters + kM3N11SubtileCounterBase + 9U, mappedSegments);
    StoreScalarI32(localPeer.debugCounters + kM3N11SubtileCounterBase + 10U, 1);

    M2PublishReturnSegmentCountersScalar(shape, localPeer, ctx, peerWindow, myRank, peerWindowLayout);
    for (uint32_t tokenOwner = 0; tokenOwner < shape.rankNum; ++tokenOwner) {
        M2PeerWindowViewDevice remotePeer =
            MakeM2RemotePeerWindowViewDevice(ctx, peerWindow, tokenOwner, peerWindowLayout);
        M2NotifyCombineDone(remotePeer, localPeer, myRank, tokenOwner);
    }
    M2WaitCombineDonePeers(shape, localPeer, myRank);
    int32_t expectedReturnSegments = M2WaitReturnSegmentCounters(shape, localPeer, myRank);
    InvalidateGmCacheLines(localPeer.returnPayload,
                           static_cast<uint32_t>(shape.m * shape.topK * returnRowStride * sizeof(half)));
    StoreScalarI32(localPeer.debugCounters, totalSegments);
    StoreScalarI32(localPeer.debugCounters + 16U, 1);
    StoreScalarI32(localPeer.debugCounters + 6U * 16U, expectedReturnSegments);
}

__global__ AICORE void M2Gmm2EpilogueAndReturn(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                               moe_new_dispatch_combine_a8w8::RankConfig rank, GM_ADDR peerWindow,
                                               GM_ADDR hcclCtx, GM_ADDR workspace)
{
    uint32_t blockId = static_cast<uint32_t>(get_block_idx());
    if (blockId != 0 || shape.rankNum == 0 || shape.hiddenSize == 0 || shape.expertPerRank == 0) {
        return;
    }
    auto workspaceLayout = MakeM2WorkspaceLayoutDevice(shape);
    auto peerWindowLayout = MakeM2PeerWindowLayoutDevice(shape);
    M2WorkspaceViewDevice workspaceView = MakeM2WorkspaceViewDevice(workspace, workspaceLayout);
    M2PeerWindowViewDevice localPeer = MakeM2PeerWindowViewDevice(peerWindow, peerWindowLayout);
    __gm__ HcclDeviceContext *ctx = reinterpret_cast<__gm__ HcclDeviceContext *>(hcclCtx);
    M2RunGmm2EpilogueAndReturn(shape, workspaceView, localPeer, ctx, peerWindow, rank.rankId, peerWindowLayout);
}

AICORE inline void WaitCombinePhase(DispatchCombineTileShape shape, LocalPeerWindowView localPeer, uint32_t blockId,
                                    uint32_t blockNum, int32_t value)
{
    for (uint32_t peer = blockId; peer < shape.ep; peer += blockNum) {
        WaitSignal(localPeer.combineDoneSignal + peer, value);
    }
}

AICORE inline void ReturnExpertRowsToOwners(DispatchCombineTileShape shape, LocalWorkspaceView workspaceView,
                                            LocalPeerWindowView localPeer, __gm__ HcclDeviceContext *ctx,
                                            GM_ADDR peerWindow, GM_ADDR expertOutput, uint32_t myRank, uint32_t blockId,
                                            uint32_t blockNum, const PeerWindowLayout &peerWindowLayout)
{
    uint32_t expertNumPadded = ExpertNumPaddedDevice(shape);
    __gm__ half *localExpertOutput = reinterpret_cast<__gm__ half *>(expertOutput);
    InvalidateGmCacheLines(localPeer.peerTokenPerExpert,
                           static_cast<uint32_t>(shape.ep * expertNumPadded * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.cumsumPerExpert,
                           static_cast<uint32_t>(shape.ep * expertNumPadded * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.dispatchOffset, static_cast<uint32_t>(shape.expertPerRank * sizeof(int32_t)));
    InvalidateGmCacheLines(workspaceView.prevSumBeforeRank,
                           static_cast<uint32_t>(shape.ep * shape.expertPerRank * sizeof(int32_t)));
    for (uint32_t src = blockId; src < shape.ep; src += blockNum) {
        LocalPeerWindowView remotePeer = MakeRemotePeerWindowView(ctx, peerWindow, src, peerWindowLayout);
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            uint32_t globalExpert = myRank * shape.expertPerRank + localExpert;
            int32_t rows = LoadScalarI32(localPeer.peerTokenPerExpert + static_cast<uint64_t>(src) * expertNumPadded +
                                         globalExpert);
            if (rows <= 0) {
                continue;
            }
            int32_t srcStart = LoadScalarI32(workspaceView.dispatchOffset + localExpert) +
                               LoadScalarI32(workspaceView.prevSumBeforeRank +
                                             static_cast<uint64_t>(src) * shape.expertPerRank + localExpert);
            int32_t dstStart = globalExpert == 0 ?
                                   0 :
                                   LoadScalarI32(workspaceView.cumsumPerExpert +
                                                 static_cast<uint64_t>(src) * expertNumPadded + globalExpert - 1);
            if (src == myRank) {
                for (int32_t row = 0; row < rows; ++row) {
                    CopyRowHalf(localPeer.ptrD, static_cast<int32_t>(shape.k), dstStart + row, localExpertOutput,
                                static_cast<int32_t>(shape.k), srcStart + row, static_cast<int32_t>(shape.k));
                }
            } else {
                TPutRowsHalf(remotePeer.ptrD, static_cast<int32_t>(shape.k), dstStart, localExpertOutput,
                             static_cast<int32_t>(shape.k), srcStart, rows, static_cast<int32_t>(shape.k));
            }
        }
        NotifySignal(remotePeer.combineDoneSignal + myRank, static_cast<int32_t>(shape.signalValue));
    }
}

AICORE inline void StoreZeroRowHalf(__gm__ half *dstBase, int32_t dstRowStride, int32_t dstRow, int32_t rowLen)
{
    for (int32_t col = 0; col < rowLen; col += kDefaultTileCols) {
        int32_t cols = rowLen - col < kDefaultTileCols ? rowLen - col : kDefaultTileCols;
        VecTile<half, kDefaultTileCols> zeroTile(1, cols);
        TASSIGN(zeroTile, kPingUbAddr);
        GlobalNd<half> dst =
            MakeGlobal2D(dstBase + static_cast<int64_t>(dstRow) * dstRowStride + col, 1, cols, dstRowStride);
        TEXPANDS(zeroTile, static_cast<half>(0.0));
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(dst, zeroTile);
        WaitStoreTileReusable();
    }
}

AICORE inline void AddWeightedRowHalf(__gm__ half *outputBase, __gm__ half *ptrDBase, int32_t rowStride,
                                      int32_t outputRow, int32_t ptrDRow, int32_t rowLen, half prob)
{
    for (int32_t col = 0; col < rowLen; col += kDefaultTileCols) {
        int32_t cols = rowLen - col < kDefaultTileCols ? rowLen - col : kDefaultTileCols;
        VecTile<half, kDefaultTileCols> outTile(1, cols);
        VecTile<half, kDefaultTileCols> ptrTile(1, cols);
        TASSIGN(outTile, kPingUbAddr);
        TASSIGN(ptrTile, kPongUbAddr);
        GlobalNd<half> outGlobal =
            MakeGlobal2D(outputBase + static_cast<int64_t>(outputRow) * rowStride + col, 1, cols, rowStride);
        __gm__ half *ptrChunk = ptrDBase + static_cast<int64_t>(ptrDRow) * rowStride + col;
        InvalidateGmCacheLines(ptrChunk, static_cast<uint32_t>(cols) * sizeof(half));
        GlobalNd<half> ptrGlobal = MakeGlobal2D(ptrChunk, 1, cols, rowStride);
        TLOAD(ptrTile, ptrGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TLOAD(outTile, outGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TAXPY(outTile, ptrTile, prob);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(outGlobal, outTile);
        WaitStoreTileReusable();
    }
}

AICORE inline void StoreZeroRowHalfScalar(__gm__ half *dstBase, int32_t dstRowStride, int32_t dstRow, int32_t rowLen)
{
    __gm__ half *dst = dstBase + static_cast<int64_t>(dstRow) * dstRowStride;
    for (int32_t col = 0; col < rowLen; ++col) {
        dst[col] = static_cast<half>(0.0);
    }
}

AICORE inline void AddWeightedRowHalfStrideScalar(__gm__ half *outputBase, int32_t outputStride, int32_t outputRow,
                                                  __gm__ half *srcBase, int32_t srcStride, int32_t srcRow,
                                                  int32_t rowLen, float prob)
{
    __gm__ half *outputRowPtr = outputBase + static_cast<int64_t>(outputRow) * outputStride;
    __gm__ half *srcRowPtr = srcBase + static_cast<int64_t>(srcRow) * srcStride;
    InvalidateGmCacheLines(srcRowPtr, static_cast<uint32_t>(rowLen * sizeof(half)));
    half halfProb = static_cast<half>(prob);
    for (int32_t col = 0; col < rowLen; ++col) {
        float current = static_cast<float>(outputRowPtr[col]);
        float product =
            static_cast<float>(static_cast<half>(static_cast<float>(srcRowPtr[col]) * static_cast<float>(halfProb)));
        outputRowPtr[col] = static_cast<half>(current + product);
    }
}

AICORE inline void RestoreOutputRows(DispatchCombineTileShape shape, LocalPeerWindowView localPeer, GM_ADDR probs,
                                     GM_ADDR outputC, uint32_t blockId, uint32_t blockNum)
{
    __gm__ float *probValues = reinterpret_cast<__gm__ float *>(probs);
    __gm__ half *output = reinterpret_cast<__gm__ half *>(outputC);
    uint32_t tokenBegin = TokenShardBegin(shape.m, blockId, blockNum);
    uint32_t tokenEnd = TokenShardEnd(shape.m, blockId, blockNum);
    for (uint32_t token = tokenBegin; token < tokenEnd; ++token) {
        StoreZeroRowHalfScalar(output, static_cast<int32_t>(shape.k), static_cast<int32_t>(token),
                               static_cast<int32_t>(shape.k));
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            uint32_t routeIndex = token * shape.topK + slot;
            int32_t ptrDRow = LoadScalarI32(localPeer.expandedRowIdx + routeIndex);
            if (ptrDRow < 0 || static_cast<uint32_t>(ptrDRow) >= shape.maxOutputSize) {
                continue;
            }
            float prob = probValues[routeIndex];
            AddWeightedRowHalfStrideScalar(output, static_cast<int32_t>(shape.k), static_cast<int32_t>(token),
                                           localPeer.ptrD, static_cast<int32_t>(shape.k), ptrDRow,
                                           static_cast<int32_t>(shape.k), prob);
        }
    }
}

__global__ AICORE void M2RestoreOutput(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                       moe_new_dispatch_combine_a8w8::RankConfig rank, GM_ADDR probs, GM_ADDR outputC,
                                       GM_ADDR peerWindow, GM_ADDR workspace)
{
    uint32_t blockId = static_cast<uint32_t>(get_block_idx());
    uint32_t blockNum = static_cast<uint32_t>(get_block_num());
    if (shape.rankNum == 0 || shape.m == 0 || shape.hiddenSize == 0 || shape.topK == 0 || blockNum == 0 ||
        blockId >= blockNum) {
        return;
    }
    auto workspaceLayout = MakeM2WorkspaceLayoutDevice(shape);
    auto peerWindowLayout = MakeM2PeerWindowLayoutDevice(shape);
    M2WorkspaceViewDevice workspaceView = MakeM2WorkspaceViewDevice(workspace, workspaceLayout);
    M2PeerWindowViewDevice localPeer = MakeM2PeerWindowViewDevice(peerWindow, peerWindowLayout);
    __gm__ float *probValues = reinterpret_cast<__gm__ float *>(probs);
    __gm__ half *output = reinterpret_cast<__gm__ half *>(outputC);
    int32_t outputRowStride = static_cast<int32_t>(shape.hiddenSize);
    int32_t returnRowStride = static_cast<int32_t>(peerWindowLayout.returnPayloadRowBytes / sizeof(half));

    for (uint32_t peer = 0; peer < shape.rankNum; ++peer) {
        __gm__ int32_t *signal = localPeer.combineDoneSignal + peer * 16U;
        while (true) {
            InvalidateGmCacheLines(signal, sizeof(int32_t));
            if (LoadScalarI32(signal) >= 1) {
                break;
            }
        }
    }

    uint32_t tokenBegin = TokenShardBegin(shape.m, blockId, blockNum);
    uint32_t tokenEnd = TokenShardEnd(shape.m, blockId, blockNum);
    for (uint32_t token = tokenBegin; token < tokenEnd; ++token) {
        StoreZeroRowHalfScalar(output, outputRowStride, static_cast<int32_t>(token),
                               static_cast<int32_t>(shape.hiddenSize));
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            uint32_t routeIndex = token * shape.topK + slot;
            int32_t ptrDRow = LoadScalarI32(workspaceView.expandedRowIdx + routeIndex);
            if (ptrDRow < 0 || static_cast<uint32_t>(ptrDRow) >= static_cast<uint32_t>(M2LocalRows(shape))) {
                continue;
            }
            float prob = probValues[routeIndex];
            AddWeightedRowHalfStrideScalar(output, outputRowStride, static_cast<int32_t>(token),
                                           localPeer.returnPayload, returnRowStride, ptrDRow,
                                           static_cast<int32_t>(shape.hiddenSize), prob);
        }
    }
    if (blockId == 0) {
        StoreScalarI32(localPeer.debugCounters + 6U * 16U, 1);
    }
    (void)rank;
}

} // namespace

#ifndef M2_FUSED_INCLUDE_DEVICE_BODY_ONLY
__global__ AICORE void DispatchCombineTileDispatch(DispatchCombineTileShape shape, uint32_t myRank, GM_ADDR inputA,
                                                   GM_ADDR expertIdx, GM_ADDR peerWindow, GM_ADDR hcclCtx,
                                                   GM_ADDR workspace)
{
    WorkspaceLayout workspaceLayout = MakeWorkspaceLayout(shape);
    PeerWindowLayout peerWindowLayout = MakePeerWindowLayout(shape);
    LocalWorkspaceView workspaceView = MakeLocalWorkspaceView(workspace, workspaceLayout);
    LocalPeerWindowView localPeer = MakeLocalPeerWindowView(peerWindow, peerWindowLayout);
    __gm__ HcclDeviceContext *ctx = reinterpret_cast<__gm__ HcclDeviceContext *>(hcclCtx);
    uint32_t blockId = static_cast<uint32_t>(get_block_idx());
    uint32_t blockNum = shape.aivBlocks == 0 ? 1 : shape.aivBlocks;
    if (shape.ep == 0 || shape.m == 0 || shape.topK == 0 || shape.expertPerRank == 0 || shape.expertNum == 0 ||
        shape.metadataPad == 0 || blockId >= blockNum) {
        return;
    }

    ClearDispatchState(shape, workspaceView, localPeer, blockId, blockNum);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    CountLocalRoutes(shape, workspaceView, localPeer, expertIdx, blockId, blockNum);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    BuildBlockPrefixAndLocalCounts(shape, workspaceView, blockId, blockNum);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    BuildPackedExpertOffset(shape, workspaceView, myRank, blockId);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    InitPackCursors(shape, workspaceView, blockId, blockNum);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    PackLocalRowsToWindow(shape, workspaceView, localPeer, inputA, expertIdx, myRank, blockId, blockNum);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    RebuildExpandedRowIdx(shape, workspaceView, localPeer, expertIdx, myRank, blockId, blockNum);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    PublishCountRows(shape, workspaceView, localPeer, ctx, peerWindow, myRank, blockId, blockNum, peerWindowLayout);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    WaitCountRows(shape, localPeer, blockId, blockNum);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    BuildPrefixMetadata(shape, workspaceView, localPeer, myRank, blockId, blockNum);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    GatherLocalExpertPayload(shape, workspaceView, localPeer, ctx, peerWindow, myRank, blockId, blockNum,
                             peerWindowLayout);
    SoftSyncAiv(workspaceView.localSync, blockNum);
}

__global__ AICORE void DispatchCombineTileCombine(DispatchCombineTileShape shape, uint32_t myRank, GM_ADDR expertOutput,
                                                  GM_ADDR probs, GM_ADDR outputC, GM_ADDR peerWindow, GM_ADDR hcclCtx,
                                                  GM_ADDR workspace)
{
    WorkspaceLayout workspaceLayout = MakeWorkspaceLayout(shape);
    PeerWindowLayout peerWindowLayout = MakePeerWindowLayout(shape);
    LocalWorkspaceView workspaceView = MakeLocalWorkspaceView(workspace, workspaceLayout);
    LocalPeerWindowView localPeer = MakeLocalPeerWindowView(peerWindow, peerWindowLayout);
    __gm__ HcclDeviceContext *ctx = reinterpret_cast<__gm__ HcclDeviceContext *>(hcclCtx);
    uint32_t blockId = static_cast<uint32_t>(get_block_idx());
    uint32_t blockNum = shape.aivBlocks == 0 ? 1 : shape.aivBlocks;
    if (shape.ep == 0 || shape.m == 0 || shape.k == 0 || shape.topK == 0 || shape.expertPerRank == 0 ||
        shape.expertNum == 0 || shape.metadataPad == 0 || blockId >= blockNum) {
        return;
    }

    ReturnExpertRowsToOwners(shape, workspaceView, localPeer, ctx, peerWindow, expertOutput, myRank, blockId, blockNum,
                             peerWindowLayout);
    SoftSyncAiv(workspaceView.localSync, blockNum);
    WaitCombinePhase(shape, localPeer, blockId, blockNum, static_cast<int32_t>(shape.signalValue));
    SoftSyncAiv(workspaceView.localSync, blockNum);
    RestoreOutputRows(shape, localPeer, probs, outputC, blockId, blockNum);
    SoftSyncAiv(workspaceView.localSync, blockNum);
}

namespace dispatch_combine_tile {

void LaunchDispatchCombineTileDispatch(DispatchCombineTileShape shape, uint32_t myRank, uint8_t *inputA,
                                       uint8_t *expertIdx, uint8_t *peerWindow, uint8_t *hcclCtx, uint8_t *workspace,
                                       void *stream, uint32_t launchBlockCount)
{
    DispatchCombineTileDispatch<<<launchBlockCount, nullptr, stream>>>(shape, myRank, inputA, expertIdx, peerWindow,
                                                                       hcclCtx, workspace);
}

void LaunchDispatchCombineTileCombine(DispatchCombineTileShape shape, uint32_t myRank, uint8_t *expertOutput,
                                      uint8_t *probs, uint8_t *outputC, uint8_t *peerWindow, uint8_t *hcclCtx,
                                      uint8_t *workspace, void *stream, uint32_t launchBlockCount)
{
    DispatchCombineTileCombine<<<launchBlockCount, nullptr, stream>>>(shape, myRank, expertOutput, probs, outputC,
                                                                      peerWindow, hcclCtx, workspace);
}

void LaunchM2Int8Dispatch(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                          moe_new_dispatch_combine_a8w8::RankConfig rank, uint8_t *inputA, uint8_t *expertIdx,
                          uint8_t *xActiveMask, uint8_t *peerWindow, uint8_t *hcclCtx, uint8_t *workspace, void *stream,
                          uint32_t launchBlockCount)
{
    M2Int8Dispatch<<<launchBlockCount, nullptr, stream>>>(shape, rank, inputA, expertIdx, xActiveMask, peerWindow,
                                                          hcclCtx, workspace);
}

void LaunchM2Gmm1Epilogue(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                          moe_new_dispatch_combine_a8w8::RankConfig rank, uint8_t *workspace, void *stream,
                          uint32_t launchBlockCount)
{
    M2Gmm1Epilogue<<<launchBlockCount, nullptr, stream>>>(shape, rank, workspace);
}

void LaunchM2ActivationQuant(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                             moe_new_dispatch_combine_a8w8::RankConfig rank, uint8_t *workspace, void *stream,
                             uint32_t launchBlockCount)
{
    M2ActivationQuant<<<launchBlockCount, nullptr, stream>>>(shape, rank, workspace);
}

void LaunchM2Gmm2EpilogueAndReturn(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                                   moe_new_dispatch_combine_a8w8::RankConfig rank, uint8_t *peerWindow,
                                   uint8_t *hcclCtx, uint8_t *workspace, void *stream, uint32_t launchBlockCount)
{
    M2Gmm2EpilogueAndReturn<<<launchBlockCount, nullptr, stream>>>(shape, rank, peerWindow, hcclCtx, workspace);
}

void LaunchM2RestoreOutput(moe_new_dispatch_combine_a8w8::ShapeConfig shape,
                           moe_new_dispatch_combine_a8w8::RankConfig rank, uint8_t *probs, uint8_t *outputC,
                           uint8_t *peerWindow, uint8_t *workspace, void *stream, uint32_t launchBlockCount)
{
    M2RestoreOutput<<<launchBlockCount, nullptr, stream>>>(shape, rank, probs, outputC, peerWindow, workspace);
}

} // namespace dispatch_combine_tile
#endif // M2_FUSED_INCLUDE_DEVICE_BODY_ONLY
