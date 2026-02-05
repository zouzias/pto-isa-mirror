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
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace std;
using namespace pto;

template <typename T, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
__global__ AICORE void runTCVT(__gm__ T *out, __gm__ S *src) {

    //if (block_idx > 0) return;

    using DynShapeDim4 = pto::Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim4 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData_src = GlobalTensor<S, DynShapeDim4, DynStridDim4>;
    using GlobalData_dst = GlobalTensor<T, DynShapeDim4, DynStridDim4>;

    using TileDataSrc = Tile<TileType::Vec, S, kTRows_, kTCols_, BLayout::RowMajor>;
    using TileDataDst = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor>;

    TileDataSrc srcTile;
    TileDataDst dstTile;


    TASSIGN(srcTile, 0x0 + 0x400 * block_idx);
    TASSIGN(dstTile, 0x20000 + 0x400 * block_idx);

    GlobalData_src srcGlobal(src);

    GlobalData_dst dstGlobal(out);

    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Default usage: saturation mode ON (default)
    TCVT(dstTile, srcTile, RoundMode::CAST_RINT);
    
    // Explicit usage: saturation mode ON
    // TCVT(dstTile, srcTile, RoundMode::CAST_RINT, SaturationMode::ON);
    
    // Explicit usage: saturation mode OFF (truncation)
    // TCVT(dstTile, srcTile, RoundMode::CAST_RINT, SaturationMode::OFF);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);
    
    out = dstGlobal.data();
}

template <typename D, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void launchTCVT(D *dst, S *src, void *stream) {
    if constexpr ( std::is_same_v<D, aclFloat16> ) {
        runTCVT<half, S, kGRows_, kGCols_, kTRows_, kTCols_><<<1, nullptr, stream>>>((half*) dst, src);
    } else if constexpr ( std::is_same_v<S, aclFloat16> ) {
        runTCVT<D, half, kGRows_, kGCols_, kTRows_, kTCols_><<<1, nullptr, stream>>>(dst, (half*)src);
    } else {
         runTCVT<D, S, kGRows_, kGCols_, kTRows_, kTCols_><<<1, nullptr, stream>>>(dst, src);
    }
}

// Macro to generate template instantiations for all shapes for a given type pair
#define INSTANTIATE_TCVT(dst_type, src_type) \
    template void launchTCVT<dst_type, src_type, 2, 128, 2, 128>(dst_type *dst, src_type *src, void *stream); \
    template void launchTCVT<dst_type, src_type, 2, 32, 2, 32>(dst_type *dst, src_type *src, void *stream); \
    template void launchTCVT<dst_type, src_type, 1, 64, 1, 64>(dst_type *dst, src_type *src, void *stream); \
    template void launchTCVT<dst_type, src_type, 4, 64, 4, 64>(dst_type *dst, src_type *src, void *stream);

// FP32 Source
INSTANTIATE_TCVT(float, float)
INSTANTIATE_TCVT(aclFloat16, float)
INSTANTIATE_TCVT(int32_t, float)
INSTANTIATE_TCVT(int16_t, float)
INSTANTIATE_TCVT(int64_t, float)

// FP16 Source
INSTANTIATE_TCVT(float, aclFloat16)
INSTANTIATE_TCVT(int32_t, aclFloat16)
INSTANTIATE_TCVT(int16_t, aclFloat16)
INSTANTIATE_TCVT(int8_t, aclFloat16)
INSTANTIATE_TCVT(uint8_t, aclFloat16)

// INT32 Source
INSTANTIATE_TCVT(float, int32_t)
INSTANTIATE_TCVT(int16_t, int32_t)
INSTANTIATE_TCVT(int64_t, int32_t)

// INT16 Source
INSTANTIATE_TCVT(aclFloat16, int16_t)
INSTANTIATE_TCVT(float, int16_t)

// INT8 Source
INSTANTIATE_TCVT(aclFloat16, int8_t)

// UINT8 Source
INSTANTIATE_TCVT(aclFloat16, uint8_t)

// INT64 Source
INSTANTIATE_TCVT(float, int64_t)
INSTANTIATE_TCVT(int32_t, int64_t)

// ============================================================================
// Saturation Mode Test Kernels
// ============================================================================
// Test kernel to demonstrate saturation mode behavior
// Tests saturation ON, OFF, and DEFAULT modes
template <typename T, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
__global__ AICORE void runTCVTSaturationTest(__gm__ T *outSaturated, __gm__ T *outTruncated, __gm__ T *outDefault, __gm__ S *src) {
    using DynShapeDim4 = pto::Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim4 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData_src = GlobalTensor<S, DynShapeDim4, DynStridDim4>;
    using GlobalData_dst = GlobalTensor<T, DynShapeDim4, DynStridDim4>;

    using TileDataSrc = Tile<TileType::Vec, S, kTRows_, kTCols_, BLayout::RowMajor>;
    using TileDataDst = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor>;

    TileDataSrc srcTile;
    TileDataDst dstTileSat;
    TileDataDst dstTileTrunc;
    TileDataDst dstTileDefault;

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTileSat, 0x20000);
    TASSIGN(dstTileTrunc, 0x40000);
    TASSIGN(dstTileDefault, 0x60000);

    GlobalData_src srcGlobal(src);
    GlobalData_dst dstGlobalSat(outSaturated);
    GlobalData_dst dstGlobalTrunc(outTruncated);
    GlobalData_dst dstGlobalDefault(outDefault);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Test 1: Saturation mode ON (default)
    // Out-of-range values clamp to [min, max]
    // Example: 300.0f -> int8 = 127 (max for int8)
    TCVT(dstTileSat, srcTile, RoundMode::CAST_TRUNC, SaturationMode::ON);
    
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);

    // Test 2: Saturation mode OFF (truncation)
    // Convert to int64, then extract low N bits
    // Example: 300.0f -> int8 = 44 (0x12C & 0xFF = 0x2C = 44)
    TCVT(dstTileTrunc, srcTile, RoundMode::CAST_TRUNC, SaturationMode::OFF);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);

    // Test 3: Default mode (no explicit saturation parameter)
    // For fp16→uint8: should use SaturationMode::OFF
    // For other conversions: should use SaturationMode::ON
    TCVT(dstTileDefault, srcTile, RoundMode::CAST_TRUNC);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);

    TSTORE(dstGlobalSat, dstTileSat);
    TSTORE(dstGlobalTrunc, dstTileTrunc);
    TSTORE(dstGlobalDefault, dstTileDefault);
}

// Launcher for saturation mode tests (including default mode)
template <typename D, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void launchTCVTSaturationTest(D *dstSaturated, D *dstTruncated, D *dstDefault, S *src, void *stream) {
    if constexpr ( std::is_same_v<D, aclFloat16> ) {
        runTCVTSaturationTest<half, S, kGRows_, kGCols_, kTRows_, kTCols_><<<1, nullptr, stream>>>((half*)dstSaturated, (half*)dstTruncated, (half*)dstDefault, src);
    } else if constexpr ( std::is_same_v<S, aclFloat16> ) {
        runTCVTSaturationTest<D, half, kGRows_, kGCols_, kTRows_, kTCols_><<<1, nullptr, stream>>>(dstSaturated, dstTruncated, dstDefault, (half*)src);
    } else {
        runTCVTSaturationTest<D, S, kGRows_, kGCols_, kTRows_, kTCols_><<<1, nullptr, stream>>>(dstSaturated, dstTruncated, dstDefault, src);
    }
}

// Minimal saturation test instantiations (1x32 shape for fast testing)
// Note: fp32→int8 is NOT supported on A2A3 hardware
template void launchTCVTSaturationTest<int8_t, aclFloat16, 1, 32, 1, 32>(int8_t *dstSat, int8_t *dstTrunc, int8_t *dstDefault, aclFloat16 *src, void *stream);
template void launchTCVTSaturationTest<uint8_t, aclFloat16, 1, 32, 1, 32>(uint8_t *dstSat, uint8_t *dstTrunc, uint8_t *dstDefault, aclFloat16 *src, void *stream);
template void launchTCVTSaturationTest<int16_t, float, 1, 32, 1, 32>(int16_t *dstSat, int16_t *dstTrunc, int16_t *dstDefault, float *src, void *stream);
template void launchTCVTSaturationTest<int32_t, float, 1, 32, 1, 32>(int32_t *dstSat, int32_t *dstTrunc, int32_t *dstDefault, float *src, void *stream);
template void launchTCVTSaturationTest<int32_t, aclFloat16, 1, 32, 1, 32>(int32_t *dstSat, int32_t *dstTrunc, int32_t *dstDefault, aclFloat16 *src, void *stream);
template void launchTCVTSaturationTest<int32_t, int64_t, 1, 32, 1, 32>(int32_t *dstSat, int32_t *dstTrunc, int32_t *dstDefault, int64_t *src, void *stream);
template void launchTCVTSaturationTest<int16_t, int32_t, 1, 32, 1, 32>(int16_t *dstSat, int16_t *dstTrunc, int16_t *dstDefault, int32_t *src, void *stream);
