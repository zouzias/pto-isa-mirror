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
#include <acl/acl.h>

using namespace std;
using namespace pto;

template <typename T, int row, int validRow, int srcCol, int srcValidCol, int dstCol>
PTO_INTERNAL void runTRowSum(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2StrideSrc = pto::Stride<1, 1, -1, -1, 1>;
    using DynDim2StrideDst = pto::Stride<1, 1, 1, -1, -1>;

    using GlobalDataSrc = GlobalTensor<T, DynDim2Shape, DynDim2StrideSrc>;
    using GlobalDataDst = GlobalTensor<T, DynDim2Shape, DynDim2StrideDst>;
    GlobalDataSrc srcGlobal(src, DynDim2Shape(validRow, srcValidCol), DynDim2StrideSrc(row, srcCol));
    GlobalDataDst dstGlobal(out, DynDim2Shape(validRow, dstCol), DynDim2StrideDst(dstCol, row));
    using srcTileData = Tile<TileType::Vec, T, row, srcCol, BLayout::RowMajor, -1, -1>;
    using dstTileData = Tile<TileType::Vec, T, row, 16, BLayout::RowMajor, -1, -1>;
    srcTileData srcTile(validRow, srcValidCol);
    srcTileData tmpTile(validRow, srcValidCol);
    dstTileData dstTile(validRow, dstCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, row * srcCol * sizeof(T));
    TASSIGN(dstTile, 2 * row * srcCol * sizeof(T));

    // 搬运数据
    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TROWSUM(dstTile, srcTile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
}

template <typename T, int row, int validRow, int srcCol, int srcValidCol, int dstCol>
PTO_INTERNAL void runTRowSumDNDst(__gm__ T *out, __gm__ T *src)
{
    using ValidSrcShape = TileShape2D<T, validRow, srcValidCol>;
    using NDSrcShape = BaseShape2D<T, row, srcCol>;
    using GlobalDataSrc = GlobalTensor<T, ValidSrcShape, NDSrcShape>;
    GlobalDataSrc srcGlobal(src);

    using ValidDstShape = TileShape2D<T, dstCol, validRow>;
    using NDDstShape = BaseShape2D<T, row, dstCol>;
    using GlobalDataDst = GlobalTensor<T, ValidDstShape, NDDstShape>;
    GlobalDataDst dstGlobal(out);

    using srcTileData = Tile<TileType::Vec, T, row, srcCol, BLayout::RowMajor, row, srcCol>;
    using dstTileDataDN = Tile<TileType::Vec, T, row, 1, BLayout::ColMajor, row, 1>;
    srcTileData srcTile;
    srcTileData tmpTile;
    dstTileDataDN dstTile;
    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, row * srcCol * sizeof(T));
    TASSIGN(dstTile, 2 * row * srcCol * sizeof(T));

    // 搬运数据
    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TROWSUM(dstTile, srcTile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    using dstTileDataND = Tile<TileType::Vec, T, 1, row, BLayout::RowMajor, 1, row>;
    dstTileDataND dstTileND;
    TRESHAPE(dstTileND, dstTile);
    TSTORE(dstGlobal, dstTileND);
}

// Demo dimensions — must match main.py.
static constexpr int N = 2;
static constexpr int T = 8;
static constexpr int TOPK = 2;
static constexpr int D = 64;
static constexpr int L = 4;
static constexpr int R = 32;
static constexpr int N_ROUTES = T * TOPK;  // 16

// Weight payload tile width. The protocol contract is one FP32 weight per
// (e, slot) — recv_w[L, R] FP32. AIV vector tiles have a hardware minimum
// granularity of 1x8 FP32 (32 B = one MTE burst), so the on-window /
// staged-out layout is [L, R, W_PAD] FP32 with the actual weight at
// slot [0] and zeros in [1, W_PAD). Host extracts column 0 to recover the
// [L, R] grid; production with proper stride-1 ops or a scalar-write path
// can drop W_PAD entirely.
static constexpr int W_PAD = 8;
// Same minimum-tile rationale for the idx channel — INT32 [L, R] in spirit,
// but materialized as [L, R, IDX_PAD] with the actual r=t*TOPK+k at slot [0].
static constexpr int IDX_PAD = 8;

PTO_INTERNAL void case21(__gm__ float *recv_w_out, __gm__ float *recv_w_local, __gm__ int32_t *recv_idx_out, __gm__ int32_t *recv_idx_local)
{
    // Wide-window tile types (full RxPAD grid TLOADed in one shot).
    // Sum-output tiles use Layout::DN — TROWSUM produces a column tile
    // (Rx1 ColMajor); writing it back to GM needs DN layout.
    using WWideShape = pto::Shape<1, 1, 1, R, W_PAD>;
    using WWideStride = pto::Stride<R * W_PAD, R * W_PAD, R * W_PAD, W_PAD, 1>;
    using WWideG = GlobalTensor<float, WWideShape, WWideStride>;    // [32, 8]
    using WWideTile = Tile<TileType::Vec, float, R, W_PAD, BLayout::RowMajor, R, W_PAD>;    // [32, 8]
    using WSumShape = pto::Shape<1, 1, 1, R, 1>;
    using WSumStride = pto::Stride<1, 1, 1, 1, 1>;
    using WSumG = GlobalTensor<float, WSumShape, WSumStride, Layout::DN>;   // [32, 1]
    using WSumTile = Tile<TileType::Vec, float, R, 1, BLayout::ColMajor, R, 1>; // [32, 1]

    using IWideShape = pto::Shape<1, 1, 1, R, IDX_PAD>;
    using IWideStride = pto::Stride<R * IDX_PAD, R * IDX_PAD, R * IDX_PAD, IDX_PAD, 1>;
    using IWideG = GlobalTensor<int32_t, IWideShape, IWideStride>;    // [32, 8]
    using IWideTile = Tile<TileType::Vec, int32_t, R, IDX_PAD, BLayout::RowMajor, R, IDX_PAD>;    // [32, 8]
    using ISumShape = pto::Shape<1, 1, 1, R, 1>;
    using ISumStride = pto::Stride<1, 1, 1, 1, 1>;
    using ISumG = GlobalTensor<int32_t, ISumShape, ISumStride, Layout::DN>;    // [32, 1]
    using ISumTile = Tile<TileType::Vec, int32_t, R, 1, BLayout::ColMajor, R, 1>;    // [32, 1]

    // UB allocation. The payload_push phase's x_tile/w_tile/idx_tile have
    // drained, so we can reuse those slots. Slot pitch is 64 KB; the largest
    // tile here is R*W_PAD FP32 = 1 KB or R*IDX_PAD INT32 = 1 KB.
    //
    // TROWSUM's tmp tile must be the SAME SHAPE as the source (RxPAD), not
    // the destination (Rx1) — pto-isa uses it as scratch for partial
    // reductions.
    WWideTile w_wide_tile;
    WSumTile w_sum_tile;
    WWideTile w_tmp_tile;
    IWideTile idx_wide_tile;
    ISumTile idx_sum_tile;
    IWideTile idx_tmp_tile;


    // TASSIGN<0x10000>(w_wide_tile);
    // TASSIGN<0x20000>(w_sum_tile);
    // TASSIGN<0x21000>(w_tmp_tile);
    // TASSIGN<0x30000>(idx_wide_tile);
    // TASSIGN<0x40000>(idx_sum_tile);
    // TASSIGN<0x41000>(idx_tmp_tile);
    TASSIGN<0x0>(w_wide_tile);
    TASSIGN<sizeof(float) * R * W_PAD>(w_sum_tile);
    TASSIGN<sizeof(float) * R * (W_PAD + 1)>(w_tmp_tile);
    TASSIGN<sizeof(float) * R * (2 * W_PAD + 1)>(idx_wide_tile);
    TASSIGN<sizeof(float) * R * (3 * W_PAD + 1)>(idx_sum_tile);
    TASSIGN<sizeof(float) * R * (3 * W_PAD + 2)>(idx_tmp_tile);
    // Stage out compact w / idx: per-expert TLOAD + TROWSUM + TSTORE.
    // pipe_barrier(PIPE_V) brackets the TROWSUM, plus MTE2/MTE3 fences for
    // the GM legs.
    for (int e = 0; e < L; ++e) {
        // weight channel
        __gm__ float *w_win = recv_w_local + e * R * W_PAD;
        __gm__ float *w_out = recv_w_out + e * R;
        WWideG w_win_g(w_win);
        WSumG w_out_g(w_out);
        TLOAD(w_wide_tile, w_win_g);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        pipe_barrier(PIPE_V);
        TROWSUM(w_sum_tile, w_wide_tile, w_tmp_tile);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        TSTORE(w_out_g, w_sum_tile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    }

    // Stage out idx: same TROWSUM compaction as the weight channel, on the
    // INT32 [R, IDX_PAD] wide window. sum-along-PAD recovers slot [0] because
    // columns [1, IDX_PAD) are zero by design.
    for (int e = 0; e < L; ++e) {
        __gm__ int32_t *idx_win = recv_idx_local + e * R * IDX_PAD;
        __gm__ int32_t *idx_out = recv_idx_out + e * R;
        IWideG idx_win_g(idx_win);
        ISumG idx_out_g(idx_out);
        TLOAD(idx_wide_tile, idx_win_g);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        pipe_barrier(PIPE_V);
        TROWSUM(idx_sum_tile, idx_wide_tile, idx_tmp_tile);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        TSTORE(idx_out_g, idx_sum_tile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    }
    pipe_barrier(PIPE_ALL);
}

extern "C" __global__ AICORE void launchTROWSUMCase1(__gm__ float *out, __gm__ float *src)
{
    runTRowSum<float, 127, 127, 64, 63, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase2(__gm__ float *out, __gm__ float *src)
{
    runTRowSum<float, 63, 63, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase3(__gm__ float *out, __gm__ float *src)
{
    runTRowSum<float, 31, 31, 128, 127, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase4(__gm__ float *out, __gm__ float *src)
{
    runTRowSum<float, 15, 15, 192, 192, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase5(__gm__ float *out, __gm__ float *src)
{
    runTRowSum<float, 7, 7, 448, 447, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase6(__gm__ half *out, __gm__ half *src)
{
    runTRowSum<half, 256, 256, 16, 15, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase7(__gm__ float *out, __gm__ float *src)
{
    runTRowSumDNDst<float, 64, 64, 128, 128, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase8(__gm__ float *out, __gm__ float *src)
{
    runTRowSumDNDst<float, 32, 32, 256, 256, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase9(__gm__ float *out, __gm__ float *src)
{
    runTRowSumDNDst<float, 16, 16, 512, 512, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase10(__gm__ float *out, __gm__ float *src)
{
    runTRowSumDNDst<float, 8, 8, 1024, 1024, 1>(out, src);
}

// int32 test cases
extern "C" __global__ AICORE void launchTROWSUMCase11(__gm__ int32_t *out, __gm__ int32_t *src)
{
    runTRowSum<int32_t, 127, 127, 64, 63, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase12(__gm__ int32_t *out, __gm__ int32_t *src)
{
    runTRowSum<int32_t, 63, 63, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase13(__gm__ int32_t *out, __gm__ int32_t *src)
{
    runTRowSum<int32_t, 31, 31, 128, 127, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase14(__gm__ int32_t *out, __gm__ int32_t *src)
{
    runTRowSum<int32_t, 15, 15, 192, 192, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase15(__gm__ int32_t *out, __gm__ int32_t *src)
{
    runTRowSum<int32_t, 7, 7, 448, 447, 1>(out, src);
}

// int16 test cases - need 32-byte alignment for int16_t (2 bytes), so cols must be multiple of 16
extern "C" __global__ AICORE void launchTROWSUMCase16(__gm__ int16_t *out, __gm__ int16_t *src)
{
    runTRowSum<int16_t, 128, 128, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase17(__gm__ int16_t *out, __gm__ int16_t *src)
{
    runTRowSum<int16_t, 64, 64, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase18(__gm__ int16_t *out, __gm__ int16_t *src)
{
    runTRowSum<int16_t, 32, 32, 128, 128, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase19(__gm__ int16_t *out, __gm__ int16_t *src)
{
    runTRowSum<int16_t, 16, 16, 192, 192, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase20(__gm__ int16_t *out, __gm__ int16_t *src)
{
    runTRowSum<int16_t, 8, 8, 448, 448, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase21(__gm__ float *out_val, __gm__ float *src_val,
                                                      __gm__ int32_t *out_idx, __gm__ int32_t *src_idx)
{
    case21(out_val, src_val, out_idx, src_idx);
}

void launchTROWSUMTestCase21(float *out_val, float *src_val, int32_t *out_idx, int32_t *src_idx, aclrtStream stream)
{
    launchTROWSUMCase21<<<1, nullptr, stream>>>(out_val, src_val, out_idx, src_idx);
}

template <uint32_t caseId>
void launchTROWSUMTestCase(void *out, void *src, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTROWSUMCase1<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 2: {
            launchTROWSUMCase2<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 3: {
            launchTROWSUMCase3<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 4: {
            launchTROWSUMCase4<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 5: {
            launchTROWSUMCase5<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 6: {
            launchTROWSUMCase6<<<1, nullptr, stream>>>((half *)out, (half *)src);
            break;
        }
        case 7: {
            launchTROWSUMCase7<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 8: {
            launchTROWSUMCase8<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 9: {
            launchTROWSUMCase9<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 10: {
            launchTROWSUMCase10<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 11: {
            launchTROWSUMCase11<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src);
            break;
        }
        case 12: {
            launchTROWSUMCase12<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src);
            break;
        }
        case 13: {
            launchTROWSUMCase13<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src);
            break;
        }
        case 14: {
            launchTROWSUMCase14<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src);
            break;
        }
        case 15: {
            launchTROWSUMCase15<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src);
            break;
        }
        case 16: {
            launchTROWSUMCase16<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src);
            break;
        }
        case 17: {
            launchTROWSUMCase17<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src);
            break;
        }
        case 18: {
            launchTROWSUMCase18<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src);
            break;
        }
        case 19: {
            launchTROWSUMCase19<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src);
            break;
        }
        case 20: {
            launchTROWSUMCase20<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src);
            break;
        }
        default: {
        }
    }
}

template void launchTROWSUMTestCase<1>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<2>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<3>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<4>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<5>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<6>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<7>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<8>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<9>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<10>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<11>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<12>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<13>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<14>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<15>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<16>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<17>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<18>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<19>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<20>(void *out, void *src, aclrtStream stream);