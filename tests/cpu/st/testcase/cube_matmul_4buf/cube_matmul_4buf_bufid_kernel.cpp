/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
CANN Open Software License Agreement Version 2.0

4-Buffer Matmul Kernel with Buffer-ID Based Sync (get_buf/rls_buf)

Key design:
- Both A and B tiles use 4-buffer rotation
- Uses get_buf/rls_buf instead of set_flag/wait_flag
- NO pipe_barrier calls needed
- NO pre-loop priming or post-loop draining needed

Buffer ID assignments:
  - A+B tiles: buf_id 0-3 (merged since always loaded together)
  - C accumulator: buf_id 8

Pipeline flow with buffer IDs:
  MTE2: get_buf<MTE2>(id) -> TLOAD A,B -> rls_buf<MTE2>(id)
  MTE1: get_buf<MTE1>(id) -> TMOV A,B -> rls_buf<MTE1>(id)
  M:    get_buf<M>(id) -> TMATMUL -> rls_buf<M>(id)
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

constexpr int M = 32;
constexpr int K = 16;
constexpr int N = 256;
constexpr int K_ITERS = 64;
constexpr int NUM_BUFS = 4;
constexpr int C_BUF_ID = 8;  // Accumulator buffer ID

// CPU stub for get_buf/rls_buf (no-op in CPU sim)
template <pipe_t pipe>
inline void get_buf(int buf_id)
{
    (void)buf_id;
}

template <pipe_t pipe>
inline void rls_buf(int buf_id)
{
    (void)buf_id;
}

template <typename outType, typename inType>
__global__ AICORE void RunCubeMatmul4BufWithBufIdSync(__gm__ outType *out, __gm__ inType *src0, __gm__ inType *src1)
{
    using GlobalDataSrc0 = GlobalTensor<inType, Shape<1, 1, 1, M, K>, Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<inType, Shape<1, 1, 1, K, N>, Stride<K * N, K * N, K * N, N, 1>>;
    using GlobalDataOut = GlobalTensor<outType, Shape<1, 1, 1, M, N>, Stride<M * N, M * N, M * N, N, 1>>;

    GlobalDataOut dstGlobal(out);

    // L1 buffer types
    using TileMatAData = Tile<TileType::Mat, inType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, inType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;

    // L0 buffer types
    using LeftTile = TileLeft<inType, M, K, M, K>;
    using RightTile = TileRight<inType, K, N, K, N>;
    using AccTile = TileAcc<outType, M, N, M, N>;

    // 4 buffers for A tiles in L1
    TileMatAData aMatTile0, aMatTile1, aMatTile2, aMatTile3;
    TASSIGN(aMatTile0, 0x0);
    TASSIGN(aMatTile1, 0x800);
    TASSIGN(aMatTile2, 0x1000);
    TASSIGN(aMatTile3, 0x1800);

    // 4 buffers for B tiles in L1
    TileMatBData bMatTile0, bMatTile1, bMatTile2, bMatTile3;
    TASSIGN(bMatTile0, 0x10000);
    TASSIGN(bMatTile1, 0x12000);
    TASSIGN(bMatTile2, 0x14000);
    TASSIGN(bMatTile3, 0x16000);

    // 4 buffers for A tiles in L0A
    LeftTile aTile0, aTile1, aTile2, aTile3;
    TASSIGN(aTile0, 0x0);
    TASSIGN(aTile1, 0x400);
    TASSIGN(aTile2, 0x800);
    TASSIGN(aTile3, 0xC00);

    // 4 buffers for B tiles in L0B
    RightTile bTile0, bTile1, bTile2, bTile3;
    TASSIGN(bTile0, 0x0);
    TASSIGN(bTile1, 0x2000);
    TASSIGN(bTile2, 0x4000);
    TASSIGN(bTile3, 0x6000);

    // Accumulator in L0C
    AccTile cTile;
    TASSIGN(cTile, 0x0);

    // Main K-loop with 4-buffer rotation
    for (uint32_t k = 0; k < K_ITERS; k++) {
        int buf_id = k % NUM_BUFS;  // A+B share same buf_id 0-3

        GlobalDataSrc0 src0Global(src0 + k * M * K);
        GlobalDataSrc1 src1Global(src1 + k * K * N);

        // ===== MTE2 Stage: Load A and B to L1 =====
#ifndef __PTO_AUTO__
        get_buf<PIPE_MTE2>(buf_id);
#endif

        if (buf_id == 0) {
            TLOAD(aMatTile0, src0Global);
            TLOAD(bMatTile0, src1Global);
        } else if (buf_id == 1) {
            TLOAD(aMatTile1, src0Global);
            TLOAD(bMatTile1, src1Global);
        } else if (buf_id == 2) {
            TLOAD(aMatTile2, src0Global);
            TLOAD(bMatTile2, src1Global);
        } else {
            TLOAD(aMatTile3, src0Global);
            TLOAD(bMatTile3, src1Global);
        }

#ifndef __PTO_AUTO__
        rls_buf<PIPE_MTE2>(buf_id);
#endif

        // ===== MTE1 Stage: Move A and B to L0 =====
#ifndef __PTO_AUTO__
        get_buf<PIPE_MTE1>(buf_id);
#endif

        if (buf_id == 0) {
            TMOV(aTile0, aMatTile0);
            TMOV(bTile0, bMatTile0);
        } else if (buf_id == 1) {
            TMOV(aTile1, aMatTile1);
            TMOV(bTile1, bMatTile1);
        } else if (buf_id == 2) {
            TMOV(aTile2, aMatTile2);
            TMOV(bTile2, bMatTile2);
        } else {
            TMOV(aTile3, aMatTile3);
            TMOV(bTile3, bMatTile3);
        }

#ifndef __PTO_AUTO__
        rls_buf<PIPE_MTE1>(buf_id);
#endif

        // ===== Cube M Stage: Matmul =====
#ifndef __PTO_AUTO__
        get_buf<PIPE_M>(buf_id);
        get_buf<PIPE_M>(C_BUF_ID);
#endif

        if (k == 0) {
            if (buf_id == 0) {
                TMATMUL(cTile, aTile0, bTile0);
            } else if (buf_id == 1) {
                TMATMUL(cTile, aTile1, bTile1);
            } else if (buf_id == 2) {
                TMATMUL(cTile, aTile2, bTile2);
            } else {
                TMATMUL(cTile, aTile3, bTile3);
            }
        } else {
            if (buf_id == 0) {
                TMATMUL_ACC(cTile, cTile, aTile0, bTile0);
            } else if (buf_id == 1) {
                TMATMUL_ACC(cTile, cTile, aTile1, bTile1);
            } else if (buf_id == 2) {
                TMATMUL_ACC(cTile, cTile, aTile2, bTile2);
            } else {
                TMATMUL_ACC(cTile, cTile, aTile3, bTile3);
            }
        }

#ifndef __PTO_AUTO__
        rls_buf<PIPE_M>(buf_id);
        rls_buf<PIPE_M>(C_BUF_ID);
#endif
    }

    // ===== Store result =====
#ifndef __PTO_AUTO__
    get_buf<PIPE_MTE3>(C_BUF_ID);
#endif

    TSTORE(dstGlobal, cTile);

#ifndef __PTO_AUTO__
    rls_buf<PIPE_MTE3>(C_BUF_ID);
#endif

    out = dstGlobal.data();
}

template <typename T_A, typename T_B, typename T_C>
void LaunchCubeMatmul4BufWithEventSync(T_A *a, T_B *b, T_C *c, void *stream)
{
    RunCubeMatmul4BufWithBufIdSync<T_C, T_A>(c, a, b);
}

template void LaunchCubeMatmul4BufWithEventSync<half, half, float>(half *, half *, float *, void *);
