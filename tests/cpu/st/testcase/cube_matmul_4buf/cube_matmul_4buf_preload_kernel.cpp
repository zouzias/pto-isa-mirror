/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
CANN Open Software License Agreement Version 2.0
*/

/**
 * @file cube_matmul_4buf_preload_kernel.cpp
 * @brief Cube matmul kernel with 4-buffer CORRECT preload
 * 
 * Key insight: preload B[k+4] into buffer (k+4)%4 = k%4, which is the SAME
 * buffer we're computing with. So we MUST ensure the preload happens AFTER
 * the matmul of that iteration.
 * 
 * Correct order for each iteration k:
 * 1. Load A[k]
 * 2. Wait for A ready
 * 3. Move A to L0A
 * 4. Wait for B[k] ready in L0B (was preloaded earlier)
 * 5. TMATMUL with A[k], B[k]
 * 6. Wait for matmul done
 * 7. NOW safe to preload B[k+4] into the same buffer
 */

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

constexpr int M = 32;
constexpr int K = 16;
constexpr int N = 256;
constexpr int K_ITERS = 64;

template <typename outType, typename inType>
__global__ AICORE void RunCubeMatmul4BufPreload(__gm__ outType *out, __gm__ inType *src0, __gm__ inType *src1)
{
    using GlobalDataSrc0 = GlobalTensor<inType, Shape<1, 1, 1, M, K>, Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<inType, Shape<1, 1, 1, K, N>, Stride<K * N, K * N, K * N, N, 1>>;
    using GlobalDataOut = GlobalTensor<outType, Shape<1, 1, 1, M, N>, Stride<M * N, M * N, M * N, N, 1>>;

    GlobalDataOut dstGlobal(out);

    // L1 tile types
    using TileMatAData = Tile<TileType::Mat, inType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, inType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;

    // L0 tile types
    using LeftTile = TileLeft<inType, M, K, M, K>;
    using RightTile = TileRight<inType, K, N, K, N>;
    using AccTile = TileAcc<outType, M, N, M, N>;

    // L1 buffers
    TileMatAData aMatTile;
    TileMatBData bMatTile0, bMatTile1, bMatTile2, bMatTile3;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile0, 0x10000);
    TASSIGN(bMatTile1, 0x12000);
    TASSIGN(bMatTile2, 0x14000);
    TASSIGN(bMatTile3, 0x16000);

    // L0 tiles
    LeftTile aTile;
    RightTile bTile0, bTile1, bTile2, bTile3;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile0, 0x0);
    TASSIGN(bTile1, 0x2000);
    TASSIGN(bTile2, 0x4000);
    TASSIGN(bTile3, 0x6000);
    TASSIGN(cTile, 0x0);

    // Helper lambda to load B tile to L1 and L0
    auto loadBTile = [&](int buf, GlobalDataSrc1& src) {
        if (buf == 0) {
            TLOAD(bMatTile0, src);
            pipe_barrier(PIPE_MTE2);
            TMOV(bTile0, bMatTile0);
        } else if (buf == 1) {
            TLOAD(bMatTile1, src);
            pipe_barrier(PIPE_MTE2);
            TMOV(bTile1, bMatTile1);
        } else if (buf == 2) {
            TLOAD(bMatTile2, src);
            pipe_barrier(PIPE_MTE2);
            TMOV(bTile2, bMatTile2);
        } else {
            TLOAD(bMatTile3, src);
            pipe_barrier(PIPE_MTE2);
            TMOV(bTile3, bMatTile3);
        }
        pipe_barrier(PIPE_MTE1);
    };

    // ==== PRIMING PHASE: Load B[0..3] ====
    for (int i = 0; i < 4; i++) {
        GlobalDataSrc1 srcB(src1 + i * K * N);
        loadBTile(i, srcB);
    }

    // ==== MAIN LOOP ====
    for (uint32_t k = 0; k < K_ITERS; k++) {
        int buf_idx = k % 4;
        
        // 1. Load A[k] 
        GlobalDataSrc0 src0Global(src0 + k * M * K);
        TLOAD(aMatTile, src0Global);
        pipe_barrier(PIPE_MTE2);
        
        // 2. Move A[k] to L0A
        TMOV(aTile, aMatTile);
        pipe_barrier(PIPE_MTE1);

        // 3. Matmul with B[buf_idx] (already in L0B from priming or previous preload)
        if (k == 0) {
            if (buf_idx == 0) TMATMUL(cTile, aTile, bTile0);
            else if (buf_idx == 1) TMATMUL(cTile, aTile, bTile1);
            else if (buf_idx == 2) TMATMUL(cTile, aTile, bTile2);
            else TMATMUL(cTile, aTile, bTile3);
        } else {
            if (buf_idx == 0) TMATMUL_ACC(cTile, cTile, aTile, bTile0);
            else if (buf_idx == 1) TMATMUL_ACC(cTile, cTile, aTile, bTile1);
            else if (buf_idx == 2) TMATMUL_ACC(cTile, cTile, aTile, bTile2);
            else TMATMUL_ACC(cTile, cTile, aTile, bTile3);
        }
        pipe_barrier(PIPE_M);  // Wait for matmul to finish

        // 4. NOW safe to preload B[k+4] into same buffer (for iteration k+4)
        if (k + 4 < K_ITERS) {
            GlobalDataSrc1 srcBPreload(src1 + (k + 4) * K * N);
            loadBTile(buf_idx, srcBPreload);  // (k+4) % 4 == k % 4 == buf_idx
        }
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T_A, typename T_B, typename T_C>
void LaunchCubeMatmul4BufPreload(T_A *a, T_B *b, T_C *c, void *stream)
{
    RunCubeMatmul4BufPreload<T_C, T_A>(c, a, b);
}

template void LaunchCubeMatmul4BufPreload<half, half, float>(half *, half *, float *, void *);
