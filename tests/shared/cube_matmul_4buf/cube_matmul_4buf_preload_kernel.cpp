/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
CANN Open Software License Agreement Version 2.0
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
    using GlobalDataSrc0 = GlobalTensor<inType, pto::Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<inType, pto::Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, N, 1>>;
    using GlobalDataOut = GlobalTensor<outType, pto::Shape<1, 1, 1, M, N>, pto::Stride<M * N, M * N, M * N, N, 1>>;

    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, inType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, inType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;

    using LeftTile = TileLeft<inType, M, K, M, K>;
    using RightTile = TileRight<inType, K, N, K, N>;
    using AccTile = TileAcc<outType, M, N, M, N>;

    TileMatAData aMatTile;
    TileMatBData bMatTile0, bMatTile1, bMatTile2, bMatTile3;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile0, 0x10000);
    TASSIGN(bMatTile1, 0x12000);
    TASSIGN(bMatTile2, 0x14000);
    TASSIGN(bMatTile3, 0x16000);

    LeftTile aTile;
    RightTile bTile0, bTile1, bTile2, bTile3;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile0, 0x0);
    TASSIGN(bTile1, 0x2000);
    TASSIGN(bTile2, 0x4000);
    TASSIGN(bTile3, 0x6000);
    TASSIGN(cTile, 0x0);

    // ==== PRIMING PHASE: Load B[0..3] ====
    {
        GlobalDataSrc1 srcB0(src1 + 0 * K * N);
        GlobalDataSrc1 srcB1(src1 + 1 * K * N);
        GlobalDataSrc1 srcB2(src1 + 2 * K * N);
        GlobalDataSrc1 srcB3(src1 + 3 * K * N);
        
        TLOAD(bMatTile0, srcB0);
        TLOAD(bMatTile1, srcB1);
        TLOAD(bMatTile2, srcB2);
        TLOAD(bMatTile3, srcB3);
    }
    
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_MTE2);
#endif
    
    TMOV(bTile0, bMatTile0);
    TMOV(bTile1, bMatTile1);
    TMOV(bTile2, bMatTile2);
    TMOV(bTile3, bMatTile3);
    
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_MTE1);
#endif

    // ==== MAIN LOOP ====
    for (uint32_t k = 0; k < K_ITERS; k++) {
        int buf_idx = k % 4;
        
        GlobalDataSrc0 src0Global(src0 + k * M * K);
        TLOAD(aMatTile, src0Global);

#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_MTE2);
#endif
        
        TMOV(aTile, aMatTile);

#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_MTE1);
#endif

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

#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_M);
#endif

        // Preload B[k+4]
        if (k + 4 < K_ITERS) {
            GlobalDataSrc1 srcBPreload(src1 + (k + 4) * K * N);
            
            if (buf_idx == 0) {
                TLOAD(bMatTile0, srcBPreload);
#ifndef __PTO_AUTO__
                pipe_barrier(PIPE_MTE2);
#endif
                TMOV(bTile0, bMatTile0);
            } else if (buf_idx == 1) {
                TLOAD(bMatTile1, srcBPreload);
#ifndef __PTO_AUTO__
                pipe_barrier(PIPE_MTE2);
#endif
                TMOV(bTile1, bMatTile1);
            } else if (buf_idx == 2) {
                TLOAD(bMatTile2, srcBPreload);
#ifndef __PTO_AUTO__
                pipe_barrier(PIPE_MTE2);
#endif
                TMOV(bTile2, bMatTile2);
            } else {
                TLOAD(bMatTile3, srcBPreload);
#ifndef __PTO_AUTO__
                pipe_barrier(PIPE_MTE2);
#endif
                TMOV(bTile3, bMatTile3);
            }
            
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_MTE1);
#endif
        }
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T_A, typename T_B, typename T_C>
void LaunchCubeMatmul4BufPreload(T_A *a, T_B *b, T_C *c, void *stream)
{
    RunCubeMatmul4BufPreload<T_C, T_A><<<1, nullptr, stream>>>(c, a, b);
}

template void LaunchCubeMatmul4BufPreload<half, half, float>(half *, half *, float *, void *);
