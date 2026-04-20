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
__global__ AICORE void RunCubeMatmul4Buf(__gm__ outType *out, __gm__ inType *src0, __gm__ inType *src1)
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

    for (uint32_t i = 0; i < K_ITERS; i++) {
        GlobalDataSrc0 src0Global(src0 + i * M * K);
        GlobalDataSrc1 src1Global(src1 + i * K * N);

        int buf_idx = i % 4;

        if (buf_idx == 0) {
            TLOAD(bMatTile0, src1Global);
        } else if (buf_idx == 1) {
            TLOAD(bMatTile1, src1Global);
        } else if (buf_idx == 2) {
            TLOAD(bMatTile2, src1Global);
        } else {
            TLOAD(bMatTile3, src1Global);
        }

        TLOAD(aMatTile, src0Global);

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

        TMOV(aTile, aMatTile);

        if (buf_idx == 0) {
            TMOV(bTile0, bMatTile0);
        } else if (buf_idx == 1) {
            TMOV(bTile1, bMatTile1);
        } else if (buf_idx == 2) {
            TMOV(bTile2, bMatTile2);
        } else {
            TMOV(bTile3, bMatTile3);
        }

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

        if (i == 0) {
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
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

void LaunchCubeMatmul4Buf(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunCubeMatmul4Buf<float, half><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out),
        reinterpret_cast<half *>(src0),
        reinterpret_cast<half *>(src1));
}
