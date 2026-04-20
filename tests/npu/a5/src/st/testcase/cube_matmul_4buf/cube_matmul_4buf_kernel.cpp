/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
CANN Open Software License Agreement Version 2.0

4-Buffer Matmul Kernel with Buffer-ID Based Sync (get_buf/rls_buf)
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

constexpr int M = 32;
constexpr int K = 16;
constexpr int N = 256;
constexpr int K_ITERS = 64;
constexpr int NUM_BUFS = 4;
constexpr int C_BUF_ID = 8;

template <int pipe>
AICORE void get_buffer(int id)
{
	get_buf(pipe,id,0);
	return;
}
template <int pipe>
AICORE void rls_buffer(int id)
{
	rls_buf(pipe,id,0);
	return;
}

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

    TileMatAData aMatTile0, aMatTile1, aMatTile2, aMatTile3;
    TASSIGN(aMatTile0, 0x0);
    TASSIGN(aMatTile1, 0x800);
    TASSIGN(aMatTile2, 0x1000);
    TASSIGN(aMatTile3, 0x1800);

    TileMatBData bMatTile0, bMatTile1, bMatTile2, bMatTile3;
    TASSIGN(bMatTile0, 0x10000);
    TASSIGN(bMatTile1, 0x12000);
    TASSIGN(bMatTile2, 0x14000);
    TASSIGN(bMatTile3, 0x16000);

    LeftTile aTile0, aTile1, aTile2, aTile3;
    TASSIGN(aTile0, 0x0);
    TASSIGN(aTile1, 0x400);
    TASSIGN(aTile2, 0x800);
    TASSIGN(aTile3, 0xC00);

    RightTile bTile0, bTile1, bTile2, bTile3;
    TASSIGN(bTile0, 0x0);
    TASSIGN(bTile1, 0x2000);
    TASSIGN(bTile2, 0x4000);
    TASSIGN(bTile3, 0x6000);

    AccTile cTile;
    TASSIGN(cTile, 0x0);

    for (uint32_t k = 0; k < K_ITERS; k++) {
        int buf_id = k % NUM_BUFS;

        GlobalDataSrc0 src0Global(src0 + k * M * K);
        GlobalDataSrc1 src1Global(src1 + k * K * N);

        // MTE2: Load A and B
#ifndef __PTO_AUTO__
        get_buffer<PIPE_MTE2>(buf_id);
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
        rls_buffer<PIPE_MTE2>(buf_id);
#endif

        // MTE1: Move to L0
#ifndef __PTO_AUTO__
        get_buffer<PIPE_MTE1>(buf_id);
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
        rls_buffer<PIPE_MTE1>(buf_id);
#endif

        // M: Matmul
#ifndef __PTO_AUTO__
        get_buffer<PIPE_M>(buf_id);
        get_buffer<PIPE_M>(C_BUF_ID);
#endif

        if (k == 0) {
            if (buf_id == 0) TMATMUL(cTile, aTile0, bTile0);
            else if (buf_id == 1) TMATMUL(cTile, aTile1, bTile1);
            else if (buf_id == 2) TMATMUL(cTile, aTile2, bTile2);
            else TMATMUL(cTile, aTile3, bTile3);
        } else {
            if (buf_id == 0) TMATMUL_ACC(cTile, cTile, aTile0, bTile0);
            else if (buf_id == 1) TMATMUL_ACC(cTile, cTile, aTile1, bTile1);
            else if (buf_id == 2) TMATMUL_ACC(cTile, cTile, aTile2, bTile2);
            else TMATMUL_ACC(cTile, cTile, aTile3, bTile3);
        }

#ifndef __PTO_AUTO__
        rls_buffer<PIPE_M>(buf_id);
        rls_buffer<PIPE_M>(C_BUF_ID);
#endif
    }

    // Store
#ifndef __PTO_AUTO__
    get_buffer<PIPE_MTE3>(C_BUF_ID);
#endif

    TSTORE(dstGlobal, cTile);

#ifndef __PTO_AUTO__
    rls_buffer<PIPE_MTE3>(C_BUF_ID);
#endif

    out = dstGlobal.data();
}

void LaunchCubeMatmul4Buf(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunCubeMatmul4Buf<float, half><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out),
        reinterpret_cast<half *>(src0),
        reinterpret_cast<half *>(src1));
}
