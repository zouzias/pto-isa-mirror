/**
 * Simple TBMM_QK kernel for M=128,K=128,N=128.
 */

#include <pto/common/constants.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>
#include "pto_macro_matmul.hpp"

using namespace pto;


// NN variant: both matrices are Normal (row-major)
template<uint32_t TM, uint32_t TK, uint32_t TN>
__global__ AICORE void LaunchTBMM_QK_kern_NN(__gm__ float *out, __gm__ half *q, __gm__ half *k) {
    constexpr uint32_t M = TM; 
    constexpr uint32_t N = TN; 
    constexpr uint32_t K = TK; 
    constexpr uint32_t Cube_M = M;
    constexpr uint32_t Cube_N = N;
    constexpr uint32_t Cube_K = calculateFittingCubeK(Cube_M, Cube_N);

    // Global tensor typedefs (static shapes)
    using GlobalDataSrc0 = GlobalTensor<half, pto::Shape<1, 1, 1, M, Cube_K>,
        pto::Stride<1, 1 , 1, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<half, pto::Shape<1, 1, 1, Cube_K, N>,
        pto::Stride<1 , 1 , 1 , N, 1>>;
    using GlobalDataOut = GlobalTensor<float, pto::Shape<1, 1, 1, M, N>,
        pto::Stride<1, 1 , 1, N, 1>>;

    GlobalDataOut dstGlobal(out);

    // Mat tiles: half inputs, float accumulation
    using TileMatAData = Tile<TileType::Mat, half, M, Cube_K, BLayout::ColMajor, M, Cube_K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, half, Cube_K, N, BLayout::ColMajor, Cube_K, N, SLayout::RowMajor, 512>;
    using AccTile = TileAcc<float, M, N, M, N>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    AccTile cTile;
    TASSIGN(cTile, 0x0);

    constexpr int iter = K / Cube_K;
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

    for (int16_t i = 0; i < iter; i++) {
        GlobalDataSrc0 src0Global(q + i * Cube_K);
        GlobalDataSrc1 src1Global(k + N * i * Cube_K);

        /******************************TLOAD*****************************/
        TLOAD(aMatTile, src0Global);
        TLOAD(bMatTile, src1Global);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        /**************************TMOV && TEXTRACT**************************/
        if (i == 0) {
            pto_macro_matmul<Cube_M, Cube_K, Cube_N>(aMatTile, bMatTile, cTile, false);
        } else {
            pto_macro_matmul<Cube_M, Cube_K, Cube_N>(aMatTile, bMatTile, cTile, true);
        }
        

        set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    }

    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    
    /********************************TSTORE****************************/

    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

//NT_inner is the only test can split k in inner loop, due to TEXTRACT only can extract from continuous memory
template<uint32_t TM, uint32_t TK, uint32_t TN>
__global__ AICORE void LaunchTBMM_QK_kern_NT_inner(__gm__ float *out, __gm__ half *q, __gm__ half *k) {
    constexpr uint32_t M = TM; 
    constexpr uint32_t N = TN; 
    constexpr uint32_t K = TK; 
    constexpr uint32_t Cube_M = M;
    constexpr uint32_t Cube_N = N;

    // Global tensor typedefs (static shapes)
    using GlobalDataSrc0 = GlobalTensor<half, pto::Shape<1, 1, 1, M, K>,
        pto::Stride<1 , 1 , 1, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<half, pto::Shape<1, 1, 1, K, N>,
        pto::Stride<1 , 1 , 1, 1, K>, Layout::DN>;
    using GlobalDataOut = GlobalTensor<float, pto::Shape<1, 1, 1, M, N>,
        pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

    GlobalDataSrc0 src0Global(q);
    GlobalDataSrc1 src1Global(k);
    GlobalDataOut dstGlobal(out);

    // Mat tiles: half inputs, float accumulation
    using TileMatAData = Tile<TileType::Mat, half, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, half, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>;
    using AccTile = TileAcc<float, M, N, M, N>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    AccTile cTile;
    TASSIGN(cTile, 0x0);

    /******************************TLOAD*****************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

    /**************************TMOV && TEXTRACT**************************/
    pto_macro_matmul<Cube_M, K, Cube_N>(aMatTile, bMatTile, cTile);

    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    
    /********************************TSTORE****************************/

    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}


// NT variant with outer K loop (default NT version)
template<uint32_t TM, uint32_t TK, uint32_t TN>
__global__ AICORE void LaunchTBMM_QK_kern_NT(__gm__ float *out, __gm__ half *q, __gm__ half *k) {
    constexpr uint32_t M = TM; 
    constexpr uint32_t N = TN; 
    constexpr uint32_t K = TK; 
    constexpr uint32_t Cube_M = M;
    constexpr uint32_t Cube_N = N;
    constexpr uint32_t Cube_K = calculateFittingCubeK(Cube_M, Cube_N);

    // Global tensor typedefs (static shapes)
    using GlobalDataSrc0 = GlobalTensor<half, pto::Shape<1, 1, 1, M, Cube_K>,
        pto::Stride<1, 1 , 1, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<half, pto::Shape<1, 1, 1, Cube_K, N>,
        pto::Stride<1 , 1 , 1, 1, K>, Layout::DN>;
    using GlobalDataOut = GlobalTensor<float, pto::Shape<1, 1, 1, M, N>,
        pto::Stride<1, 1 , 1, N, 1>>;

    GlobalDataOut dstGlobal(out);

    // Mat tiles: half inputs, float accumulation
    using TileMatAData = Tile<TileType::Mat, half, M, Cube_K, BLayout::ColMajor, M, Cube_K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, half, Cube_K, N, BLayout::RowMajor, Cube_K, N, SLayout::ColMajor, 512>;
    using AccTile = TileAcc<float, M, N, M, N>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    AccTile cTile;
    TASSIGN(cTile, 0x0);

    constexpr int iter = K / Cube_K;
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

    for (int16_t i = 0; i < iter; i++) {
        GlobalDataSrc0 src0Global(q + i * Cube_K);
        GlobalDataSrc1 src1Global(k + i * Cube_K);

        /******************************TLOAD*****************************/
        TLOAD(aMatTile, src0Global);
        TLOAD(bMatTile, src1Global);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        /**************************TMOV && TEXTRACT**************************/
        if (i == 0) {
            pto_macro_matmul<Cube_M, Cube_K, Cube_N>(aMatTile, bMatTile, cTile, false);
        } else {
            pto_macro_matmul<Cube_M, Cube_K, Cube_N>(aMatTile, bMatTile, cTile, true);
        }
        

        set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    }

    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    
    /********************************TSTORE****************************/

    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}


// TN cannot split K inner loop, so the testcase will only either split k in outer loop or not split K at all.
template<uint32_t TM, uint32_t TK, uint32_t TN>
__global__ AICORE void LaunchTBMM_QK_kern_TN(__gm__ float *out, __gm__ half *q, __gm__ half *k) {
    constexpr uint32_t M = TM; 
    constexpr uint32_t N = TN; 
    constexpr uint32_t K = TK; 
    constexpr uint32_t Cube_M = M;
    constexpr uint32_t Cube_N = N;
    constexpr uint32_t Cube_K = calculateFittingCubeK(Cube_M, Cube_N);

    // Global tensor typedefs (static shapes)
    using GlobalDataSrc0 = GlobalTensor<half, pto::Shape<1, 1, 1, M, Cube_K>,
        pto::Stride<1 , 1 , 1 , 1, M>, Layout::DN>;
    using GlobalDataSrc1 = GlobalTensor<half, pto::Shape<1, 1, 1, Cube_K, N>,
        pto::Stride<1 , 1 , 1, N, 1>>;
    using GlobalDataOut = GlobalTensor<float, pto::Shape<1, 1, 1, M, N>,
        pto::Stride<1 , 1 , 1, N, 1>>;

    GlobalDataOut dstGlobal(out);

    // Mat tiles: half inputs, float accumulation
    using TileMatAData = Tile<TileType::Mat, half, M, Cube_K, BLayout::RowMajor, M, Cube_K, SLayout::ColMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, half, Cube_K, N, BLayout::ColMajor, Cube_K, N, SLayout::RowMajor, 512>;
    using AccTile = TileAcc<float, M, N, M, N>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    AccTile cTile;
    TASSIGN(cTile, 0x0);

    constexpr int iter = K / Cube_K;
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

    for (int16_t i = 0; i < iter; i++) {
        GlobalDataSrc0 src0Global(q + i * Cube_K * M);
        GlobalDataSrc1 src1Global(k + N * i * Cube_K);

        /******************************TLOAD*****************************/
        TLOAD(aMatTile, src0Global);
        TLOAD(bMatTile, src1Global);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        /**************************TMOV && TEXTRACT**************************/
        if (i == 0) {
            pto_macro_matmul<Cube_M, Cube_K, Cube_N>(aMatTile, bMatTile, cTile, false);
        } else {
            pto_macro_matmul<Cube_M, Cube_K, Cube_N>(aMatTile, bMatTile, cTile, true);
        }
        

        set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    }

    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    
    /********************************TSTORE****************************/

    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}



// Explicit wrappers - use macros to reduce boilerplate
#define DEFINE_KERNEL_WRAPPER(M, K, N, VARIANT) \
    extern "C" void LaunchTBMM_QK_##M##_##K##_##N##_##VARIANT(float *out, uint16_t *q, uint16_t *k, void *stream) { \
        LaunchTBMM_QK_kern_##VARIANT<M, K, N><<<1, nullptr, stream>>>(out, (half*)q, (half*)k); \
    }

// NT variant wrappers (outer K loop - default NT version)
DEFINE_KERNEL_WRAPPER(128, 128, 128, NT)
DEFINE_KERNEL_WRAPPER(256, 128, 64, NT)
DEFINE_KERNEL_WRAPPER(64, 256, 64, NT)
DEFINE_KERNEL_WRAPPER(64, 128, 128, NT)
DEFINE_KERNEL_WRAPPER(256, 128, 128, NT)
DEFINE_KERNEL_WRAPPER(128, 256, 64, NT)
DEFINE_KERNEL_WRAPPER(128, 128, 64, NT)

// NT_inner variant wrappers (inner K loop version)
DEFINE_KERNEL_WRAPPER(128, 128, 128, NT_inner)
DEFINE_KERNEL_WRAPPER(256, 128, 64, NT_inner)
DEFINE_KERNEL_WRAPPER(64, 256, 64, NT_inner)
DEFINE_KERNEL_WRAPPER(64, 128, 128, NT_inner)
DEFINE_KERNEL_WRAPPER(256, 128, 128, NT_inner)
DEFINE_KERNEL_WRAPPER(128, 256, 64, NT_inner)
DEFINE_KERNEL_WRAPPER(128, 128, 64, NT_inner)

// NN variant wrappers
DEFINE_KERNEL_WRAPPER(128, 128, 128, NN)
DEFINE_KERNEL_WRAPPER(256, 128, 64, NN)
DEFINE_KERNEL_WRAPPER(64, 256, 64, NN)
DEFINE_KERNEL_WRAPPER(64, 128, 128, NN)
DEFINE_KERNEL_WRAPPER(256, 128, 128, NN)
DEFINE_KERNEL_WRAPPER(128, 256, 64, NN)
DEFINE_KERNEL_WRAPPER(128, 128, 64, NN)

// TN variant wrappers
DEFINE_KERNEL_WRAPPER(128, 128, 128, TN)
DEFINE_KERNEL_WRAPPER(256, 128, 64, TN)
DEFINE_KERNEL_WRAPPER(64, 256, 64, TN)
DEFINE_KERNEL_WRAPPER(64, 128, 128, TN)
DEFINE_KERNEL_WRAPPER(256, 128, 128, TN)
DEFINE_KERNEL_WRAPPER(128, 256, 64, TN)
DEFINE_KERNEL_WRAPPER(128, 128, 64, TN)

#undef DEFINE_KERNEL_WRAPPER