/**
 * Simple TBMM_QK kernel for M=128,K=128,N=128.
 */

#include <pto/common/constants.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>
#include "pto_macro_matmul.hpp"

using namespace pto;


// Templated kernel implementing mixed-precision matmul:
// inputs `q` and `k` are FP16 (half), output `out` is FP32 (float).
template<uint32_t TM, uint32_t TK, uint32_t TN>
__global__ AICORE void LaunchTBMM_QK_kern(__gm__ float *out, __gm__ half *q, __gm__ half *k) {
    constexpr uint32_t M = TM; 
    constexpr uint32_t N = TN; 
    constexpr uint32_t K = TK; 
    constexpr uint32_t validM = M;
    constexpr uint32_t validN = N;
    constexpr uint32_t validK = K;
    constexpr uint32_t Cube_M = M;
    constexpr uint32_t Cube_N = N;
    constexpr uint32_t Cube_K = K;

    // Global tensor typedefs (static shapes)
    using GlobalDataSrc0 = GlobalTensor<half, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<half, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataOut = GlobalTensor<float, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;

    GlobalDataSrc0 src0Global(q);
    GlobalDataSrc1 src1Global(k);
    GlobalDataOut dstGlobal(out);

    // Mat tiles: half inputs, float accumulation
    using TileMatAData = Tile<TileType::Mat, half, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, half, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using AccTile = TileAcc<float, M, N, validM, validN>;

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
    pto_macro_matmul<Cube_M, Cube_K, Cube_N>(aMatTile, bMatTile, cTile);

    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    
    /********************************TSTORE****************************/

    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template<uint32_t TM, uint32_t TK, uint32_t TN>
__global__ AICORE void LaunchTBMM_QK_kern_NT(__gm__ float *out, __gm__ half *q, __gm__ half *k) {
    constexpr uint32_t M = TM; 
    constexpr uint32_t N = TN; 
    constexpr uint32_t K = TK; 
    constexpr uint32_t validM = M;
    constexpr uint32_t validN = N;
    constexpr uint32_t validK = K;
    constexpr uint32_t Cube_M = M;
    constexpr uint32_t Cube_N = N;
    constexpr uint32_t Cube_K = K > 128 ? 64 : K;

    // Global tensor typedefs (static shapes)
    using GlobalDataSrc0 = GlobalTensor<half, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<1 , 1 , 1, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<half, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<1 , 1 , 1, 1, validK>, Layout::DN>;
    using GlobalDataOut = GlobalTensor<float, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;

    GlobalDataSrc0 src0Global(q);
    GlobalDataSrc1 src1Global(k);
    GlobalDataOut dstGlobal(out);

    // Mat tiles: half inputs, float accumulation
    using TileMatAData = Tile<TileType::Mat, half, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, half, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>;
    using AccTile = TileAcc<float, M, N, validM, validN>;

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
    pto_macro_matmul<Cube_M, validK, Cube_N>(aMatTile, bMatTile, cTile);

    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    
    /********************************TSTORE****************************/

    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template<uint32_t TM, uint32_t TK, uint32_t TN>
__global__ AICORE void LaunchTBMM_QK_kern_split_K(__gm__ float *out, __gm__ half *q, __gm__ half *k) {
    constexpr uint32_t M = TM; 
    constexpr uint32_t N = TN; 
    constexpr uint32_t K = TK; 
    constexpr uint32_t validM = M;
    constexpr uint32_t validN = N;
    constexpr uint32_t validK = K;
    constexpr uint32_t Cube_M = M;
    constexpr uint32_t Cube_N = N;
    constexpr uint32_t Cube_K = 64; //calculateFittingCubeK(Cube_M, Cube_N)

    // Global tensor typedefs (static shapes)
    using GlobalDataSrc0 = GlobalTensor<half, pto::Shape<1, 1, 1, validM, Cube_K>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<half, pto::Shape<1, 1, 1, Cube_K, validN>,
        pto::Stride<1 * Cube_K * validN, 1 * Cube_K * validN, Cube_K * validN, validN, 1>>;
    using GlobalDataOut = GlobalTensor<float, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;


    GlobalDataOut dstGlobal(out);

    // Mat tiles: half inputs, float accumulation
    using TileMatAData = Tile<TileType::Mat, half, M, Cube_K, BLayout::ColMajor, M, Cube_K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, half, Cube_K, N, BLayout::ColMajor, Cube_K, N, SLayout::RowMajor, 512>;
    using AccTile = TileAcc<float, M, N, validM, validN>;

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
        GlobalDataSrc1 src1Global(k + validN * i * Cube_K);

        /******************************TLOAD*****************************/
        TLOAD(aMatTile, src0Global);
        TLOAD(bMatTile, src1Global);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);


        /**************************TMOV && TEXTRACT**************************/
        pto_macro_matmul<Cube_M, Cube_K, Cube_N>(aMatTile, bMatTile, cTile);

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

// Explicit wrappers for specific (M,K,N) instantiations so tests can
// call the correct kernel variant.
// extern "C" void LaunchTBMM_QK_128_128_128(float *out, uint16_t *q, uint16_t *k, void *stream) {
//     LaunchTBMM_QK_kern<128, 128, 128><<<1, nullptr, stream>>>(out, (half*)q, (half*)k);
// }

extern "C" void LaunchTBMM_QK_128_128_128_NT(float *out, uint16_t *q, uint16_t *k, void *stream) {
    LaunchTBMM_QK_kern_NT<128, 128, 128><<<1, nullptr, stream>>>(out, (half*)q, (half*)k);
}

extern "C" void LaunchTBMM_QK_256_128_64_NT(float *out, uint16_t *q, uint16_t *k, void *stream) {
    LaunchTBMM_QK_kern_NT<256, 128, 64><<<1, nullptr, stream>>>(out, (half*)q, (half*)k);
}

extern "C" void LaunchTBMM_QK_64_256_64_NT(float *out, uint16_t *q, uint16_t *k, void *stream) {
    LaunchTBMM_QK_kern_NT<64, 256, 64><<<1, nullptr, stream>>>(out, (half*)q, (half*)k);
}

extern "C" void LaunchTBMM_QK_64_128_128_NT(float *out, uint16_t *q, uint16_t *k, void *stream) {
    LaunchTBMM_QK_kern_NT<64, 128, 128><<<1, nullptr, stream>>>(out, (half*)q, (half*)k);
}

extern "C" void LaunchTBMM_QK_256_128_128_NT(float *out, uint16_t *q, uint16_t *k, void *stream) {
    LaunchTBMM_QK_kern_NT<256, 128, 128><<<1, nullptr, stream>>>(out, (half*)q, (half*)k);
}

extern "C" void LaunchTBMM_QK_128_256_64_NT(float *out, uint16_t *q, uint16_t *k, void *stream) {
    LaunchTBMM_QK_kern_NT<128, 256, 64><<<1, nullptr, stream>>>(out, (half*)q, (half*)k);
}

extern "C" void LaunchTBMM_QK_128_64_128_NT(float *out, uint16_t *q, uint16_t *k, void *stream) {
    LaunchTBMM_QK_kern_NT<128, 64, 128><<<1, nullptr, stream>>>(out, (half*)q, (half*)k);
}

extern "C" void LaunchTBMM_QK_128_128_64_NT(float *out, uint16_t *q, uint16_t *k, void *stream) {
    LaunchTBMM_QK_kern_NT<128, 128, 64><<<1, nullptr, stream>>>(out, (half*)q, (half*)k);
}

extern "C" void LaunchTBMM_QK_64_64_128_NT(float *out, uint16_t *q, uint16_t *k, void *stream) {
    LaunchTBMM_QK_kern_NT<64, 64, 128><<<1, nullptr, stream>>>(out, (half*)q, (half*)k);
}

extern "C" void LaunchTBMM_QK_128_128_128_split(float *out, uint16_t *q, uint16_t *k, void *stream) {
    LaunchTBMM_QK_kern_split_K<128, 128, 128><<<1, nullptr, stream>>>(out, (half*)q, (half*)k);
}

// extern "C" void LaunchTBMM_QK_128_64_128(float *out, uint16_t *q, uint16_t *k, void *stream) {
//     LaunchTBMM_QK_kern<128, 64, 128><<<1, nullptr, stream>>>(out, (half*)q, (half*)k);
// }