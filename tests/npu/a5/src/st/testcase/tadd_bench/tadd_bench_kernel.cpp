/**
 * TADD Benchmark Kernel - Auto-generated from input.csv
 * 10x chained TADD with pipe_barrier for isolated VF measurement
 */
#include "kernel_operator.h"

using namespace AscendC;

template <>
__aicore__ inline void LaunchTAddBench<float, 1, 4096, 1, 4096>(float *dst, float *src0, float *src1, void *stream)
{
    constexpr int tileH = 1;
    constexpr int tileW = 4096;
    constexpr int vRows = 1;
    constexpr int vCols = 4096;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc0(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc1(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc0 = (GM_ADDR)src0;
    GM_ADDR gmSrc1 = (GM_ADDR)src1;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    // 10x TADD with barriers for isolated VF measurement
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc0, gmSrc0, tileW * sizeof(float));
        TLOAD(tileSrc1, gmSrc1, tileW * sizeof(float));
        TADD(tileDst, tileSrc0, tileSrc1);
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

template <>
__aicore__ inline void LaunchTAddBench<float, 1, 8192, 1, 8192>(float *dst, float *src0, float *src1, void *stream)
{
    constexpr int tileH = 1;
    constexpr int tileW = 8192;
    constexpr int vRows = 1;
    constexpr int vCols = 8192;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc0(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc1(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc0 = (GM_ADDR)src0;
    GM_ADDR gmSrc1 = (GM_ADDR)src1;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    // 10x TADD with barriers for isolated VF measurement
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc0, gmSrc0, tileW * sizeof(float));
        TLOAD(tileSrc1, gmSrc1, tileW * sizeof(float));
        TADD(tileDst, tileSrc0, tileSrc1);
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

template <>
__aicore__ inline void LaunchTAddBench<float, 1, 16384, 1, 16384>(float *dst, float *src0, float *src1, void *stream)
{
    constexpr int tileH = 1;
    constexpr int tileW = 16384;
    constexpr int vRows = 1;
    constexpr int vCols = 16384;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc0(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc1(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc0 = (GM_ADDR)src0;
    GM_ADDR gmSrc1 = (GM_ADDR)src1;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    // 10x TADD with barriers for isolated VF measurement
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc0, gmSrc0, tileW * sizeof(float));
        TLOAD(tileSrc1, gmSrc1, tileW * sizeof(float));
        TADD(tileDst, tileSrc0, tileSrc1);
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

template <>
__aicore__ inline void LaunchTAddBench<float, 8, 512, 8, 512>(float *dst, float *src0, float *src1, void *stream)
{
    constexpr int tileH = 8;
    constexpr int tileW = 512;
    constexpr int vRows = 8;
    constexpr int vCols = 512;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc0(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc1(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc0 = (GM_ADDR)src0;
    GM_ADDR gmSrc1 = (GM_ADDR)src1;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    // 10x TADD with barriers for isolated VF measurement
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc0, gmSrc0, tileW * sizeof(float));
        TLOAD(tileSrc1, gmSrc1, tileW * sizeof(float));
        TADD(tileDst, tileSrc0, tileSrc1);
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

template <>
__aicore__ inline void LaunchTAddBench<float, 16, 512, 16, 512>(float *dst, float *src0, float *src1, void *stream)
{
    constexpr int tileH = 16;
    constexpr int tileW = 512;
    constexpr int vRows = 16;
    constexpr int vCols = 512;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc0(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc1(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc0 = (GM_ADDR)src0;
    GM_ADDR gmSrc1 = (GM_ADDR)src1;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    // 10x TADD with barriers for isolated VF measurement
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc0, gmSrc0, tileW * sizeof(float));
        TLOAD(tileSrc1, gmSrc1, tileW * sizeof(float));
        TADD(tileDst, tileSrc0, tileSrc1);
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

template <>
__aicore__ inline void LaunchTAddBench<float, 32, 512, 32, 512>(float *dst, float *src0, float *src1, void *stream)
{
    constexpr int tileH = 32;
    constexpr int tileW = 512;
    constexpr int vRows = 32;
    constexpr int vCols = 512;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc0(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc1(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc0 = (GM_ADDR)src0;
    GM_ADDR gmSrc1 = (GM_ADDR)src1;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    // 10x TADD with barriers for isolated VF measurement
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc0, gmSrc0, tileW * sizeof(float));
        TLOAD(tileSrc1, gmSrc1, tileW * sizeof(float));
        TADD(tileDst, tileSrc0, tileSrc1);
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

template <>
__aicore__ inline void LaunchTAddBench<float, 64, 64, 64, 64>(float *dst, float *src0, float *src1, void *stream)
{
    constexpr int tileH = 64;
    constexpr int tileW = 64;
    constexpr int vRows = 64;
    constexpr int vCols = 64;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc0(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc1(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc0 = (GM_ADDR)src0;
    GM_ADDR gmSrc1 = (GM_ADDR)src1;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    // 10x TADD with barriers for isolated VF measurement
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc0, gmSrc0, tileW * sizeof(float));
        TLOAD(tileSrc1, gmSrc1, tileW * sizeof(float));
        TADD(tileDst, tileSrc0, tileSrc1);
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

template <>
__aicore__ inline void LaunchTAddBench<float, 64, 128, 64, 128>(float *dst, float *src0, float *src1, void *stream)
{
    constexpr int tileH = 64;
    constexpr int tileW = 128;
    constexpr int vRows = 64;
    constexpr int vCols = 128;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc0(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc1(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc0 = (GM_ADDR)src0;
    GM_ADDR gmSrc1 = (GM_ADDR)src1;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    // 10x TADD with barriers for isolated VF measurement
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc0, gmSrc0, tileW * sizeof(float));
        TLOAD(tileSrc1, gmSrc1, tileW * sizeof(float));
        TADD(tileDst, tileSrc0, tileSrc1);
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

template <>
__aicore__ inline void LaunchTAddBench<float, 128, 128, 128, 128>(float *dst, float *src0, float *src1, void *stream)
{
    constexpr int tileH = 128;
    constexpr int tileW = 128;
    constexpr int vRows = 128;
    constexpr int vCols = 128;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc0(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc1(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc0 = (GM_ADDR)src0;
    GM_ADDR gmSrc1 = (GM_ADDR)src1;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    // 10x TADD with barriers for isolated VF measurement
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc0, gmSrc0, tileW * sizeof(float));
        TLOAD(tileSrc1, gmSrc1, tileW * sizeof(float));
        TADD(tileDst, tileSrc0, tileSrc1);
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

