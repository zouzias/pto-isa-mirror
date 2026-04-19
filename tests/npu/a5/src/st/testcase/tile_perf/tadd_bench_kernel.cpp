/**
 * Tile Op Benchmark Kernel - Auto-generated from input.csv
 * Supports: binary (tadd), unary (texp), scalar (tadds) ops
 * 10x ops with pipe_barrier for isolated VF measurement
 */
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

// Binary op kernel: 2 tiles -> 1 tile
template <>
__aicore__ inline void LaunchBinaryBench<float, 1, 4096, 1, 4096>(const char* op, float *dst, float *src0, float *src1, void *stream)
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
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc0, gmSrc0, tileW * sizeof(float));
        TLOAD(tileSrc1, gmSrc1, tileW * sizeof(float));
        TADD(tileDst, tileSrc0, tileSrc1);  // Binary op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Binary op kernel: 2 tiles -> 1 tile
template <>
__aicore__ inline void LaunchBinaryBench<float, 1, 8192, 1, 8192>(const char* op, float *dst, float *src0, float *src1, void *stream)
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
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc0, gmSrc0, tileW * sizeof(float));
        TLOAD(tileSrc1, gmSrc1, tileW * sizeof(float));
        TADD(tileDst, tileSrc0, tileSrc1);  // Binary op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Binary op kernel: 2 tiles -> 1 tile
template <>
__aicore__ inline void LaunchBinaryBench<float, 1, 16384, 1, 16384>(const char* op, float *dst, float *src0, float *src1, void *stream)
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
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc0, gmSrc0, tileW * sizeof(float));
        TLOAD(tileSrc1, gmSrc1, tileW * sizeof(float));
        TADD(tileDst, tileSrc0, tileSrc1);  // Binary op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Binary op kernel: 2 tiles -> 1 tile
template <>
__aicore__ inline void LaunchBinaryBench<float, 1, 32768, 1, 32768>(const char* op, float *dst, float *src0, float *src1, void *stream)
{
    constexpr int tileH = 1;
    constexpr int tileW = 32768;
    constexpr int vRows = 1;
    constexpr int vCols = 32768;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc0(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc1(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc0 = (GM_ADDR)src0;
    GM_ADDR gmSrc1 = (GM_ADDR)src1;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc0, gmSrc0, tileW * sizeof(float));
        TLOAD(tileSrc1, gmSrc1, tileW * sizeof(float));
        TADD(tileDst, tileSrc0, tileSrc1);  // Binary op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Binary op kernel: 2 tiles -> 1 tile
template <>
__aicore__ inline void LaunchBinaryBench<float, 8, 512, 8, 512>(const char* op, float *dst, float *src0, float *src1, void *stream)
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
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc0, gmSrc0, tileW * sizeof(float));
        TLOAD(tileSrc1, gmSrc1, tileW * sizeof(float));
        TADD(tileDst, tileSrc0, tileSrc1);  // Binary op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Binary op kernel: 2 tiles -> 1 tile
template <>
__aicore__ inline void LaunchBinaryBench<float, 16, 256, 16, 256>(const char* op, float *dst, float *src0, float *src1, void *stream)
{
    constexpr int tileH = 16;
    constexpr int tileW = 256;
    constexpr int vRows = 16;
    constexpr int vCols = 256;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc0(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc1(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc0 = (GM_ADDR)src0;
    GM_ADDR gmSrc1 = (GM_ADDR)src1;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc0, gmSrc0, tileW * sizeof(float));
        TLOAD(tileSrc1, gmSrc1, tileW * sizeof(float));
        TADD(tileDst, tileSrc0, tileSrc1);  // Binary op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Binary op kernel: 2 tiles -> 1 tile
template <>
__aicore__ inline void LaunchBinaryBench<float, 16, 512, 16, 512>(const char* op, float *dst, float *src0, float *src1, void *stream)
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
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc0, gmSrc0, tileW * sizeof(float));
        TLOAD(tileSrc1, gmSrc1, tileW * sizeof(float));
        TADD(tileDst, tileSrc0, tileSrc1);  // Binary op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Binary op kernel: 2 tiles -> 1 tile
template <>
__aicore__ inline void LaunchBinaryBench<float, 32, 256, 32, 256>(const char* op, float *dst, float *src0, float *src1, void *stream)
{
    constexpr int tileH = 32;
    constexpr int tileW = 256;
    constexpr int vRows = 32;
    constexpr int vCols = 256;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc0(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc1(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc0 = (GM_ADDR)src0;
    GM_ADDR gmSrc1 = (GM_ADDR)src1;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc0, gmSrc0, tileW * sizeof(float));
        TLOAD(tileSrc1, gmSrc1, tileW * sizeof(float));
        TADD(tileDst, tileSrc0, tileSrc1);  // Binary op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Binary op kernel: 2 tiles -> 1 tile
template <>
__aicore__ inline void LaunchBinaryBench<float, 32, 512, 32, 512>(const char* op, float *dst, float *src0, float *src1, void *stream)
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
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc0, gmSrc0, tileW * sizeof(float));
        TLOAD(tileSrc1, gmSrc1, tileW * sizeof(float));
        TADD(tileDst, tileSrc0, tileSrc1);  // Binary op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Binary op kernel: 2 tiles -> 1 tile
template <>
__aicore__ inline void LaunchBinaryBench<float, 64, 256, 64, 256>(const char* op, float *dst, float *src0, float *src1, void *stream)
{
    constexpr int tileH = 64;
    constexpr int tileW = 256;
    constexpr int vRows = 64;
    constexpr int vCols = 256;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc0(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc1(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc0 = (GM_ADDR)src0;
    GM_ADDR gmSrc1 = (GM_ADDR)src1;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc0, gmSrc0, tileW * sizeof(float));
        TLOAD(tileSrc1, gmSrc1, tileW * sizeof(float));
        TADD(tileDst, tileSrc0, tileSrc1);  // Binary op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Binary op kernel: 2 tiles -> 1 tile
template <>
__aicore__ inline void LaunchBinaryBench<float, 64, 512, 64, 512>(const char* op, float *dst, float *src0, float *src1, void *stream)
{
    constexpr int tileH = 64;
    constexpr int tileW = 512;
    constexpr int vRows = 64;
    constexpr int vCols = 512;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc0(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc1(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc0 = (GM_ADDR)src0;
    GM_ADDR gmSrc1 = (GM_ADDR)src1;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc0, gmSrc0, tileW * sizeof(float));
        TLOAD(tileSrc1, gmSrc1, tileW * sizeof(float));
        TADD(tileDst, tileSrc0, tileSrc1);  // Binary op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Binary op kernel: 2 tiles -> 1 tile
template <>
__aicore__ inline void LaunchBinaryBench<float, 128, 256, 128, 256>(const char* op, float *dst, float *src0, float *src1, void *stream)
{
    constexpr int tileH = 128;
    constexpr int tileW = 256;
    constexpr int vRows = 128;
    constexpr int vCols = 256;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc0(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc1(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc0 = (GM_ADDR)src0;
    GM_ADDR gmSrc1 = (GM_ADDR)src1;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc0, gmSrc0, tileW * sizeof(float));
        TLOAD(tileSrc1, gmSrc1, tileW * sizeof(float));
        TADD(tileDst, tileSrc0, tileSrc1);  // Binary op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Unary op kernel: 1 tile -> 1 tile
template <>
__aicore__ inline void LaunchUnaryBench<float, 1, 4096, 1, 4096>(const char* op, float *dst, float *src, void *stream)
{
    constexpr int tileH = 1;
    constexpr int tileW = 4096;
    constexpr int vRows = 1;
    constexpr int vCols = 4096;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc = (GM_ADDR)src;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc, gmSrc, tileW * sizeof(float));
        TEXP(tileDst, tileSrc);  // Unary op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Unary op kernel: 1 tile -> 1 tile
template <>
__aicore__ inline void LaunchUnaryBench<float, 1, 8192, 1, 8192>(const char* op, float *dst, float *src, void *stream)
{
    constexpr int tileH = 1;
    constexpr int tileW = 8192;
    constexpr int vRows = 1;
    constexpr int vCols = 8192;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc = (GM_ADDR)src;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc, gmSrc, tileW * sizeof(float));
        TEXP(tileDst, tileSrc);  // Unary op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Unary op kernel: 1 tile -> 1 tile
template <>
__aicore__ inline void LaunchUnaryBench<float, 1, 16384, 1, 16384>(const char* op, float *dst, float *src, void *stream)
{
    constexpr int tileH = 1;
    constexpr int tileW = 16384;
    constexpr int vRows = 1;
    constexpr int vCols = 16384;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc = (GM_ADDR)src;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc, gmSrc, tileW * sizeof(float));
        TEXP(tileDst, tileSrc);  // Unary op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Unary op kernel: 1 tile -> 1 tile
template <>
__aicore__ inline void LaunchUnaryBench<float, 1, 32768, 1, 32768>(const char* op, float *dst, float *src, void *stream)
{
    constexpr int tileH = 1;
    constexpr int tileW = 32768;
    constexpr int vRows = 1;
    constexpr int vCols = 32768;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc = (GM_ADDR)src;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc, gmSrc, tileW * sizeof(float));
        TEXP(tileDst, tileSrc);  // Unary op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Unary op kernel: 1 tile -> 1 tile
template <>
__aicore__ inline void LaunchUnaryBench<float, 8, 512, 8, 512>(const char* op, float *dst, float *src, void *stream)
{
    constexpr int tileH = 8;
    constexpr int tileW = 512;
    constexpr int vRows = 8;
    constexpr int vCols = 512;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc = (GM_ADDR)src;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc, gmSrc, tileW * sizeof(float));
        TEXP(tileDst, tileSrc);  // Unary op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Unary op kernel: 1 tile -> 1 tile
template <>
__aicore__ inline void LaunchUnaryBench<float, 16, 256, 16, 256>(const char* op, float *dst, float *src, void *stream)
{
    constexpr int tileH = 16;
    constexpr int tileW = 256;
    constexpr int vRows = 16;
    constexpr int vCols = 256;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc = (GM_ADDR)src;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc, gmSrc, tileW * sizeof(float));
        TEXP(tileDst, tileSrc);  // Unary op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Unary op kernel: 1 tile -> 1 tile
template <>
__aicore__ inline void LaunchUnaryBench<float, 16, 512, 16, 512>(const char* op, float *dst, float *src, void *stream)
{
    constexpr int tileH = 16;
    constexpr int tileW = 512;
    constexpr int vRows = 16;
    constexpr int vCols = 512;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc = (GM_ADDR)src;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc, gmSrc, tileW * sizeof(float));
        TEXP(tileDst, tileSrc);  // Unary op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Unary op kernel: 1 tile -> 1 tile
template <>
__aicore__ inline void LaunchUnaryBench<float, 32, 256, 32, 256>(const char* op, float *dst, float *src, void *stream)
{
    constexpr int tileH = 32;
    constexpr int tileW = 256;
    constexpr int vRows = 32;
    constexpr int vCols = 256;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc = (GM_ADDR)src;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc, gmSrc, tileW * sizeof(float));
        TEXP(tileDst, tileSrc);  // Unary op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Unary op kernel: 1 tile -> 1 tile
template <>
__aicore__ inline void LaunchUnaryBench<float, 32, 512, 32, 512>(const char* op, float *dst, float *src, void *stream)
{
    constexpr int tileH = 32;
    constexpr int tileW = 512;
    constexpr int vRows = 32;
    constexpr int vCols = 512;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc = (GM_ADDR)src;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc, gmSrc, tileW * sizeof(float));
        TEXP(tileDst, tileSrc);  // Unary op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Unary op kernel: 1 tile -> 1 tile
template <>
__aicore__ inline void LaunchUnaryBench<float, 64, 256, 64, 256>(const char* op, float *dst, float *src, void *stream)
{
    constexpr int tileH = 64;
    constexpr int tileW = 256;
    constexpr int vRows = 64;
    constexpr int vCols = 256;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc = (GM_ADDR)src;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc, gmSrc, tileW * sizeof(float));
        TEXP(tileDst, tileSrc);  // Unary op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Unary op kernel: 1 tile -> 1 tile
template <>
__aicore__ inline void LaunchUnaryBench<float, 64, 512, 64, 512>(const char* op, float *dst, float *src, void *stream)
{
    constexpr int tileH = 64;
    constexpr int tileW = 512;
    constexpr int vRows = 64;
    constexpr int vCols = 512;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc = (GM_ADDR)src;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc, gmSrc, tileW * sizeof(float));
        TEXP(tileDst, tileSrc);  // Unary op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Unary op kernel: 1 tile -> 1 tile
template <>
__aicore__ inline void LaunchUnaryBench<float, 128, 256, 128, 256>(const char* op, float *dst, float *src, void *stream)
{
    constexpr int tileH = 128;
    constexpr int tileW = 256;
    constexpr int vRows = 128;
    constexpr int vCols = 256;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc = (GM_ADDR)src;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc, gmSrc, tileW * sizeof(float));
        TEXP(tileDst, tileSrc);  // Unary op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Scalar op kernel: 1 tile + scalar -> 1 tile
template <>
__aicore__ inline void LaunchScalarBench<float, 1, 4096, 1, 4096>(const char* op, float *dst, float *src, float scalar, void *stream)
{
    constexpr int tileH = 1;
    constexpr int tileW = 4096;
    constexpr int vRows = 1;
    constexpr int vCols = 4096;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc = (GM_ADDR)src;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc, gmSrc, tileW * sizeof(float));
        TADDS(tileDst, tileSrc, scalar);  // Scalar op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Scalar op kernel: 1 tile + scalar -> 1 tile
template <>
__aicore__ inline void LaunchScalarBench<float, 1, 8192, 1, 8192>(const char* op, float *dst, float *src, float scalar, void *stream)
{
    constexpr int tileH = 1;
    constexpr int tileW = 8192;
    constexpr int vRows = 1;
    constexpr int vCols = 8192;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc = (GM_ADDR)src;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc, gmSrc, tileW * sizeof(float));
        TADDS(tileDst, tileSrc, scalar);  // Scalar op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Scalar op kernel: 1 tile + scalar -> 1 tile
template <>
__aicore__ inline void LaunchScalarBench<float, 1, 16384, 1, 16384>(const char* op, float *dst, float *src, float scalar, void *stream)
{
    constexpr int tileH = 1;
    constexpr int tileW = 16384;
    constexpr int vRows = 1;
    constexpr int vCols = 16384;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc = (GM_ADDR)src;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc, gmSrc, tileW * sizeof(float));
        TADDS(tileDst, tileSrc, scalar);  // Scalar op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Scalar op kernel: 1 tile + scalar -> 1 tile
template <>
__aicore__ inline void LaunchScalarBench<float, 1, 32768, 1, 32768>(const char* op, float *dst, float *src, float scalar, void *stream)
{
    constexpr int tileH = 1;
    constexpr int tileW = 32768;
    constexpr int vRows = 1;
    constexpr int vCols = 32768;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc = (GM_ADDR)src;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc, gmSrc, tileW * sizeof(float));
        TADDS(tileDst, tileSrc, scalar);  // Scalar op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Scalar op kernel: 1 tile + scalar -> 1 tile
template <>
__aicore__ inline void LaunchScalarBench<float, 8, 512, 8, 512>(const char* op, float *dst, float *src, float scalar, void *stream)
{
    constexpr int tileH = 8;
    constexpr int tileW = 512;
    constexpr int vRows = 8;
    constexpr int vCols = 512;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc = (GM_ADDR)src;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc, gmSrc, tileW * sizeof(float));
        TADDS(tileDst, tileSrc, scalar);  // Scalar op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Scalar op kernel: 1 tile + scalar -> 1 tile
template <>
__aicore__ inline void LaunchScalarBench<float, 16, 256, 16, 256>(const char* op, float *dst, float *src, float scalar, void *stream)
{
    constexpr int tileH = 16;
    constexpr int tileW = 256;
    constexpr int vRows = 16;
    constexpr int vCols = 256;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc = (GM_ADDR)src;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc, gmSrc, tileW * sizeof(float));
        TADDS(tileDst, tileSrc, scalar);  // Scalar op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Scalar op kernel: 1 tile + scalar -> 1 tile
template <>
__aicore__ inline void LaunchScalarBench<float, 16, 512, 16, 512>(const char* op, float *dst, float *src, float scalar, void *stream)
{
    constexpr int tileH = 16;
    constexpr int tileW = 512;
    constexpr int vRows = 16;
    constexpr int vCols = 512;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc = (GM_ADDR)src;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc, gmSrc, tileW * sizeof(float));
        TADDS(tileDst, tileSrc, scalar);  // Scalar op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Scalar op kernel: 1 tile + scalar -> 1 tile
template <>
__aicore__ inline void LaunchScalarBench<float, 32, 256, 32, 256>(const char* op, float *dst, float *src, float scalar, void *stream)
{
    constexpr int tileH = 32;
    constexpr int tileW = 256;
    constexpr int vRows = 32;
    constexpr int vCols = 256;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc = (GM_ADDR)src;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc, gmSrc, tileW * sizeof(float));
        TADDS(tileDst, tileSrc, scalar);  // Scalar op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Scalar op kernel: 1 tile + scalar -> 1 tile
template <>
__aicore__ inline void LaunchScalarBench<float, 32, 512, 32, 512>(const char* op, float *dst, float *src, float scalar, void *stream)
{
    constexpr int tileH = 32;
    constexpr int tileW = 512;
    constexpr int vRows = 32;
    constexpr int vCols = 512;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc = (GM_ADDR)src;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc, gmSrc, tileW * sizeof(float));
        TADDS(tileDst, tileSrc, scalar);  // Scalar op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Scalar op kernel: 1 tile + scalar -> 1 tile
template <>
__aicore__ inline void LaunchScalarBench<float, 64, 256, 64, 256>(const char* op, float *dst, float *src, float scalar, void *stream)
{
    constexpr int tileH = 64;
    constexpr int tileW = 256;
    constexpr int vRows = 64;
    constexpr int vCols = 256;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc = (GM_ADDR)src;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc, gmSrc, tileW * sizeof(float));
        TADDS(tileDst, tileSrc, scalar);  // Scalar op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Scalar op kernel: 1 tile + scalar -> 1 tile
template <>
__aicore__ inline void LaunchScalarBench<float, 64, 512, 64, 512>(const char* op, float *dst, float *src, float scalar, void *stream)
{
    constexpr int tileH = 64;
    constexpr int tileW = 512;
    constexpr int vRows = 64;
    constexpr int vCols = 512;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc = (GM_ADDR)src;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc, gmSrc, tileW * sizeof(float));
        TADDS(tileDst, tileSrc, scalar);  // Scalar op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

// Scalar op kernel: 1 tile + scalar -> 1 tile
template <>
__aicore__ inline void LaunchScalarBench<float, 128, 256, 128, 256>(const char* op, float *dst, float *src, float scalar, void *stream)
{
    constexpr int tileH = 128;
    constexpr int tileW = 256;
    constexpr int vRows = 128;
    constexpr int vCols = 256;
    
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileSrc(vRows, vCols);
    TileDyn<float, TileFormat::RowMajor, tileH, tileW, PadValue::Null> tileDst(vRows, vCols);
    
    GM_ADDR gmSrc = (GM_ADDR)src;
    GM_ADDR gmDst = (GM_ADDR)dst;
    
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        TLOAD(tileSrc, gmSrc, tileW * sizeof(float));
        TADDS(tileDst, tileSrc, scalar);  // Scalar op
        TSTORE(gmDst, tileDst, tileW * sizeof(float));
        pipe_barrier(PIPE_ALL);
    }
}

