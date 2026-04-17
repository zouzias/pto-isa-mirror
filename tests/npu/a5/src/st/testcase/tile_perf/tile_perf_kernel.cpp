/**
 * Tile Performance Benchmark
 * Supports: TADD, TADDS, TEXP
 * Types: float (fp32), half (fp16)
 * 
 * UB Address Layout (A2A3 bank conflict free):
 *   Binary ops: src0=0x0, src1=0x100 (256B), dst=0x10000 (64KB)
 *   Unary ops:  src=0x0, dst=0x10000 (64KB)
 */
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <acl/acl.h>

using namespace pto;

// ========== TADD ==========
template <typename T, int tileH, int tileW, int vRows, int vCols>
PTO_INTERNAL void runTAdd(__gm__ T *out, __gm__ T *src0, __gm__ T *src1)
{
    using DynShape = Shape<1, 1, 1, -1, -1>;
    using DynStride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    using TileData = Tile<TileType::Vec, T, tileH, tileW, BLayout::RowMajor, -1, -1>;

    GlobalData src0Global(src0, DynShape(vRows, vCols), DynStride(tileH, tileW));
    GlobalData src1Global(src1, DynShape(vRows, vCols), DynStride(tileH, tileW));
    GlobalData dstGlobal(out, DynShape(vRows, vCols), DynStride(tileH, tileW));
    TileData src0Tile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x100);
    TASSIGN(dstTile, 0x10000);

    for (int i = 0; i < 10; i++) {
        TLOAD(src0Tile, src0Global);
        TLOAD(src1Tile, src1Global);
        TADD(dstTile, src0Tile, src1Tile);
        TSTORE(dstGlobal, dstTile);
        pipe_barrier(PIPE_ALL);
    }
}

// ========== TADDS ==========
template <typename T, int tileH, int tileW, int vRows, int vCols>
PTO_INTERNAL void runTAddS(__gm__ T *out, __gm__ T *src, T scalar)
{
    using DynShape = Shape<1, 1, 1, -1, -1>;
    using DynStride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    using TileData = Tile<TileType::Vec, T, tileH, tileW, BLayout::RowMajor, -1, -1>;

    GlobalData srcGlobal(src, DynShape(vRows, vCols), DynStride(tileH, tileW));
    GlobalData dstGlobal(out, DynShape(vRows, vCols), DynStride(tileH, tileW));
    TileData srcTile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);

    for (int i = 0; i < 10; i++) {
        TLOAD(srcTile, srcGlobal);
        TADDS(dstTile, srcTile, scalar);
        TSTORE(dstGlobal, dstTile);
        pipe_barrier(PIPE_ALL);
    }
}

// ========== TEXP ==========
template <typename T, int tileH, int tileW, int vRows, int vCols>
PTO_INTERNAL void runTExp(__gm__ T *out, __gm__ T *src)
{
    using DynShape = Shape<1, 1, 1, -1, -1>;
    using DynStride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    using TileData = Tile<TileType::Vec, T, tileH, tileW, BLayout::RowMajor, -1, -1>;

    GlobalData srcGlobal(src, DynShape(vRows, vCols), DynStride(tileH, tileW));
    GlobalData dstGlobal(out, DynShape(vRows, vCols), DynStride(tileH, tileW));
    TileData srcTile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);

    for (int i = 0; i < 10; i++) {
        TLOAD(srcTile, srcGlobal);
        TEXP(dstTile, srcTile);
        TSTORE(dstGlobal, dstTile);
        pipe_barrier(PIPE_ALL);
    }
}

// ===== TADD Launch Functions =====

// fp32 16KB (4096 elem)
extern "C" __global__ AICORE void launchTADD_float_1x4096(__gm__ float *out, __gm__ float *src0, __gm__ float *src1) { runTAdd<float, 1, 4096, 1, 4096>(out, src0, src1); }
extern "C" __global__ AICORE void launchTADD_float_64x64(__gm__ float *out, __gm__ float *src0, __gm__ float *src1) { runTAdd<float, 64, 64, 64, 64>(out, src0, src1); }
extern "C" __global__ AICORE void launchTADD_float_128x32(__gm__ float *out, __gm__ float *src0, __gm__ float *src1) { runTAdd<float, 128, 32, 128, 32>(out, src0, src1); }
extern "C" __global__ AICORE void launchTADD_float_32x128(__gm__ float *out, __gm__ float *src0, __gm__ float *src1) { runTAdd<float, 32, 128, 32, 128>(out, src0, src1); }

// fp32 32KB (8192 elem)
extern "C" __global__ AICORE void launchTADD_float_1x8192(__gm__ float *out, __gm__ float *src0, __gm__ float *src1) { runTAdd<float, 1, 8192, 1, 8192>(out, src0, src1); }
extern "C" __global__ AICORE void launchTADD_float_64x128(__gm__ float *out, __gm__ float *src0, __gm__ float *src1) { runTAdd<float, 64, 128, 64, 128>(out, src0, src1); }
extern "C" __global__ AICORE void launchTADD_float_256x32(__gm__ float *out, __gm__ float *src0, __gm__ float *src1) { runTAdd<float, 256, 32, 256, 32>(out, src0, src1); }
extern "C" __global__ AICORE void launchTADD_float_32x256(__gm__ float *out, __gm__ float *src0, __gm__ float *src1) { runTAdd<float, 32, 256, 32, 256>(out, src0, src1); }

// fp32 64KB (16384 elem)
extern "C" __global__ AICORE void launchTADD_float_1x16384(__gm__ float *out, __gm__ float *src0, __gm__ float *src1) { runTAdd<float, 1, 16384, 1, 16384>(out, src0, src1); }
extern "C" __global__ AICORE void launchTADD_float_128x128(__gm__ float *out, __gm__ float *src0, __gm__ float *src1) { runTAdd<float, 128, 128, 128, 128>(out, src0, src1); }
extern "C" __global__ AICORE void launchTADD_float_512x32(__gm__ float *out, __gm__ float *src0, __gm__ float *src1) { runTAdd<float, 512, 32, 512, 32>(out, src0, src1); }
extern "C" __global__ AICORE void launchTADD_float_32x512(__gm__ float *out, __gm__ float *src0, __gm__ float *src1) { runTAdd<float, 32, 512, 32, 512>(out, src0, src1); }

// fp16 16KB (8192 elem)
extern "C" __global__ AICORE void launchTADD_half_1x8192(__gm__ half *out, __gm__ half *src0, __gm__ half *src1) { runTAdd<half, 1, 8192, 1, 8192>(out, src0, src1); }
extern "C" __global__ AICORE void launchTADD_half_64x128(__gm__ half *out, __gm__ half *src0, __gm__ half *src1) { runTAdd<half, 64, 128, 64, 128>(out, src0, src1); }
extern "C" __global__ AICORE void launchTADD_half_256x32(__gm__ half *out, __gm__ half *src0, __gm__ half *src1) { runTAdd<half, 256, 32, 256, 32>(out, src0, src1); }
extern "C" __global__ AICORE void launchTADD_half_32x256(__gm__ half *out, __gm__ half *src0, __gm__ half *src1) { runTAdd<half, 32, 256, 32, 256>(out, src0, src1); }

// fp16 32KB (16384 elem)
extern "C" __global__ AICORE void launchTADD_half_1x16384(__gm__ half *out, __gm__ half *src0, __gm__ half *src1) { runTAdd<half, 1, 16384, 1, 16384>(out, src0, src1); }
extern "C" __global__ AICORE void launchTADD_half_128x128(__gm__ half *out, __gm__ half *src0, __gm__ half *src1) { runTAdd<half, 128, 128, 128, 128>(out, src0, src1); }
extern "C" __global__ AICORE void launchTADD_half_512x32(__gm__ half *out, __gm__ half *src0, __gm__ half *src1) { runTAdd<half, 512, 32, 512, 32>(out, src0, src1); }
extern "C" __global__ AICORE void launchTADD_half_32x512(__gm__ half *out, __gm__ half *src0, __gm__ half *src1) { runTAdd<half, 32, 512, 32, 512>(out, src0, src1); }

// fp16 64KB (32768 elem)
extern "C" __global__ AICORE void launchTADD_half_1x32768(__gm__ half *out, __gm__ half *src0, __gm__ half *src1) { runTAdd<half, 1, 32768, 1, 32768>(out, src0, src1); }
extern "C" __global__ AICORE void launchTADD_half_128x256(__gm__ half *out, __gm__ half *src0, __gm__ half *src1) { runTAdd<half, 128, 256, 128, 256>(out, src0, src1); }
extern "C" __global__ AICORE void launchTADD_half_1024x32(__gm__ half *out, __gm__ half *src0, __gm__ half *src1) { runTAdd<half, 1024, 32, 1024, 32>(out, src0, src1); }
extern "C" __global__ AICORE void launchTADD_half_32x1024(__gm__ half *out, __gm__ half *src0, __gm__ half *src1) { runTAdd<half, 32, 1024, 32, 1024>(out, src0, src1); }

// ===== TEXP Launch Functions =====

// fp32 16KB (4096 elem)
extern "C" __global__ AICORE void launchTEXP_float_1x4096(__gm__ float *out, __gm__ float *src) { runTExp<float, 1, 4096, 1, 4096>(out, src); }
extern "C" __global__ AICORE void launchTEXP_float_64x64(__gm__ float *out, __gm__ float *src) { runTExp<float, 64, 64, 64, 64>(out, src); }
extern "C" __global__ AICORE void launchTEXP_float_128x32(__gm__ float *out, __gm__ float *src) { runTExp<float, 128, 32, 128, 32>(out, src); }
extern "C" __global__ AICORE void launchTEXP_float_32x128(__gm__ float *out, __gm__ float *src) { runTExp<float, 32, 128, 32, 128>(out, src); }

// fp32 32KB (8192 elem)
extern "C" __global__ AICORE void launchTEXP_float_1x8192(__gm__ float *out, __gm__ float *src) { runTExp<float, 1, 8192, 1, 8192>(out, src); }
extern "C" __global__ AICORE void launchTEXP_float_64x128(__gm__ float *out, __gm__ float *src) { runTExp<float, 64, 128, 64, 128>(out, src); }
extern "C" __global__ AICORE void launchTEXP_float_256x32(__gm__ float *out, __gm__ float *src) { runTExp<float, 256, 32, 256, 32>(out, src); }
extern "C" __global__ AICORE void launchTEXP_float_32x256(__gm__ float *out, __gm__ float *src) { runTExp<float, 32, 256, 32, 256>(out, src); }

// fp32 64KB (16384 elem)
extern "C" __global__ AICORE void launchTEXP_float_1x16384(__gm__ float *out, __gm__ float *src) { runTExp<float, 1, 16384, 1, 16384>(out, src); }
extern "C" __global__ AICORE void launchTEXP_float_128x128(__gm__ float *out, __gm__ float *src) { runTExp<float, 128, 128, 128, 128>(out, src); }
extern "C" __global__ AICORE void launchTEXP_float_512x32(__gm__ float *out, __gm__ float *src) { runTExp<float, 512, 32, 512, 32>(out, src); }
extern "C" __global__ AICORE void launchTEXP_float_32x512(__gm__ float *out, __gm__ float *src) { runTExp<float, 32, 512, 32, 512>(out, src); }

// fp16 16KB (8192 elem)
extern "C" __global__ AICORE void launchTEXP_half_1x8192(__gm__ half *out, __gm__ half *src) { runTExp<half, 1, 8192, 1, 8192>(out, src); }
extern "C" __global__ AICORE void launchTEXP_half_64x128(__gm__ half *out, __gm__ half *src) { runTExp<half, 64, 128, 64, 128>(out, src); }
extern "C" __global__ AICORE void launchTEXP_half_256x32(__gm__ half *out, __gm__ half *src) { runTExp<half, 256, 32, 256, 32>(out, src); }
extern "C" __global__ AICORE void launchTEXP_half_32x256(__gm__ half *out, __gm__ half *src) { runTExp<half, 32, 256, 32, 256>(out, src); }

// fp16 32KB (16384 elem)
extern "C" __global__ AICORE void launchTEXP_half_1x16384(__gm__ half *out, __gm__ half *src) { runTExp<half, 1, 16384, 1, 16384>(out, src); }
extern "C" __global__ AICORE void launchTEXP_half_128x128(__gm__ half *out, __gm__ half *src) { runTExp<half, 128, 128, 128, 128>(out, src); }
extern "C" __global__ AICORE void launchTEXP_half_512x32(__gm__ half *out, __gm__ half *src) { runTExp<half, 512, 32, 512, 32>(out, src); }
extern "C" __global__ AICORE void launchTEXP_half_32x512(__gm__ half *out, __gm__ half *src) { runTExp<half, 32, 512, 32, 512>(out, src); }

// fp16 64KB (32768 elem)
extern "C" __global__ AICORE void launchTEXP_half_1x32768(__gm__ half *out, __gm__ half *src) { runTExp<half, 1, 32768, 1, 32768>(out, src); }
extern "C" __global__ AICORE void launchTEXP_half_128x256(__gm__ half *out, __gm__ half *src) { runTExp<half, 128, 256, 128, 256>(out, src); }
extern "C" __global__ AICORE void launchTEXP_half_1024x32(__gm__ half *out, __gm__ half *src) { runTExp<half, 1024, 32, 1024, 32>(out, src); }
extern "C" __global__ AICORE void launchTEXP_half_32x1024(__gm__ half *out, __gm__ half *src) { runTExp<half, 32, 1024, 32, 1024>(out, src); }

// ===== TADDS Launch Functions =====

// fp32 16KB (4096 elem)
extern "C" __global__ AICORE void launchTADDS_float_1x4096(__gm__ float *out, __gm__ float *src, float scalar) { runTAddS<float, 1, 4096, 1, 4096>(out, src, scalar); }
extern "C" __global__ AICORE void launchTADDS_float_64x64(__gm__ float *out, __gm__ float *src, float scalar) { runTAddS<float, 64, 64, 64, 64>(out, src, scalar); }
extern "C" __global__ AICORE void launchTADDS_float_128x32(__gm__ float *out, __gm__ float *src, float scalar) { runTAddS<float, 128, 32, 128, 32>(out, src, scalar); }
extern "C" __global__ AICORE void launchTADDS_float_32x128(__gm__ float *out, __gm__ float *src, float scalar) { runTAddS<float, 32, 128, 32, 128>(out, src, scalar); }

// fp32 32KB (8192 elem)
extern "C" __global__ AICORE void launchTADDS_float_1x8192(__gm__ float *out, __gm__ float *src, float scalar) { runTAddS<float, 1, 8192, 1, 8192>(out, src, scalar); }
extern "C" __global__ AICORE void launchTADDS_float_64x128(__gm__ float *out, __gm__ float *src, float scalar) { runTAddS<float, 64, 128, 64, 128>(out, src, scalar); }
extern "C" __global__ AICORE void launchTADDS_float_256x32(__gm__ float *out, __gm__ float *src, float scalar) { runTAddS<float, 256, 32, 256, 32>(out, src, scalar); }
extern "C" __global__ AICORE void launchTADDS_float_32x256(__gm__ float *out, __gm__ float *src, float scalar) { runTAddS<float, 32, 256, 32, 256>(out, src, scalar); }

// fp32 64KB (16384 elem)
extern "C" __global__ AICORE void launchTADDS_float_1x16384(__gm__ float *out, __gm__ float *src, float scalar) { runTAddS<float, 1, 16384, 1, 16384>(out, src, scalar); }
extern "C" __global__ AICORE void launchTADDS_float_128x128(__gm__ float *out, __gm__ float *src, float scalar) { runTAddS<float, 128, 128, 128, 128>(out, src, scalar); }
extern "C" __global__ AICORE void launchTADDS_float_512x32(__gm__ float *out, __gm__ float *src, float scalar) { runTAddS<float, 512, 32, 512, 32>(out, src, scalar); }
extern "C" __global__ AICORE void launchTADDS_float_32x512(__gm__ float *out, __gm__ float *src, float scalar) { runTAddS<float, 32, 512, 32, 512>(out, src, scalar); }

// fp16 16KB (8192 elem)
extern "C" __global__ AICORE void launchTADDS_half_1x8192(__gm__ half *out, __gm__ half *src, half scalar) { runTAddS<half, 1, 8192, 1, 8192>(out, src, scalar); }
extern "C" __global__ AICORE void launchTADDS_half_64x128(__gm__ half *out, __gm__ half *src, half scalar) { runTAddS<half, 64, 128, 64, 128>(out, src, scalar); }
extern "C" __global__ AICORE void launchTADDS_half_256x32(__gm__ half *out, __gm__ half *src, half scalar) { runTAddS<half, 256, 32, 256, 32>(out, src, scalar); }
extern "C" __global__ AICORE void launchTADDS_half_32x256(__gm__ half *out, __gm__ half *src, half scalar) { runTAddS<half, 32, 256, 32, 256>(out, src, scalar); }

// fp16 32KB (16384 elem)
extern "C" __global__ AICORE void launchTADDS_half_1x16384(__gm__ half *out, __gm__ half *src, half scalar) { runTAddS<half, 1, 16384, 1, 16384>(out, src, scalar); }
extern "C" __global__ AICORE void launchTADDS_half_128x128(__gm__ half *out, __gm__ half *src, half scalar) { runTAddS<half, 128, 128, 128, 128>(out, src, scalar); }
extern "C" __global__ AICORE void launchTADDS_half_512x32(__gm__ half *out, __gm__ half *src, half scalar) { runTAddS<half, 512, 32, 512, 32>(out, src, scalar); }
extern "C" __global__ AICORE void launchTADDS_half_32x512(__gm__ half *out, __gm__ half *src, half scalar) { runTAddS<half, 32, 512, 32, 512>(out, src, scalar); }

// fp16 64KB (32768 elem)
extern "C" __global__ AICORE void launchTADDS_half_1x32768(__gm__ half *out, __gm__ half *src, half scalar) { runTAddS<half, 1, 32768, 1, 32768>(out, src, scalar); }
extern "C" __global__ AICORE void launchTADDS_half_128x256(__gm__ half *out, __gm__ half *src, half scalar) { runTAddS<half, 128, 256, 128, 256>(out, src, scalar); }
extern "C" __global__ AICORE void launchTADDS_half_1024x32(__gm__ half *out, __gm__ half *src, half scalar) { runTAddS<half, 1024, 32, 1024, 32>(out, src, scalar); }
extern "C" __global__ AICORE void launchTADDS_half_32x1024(__gm__ half *out, __gm__ half *src, half scalar) { runTAddS<half, 32, 1024, 32, 1024>(out, src, scalar); }
