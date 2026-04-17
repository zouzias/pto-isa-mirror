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

// ===== Internal kernel entry points =====

// TADD fp32
extern "C" __global__ AICORE void kernel_TADD_float_1x4096(__gm__ float *out, __gm__ float *src0, __gm__ float *src1) { runTAdd<float, 1, 4096, 1, 4096>(out, src0, src1); }
extern "C" __global__ AICORE void kernel_TADD_float_64x64(__gm__ float *out, __gm__ float *src0, __gm__ float *src1) { runTAdd<float, 64, 64, 64, 64>(out, src0, src1); }
extern "C" __global__ AICORE void kernel_TADD_float_128x32(__gm__ float *out, __gm__ float *src0, __gm__ float *src1) { runTAdd<float, 128, 32, 128, 32>(out, src0, src1); }
extern "C" __global__ AICORE void kernel_TADD_float_32x128(__gm__ float *out, __gm__ float *src0, __gm__ float *src1) { runTAdd<float, 32, 128, 32, 128>(out, src0, src1); }
extern "C" __global__ AICORE void kernel_TADD_float_1x8192(__gm__ float *out, __gm__ float *src0, __gm__ float *src1) { runTAdd<float, 1, 8192, 1, 8192>(out, src0, src1); }
extern "C" __global__ AICORE void kernel_TADD_float_64x128(__gm__ float *out, __gm__ float *src0, __gm__ float *src1) { runTAdd<float, 64, 128, 64, 128>(out, src0, src1); }
extern "C" __global__ AICORE void kernel_TADD_float_256x32(__gm__ float *out, __gm__ float *src0, __gm__ float *src1) { runTAdd<float, 256, 32, 256, 32>(out, src0, src1); }
extern "C" __global__ AICORE void kernel_TADD_float_32x256(__gm__ float *out, __gm__ float *src0, __gm__ float *src1) { runTAdd<float, 32, 256, 32, 256>(out, src0, src1); }
extern "C" __global__ AICORE void kernel_TADD_float_1x16384(__gm__ float *out, __gm__ float *src0, __gm__ float *src1) { runTAdd<float, 1, 16384, 1, 16384>(out, src0, src1); }
extern "C" __global__ AICORE void kernel_TADD_float_128x128(__gm__ float *out, __gm__ float *src0, __gm__ float *src1) { runTAdd<float, 128, 128, 128, 128>(out, src0, src1); }
extern "C" __global__ AICORE void kernel_TADD_float_512x32(__gm__ float *out, __gm__ float *src0, __gm__ float *src1) { runTAdd<float, 512, 32, 512, 32>(out, src0, src1); }
extern "C" __global__ AICORE void kernel_TADD_float_32x512(__gm__ float *out, __gm__ float *src0, __gm__ float *src1) { runTAdd<float, 32, 512, 32, 512>(out, src0, src1); }

// TADD fp16

// TEXP fp32
extern "C" __global__ AICORE void kernel_TEXP_float_1x4096(__gm__ float *out, __gm__ float *src) { runTExp<float, 1, 4096, 1, 4096>(out, src); }
extern "C" __global__ AICORE void kernel_TEXP_float_64x64(__gm__ float *out, __gm__ float *src) { runTExp<float, 64, 64, 64, 64>(out, src); }
extern "C" __global__ AICORE void kernel_TEXP_float_128x32(__gm__ float *out, __gm__ float *src) { runTExp<float, 128, 32, 128, 32>(out, src); }
extern "C" __global__ AICORE void kernel_TEXP_float_32x128(__gm__ float *out, __gm__ float *src) { runTExp<float, 32, 128, 32, 128>(out, src); }
extern "C" __global__ AICORE void kernel_TEXP_float_1x8192(__gm__ float *out, __gm__ float *src) { runTExp<float, 1, 8192, 1, 8192>(out, src); }
extern "C" __global__ AICORE void kernel_TEXP_float_64x128(__gm__ float *out, __gm__ float *src) { runTExp<float, 64, 128, 64, 128>(out, src); }
extern "C" __global__ AICORE void kernel_TEXP_float_256x32(__gm__ float *out, __gm__ float *src) { runTExp<float, 256, 32, 256, 32>(out, src); }
extern "C" __global__ AICORE void kernel_TEXP_float_32x256(__gm__ float *out, __gm__ float *src) { runTExp<float, 32, 256, 32, 256>(out, src); }
extern "C" __global__ AICORE void kernel_TEXP_float_1x16384(__gm__ float *out, __gm__ float *src) { runTExp<float, 1, 16384, 1, 16384>(out, src); }
extern "C" __global__ AICORE void kernel_TEXP_float_128x128(__gm__ float *out, __gm__ float *src) { runTExp<float, 128, 128, 128, 128>(out, src); }
extern "C" __global__ AICORE void kernel_TEXP_float_512x32(__gm__ float *out, __gm__ float *src) { runTExp<float, 512, 32, 512, 32>(out, src); }
extern "C" __global__ AICORE void kernel_TEXP_float_32x512(__gm__ float *out, __gm__ float *src) { runTExp<float, 32, 512, 32, 512>(out, src); }

// TEXP fp16

// TADDS fp32
extern "C" __global__ AICORE void kernel_TADDS_float_1x4096(__gm__ float *out, __gm__ float *src, float scalar) { runTAddS<float, 1, 4096, 1, 4096>(out, src, scalar); }
extern "C" __global__ AICORE void kernel_TADDS_float_64x64(__gm__ float *out, __gm__ float *src, float scalar) { runTAddS<float, 64, 64, 64, 64>(out, src, scalar); }
extern "C" __global__ AICORE void kernel_TADDS_float_128x32(__gm__ float *out, __gm__ float *src, float scalar) { runTAddS<float, 128, 32, 128, 32>(out, src, scalar); }
extern "C" __global__ AICORE void kernel_TADDS_float_32x128(__gm__ float *out, __gm__ float *src, float scalar) { runTAddS<float, 32, 128, 32, 128>(out, src, scalar); }
extern "C" __global__ AICORE void kernel_TADDS_float_1x8192(__gm__ float *out, __gm__ float *src, float scalar) { runTAddS<float, 1, 8192, 1, 8192>(out, src, scalar); }
extern "C" __global__ AICORE void kernel_TADDS_float_64x128(__gm__ float *out, __gm__ float *src, float scalar) { runTAddS<float, 64, 128, 64, 128>(out, src, scalar); }
extern "C" __global__ AICORE void kernel_TADDS_float_256x32(__gm__ float *out, __gm__ float *src, float scalar) { runTAddS<float, 256, 32, 256, 32>(out, src, scalar); }
extern "C" __global__ AICORE void kernel_TADDS_float_32x256(__gm__ float *out, __gm__ float *src, float scalar) { runTAddS<float, 32, 256, 32, 256>(out, src, scalar); }
extern "C" __global__ AICORE void kernel_TADDS_float_1x16384(__gm__ float *out, __gm__ float *src, float scalar) { runTAddS<float, 1, 16384, 1, 16384>(out, src, scalar); }
extern "C" __global__ AICORE void kernel_TADDS_float_128x128(__gm__ float *out, __gm__ float *src, float scalar) { runTAddS<float, 128, 128, 128, 128>(out, src, scalar); }
extern "C" __global__ AICORE void kernel_TADDS_float_512x32(__gm__ float *out, __gm__ float *src, float scalar) { runTAddS<float, 512, 32, 512, 32>(out, src, scalar); }
extern "C" __global__ AICORE void kernel_TADDS_float_32x512(__gm__ float *out, __gm__ float *src, float scalar) { runTAddS<float, 32, 512, 32, 512>(out, src, scalar); }

// TADDS fp16

// ===== Template launcher functions (called from host) =====

// TADD float launchers
template <int caseId> void launchTADD_float(void *out, void *src0, void *src1, aclrtStream stream);
template<> void launchTADD_float<1>(void *out, void *src0, void *src1, aclrtStream stream) { kernel_TADD_float_1x4096<<<1, nullptr, stream>>>((float*)out, (float*)src0, (float*)src1); }
template<> void launchTADD_float<2>(void *out, void *src0, void *src1, aclrtStream stream) { kernel_TADD_float_64x64<<<1, nullptr, stream>>>((float*)out, (float*)src0, (float*)src1); }
template<> void launchTADD_float<3>(void *out, void *src0, void *src1, aclrtStream stream) { kernel_TADD_float_128x32<<<1, nullptr, stream>>>((float*)out, (float*)src0, (float*)src1); }
template<> void launchTADD_float<4>(void *out, void *src0, void *src1, aclrtStream stream) { kernel_TADD_float_32x128<<<1, nullptr, stream>>>((float*)out, (float*)src0, (float*)src1); }
template<> void launchTADD_float<5>(void *out, void *src0, void *src1, aclrtStream stream) { kernel_TADD_float_1x8192<<<1, nullptr, stream>>>((float*)out, (float*)src0, (float*)src1); }
template<> void launchTADD_float<6>(void *out, void *src0, void *src1, aclrtStream stream) { kernel_TADD_float_64x128<<<1, nullptr, stream>>>((float*)out, (float*)src0, (float*)src1); }
template<> void launchTADD_float<7>(void *out, void *src0, void *src1, aclrtStream stream) { kernel_TADD_float_256x32<<<1, nullptr, stream>>>((float*)out, (float*)src0, (float*)src1); }
template<> void launchTADD_float<8>(void *out, void *src0, void *src1, aclrtStream stream) { kernel_TADD_float_32x256<<<1, nullptr, stream>>>((float*)out, (float*)src0, (float*)src1); }
template<> void launchTADD_float<9>(void *out, void *src0, void *src1, aclrtStream stream) { kernel_TADD_float_1x16384<<<1, nullptr, stream>>>((float*)out, (float*)src0, (float*)src1); }
template<> void launchTADD_float<10>(void *out, void *src0, void *src1, aclrtStream stream) { kernel_TADD_float_128x128<<<1, nullptr, stream>>>((float*)out, (float*)src0, (float*)src1); }
template<> void launchTADD_float<11>(void *out, void *src0, void *src1, aclrtStream stream) { kernel_TADD_float_512x32<<<1, nullptr, stream>>>((float*)out, (float*)src0, (float*)src1); }
template<> void launchTADD_float<12>(void *out, void *src0, void *src1, aclrtStream stream) { kernel_TADD_float_32x512<<<1, nullptr, stream>>>((float*)out, (float*)src0, (float*)src1); }

// TADD half launchers

// TEXP float launchers
template <int caseId> void launchTEXP_float(void *out, void *src, aclrtStream stream);
template<> void launchTEXP_float<1>(void *out, void *src, aclrtStream stream) { kernel_TEXP_float_1x4096<<<1, nullptr, stream>>>((float*)out, (float*)src); }
template<> void launchTEXP_float<2>(void *out, void *src, aclrtStream stream) { kernel_TEXP_float_64x64<<<1, nullptr, stream>>>((float*)out, (float*)src); }
template<> void launchTEXP_float<3>(void *out, void *src, aclrtStream stream) { kernel_TEXP_float_128x32<<<1, nullptr, stream>>>((float*)out, (float*)src); }
template<> void launchTEXP_float<4>(void *out, void *src, aclrtStream stream) { kernel_TEXP_float_32x128<<<1, nullptr, stream>>>((float*)out, (float*)src); }
template<> void launchTEXP_float<5>(void *out, void *src, aclrtStream stream) { kernel_TEXP_float_1x8192<<<1, nullptr, stream>>>((float*)out, (float*)src); }
template<> void launchTEXP_float<6>(void *out, void *src, aclrtStream stream) { kernel_TEXP_float_64x128<<<1, nullptr, stream>>>((float*)out, (float*)src); }
template<> void launchTEXP_float<7>(void *out, void *src, aclrtStream stream) { kernel_TEXP_float_256x32<<<1, nullptr, stream>>>((float*)out, (float*)src); }
template<> void launchTEXP_float<8>(void *out, void *src, aclrtStream stream) { kernel_TEXP_float_32x256<<<1, nullptr, stream>>>((float*)out, (float*)src); }
template<> void launchTEXP_float<9>(void *out, void *src, aclrtStream stream) { kernel_TEXP_float_1x16384<<<1, nullptr, stream>>>((float*)out, (float*)src); }
template<> void launchTEXP_float<10>(void *out, void *src, aclrtStream stream) { kernel_TEXP_float_128x128<<<1, nullptr, stream>>>((float*)out, (float*)src); }
template<> void launchTEXP_float<11>(void *out, void *src, aclrtStream stream) { kernel_TEXP_float_512x32<<<1, nullptr, stream>>>((float*)out, (float*)src); }
template<> void launchTEXP_float<12>(void *out, void *src, aclrtStream stream) { kernel_TEXP_float_32x512<<<1, nullptr, stream>>>((float*)out, (float*)src); }

// TEXP half launchers

// TADDS float launchers
template <int caseId> void launchTADDS_float(void *out, void *src, float scalar, aclrtStream stream);
template<> void launchTADDS_float<1>(void *out, void *src, float scalar, aclrtStream stream) { kernel_TADDS_float_1x4096<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar); }
template<> void launchTADDS_float<2>(void *out, void *src, float scalar, aclrtStream stream) { kernel_TADDS_float_64x64<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar); }
template<> void launchTADDS_float<3>(void *out, void *src, float scalar, aclrtStream stream) { kernel_TADDS_float_128x32<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar); }
template<> void launchTADDS_float<4>(void *out, void *src, float scalar, aclrtStream stream) { kernel_TADDS_float_32x128<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar); }
template<> void launchTADDS_float<5>(void *out, void *src, float scalar, aclrtStream stream) { kernel_TADDS_float_1x8192<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar); }
template<> void launchTADDS_float<6>(void *out, void *src, float scalar, aclrtStream stream) { kernel_TADDS_float_64x128<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar); }
template<> void launchTADDS_float<7>(void *out, void *src, float scalar, aclrtStream stream) { kernel_TADDS_float_256x32<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar); }
template<> void launchTADDS_float<8>(void *out, void *src, float scalar, aclrtStream stream) { kernel_TADDS_float_32x256<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar); }
template<> void launchTADDS_float<9>(void *out, void *src, float scalar, aclrtStream stream) { kernel_TADDS_float_1x16384<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar); }
template<> void launchTADDS_float<10>(void *out, void *src, float scalar, aclrtStream stream) { kernel_TADDS_float_128x128<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar); }
template<> void launchTADDS_float<11>(void *out, void *src, float scalar, aclrtStream stream) { kernel_TADDS_float_512x32<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar); }
template<> void launchTADDS_float<12>(void *out, void *src, float scalar, aclrtStream stream) { kernel_TADDS_float_32x512<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar); }

// TADDS half launchers
