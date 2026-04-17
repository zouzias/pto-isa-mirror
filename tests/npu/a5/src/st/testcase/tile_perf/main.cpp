/**
 * Tile Performance Benchmark - Test Main
 * Supports: TADD, TADDS, TEXP for fp32/fp16
 * Shapes: Width >= 256 (4×VL) for proper vector utilization
 */
#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>
#include <cstring>

using namespace std;
using namespace PtoTestCommon;

// Forward declare kernel launcher templates
template <typename T, int tileH, int tileW, int vRows, int vCols>
void launchTADD(void *out, void *src0, void *src1, aclrtStream stream);

template <typename T, int tileH, int tileW, int vRows, int vCols>
void launchTEXP(void *out, void *src, aclrtStream stream);

template <typename T, int tileH, int tileW, int vRows, int vCols>
void launchTADDS(void *out, void *src, float scalar, aclrtStream stream);

class TilePerfTest : public testing::Test {};

// ========== TADD Tests ==========

#define TADD_FLOAT_TEST(NAME, H, W, ELEM) \
TEST_F(TilePerfTest, TADD_float_##NAME) { \
    size_t bytes = (ELEM) * sizeof(float); \
    aclInit(nullptr); aclrtSetDevice(0); \
    void *devOut, *devSrc0, *devSrc1; \
    aclrtMalloc(&devOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtMalloc(&devSrc0, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtMalloc(&devSrc1, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtStream stream = nullptr; aclrtCreateStream(&stream); \
    launchTADD<float, H, W, H, W>(devOut, devSrc0, devSrc1, stream); \
    aclrtSynchronizeStream(stream); aclrtDestroyStream(stream); \
    aclrtFree(devOut); aclrtFree(devSrc0); aclrtFree(devSrc1); \
    aclrtResetDevice(0); aclFinalize(); \
}


#define TADD_HALF_TEST(NAME, H, W, ELEM) \
TEST_F(TilePerfTest, TADD_half_##NAME) { \
    size_t bytes = (ELEM) * sizeof(aclFloat16); \
    aclInit(nullptr); aclrtSetDevice(0); \
    void *devOut, *devSrc0, *devSrc1; \
    aclrtMalloc(&devOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtMalloc(&devSrc0, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtMalloc(&devSrc1, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtStream stream = nullptr; aclrtCreateStream(&stream); \
    launchTADD<aclFloat16, H, W, H, W>(devOut, devSrc0, devSrc1, stream); \
    aclrtSynchronizeStream(stream); aclrtDestroyStream(stream); \
    aclrtFree(devOut); aclrtFree(devSrc0); aclrtFree(devSrc1); \
    aclrtResetDevice(0); aclFinalize(); \
}

// float 16KB
TADD_FLOAT_TEST(1x4096, 1, 4096, 4096)
TADD_FLOAT_TEST(16x256, 16, 256, 4096)
TADD_FLOAT_TEST(8x512, 8, 512, 4096)
// float 32KB
TADD_FLOAT_TEST(1x8192, 1, 8192, 8192)
TADD_FLOAT_TEST(32x256, 32, 256, 8192)
TADD_FLOAT_TEST(16x512, 16, 512, 8192)
// float 64KB
TADD_FLOAT_TEST(1x16384, 1, 16384, 16384)
TADD_FLOAT_TEST(64x256, 64, 256, 16384)
TADD_FLOAT_TEST(32x512, 32, 512, 16384)
// half 16KB
TADD_HALF_TEST(1x8192, 1, 8192, 8192)
TADD_HALF_TEST(32x256, 32, 256, 8192)
TADD_HALF_TEST(16x512, 16, 512, 8192)
// half 32KB
TADD_HALF_TEST(1x16384, 1, 16384, 16384)
TADD_HALF_TEST(64x256, 64, 256, 16384)
TADD_HALF_TEST(32x512, 32, 512, 16384)
// half 64KB
TADD_HALF_TEST(1x32768, 1, 32768, 32768)
TADD_HALF_TEST(128x256, 128, 256, 32768)
TADD_HALF_TEST(64x512, 64, 512, 32768)

// ========== TEXP Tests ==========

#define TEXP_FLOAT_TEST(NAME, H, W, ELEM) \
TEST_F(TilePerfTest, TEXP_float_##NAME) { \
    size_t bytes = (ELEM) * sizeof(float); \
    aclInit(nullptr); aclrtSetDevice(0); \
    void *devOut, *devSrc; \
    aclrtMalloc(&devOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtMalloc(&devSrc, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtStream stream = nullptr; aclrtCreateStream(&stream); \
    launchTEXP<float, H, W, H, W>(devOut, devSrc, stream); \
    aclrtSynchronizeStream(stream); aclrtDestroyStream(stream); \
    aclrtFree(devOut); aclrtFree(devSrc); \
    aclrtResetDevice(0); aclFinalize(); \
}


#define TEXP_HALF_TEST(NAME, H, W, ELEM) \
TEST_F(TilePerfTest, TEXP_half_##NAME) { \
    size_t bytes = (ELEM) * sizeof(aclFloat16); \
    aclInit(nullptr); aclrtSetDevice(0); \
    void *devOut, *devSrc; \
    aclrtMalloc(&devOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtMalloc(&devSrc, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtStream stream = nullptr; aclrtCreateStream(&stream); \
    launchTEXP<aclFloat16, H, W, H, W>(devOut, devSrc, stream); \
    aclrtSynchronizeStream(stream); aclrtDestroyStream(stream); \
    aclrtFree(devOut); aclrtFree(devSrc); \
    aclrtResetDevice(0); aclFinalize(); \
}

// float 16KB
TEXP_FLOAT_TEST(1x4096, 1, 4096, 4096)
TEXP_FLOAT_TEST(16x256, 16, 256, 4096)
TEXP_FLOAT_TEST(8x512, 8, 512, 4096)
// float 32KB
TEXP_FLOAT_TEST(1x8192, 1, 8192, 8192)
TEXP_FLOAT_TEST(32x256, 32, 256, 8192)
TEXP_FLOAT_TEST(16x512, 16, 512, 8192)
// float 64KB
TEXP_FLOAT_TEST(1x16384, 1, 16384, 16384)
TEXP_FLOAT_TEST(64x256, 64, 256, 16384)
TEXP_FLOAT_TEST(32x512, 32, 512, 16384)
// half 16KB
TEXP_HALF_TEST(1x8192, 1, 8192, 8192)
TEXP_HALF_TEST(32x256, 32, 256, 8192)
TEXP_HALF_TEST(16x512, 16, 512, 8192)
// half 32KB
TEXP_HALF_TEST(1x16384, 1, 16384, 16384)
TEXP_HALF_TEST(64x256, 64, 256, 16384)
TEXP_HALF_TEST(32x512, 32, 512, 16384)
// half 64KB
TEXP_HALF_TEST(1x32768, 1, 32768, 32768)
TEXP_HALF_TEST(128x256, 128, 256, 32768)
TEXP_HALF_TEST(64x512, 64, 512, 32768)

// ========== TADDS Tests ==========

#define TADDS_FLOAT_TEST(NAME, H, W, ELEM) \
TEST_F(TilePerfTest, TADDS_float_##NAME) { \
    size_t bytes = (ELEM) * sizeof(float); \
    aclInit(nullptr); aclrtSetDevice(0); \
    void *devOut, *devSrc; \
    aclrtMalloc(&devOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtMalloc(&devSrc, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtStream stream = nullptr; aclrtCreateStream(&stream); \
    launchTADDS<float, H, W, H, W>(devOut, devSrc, 1.5f, stream); \
    aclrtSynchronizeStream(stream); aclrtDestroyStream(stream); \
    aclrtFree(devOut); aclrtFree(devSrc); \
    aclrtResetDevice(0); aclFinalize(); \
}


#define TADDS_HALF_TEST(NAME, H, W, ELEM) \
TEST_F(TilePerfTest, TADDS_half_##NAME) { \
    size_t bytes = (ELEM) * sizeof(aclFloat16); \
    aclInit(nullptr); aclrtSetDevice(0); \
    void *devOut, *devSrc; \
    aclrtMalloc(&devOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtMalloc(&devSrc, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtStream stream = nullptr; aclrtCreateStream(&stream); \
    launchTADDS<aclFloat16, H, W, H, W>(devOut, devSrc, 1.5f, stream); \
    aclrtSynchronizeStream(stream); aclrtDestroyStream(stream); \
    aclrtFree(devOut); aclrtFree(devSrc); \
    aclrtResetDevice(0); aclFinalize(); \
}

// float 16KB
TADDS_FLOAT_TEST(1x4096, 1, 4096, 4096)
TADDS_FLOAT_TEST(16x256, 16, 256, 4096)
TADDS_FLOAT_TEST(8x512, 8, 512, 4096)
// float 32KB
TADDS_FLOAT_TEST(1x8192, 1, 8192, 8192)
TADDS_FLOAT_TEST(32x256, 32, 256, 8192)
TADDS_FLOAT_TEST(16x512, 16, 512, 8192)
// float 64KB
TADDS_FLOAT_TEST(1x16384, 1, 16384, 16384)
TADDS_FLOAT_TEST(64x256, 64, 256, 16384)
TADDS_FLOAT_TEST(32x512, 32, 512, 16384)
// half 16KB
TADDS_HALF_TEST(1x8192, 1, 8192, 8192)
TADDS_HALF_TEST(32x256, 32, 256, 8192)
TADDS_HALF_TEST(16x512, 16, 512, 8192)
// half 32KB
TADDS_HALF_TEST(1x16384, 1, 16384, 16384)
TADDS_HALF_TEST(64x256, 64, 256, 16384)
TADDS_HALF_TEST(32x512, 32, 512, 16384)
// half 64KB
TADDS_HALF_TEST(1x32768, 1, 32768, 32768)
TADDS_HALF_TEST(128x256, 128, 256, 32768)
TADDS_HALF_TEST(64x512, 64, 512, 32768)

