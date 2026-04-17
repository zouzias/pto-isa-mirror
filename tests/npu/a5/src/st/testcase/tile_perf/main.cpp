/**
 * Tile Performance Benchmark - Test Main
 * Supports: TADD, TADDS, TEXP for fp32/fp16
 */
#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>
#include <cstring>

using namespace std;
using namespace PtoTestCommon;

// Forward declare kernel launcher templates (use aclFloat16 for fp16 on host)
template <typename T, int tileH, int tileW, int vRows, int vCols>
void launchTADD(void *out, void *src0, void *src1, aclrtStream stream);

template <typename T, int tileH, int tileW, int vRows, int vCols>
void launchTEXP(void *out, void *src, aclrtStream stream);

template <typename T, int tileH, int tileW, int vRows, int vCols>
void launchTADDS(void *out, void *src, float scalar, aclrtStream stream);

// Test fixtures
class TilePerfTest : public testing::Test {};

// ========== TADD fp32 Tests ==========
#define TADD_FLOAT_TEST(NAME, H, W, ELEM) \
TEST_F(TilePerfTest, TADD_float_##NAME) { \
    size_t bytes = (ELEM) * sizeof(float); \
    aclInit(nullptr); \
    aclrtSetDevice(0); \
    void *devOut, *devSrc0, *devSrc1; \
    aclrtMalloc(&devOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtMalloc(&devSrc0, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtMalloc(&devSrc1, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtStream stream = nullptr; \
    aclrtCreateStream(&stream); \
    launchTADD<float, H, W, H, W>(devOut, devSrc0, devSrc1, stream); \
    aclrtSynchronizeStream(stream); \
    aclrtDestroyStream(stream); \
    aclrtFree(devOut); aclrtFree(devSrc0); aclrtFree(devSrc1); \
    aclrtResetDevice(0); \
    aclFinalize(); \
}

// 16KB (4096 elements)
TADD_FLOAT_TEST(1x4096, 1, 4096, 4096)
TADD_FLOAT_TEST(64x64, 64, 64, 4096)
TADD_FLOAT_TEST(128x32, 128, 32, 4096)
TADD_FLOAT_TEST(32x128, 32, 128, 4096)
// 32KB (8192 elements)
TADD_FLOAT_TEST(1x8192, 1, 8192, 8192)
TADD_FLOAT_TEST(64x128, 64, 128, 8192)
TADD_FLOAT_TEST(256x32, 256, 32, 8192)
TADD_FLOAT_TEST(32x256, 32, 256, 8192)
// 64KB (16384 elements)
TADD_FLOAT_TEST(1x16384, 1, 16384, 16384)
TADD_FLOAT_TEST(128x128, 128, 128, 16384)
TADD_FLOAT_TEST(512x32, 512, 32, 16384)
TADD_FLOAT_TEST(32x512, 32, 512, 16384)

// ========== TADD fp16 Tests ==========
#define TADD_HALF_TEST(NAME, H, W, ELEM) \
TEST_F(TilePerfTest, TADD_half_##NAME) { \
    size_t bytes = (ELEM) * sizeof(aclFloat16); \
    aclInit(nullptr); \
    aclrtSetDevice(0); \
    void *devOut, *devSrc0, *devSrc1; \
    aclrtMalloc(&devOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtMalloc(&devSrc0, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtMalloc(&devSrc1, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtStream stream = nullptr; \
    aclrtCreateStream(&stream); \
    launchTADD<aclFloat16, H, W, H, W>(devOut, devSrc0, devSrc1, stream); \
    aclrtSynchronizeStream(stream); \
    aclrtDestroyStream(stream); \
    aclrtFree(devOut); aclrtFree(devSrc0); aclrtFree(devSrc1); \
    aclrtResetDevice(0); \
    aclFinalize(); \
}

// 16KB (8192 elements)
TADD_HALF_TEST(1x8192, 1, 8192, 8192)
TADD_HALF_TEST(64x128, 64, 128, 8192)
TADD_HALF_TEST(256x32, 256, 32, 8192)
TADD_HALF_TEST(32x256, 32, 256, 8192)
// 32KB (16384 elements)
TADD_HALF_TEST(1x16384, 1, 16384, 16384)
TADD_HALF_TEST(128x128, 128, 128, 16384)
TADD_HALF_TEST(512x32, 512, 32, 16384)
TADD_HALF_TEST(32x512, 32, 512, 16384)
// 64KB (32768 elements)
TADD_HALF_TEST(1x32768, 1, 32768, 32768)
TADD_HALF_TEST(128x256, 128, 256, 32768)
TADD_HALF_TEST(1024x32, 1024, 32, 32768)
TADD_HALF_TEST(32x1024, 32, 1024, 32768)

// ========== TEXP fp32 Tests ==========
#define TEXP_FLOAT_TEST(NAME, H, W, ELEM) \
TEST_F(TilePerfTest, TEXP_float_##NAME) { \
    size_t bytes = (ELEM) * sizeof(float); \
    aclInit(nullptr); \
    aclrtSetDevice(0); \
    void *devOut, *devSrc; \
    aclrtMalloc(&devOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtMalloc(&devSrc, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtStream stream = nullptr; \
    aclrtCreateStream(&stream); \
    launchTEXP<float, H, W, H, W>(devOut, devSrc, stream); \
    aclrtSynchronizeStream(stream); \
    aclrtDestroyStream(stream); \
    aclrtFree(devOut); aclrtFree(devSrc); \
    aclrtResetDevice(0); \
    aclFinalize(); \
}

// 16KB (4096 elements)
TEXP_FLOAT_TEST(1x4096, 1, 4096, 4096)
TEXP_FLOAT_TEST(64x64, 64, 64, 4096)
TEXP_FLOAT_TEST(128x32, 128, 32, 4096)
TEXP_FLOAT_TEST(32x128, 32, 128, 4096)
// 32KB (8192 elements)
TEXP_FLOAT_TEST(1x8192, 1, 8192, 8192)
TEXP_FLOAT_TEST(64x128, 64, 128, 8192)
TEXP_FLOAT_TEST(256x32, 256, 32, 8192)
TEXP_FLOAT_TEST(32x256, 32, 256, 8192)
// 64KB (16384 elements)
TEXP_FLOAT_TEST(1x16384, 1, 16384, 16384)
TEXP_FLOAT_TEST(128x128, 128, 128, 16384)
TEXP_FLOAT_TEST(512x32, 512, 32, 16384)
TEXP_FLOAT_TEST(32x512, 32, 512, 16384)

// ========== TEXP fp16 Tests ==========
#define TEXP_HALF_TEST(NAME, H, W, ELEM) \
TEST_F(TilePerfTest, TEXP_half_##NAME) { \
    size_t bytes = (ELEM) * sizeof(aclFloat16); \
    aclInit(nullptr); \
    aclrtSetDevice(0); \
    void *devOut, *devSrc; \
    aclrtMalloc(&devOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtMalloc(&devSrc, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtStream stream = nullptr; \
    aclrtCreateStream(&stream); \
    launchTEXP<aclFloat16, H, W, H, W>(devOut, devSrc, stream); \
    aclrtSynchronizeStream(stream); \
    aclrtDestroyStream(stream); \
    aclrtFree(devOut); aclrtFree(devSrc); \
    aclrtResetDevice(0); \
    aclFinalize(); \
}

// 16KB (8192 elements)
TEXP_HALF_TEST(1x8192, 1, 8192, 8192)
TEXP_HALF_TEST(64x128, 64, 128, 8192)
TEXP_HALF_TEST(256x32, 256, 32, 8192)
TEXP_HALF_TEST(32x256, 32, 256, 8192)
// 32KB (16384 elements)
TEXP_HALF_TEST(1x16384, 1, 16384, 16384)
TEXP_HALF_TEST(128x128, 128, 128, 16384)
TEXP_HALF_TEST(512x32, 512, 32, 16384)
TEXP_HALF_TEST(32x512, 32, 512, 16384)
// 64KB (32768 elements)
TEXP_HALF_TEST(1x32768, 1, 32768, 32768)
TEXP_HALF_TEST(128x256, 128, 256, 32768)
TEXP_HALF_TEST(1024x32, 1024, 32, 32768)
TEXP_HALF_TEST(32x1024, 32, 1024, 32768)

// ========== TADDS fp32 Tests ==========
#define TADDS_FLOAT_TEST(NAME, H, W, ELEM) \
TEST_F(TilePerfTest, TADDS_float_##NAME) { \
    size_t bytes = (ELEM) * sizeof(float); \
    float scalar = 1.5f; \
    aclInit(nullptr); \
    aclrtSetDevice(0); \
    void *devOut, *devSrc; \
    aclrtMalloc(&devOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtMalloc(&devSrc, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtStream stream = nullptr; \
    aclrtCreateStream(&stream); \
    launchTADDS<float, H, W, H, W>(devOut, devSrc, scalar, stream); \
    aclrtSynchronizeStream(stream); \
    aclrtDestroyStream(stream); \
    aclrtFree(devOut); aclrtFree(devSrc); \
    aclrtResetDevice(0); \
    aclFinalize(); \
}

// 16KB (4096 elements)
TADDS_FLOAT_TEST(1x4096, 1, 4096, 4096)
TADDS_FLOAT_TEST(64x64, 64, 64, 4096)
TADDS_FLOAT_TEST(128x32, 128, 32, 4096)
TADDS_FLOAT_TEST(32x128, 32, 128, 4096)
// 32KB (8192 elements)
TADDS_FLOAT_TEST(1x8192, 1, 8192, 8192)
TADDS_FLOAT_TEST(64x128, 64, 128, 8192)
TADDS_FLOAT_TEST(256x32, 256, 32, 8192)
TADDS_FLOAT_TEST(32x256, 32, 256, 8192)
// 64KB (16384 elements)
TADDS_FLOAT_TEST(1x16384, 1, 16384, 16384)
TADDS_FLOAT_TEST(128x128, 128, 128, 16384)
TADDS_FLOAT_TEST(512x32, 512, 32, 16384)
TADDS_FLOAT_TEST(32x512, 32, 512, 16384)

// ========== TADDS fp16 Tests ==========
#define TADDS_HALF_TEST(NAME, H, W, ELEM) \
TEST_F(TilePerfTest, TADDS_half_##NAME) { \
    size_t bytes = (ELEM) * sizeof(aclFloat16); \
    float scalar = 1.5f; \
    aclInit(nullptr); \
    aclrtSetDevice(0); \
    void *devOut, *devSrc; \
    aclrtMalloc(&devOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtMalloc(&devSrc, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtStream stream = nullptr; \
    aclrtCreateStream(&stream); \
    launchTADDS<aclFloat16, H, W, H, W>(devOut, devSrc, scalar, stream); \
    aclrtSynchronizeStream(stream); \
    aclrtDestroyStream(stream); \
    aclrtFree(devOut); aclrtFree(devSrc); \
    aclrtResetDevice(0); \
    aclFinalize(); \
}

// 16KB (8192 elements)
TADDS_HALF_TEST(1x8192, 1, 8192, 8192)
TADDS_HALF_TEST(64x128, 64, 128, 8192)
TADDS_HALF_TEST(256x32, 256, 32, 8192)
TADDS_HALF_TEST(32x256, 32, 256, 8192)
// 32KB (16384 elements)
TADDS_HALF_TEST(1x16384, 1, 16384, 16384)
TADDS_HALF_TEST(128x128, 128, 128, 16384)
TADDS_HALF_TEST(512x32, 512, 32, 16384)
TADDS_HALF_TEST(32x512, 32, 512, 16384)
// 64KB (32768 elements)
TADDS_HALF_TEST(1x32768, 1, 32768, 32768)
TADDS_HALF_TEST(128x256, 128, 256, 32768)
TADDS_HALF_TEST(1024x32, 1024, 32, 32768)
TADDS_HALF_TEST(32x1024, 32, 1024, 32768)
