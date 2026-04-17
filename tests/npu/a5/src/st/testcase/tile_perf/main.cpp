/**
 * Tile Performance Benchmark - Test Main
 * fp32 only for now (fp16 build issue pending)
 */
#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>
#include <cstring>

using namespace std;
using namespace PtoTestCommon;

// Forward declare kernel launcher templates
template <int caseId> void launchTADD_float(void *out, void *src0, void *src1, aclrtStream stream);
template <int caseId> void launchTEXP_float(void *out, void *src, aclrtStream stream);
template <int caseId> void launchTADDS_float(void *out, void *src, float scalar, aclrtStream stream);

// Test fixtures
class TilePerfTest : public testing::Test {};

// TADD fp32 Tests
#define TADD_FLOAT_TEST(NAME, CASE_ID, ELEM) \
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
    launchTADD_float<CASE_ID>(devOut, devSrc0, devSrc1, stream); \
    aclrtSynchronizeStream(stream); \
    aclrtDestroyStream(stream); \
    aclrtFree(devOut); aclrtFree(devSrc0); aclrtFree(devSrc1); \
    aclrtResetDevice(0); \
    aclFinalize(); \
}

TADD_FLOAT_TEST(1x4096, 1, 4096)
TADD_FLOAT_TEST(64x64, 2, 4096)
TADD_FLOAT_TEST(128x32, 3, 4096)
TADD_FLOAT_TEST(32x128, 4, 4096)
TADD_FLOAT_TEST(1x8192, 5, 8192)
TADD_FLOAT_TEST(64x128, 6, 8192)
TADD_FLOAT_TEST(256x32, 7, 8192)
TADD_FLOAT_TEST(32x256, 8, 8192)
TADD_FLOAT_TEST(1x16384, 9, 16384)
TADD_FLOAT_TEST(128x128, 10, 16384)
TADD_FLOAT_TEST(512x32, 11, 16384)
TADD_FLOAT_TEST(32x512, 12, 16384)

// TEXP fp32 Tests
#define TEXP_FLOAT_TEST(NAME, CASE_ID, ELEM) \
TEST_F(TilePerfTest, TEXP_float_##NAME) { \
    size_t bytes = (ELEM) * sizeof(float); \
    aclInit(nullptr); \
    aclrtSetDevice(0); \
    void *devOut, *devSrc; \
    aclrtMalloc(&devOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtMalloc(&devSrc, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \
    aclrtStream stream = nullptr; \
    aclrtCreateStream(&stream); \
    launchTEXP_float<CASE_ID>(devOut, devSrc, stream); \
    aclrtSynchronizeStream(stream); \
    aclrtDestroyStream(stream); \
    aclrtFree(devOut); aclrtFree(devSrc); \
    aclrtResetDevice(0); \
    aclFinalize(); \
}

TEXP_FLOAT_TEST(1x4096, 1, 4096)
TEXP_FLOAT_TEST(64x64, 2, 4096)
TEXP_FLOAT_TEST(128x32, 3, 4096)
TEXP_FLOAT_TEST(32x128, 4, 4096)
TEXP_FLOAT_TEST(1x8192, 5, 8192)
TEXP_FLOAT_TEST(64x128, 6, 8192)
TEXP_FLOAT_TEST(256x32, 7, 8192)
TEXP_FLOAT_TEST(32x256, 8, 8192)
TEXP_FLOAT_TEST(1x16384, 9, 16384)
TEXP_FLOAT_TEST(128x128, 10, 16384)
TEXP_FLOAT_TEST(512x32, 11, 16384)
TEXP_FLOAT_TEST(32x512, 12, 16384)

// TADDS fp32 Tests
#define TADDS_FLOAT_TEST(NAME, CASE_ID, ELEM) \
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
    launchTADDS_float<CASE_ID>(devOut, devSrc, scalar, stream); \
    aclrtSynchronizeStream(stream); \
    aclrtDestroyStream(stream); \
    aclrtFree(devOut); aclrtFree(devSrc); \
    aclrtResetDevice(0); \
    aclFinalize(); \
}

TADDS_FLOAT_TEST(1x4096, 1, 4096)
TADDS_FLOAT_TEST(64x64, 2, 4096)
TADDS_FLOAT_TEST(128x32, 3, 4096)
TADDS_FLOAT_TEST(32x128, 4, 4096)
TADDS_FLOAT_TEST(1x8192, 5, 8192)
TADDS_FLOAT_TEST(64x128, 6, 8192)
TADDS_FLOAT_TEST(256x32, 7, 8192)
TADDS_FLOAT_TEST(32x256, 8, 8192)
TADDS_FLOAT_TEST(1x16384, 9, 16384)
TADDS_FLOAT_TEST(128x128, 10, 16384)
TADDS_FLOAT_TEST(512x32, 11, 16384)
TADDS_FLOAT_TEST(32x512, 12, 16384)
