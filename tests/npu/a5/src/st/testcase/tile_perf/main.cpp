/**
 * Tile Performance Benchmark - Test Main
 * Auto-generated test cases from input.csv
 */
#include <gtest/gtest.h>
#include <acl/acl.h>
#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <random>

// Forward declarations from kernel
// TADD fp32
extern "C" __global__ void launchTADD_float_1x4096(float*, float*, float*);
extern "C" __global__ void launchTADD_float_64x64(float*, float*, float*);
extern "C" __global__ void launchTADD_float_128x32(float*, float*, float*);
extern "C" __global__ void launchTADD_float_32x128(float*, float*, float*);
extern "C" __global__ void launchTADD_float_1x8192(float*, float*, float*);
extern "C" __global__ void launchTADD_float_64x128(float*, float*, float*);
extern "C" __global__ void launchTADD_float_256x32(float*, float*, float*);
extern "C" __global__ void launchTADD_float_32x256(float*, float*, float*);
extern "C" __global__ void launchTADD_float_1x16384(float*, float*, float*);
extern "C" __global__ void launchTADD_float_128x128(float*, float*, float*);
extern "C" __global__ void launchTADD_float_512x32(float*, float*, float*);
extern "C" __global__ void launchTADD_float_32x512(float*, float*, float*);

// TADD fp16
extern "C" __global__ void launchTADD_half_1x8192(half*, half*, half*);
extern "C" __global__ void launchTADD_half_64x128(half*, half*, half*);
extern "C" __global__ void launchTADD_half_256x32(half*, half*, half*);
extern "C" __global__ void launchTADD_half_32x256(half*, half*, half*);
extern "C" __global__ void launchTADD_half_1x16384(half*, half*, half*);
extern "C" __global__ void launchTADD_half_128x128(half*, half*, half*);
extern "C" __global__ void launchTADD_half_512x32(half*, half*, half*);
extern "C" __global__ void launchTADD_half_32x512(half*, half*, half*);
extern "C" __global__ void launchTADD_half_1x32768(half*, half*, half*);
extern "C" __global__ void launchTADD_half_128x256(half*, half*, half*);
extern "C" __global__ void launchTADD_half_1024x32(half*, half*, half*);
extern "C" __global__ void launchTADD_half_32x1024(half*, half*, half*);

// TEXP fp32
extern "C" __global__ void launchTEXP_float_1x4096(float*, float*);
extern "C" __global__ void launchTEXP_float_64x64(float*, float*);
extern "C" __global__ void launchTEXP_float_128x32(float*, float*);
extern "C" __global__ void launchTEXP_float_32x128(float*, float*);
extern "C" __global__ void launchTEXP_float_1x8192(float*, float*);
extern "C" __global__ void launchTEXP_float_64x128(float*, float*);
extern "C" __global__ void launchTEXP_float_256x32(float*, float*);
extern "C" __global__ void launchTEXP_float_32x256(float*, float*);
extern "C" __global__ void launchTEXP_float_1x16384(float*, float*);
extern "C" __global__ void launchTEXP_float_128x128(float*, float*);
extern "C" __global__ void launchTEXP_float_512x32(float*, float*);
extern "C" __global__ void launchTEXP_float_32x512(float*, float*);

// TEXP fp16
extern "C" __global__ void launchTEXP_half_1x8192(half*, half*);
extern "C" __global__ void launchTEXP_half_64x128(half*, half*);
extern "C" __global__ void launchTEXP_half_256x32(half*, half*);
extern "C" __global__ void launchTEXP_half_32x256(half*, half*);
extern "C" __global__ void launchTEXP_half_1x16384(half*, half*);
extern "C" __global__ void launchTEXP_half_128x128(half*, half*);
extern "C" __global__ void launchTEXP_half_512x32(half*, half*);
extern "C" __global__ void launchTEXP_half_32x512(half*, half*);
extern "C" __global__ void launchTEXP_half_1x32768(half*, half*);
extern "C" __global__ void launchTEXP_half_128x256(half*, half*);
extern "C" __global__ void launchTEXP_half_1024x32(half*, half*);
extern "C" __global__ void launchTEXP_half_32x1024(half*, half*);

// TADDS fp32
extern "C" __global__ void launchTADDS_float_1x4096(float*, float*, float);
extern "C" __global__ void launchTADDS_float_64x64(float*, float*, float);
extern "C" __global__ void launchTADDS_float_128x32(float*, float*, float);
extern "C" __global__ void launchTADDS_float_32x128(float*, float*, float);
extern "C" __global__ void launchTADDS_float_1x8192(float*, float*, float);
extern "C" __global__ void launchTADDS_float_64x128(float*, float*, float);
extern "C" __global__ void launchTADDS_float_256x32(float*, float*, float);
extern "C" __global__ void launchTADDS_float_32x256(float*, float*, float);
extern "C" __global__ void launchTADDS_float_1x16384(float*, float*, float);
extern "C" __global__ void launchTADDS_float_128x128(float*, float*, float);
extern "C" __global__ void launchTADDS_float_512x32(float*, float*, float);
extern "C" __global__ void launchTADDS_float_32x512(float*, float*, float);

// TADDS fp16
extern "C" __global__ void launchTADDS_half_1x8192(half*, half*, half);
extern "C" __global__ void launchTADDS_half_64x128(half*, half*, half);
extern "C" __global__ void launchTADDS_half_256x32(half*, half*, half);
extern "C" __global__ void launchTADDS_half_32x256(half*, half*, half);
extern "C" __global__ void launchTADDS_half_1x16384(half*, half*, half);
extern "C" __global__ void launchTADDS_half_128x128(half*, half*, half);
extern "C" __global__ void launchTADDS_half_512x32(half*, half*, half);
extern "C" __global__ void launchTADDS_half_32x512(half*, half*, half);
extern "C" __global__ void launchTADDS_half_1x32768(half*, half*, half);
extern "C" __global__ void launchTADDS_half_128x256(half*, half*, half);
extern "C" __global__ void launchTADDS_half_1024x32(half*, half*, half);
extern "C" __global__ void launchTADDS_half_32x1024(half*, half*, half);

class TilePerfTest : public ::testing::Test {
protected:
    void SetUp() override {
        aclInit(nullptr);
        aclrtSetDevice(0);
        aclrtCreateStream(&stream);
    }
    
    void TearDown() override {
        aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
    }
    
    aclrtStream stream;
};

// Helper to allocate and initialize device memory
template<typename T>
void allocAndInit(void** devPtr, size_t count, T initVal = T(1)) {
    size_t bytes = count * sizeof(T);
    aclrtMalloc(devPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    std::vector<T> hostData(count, initVal);
    aclrtMemcpy(*devPtr, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
}

// ========== TADD Tests - fp32 ==========
#define TADD_FLOAT_TEST(H, W, ELEM) \
TEST_F(TilePerfTest, TADD_float_##H##x##W) { \
    void *out, *src0, *src1; \
    allocAndInit<float>(&out, ELEM, 0.0f); \
    allocAndInit<float>(&src0, ELEM, 1.0f); \
    allocAndInit<float>(&src1, ELEM, 2.0f); \
    launchTADD_float_##H##x##W<<<1, nullptr, stream>>>((float*)out, (float*)src0, (float*)src1); \
    aclrtSynchronizeStream(stream); \
    aclrtFree(out); aclrtFree(src0); aclrtFree(src1); \
}

TADD_FLOAT_TEST(1, 4096, 4096)
TADD_FLOAT_TEST(64, 64, 4096)
TADD_FLOAT_TEST(128, 32, 4096)
TADD_FLOAT_TEST(32, 128, 4096)
TADD_FLOAT_TEST(1, 8192, 8192)
TADD_FLOAT_TEST(64, 128, 8192)
TADD_FLOAT_TEST(256, 32, 8192)
TADD_FLOAT_TEST(32, 256, 8192)
TADD_FLOAT_TEST(1, 16384, 16384)
TADD_FLOAT_TEST(128, 128, 16384)
TADD_FLOAT_TEST(512, 32, 16384)
TADD_FLOAT_TEST(32, 512, 16384)

// ========== TADD Tests - fp16 ==========
#define TADD_HALF_TEST(H, W, ELEM) \
TEST_F(TilePerfTest, TADD_half_##H##x##W) { \
    void *out, *src0, *src1; \
    allocAndInit<half>(&out, ELEM, half(0)); \
    allocAndInit<half>(&src0, ELEM, half(1)); \
    allocAndInit<half>(&src1, ELEM, half(2)); \
    launchTADD_half_##H##x##W<<<1, nullptr, stream>>>((half*)out, (half*)src0, (half*)src1); \
    aclrtSynchronizeStream(stream); \
    aclrtFree(out); aclrtFree(src0); aclrtFree(src1); \
}

TADD_HALF_TEST(1, 8192, 8192)
TADD_HALF_TEST(64, 128, 8192)
TADD_HALF_TEST(256, 32, 8192)
TADD_HALF_TEST(32, 256, 8192)
TADD_HALF_TEST(1, 16384, 16384)
TADD_HALF_TEST(128, 128, 16384)
TADD_HALF_TEST(512, 32, 16384)
TADD_HALF_TEST(32, 512, 16384)
TADD_HALF_TEST(1, 32768, 32768)
TADD_HALF_TEST(128, 256, 32768)
TADD_HALF_TEST(1024, 32, 32768)
TADD_HALF_TEST(32, 1024, 32768)

// ========== TEXP Tests - fp32 ==========
#define TEXP_FLOAT_TEST(H, W, ELEM) \
TEST_F(TilePerfTest, TEXP_float_##H##x##W) { \
    void *out, *src; \
    allocAndInit<float>(&out, ELEM, 0.0f); \
    allocAndInit<float>(&src, ELEM, 0.5f); \
    launchTEXP_float_##H##x##W<<<1, nullptr, stream>>>((float*)out, (float*)src); \
    aclrtSynchronizeStream(stream); \
    aclrtFree(out); aclrtFree(src); \
}

TEXP_FLOAT_TEST(1, 4096, 4096)
TEXP_FLOAT_TEST(64, 64, 4096)
TEXP_FLOAT_TEST(128, 32, 4096)
TEXP_FLOAT_TEST(32, 128, 4096)
TEXP_FLOAT_TEST(1, 8192, 8192)
TEXP_FLOAT_TEST(64, 128, 8192)
TEXP_FLOAT_TEST(256, 32, 8192)
TEXP_FLOAT_TEST(32, 256, 8192)
TEXP_FLOAT_TEST(1, 16384, 16384)
TEXP_FLOAT_TEST(128, 128, 16384)
TEXP_FLOAT_TEST(512, 32, 16384)
TEXP_FLOAT_TEST(32, 512, 16384)

// ========== TEXP Tests - fp16 ==========
#define TEXP_HALF_TEST(H, W, ELEM) \
TEST_F(TilePerfTest, TEXP_half_##H##x##W) { \
    void *out, *src; \
    allocAndInit<half>(&out, ELEM, half(0)); \
    allocAndInit<half>(&src, ELEM, half(0.5)); \
    launchTEXP_half_##H##x##W<<<1, nullptr, stream>>>((half*)out, (half*)src); \
    aclrtSynchronizeStream(stream); \
    aclrtFree(out); aclrtFree(src); \
}

TEXP_HALF_TEST(1, 8192, 8192)
TEXP_HALF_TEST(64, 128, 8192)
TEXP_HALF_TEST(256, 32, 8192)
TEXP_HALF_TEST(32, 256, 8192)
TEXP_HALF_TEST(1, 16384, 16384)
TEXP_HALF_TEST(128, 128, 16384)
TEXP_HALF_TEST(512, 32, 16384)
TEXP_HALF_TEST(32, 512, 16384)
TEXP_HALF_TEST(1, 32768, 32768)
TEXP_HALF_TEST(128, 256, 32768)
TEXP_HALF_TEST(1024, 32, 32768)
TEXP_HALF_TEST(32, 1024, 32768)

// ========== TADDS Tests - fp32 ==========
#define TADDS_FLOAT_TEST(H, W, ELEM) \
TEST_F(TilePerfTest, TADDS_float_##H##x##W) { \
    void *out, *src; \
    allocAndInit<float>(&out, ELEM, 0.0f); \
    allocAndInit<float>(&src, ELEM, 1.0f); \
    launchTADDS_float_##H##x##W<<<1, nullptr, stream>>>((float*)out, (float*)src, 1.5f); \
    aclrtSynchronizeStream(stream); \
    aclrtFree(out); aclrtFree(src); \
}

TADDS_FLOAT_TEST(1, 4096, 4096)
TADDS_FLOAT_TEST(64, 64, 4096)
TADDS_FLOAT_TEST(128, 32, 4096)
TADDS_FLOAT_TEST(32, 128, 4096)
TADDS_FLOAT_TEST(1, 8192, 8192)
TADDS_FLOAT_TEST(64, 128, 8192)
TADDS_FLOAT_TEST(256, 32, 8192)
TADDS_FLOAT_TEST(32, 256, 8192)
TADDS_FLOAT_TEST(1, 16384, 16384)
TADDS_FLOAT_TEST(128, 128, 16384)
TADDS_FLOAT_TEST(512, 32, 16384)
TADDS_FLOAT_TEST(32, 512, 16384)

// ========== TADDS Tests - fp16 ==========
#define TADDS_HALF_TEST(H, W, ELEM) \
TEST_F(TilePerfTest, TADDS_half_##H##x##W) { \
    void *out, *src; \
    allocAndInit<half>(&out, ELEM, half(0)); \
    allocAndInit<half>(&src, ELEM, half(1)); \
    launchTADDS_half_##H##x##W<<<1, nullptr, stream>>>((half*)out, (half*)src, half(1.5)); \
    aclrtSynchronizeStream(stream); \
    aclrtFree(out); aclrtFree(src); \
}

TADDS_HALF_TEST(1, 8192, 8192)
TADDS_HALF_TEST(64, 128, 8192)
TADDS_HALF_TEST(256, 32, 8192)
TADDS_HALF_TEST(32, 256, 8192)
TADDS_HALF_TEST(1, 16384, 16384)
TADDS_HALF_TEST(128, 128, 16384)
TADDS_HALF_TEST(512, 32, 16384)
TADDS_HALF_TEST(32, 512, 16384)
TADDS_HALF_TEST(1, 32768, 32768)
TADDS_HALF_TEST(128, 256, 32768)
TADDS_HALF_TEST(1024, 32, 32768)
TADDS_HALF_TEST(32, 1024, 32768)
