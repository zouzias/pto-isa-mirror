/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
CANN Open Software License Agreement Version 2.0
*/

/**
 * NPU test harness for cube_matmul_nbuf benchmark suite.
 *
 * Configurations (all: M=32, K=1024, N=256, fp16→fp32):
 *   2-buf  K_tile=16  (8KB  B tile, ping-pong)
 *   4-buf  K_tile=16  (8KB  B tile)
 *   8-buf  K_tile=16  (8KB  B tile)
 *   2-buf  K_tile=32  (16KB B tile, ping-pong)
 *   4-buf  K_tile=32  (16KB B tile)
 */

#include <gtest/gtest.h>
#include <acl/acl.h>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

static constexpr int GM_M = 32;
static constexpr int GM_N = 256;
static constexpr int GM_K = 1024;

// Kernel launchers (defined in cube_matmul_nbuf_kernel.cpp)
extern void LaunchCubeMatmul2Buf8K (uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
extern void LaunchCubeMatmul4Buf8K (uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
extern void LaunchCubeMatmul8Buf8K (uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
extern void LaunchCubeMatmul2Buf16K(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
extern void LaunchCubeMatmul4Buf16K(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

// ─── Helpers ─────────────────────────────────────────────────────────────────

static bool ReadFile(const std::string &path, size_t size, void *buf, size_t maxSize) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; return false; }
    f.read(reinterpret_cast<char *>(buf), std::min(size, maxSize));
    return true;
}

static void WriteFile(const std::string &path, const void *data, size_t size) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(data), size);
}

static bool ResultCmp(const std::vector<float> &golden,
                      const std::vector<float> &actual,
                      float threshold = 0.01f)
{
    if (golden.size() != actual.size()) return false;
    float maxDiff = 0, maxRatio = 0;
    int   bad = 0;
    for (size_t i = 0; i < golden.size(); i++) {
        float diff  = std::abs(golden[i] - actual[i]);
        float ratio = (std::abs(golden[i]) > 1e-6f) ? diff / std::abs(golden[i]) : 0;
        if (i < 10 && diff > 1e-4f)
            std::cerr << "  Mismatch at " << i << ": golden=" << golden[i]
                      << " actual=" << actual[i] << "\n";
        if (diff > maxDiff)  maxDiff  = diff;
        if (ratio > maxRatio) maxRatio = ratio;
        if (diff > threshold) bad++;
    }
    int errThresh = static_cast<int>(golden.size()) / 100 + 1;
    std::cout << "max diff: " << maxDiff << ", max ratio: " << maxRatio
              << ", bad count: " << bad << "\n";
    return bad <= errThresh;
}

// Golden data lives in the 8KB-tile gen_data dir (same A/B/golden for all configs
// because M=32, K=1024, N=256 is fixed; K_tile is a tiling choice not a shape change)
static const std::string kGoldenDir = "../CubeMatmulNBufTest.golden";

// ─── Generic test runner ──────────────────────────────────────────────────────

using LaunchFn = void (*)(uint8_t *, uint8_t *, uint8_t *, void *);

static void RunNBufTest(LaunchFn launch, const std::string &outName)
{
    size_t aBytes = GM_M * GM_K * sizeof(uint16_t);
    size_t bBytes = GM_K * GM_N * sizeof(uint16_t);
    size_t cBytes = GM_M * GM_N * sizeof(float);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint16_t *devA, *devB;
    float    *devC;
    void     *hostA, *hostB, *hostC;

    aclrtMallocHost(&hostA, aBytes);
    aclrtMallocHost(&hostB, bBytes);
    aclrtMallocHost(&hostC, cBytes);
    aclrtMalloc((void **)&devA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&devB, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&devC, cBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    ASSERT_TRUE(ReadFile(kGoldenDir + "/A_gm.bin", aBytes, hostA, aBytes));
    ASSERT_TRUE(ReadFile(kGoldenDir + "/B_gm.bin", bBytes, hostB, bBytes));

    aclrtMemcpy(devA, aBytes, hostA, aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(devB, bBytes, hostB, bBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    launch(reinterpret_cast<uint8_t *>(devC),
           reinterpret_cast<uint8_t *>(devA),
           reinterpret_cast<uint8_t *>(devB),
           stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(hostC, cBytes, devC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    system(("mkdir -p " + kGoldenDir).c_str());
    WriteFile(kGoldenDir + "/" + outName, hostC, cBytes);

    aclrtFree(devC); aclrtFree(devA); aclrtFree(devB);
    aclrtFreeHost(hostC); aclrtFreeHost(hostA); aclrtFreeHost(hostB);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(GM_M * GM_N), actual(GM_M * GM_N);
    ASSERT_TRUE(ReadFile(kGoldenDir + "/golden.bin", cBytes, golden.data(), cBytes));
    ASSERT_TRUE(ReadFile(kGoldenDir + "/" + outName,  cBytes, actual.data(), cBytes));
    EXPECT_TRUE(ResultCmp(golden, actual));
}

// ─── Test cases ───────────────────────────────────────────────────────────────

TEST(CubeMatmulNBufTest, buf2_ktile16_8KB)  { RunNBufTest(LaunchCubeMatmul2Buf8K,  "out_2buf_8k.bin");  }
TEST(CubeMatmulNBufTest, buf4_ktile16_8KB)  { RunNBufTest(LaunchCubeMatmul4Buf8K,  "out_4buf_8k.bin");  }
TEST(CubeMatmulNBufTest, buf8_ktile16_8KB)  { RunNBufTest(LaunchCubeMatmul8Buf8K,  "out_8buf_8k.bin");  }
TEST(CubeMatmulNBufTest, buf2_ktile32_16KB) { RunNBufTest(LaunchCubeMatmul2Buf16K, "out_2buf_16k.bin"); }
TEST(CubeMatmulNBufTest, buf4_ktile32_16KB) { RunNBufTest(LaunchCubeMatmul4Buf16K, "out_4buf_16k.bin"); }
