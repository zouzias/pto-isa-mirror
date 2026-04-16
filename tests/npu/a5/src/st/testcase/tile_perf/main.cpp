/**
 * TADD vs TADDS Benchmark - main.cpp
 */
#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>
#include <cstring>

using namespace std;
using namespace PtoTestCommon;

// ===== External kernel launchers =====
template <uint32_t caseId>
void launchTADDBenchTestCase(void *out, void *src0, void *src1, aclrtStream stream);

template <uint32_t caseId>
void launchTADDSBenchTestCase(void *out, void *src, float scalar, aclrtStream stream);

// ===== Test classes =====
class TADDBenchTest : public testing::Test {};
class TADDSBenchTest : public testing::Test {};

// ===== TADD 32x64 =====
TEST_F(TADDBenchTest, case_float_32x64)
{
    constexpr int H = 32, W = 64;
    constexpr size_t bytes = H * W * sizeof(float);
    
    aclInit(nullptr);
    aclrtSetDevice(0);
    
    void *devOut, *devSrc0, *devSrc1;
    aclrtMalloc(&devOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devSrc0, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devSrc1, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    
    launchTADDBenchTestCase<1>(devOut, devSrc0, devSrc1, stream);
    aclrtSynchronizeStream(stream);
    
    aclrtDestroyStream(stream);
    aclrtFree(devOut);
    aclrtFree(devSrc0);
    aclrtFree(devSrc1);
    aclrtResetDevice(0);
    aclFinalize();
}

// ===== TADD 1x2048 =====
TEST_F(TADDBenchTest, case_float_1x2048)
{
    constexpr int H = 1, W = 2048;
    constexpr size_t bytes = H * W * sizeof(float);
    
    aclInit(nullptr);
    aclrtSetDevice(0);
    
    void *devOut, *devSrc0, *devSrc1;
    aclrtMalloc(&devOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devSrc0, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devSrc1, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    
    launchTADDBenchTestCase<2>(devOut, devSrc0, devSrc1, stream);
    aclrtSynchronizeStream(stream);
    
    aclrtDestroyStream(stream);
    aclrtFree(devOut);
    aclrtFree(devSrc0);
    aclrtFree(devSrc1);
    aclrtResetDevice(0);
    aclFinalize();
}

// ===== TADDS 32x64 =====
TEST_F(TADDSBenchTest, case_float_32x64)
{
    constexpr int H = 32, W = 64;
    constexpr size_t bytes = H * W * sizeof(float);
    float scalar = 3.14f;
    
    aclInit(nullptr);
    aclrtSetDevice(0);
    
    void *devOut, *devSrc;
    aclrtMalloc(&devOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devSrc, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    
    launchTADDSBenchTestCase<1>(devOut, devSrc, scalar, stream);
    aclrtSynchronizeStream(stream);
    
    aclrtDestroyStream(stream);
    aclrtFree(devOut);
    aclrtFree(devSrc);
    aclrtResetDevice(0);
    aclFinalize();
}

// ===== TADDS 1x2048 =====
TEST_F(TADDSBenchTest, case_float_1x2048)
{
    constexpr int H = 1, W = 2048;
    constexpr size_t bytes = H * W * sizeof(float);
    float scalar = 3.14f;
    
    aclInit(nullptr);
    aclrtSetDevice(0);
    
    void *devOut, *devSrc;
    aclrtMalloc(&devOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devSrc, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    
    launchTADDSBenchTestCase<2>(devOut, devSrc, scalar, stream);
    aclrtSynchronizeStream(stream);
    
    aclrtDestroyStream(stream);
    aclrtFree(devOut);
    aclrtFree(devSrc);
    aclrtResetDevice(0);
    aclFinalize();
}
