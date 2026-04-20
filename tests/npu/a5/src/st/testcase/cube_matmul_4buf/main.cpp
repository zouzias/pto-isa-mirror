/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
CANN Open Software License Agreement Version 2.0
*/

#include <gtest/gtest.h>
#include <acl/acl.h>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

#ifndef TEST_DIR
#define TEST_DIR "."
#endif

constexpr int M = 32;
constexpr int K = 16;
constexpr int N = 256;
constexpr int K_ITERS = 64;

// External kernel launcher
extern void LaunchCubeMatmul4Buf(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

class CubeMatmul4BufTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Use current working directory (build/) for data files
std::string GetGoldenDir() {
    return "./CubeMatmul4BufTest.case_f16_32x1024_1024x256";
}

void ReadFile(const std::string &path, size_t size, void *data, size_t maxSize) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open " << path << std::endl;
        return;
    }
    file.read(reinterpret_cast<char*>(data), std::min(size, maxSize));
}

void WriteFile(const std::string &path, const void *data, size_t size) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data), size);
}

template <typename T>
bool ResultCmp(const std::vector<T> &golden, const std::vector<T> &actual, float threshold) {
    if (golden.size() != actual.size()) return false;
    
    float maxDiff = 0.0f;
    float maxRatio = 0.0f;
    int badCount = 0;
    
    for (size_t i = 0; i < golden.size(); i++) {
        float g = static_cast<float>(golden[i]);
        float a = static_cast<float>(actual[i]);
        float diff = std::abs(g - a);
        float ratio = (std::abs(g) > 1e-6f) ? diff / std::abs(g) : 0.0f;
        
        maxDiff = std::max(maxDiff, diff);
        maxRatio = std::max(maxRatio, ratio);
        
        if (diff > threshold && ratio > 0.01f) {
            if (badCount < 10) {
                std::cout << "Mismatch at " << i << ": golden=" << g << " actual=" << a << std::endl;
            }
            badCount++;
        }
    }
    
    std::cout << "max diff: " << maxDiff << ", max ratio: " << maxRatio 
              << ", bad count: " << badCount << std::endl;
    
    return maxDiff < threshold || maxRatio < 0.01f;
}

TEST_F(CubeMatmul4BufTest, case_f16_32x1024_1024x256) {
    size_t aFileSize = K_ITERS * M * K * sizeof(uint16_t);  // half
    size_t bFileSize = K_ITERS * K * N * sizeof(uint16_t);  // half
    size_t cFileSize = M * N * sizeof(float);
    
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    
    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device;
    
    aclrtMallocHost((void**)&dstHost, cFileSize);
    aclrtMallocHost((void**)&src0Host, aFileSize);
    aclrtMallocHost((void**)&src1Host, bFileSize);
    
    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    std::string goldenDir = GetGoldenDir();
    
    ReadFile(goldenDir + "/A_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(goldenDir + "/B_gm.bin", bFileSize, src1Host, bFileSize);
    
    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    
    LaunchCubeMatmul4Buf(dstDevice, src0Device, src1Device, stream);
    
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    
    WriteFile(goldenDir + "/output_npu.bin", dstHost, cFileSize);
    
    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    
    std::vector<float> golden(M * N);
    std::vector<float> actual(M * N);
    ReadFile(goldenDir + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(goldenDir + "/output_npu.bin", cFileSize, actual.data(), cFileSize);
    
    bool ret = ResultCmp(golden, actual, 0.01f);
    EXPECT_TRUE(ret);
}
