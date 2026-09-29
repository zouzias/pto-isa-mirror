/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>
#include <cstdlib>
#include <vector>

#include <gtest/gtest.h>
#include "acl/acl.h"

#include "test_common.h"

using namespace std;
using namespace PtoTestCommon;

template <int32_t tilingKey>
void launchTEXTRACTAcc2Mat(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

template <int32_t tilingKey>
void launchTEXTRACT(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

template <int32_t tilingKey>
void launchTMOV(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

template <int32_t tilingKey>
void launchTEXTRACTMX(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* srcMx0, uint8_t* srcMx1, void* stream);

template <int32_t tilingKey>
void launchTMOVMX(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* srcMx0, uint8_t* srcMx1, void* stream);

class TEXTRACTTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <int32_t key, typename T, typename U, typename S>
void textract_test(uint32_t M, uint32_t K, uint32_t N, uint16_t indexM, uint16_t indexK, uint16_t indexN)
{
    uint32_t mValid = M - indexM;
    uint32_t nValid = N - indexN;
    size_t aFileSize = M * K * sizeof(U);
    size_t bFileSize = K * N * sizeof(U);
    constexpr bool accToMat = key >= 21 && key <= 26;
    size_t cFileSize = (accToMat ? 16 * 48 * (key == 22 || key == 25 ? 2 : 1) : mValid * nValid) * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    aclrtMemset(dstDevice, cFileSize, 0, cFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if constexpr (accToMat) {
        launchTEXTRACTAcc2Mat<key>(dstDevice, src0Device, src1Device, stream);
    } else {
        launchTEXTRACT<key>(dstDevice, src0Device, src1Device, stream);
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(cFileSize / sizeof(T));
    std::vector<T> devFinal(cFileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <int32_t key, typename T, typename U, typename S>
void textract_mx_test(uint32_t M, uint32_t K, uint32_t N, uint16_t indexM, uint16_t indexK, uint16_t indexN)
{
    uint32_t mValid = M - indexM;
    uint32_t nValid = N - indexN;
    size_t aFileSize = M * K * sizeof(U) / 2;
    size_t bFileSize = K * N * sizeof(U) / 2;
    size_t amxFileSize = M * K / 32 * sizeof(int8_t);
    size_t bmxFileSize = K / 32 * N * sizeof(int8_t);
    size_t cFileSize = mValid * nValid * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host, *srcMx0Host, *srcMx1Host;
    uint8_t *dstDevice, *src0Device, *src1Device, *srcMx0Device, *srcMx1Device;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);
    aclrtMallocHost((void**)(&srcMx0Host), amxFileSize);
    aclrtMallocHost((void**)(&srcMx1Host), bmxFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcMx0Device, amxFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcMx1Device, bmxFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    ReadFile(GetGoldenDir() + "/x1_mx_gm.bin", amxFileSize, srcMx0Host, amxFileSize);
    ReadFile(GetGoldenDir() + "/x2_mx_gm.bin", bmxFileSize, srcMx1Host, bmxFileSize);
    aclrtMemset(dstDevice, cFileSize, 0, cFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcMx0Device, amxFileSize, srcMx0Host, amxFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcMx1Device, bmxFileSize, srcMx1Host, bmxFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTEXTRACTMX<key>(dstDevice, src0Device, src1Device, srcMx0Device, srcMx1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(srcMx0Device);
    aclrtFree(srcMx1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(srcMx0Host);
    aclrtFreeHost(srcMx1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(cFileSize / sizeof(T));
    std::vector<T> devFinal(cFileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TEXTRACTTest, case1) { textract_test<1, float, uint16_t, uint16_t>(32, 96, 64, 0, 0, 0); }

TEST_F(TEXTRACTTest, case2) { textract_test<2, float, float, float>(128, 48, 64, 0, 0, 0); }

TEST_F(TEXTRACTTest, case3) { textract_test<3, int32_t, int8_t, int8_t>(128, 128, 64, 0, 0, 0); }

TEST_F(TEXTRACTTest, case4) { textract_test<4, float, uint16_t, uint16_t>(64, 96, 64, 32, 16, 16); }

TEST_F(TEXTRACTTest, case5) { textract_test<5, float, float, float>(64, 128, 64, 32, 32, 16); }

TEST_F(TEXTRACTTest, case6) { textract_test<6, int32_t, int8_t, int8_t>(128, 128, 64, 32, 64, 32); }

TEST_F(TEXTRACTTest, case7) { textract_test<7, float, uint16_t, uint16_t>(64, 128, 64, 0, 64, 0); }

TEST_F(TEXTRACTTest, case8) { textract_test<8, float, float, float>(64, 64, 128, 0, 0, 32); }

TEST_F(TEXTRACTTest, case9) { textract_test<9, int32_t, int8_t, int8_t>(128, 64, 128, 32, 0, 0); }

TEST_F(TEXTRACTTest, case10) { textract_test<10, float, uint16_t, uint16_t>(64, 128, 64, 16, 0, 0); }

TEST_F(TEXTRACTTest, case11) { textract_test<11, float, int8_t, int8_t>(64, 128, 64, 0, 32, 0); }

TEST_F(TEXTRACTTest, case12) { textract_test<12, float, int8_t, int8_t>(64, 128, 64, 0, 0, 32); }

TEST_F(TEXTRACTTest, case13) { textract_test<13, float, int8_t, int8_t>(64, 128, 64, 0, 32, 0); }

TEST_F(TEXTRACTTest, case14) { textract_test<14, float, int8_t, int8_t>(64, 96, 32, 32, 0, 0); }

TEST_F(TEXTRACTTest, case15) { textract_test<15, float, uint16_t, uint16_t>(64, 48, 96, 16, 16, 0); }

TEST_F(TEXTRACTTest, case16) { textract_test<16, float, float, float>(32, 96, 48, 0, 32, 16); }

TEST_F(TEXTRACTTest, case17) { textract_mx_test<17, float, int8_t, int8_t>(256, 128, 256, 128, 64, 128); }

TEST_F(TEXTRACTTest, case18) { textract_mx_test<18, float, int8_t, int8_t>(256, 128, 256, 128, 64, 128); }

TEST_F(TEXTRACTTest, case19) { textract_mx_test<19, float, int8_t, int8_t>(256, 128, 256, 128, 64, 128); }

TEST_F(TEXTRACTTest, case20) { textract_mx_test<20, float, int8_t, int8_t>(256, 128, 256, 128, 64, 128); }

TEST_F(TEXTRACTTest, case21) { textract_test<21, float, uint16_t, uint16_t>(32, 96, 64, 0, 0, 0); }

TEST_F(TEXTRACTTest, case22) { textract_test<22, float, uint16_t, uint16_t>(32, 96, 64, 0, 0, 0); }

// issue 564: K 切分累加（TMATMUL<Partial> -> TMATMUL_ACC<Final>）配合带 unit flag 的 Acc→Mat 搬出
TEST_F(TEXTRACTTest, case23) { textract_test<23, float, uint16_t, uint16_t>(32, 96, 64, 0, 0, 0); }

// Acc-to-Mat float NZ512: Final, Partial/Final and Unspecified.
TEST_F(TEXTRACTTest, case24) { textract_test<24, float, uint16_t, uint16_t>(32, 96, 64, 0, 0, 0); }

TEST_F(TEXTRACTTest, case25) { textract_test<25, float, uint16_t, uint16_t>(32, 96, 64, 0, 0, 0); }

TEST_F(TEXTRACTTest, case26) { textract_test<26, float, uint16_t, uint16_t>(32, 96, 64, 0, 0, 0); }

class TMOVTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

static bool SetTmovTargetShape(
    uint32_t M, uint32_t K, uint32_t N, uint32_t& targetM, uint32_t& targetK, uint32_t& targetN)
{
    targetM = targetM == 0 ? M : targetM;
    targetN = targetN == 0 ? N : targetN;
    targetK = targetK == 0 ? K : targetK;
    if (targetM >= M && targetN >= N && targetK >= K) {
        return true;
    }
    printf("Error: targetM targetN targetK should large than M N K");
    return false;
}

template <typename T>
void CheckTmovResult(size_t cFileSize)
{
    std::vector<T> golden(cFileSize / sizeof(T));
    std::vector<T> devFinal(cFileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <int32_t key, typename T, typename U, typename S>
void tmov_test(uint32_t M, uint32_t K, uint32_t N, uint32_t targetM = 0, uint32_t targetK = 0, uint32_t targetN = 0)
{
    if (!SetTmovTargetShape(M, K, N, targetM, targetK, targetN)) {
        return;
    }
    size_t aFileSize = targetM * targetK * sizeof(U);
    size_t bFileSize = targetK * targetN * sizeof(U);
    size_t cFileSize = M * N * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    aclrtMemset(dstDevice, cFileSize, 0, cFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTMOV<key>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    CheckTmovResult<T>(cFileSize);
}

template <int32_t key, typename T, typename U, typename S>
void tmov_mx_test(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(U) / 2;
    size_t bFileSize = K * N * sizeof(U) / 2;
    size_t amxFileSize = M * K / 32 * sizeof(int8_t);
    size_t bmxFileSize = K / 32 * N * sizeof(int8_t);
    size_t cFileSize = M * N * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host, *srcMx0Host, *srcMx1Host;
    uint8_t *dstDevice, *src0Device, *src1Device, *srcMx0Device, *srcMx1Device;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);
    aclrtMallocHost((void**)(&srcMx0Host), amxFileSize);
    aclrtMallocHost((void**)(&srcMx1Host), bmxFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcMx0Device, amxFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcMx1Device, bmxFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    ReadFile(GetGoldenDir() + "/x1_mx_gm.bin", amxFileSize, srcMx0Host, amxFileSize);
    ReadFile(GetGoldenDir() + "/x2_mx_gm.bin", bmxFileSize, srcMx1Host, bmxFileSize);
    aclrtMemset(dstDevice, cFileSize, 0, cFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcMx0Device, amxFileSize, srcMx0Host, amxFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcMx1Device, bmxFileSize, srcMx1Host, bmxFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTMOVMX<key>(dstDevice, src0Device, src1Device, srcMx0Device, srcMx1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(srcMx0Device);
    aclrtFree(srcMx1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(srcMx0Host);
    aclrtFreeHost(srcMx1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(cFileSize / sizeof(T));
    std::vector<T> devFinal(cFileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TMOVTest, case1) { tmov_test<1, float, uint16_t, uint16_t>(32, 96, 64); }

TEST_F(TMOVTest, case2) { tmov_test<2, float, float, float>(128, 48, 64); }

TEST_F(TMOVTest, case3) { tmov_test<3, int32_t, int8_t, int8_t>(128, 128, 64); }

TEST_F(TMOVTest, case4) { tmov_test<4, float, uint16_t, uint16_t>(64, 128, 64); }

TEST_F(TMOVTest, case5) { tmov_test<5, float, int8_t, int8_t>(64, 96, 64); }

TEST_F(TMOVTest, case6) { tmov_test<6, float, int8_t, int8_t>(64, 128, 64); }

TEST_F(TMOVTest, case7) { tmov_test<7, float, int8_t, int8_t>(128, 128, 64); }

TEST_F(TMOVTest, case8) { tmov_test<8, float, int8_t, int8_t>(64, 96, 64); }

TEST_F(TMOVTest, case9) { tmov_test<9, float, uint16_t, uint16_t>(64, 128, 64); }

TEST_F(TMOVTest, case10) { tmov_test<10, float, float, float>(64, 128, 64); }

TEST_F(TMOVTest, case11) { tmov_test<11, int32_t, int8_t, int8_t>(65, 40, 66, 96, 64, 96); }

TEST_F(TMOVTest, case12) { tmov_test<12, float, uint16_t, uint16_t>(65, 40, 66, 80, 48, 80); }

TEST_F(TMOVTest, case13) { tmov_test<13, float, float, float>(65, 40, 66, 80, 48, 80); }

TEST_F(TMOVTest, case14) { tmov_mx_test<14, float, int8_t, int8_t>(128, 64, 128); }

TEST_F(TMOVTest, case15) { tmov_mx_test<15, float, int8_t, int8_t>(128, 64, 128); }

TEST_F(TMOVTest, case16) { tmov_mx_test<16, float, int8_t, int8_t>(128, 64, 128); }

TEST_F(TMOVTest, case17) { tmov_mx_test<17, float, int8_t, int8_t>(128, 64, 128); }

template <int32_t TestKey>
void launchTExtractNd2Nz(uint64_t* out, uint64_t* src, void* stream);

static void testTExtractUbToL1Nd2Nz(
    void (*launch)(uint64_t*, uint64_t*, void*), int elementBits, int srcRows, int srcCols, int dstRows, int dstCols,
    int validRows, int validCols, int indexRow, int indexCol, bool useDefaultCopy = false)
{
    const size_t srcBytes = srcRows * srcCols * elementBits / 8;
    const size_t dstBytes = dstRows * dstCols * elementBits / 8;
    std::vector<uint8_t> input(srcBytes);
    std::vector<uint8_t> output(dstBytes, 0xa5);
    std::vector<uint8_t> golden = output;
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<uint8_t>((i * 17 + i / 7) % 251);
    }
    if (useDefaultCopy) {
        // These cases use the full source valid width for the default contiguous copy.
        const size_t sourceOffset = (indexRow * srcCols + indexCol) * elementBits / 8;
        for (size_t byte = 0; byte < static_cast<size_t>(validRows * validCols * elementBits / 8); ++byte) {
            golden[byte] = input[sourceOffset + byte];
        }
    } else {
        for (int row = 0; row < validRows; ++row) {
            for (int colByte = 0; colByte < validCols * elementBits / 8; ++colByte) {
                const size_t offset = (colByte / 32 * dstRows + row) * 32 + colByte % 32;
                golden[offset] =
                    input[(row + indexRow) * srcCols * elementBits / 8 + indexCol * elementBits / 8 + colByte];
            }
        }
    }
    ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
    const char* deviceEnv = std::getenv("PTO_DEVICE_ID");
    const int deviceId = deviceEnv ? std::atoi(deviceEnv) : 0;
    ASSERT_EQ(aclrtSetDevice(deviceId), ACL_SUCCESS);
    aclrtStream stream;
    ASSERT_EQ(aclrtCreateStream(&stream), ACL_SUCCESS);
    void* srcDevice = nullptr;
    void* dstDevice = nullptr;
    ASSERT_EQ(aclrtMalloc(&srcDevice, srcBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dstDevice, dstBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(srcDevice, srcBytes, input.data(), srcBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dstDevice, dstBytes, output.data(), dstBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    launch(static_cast<uint64_t*>(dstDevice), static_cast<uint64_t*>(srcDevice), stream);
    EXPECT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    EXPECT_EQ(aclrtMemcpy(output.data(), dstBytes, dstDevice, dstBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    EXPECT_EQ(aclrtFree(srcDevice), ACL_SUCCESS);
    EXPECT_EQ(aclrtFree(dstDevice), ACL_SUCCESS);
    EXPECT_EQ(aclrtDestroyStream(stream), ACL_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(deviceId), ACL_SUCCESS);
    EXPECT_EQ(aclFinalize(), ACL_SUCCESS);
    EXPECT_EQ(output, golden);
}

TEST_F(TEXTRACTTest, nd2nz_extract_offset)
{
    testTExtractUbToL1Nd2Nz(launchTExtractNd2Nz<21>, 16, 48, 96, 32, 64, 17, 48, 3, 16, false);
}

TEST_F(TEXTRACTTest, nd2nz_extract_bfloat16)
{
    testTExtractUbToL1Nd2Nz(launchTExtractNd2Nz<30>, 16, 48, 96, 32, 64, 17, 48, 3, 16, false);
}

TEST_F(TEXTRACTTest, nd2nz_extract_float)
{
    testTExtractUbToL1Nd2Nz(launchTExtractNd2Nz<31>, 32, 48, 48, 32, 32, 17, 24, 3, 8, false);
}

TEST_F(TEXTRACTTest, nd2nz_extract_int8)
{
    testTExtractUbToL1Nd2Nz(launchTExtractNd2Nz<32>, 8, 48, 192, 32, 128, 17, 96, 3, 32, false);
}

TEST_F(TEXTRACTTest, nd2nz_extract_hifloat8)
{
    testTExtractUbToL1Nd2Nz(launchTExtractNd2Nz<33>, 8, 48, 192, 32, 128, 17, 96, 3, 32, false);
}

TEST_F(TEXTRACTTest, nd2nz_extract_fp8_e4m3)
{
    testTExtractUbToL1Nd2Nz(launchTExtractNd2Nz<34>, 8, 48, 192, 32, 128, 17, 96, 3, 32, false);
}

TEST_F(TEXTRACTTest, nd2nz_extract_fp8_e5m2)
{
    testTExtractUbToL1Nd2Nz(launchTExtractNd2Nz<35>, 8, 48, 192, 32, 128, 17, 96, 3, 32, false);
}

TEST_F(TEXTRACTTest, nd2nz_extract_fp8_e8m0)
{
    testTExtractUbToL1Nd2Nz(launchTExtractNd2Nz<36>, 8, 48, 192, 32, 128, 17, 96, 3, 32, false);
}

TEST_F(TEXTRACTTest, nd2nz_extract_fp4_e2m1)
{
    testTExtractUbToL1Nd2Nz(launchTExtractNd2Nz<37>, 4, 48, 384, 32, 256, 17, 192, 3, 64, false);
}

TEST_F(TEXTRACTTest, nd2nz_extract_fp4_e1m2)
{
    testTExtractUbToL1Nd2Nz(launchTExtractNd2Nz<38>, 4, 48, 384, 32, 256, 17, 192, 3, 64, false);
}

TEST_F(TEXTRACTTest, nd2nz_extract_static)
{
    testTExtractUbToL1Nd2Nz(launchTExtractNd2Nz<40>, 16, 48, 96, 32, 64, 17, 48, 3, 16, false);
}

TEST_F(TEXTRACTTest, nd2nz_extract_empty_rows)
{
    testTExtractUbToL1Nd2Nz(launchTExtractNd2Nz<41>, 16, 16, 64, 16, 32, 0, 16, 3, 16, false);
}

TEST_F(TEXTRACTTest, nd2nz_extract_empty_cols)
{
    testTExtractUbToL1Nd2Nz(launchTExtractNd2Nz<42>, 16, 16, 64, 16, 32, 7, 0, 3, 16, false);
}

TEST_F(TEXTRACTTest, nd2nz_extract_valid_edge)
{
    testTExtractUbToL1Nd2Nz(launchTExtractNd2Nz<43>, 32, 48, 64, 32, 32, 17, 24, 3, 8, false);
}

TEST_F(TEXTRACTTest, legacy_null_static)
{
    testTExtractUbToL1Nd2Nz(launchTExtractNd2Nz<70>, 16, 16, 64, 16, 64, 2, 32, 1, 16, true);
}

TEST_F(TEXTRACTTest, legacy_null_dynamic)
{
    testTExtractUbToL1Nd2Nz(launchTExtractNd2Nz<71>, 16, 16, 64, 16, 64, 7, 32, 3, 16, true);
}
