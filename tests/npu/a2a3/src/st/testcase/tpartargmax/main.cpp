/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

class TestResource {
private:
    void *device;
    size_t size;
    inline bool allocMem(size_t size)
    {
        if (size == 0 || this->size != 0) {
            return false;
        }
        this->size = size;
        if (this->device == nullptr) {
            aclrtMalloc(&this->device, this->size, ACL_MEM_MALLOC_HUGE_FIRST);
        }
        return true;
    }
public:
    TestResource()
    {
        this->size = 0;
        this->host = nullptr;
        this->device = nullptr;
    }
    ~TestResource()
    {
        this->close()
    }
    void init(size_t size)
    {
        if (!this->allocMem(size)) {
            return;
        }
        void *host = nullptr;
        aclrtMallocHost(&host, this->size);
        memset(host, 0, size);
        aclrtMemcpy(this->device, size, host, size, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtFreeHost(host);
    }
    void init(size_t size, const std::string &fileName)
    {
        if (!this->allocMem(size)) {
            return;
        }
        void *host = nullptr;
        aclrtMallocHost(&host, this->size);
        ReadFile(GetGoldenDir() + fileName, size, host, size);
        aclrtMemcpy(this->device, size, host, tsize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtFreeHost(host);
    }
    void close()
    {
        if (this->device != nullptr) {
            aclrtFree(this->device);
            this->device = nullptr;
        }
        this->size = 0;
    }
    void *getDevice()
    {
        return this->device;
    }
    template<typename T>
    bool checkOutput(const std::string &goldenFileName, const std::string &outputFileName, float eps = 0.0001f)
    {
        size_t size = this->size;
        std::vector<T> golden(size);
        std::vector<T> output(size);
        aclrtMemcpy(outputVal.data(), size, this->device, size, ACL_MEMCPY_DEVICE_TO_HOST);
        ReadFile(GetGoldenDir() + goldenFileName, size, golden.data(), size);
        bool res = ResultCmp<T>(golden, output, eps);
        if (!res) {
            WriteFile(GetGoldenDir() + outputFileName, output.data(), size);
        }
        return res;
    }
};

class TPARTARGMAXTest : public testing::Test {
private:
    aclrtStream stream;
    TestResource dstVal;
    TestResource dstIdx;
    TestResource src0Val;
    TestResource src0Idx;
    TestResource src1Val;
    TestResource src1Idx;

protected:
    std::string GetGoldenDir()
    {
        const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
        const std::string caseName = testInfo->name();
        std::string suiteName = testInfo->test_suite_name();
        std::string fullPath = "../" + suiteName + "." + caseName;
        return fullPath;
    }
    void SetUp() override
    {
        aclInit(nullptr);
        aclrtSetDevice(0);
        aclrtCreateStream(&this->stream);
    }
    void TearDown() override
    {
        aclrtDestroyStream(this->stream);
        aclrtResetDevice(0);
        aclFinalize();
    }
    template <bool isHalf = false, typename TVal, typename TIdx, int dstValH, int dstValW, int dstIdxH, int dstIdxW,
        int src0ValH, int src0ValW, int src0IdxH, int src0IdxW, int src1ValH, int src1ValW, int src1dxH, int src1IdxW,
        int vRows0, int vCols0, int vRows1, int vCols1>
    void Launch()
    {
        dstVal.init(sizeof(TVal) * dstValH * dstValW);
        dstIdx.init(sizeof(TIdx) * dstIdxH * dstIdxW);
        src0Val.init(sizeof(TVal) * src0ValH * src0ValW, "src_val0.bin");
        src0Idx.init(sizeof(TIdx) * src0IdxH * src0IdxW, "src_idx0.bin");
        src1Val.init(sizeof(TVal) * src1ValH * src1ValW, "src_val1.bin");
        src1Idx.init(sizeof(TIdx) * src1IdxH * src1IdxW, "src_idx1.bin");
        if constexpr (isHalf) {
            LaunchTPartArgMaxHalf<TIdx, dstValH, dstValW, dstIdxH, dstIdxW, src0ValH, src0ValW, src0IdxH, src0IdxW,
                src1ValH, src1ValW, src1dxH, src1IdxW, vRows0, vCols0, vRows1, vCols1>
                ((TVal *)dstVal.getDevice(), (TIdx *)dstIdx.getDevice(), (TVal *)src0Val.getDevice(), (TIdx *)src0Idx.getDevice(),
                (TVal *)src1Val.getDevice(), (TIdx *)src1Idx.getDevice(), this->stream);
        } else {
            LaunchTPartArgMax<TVal, TIdx, dstValH, dstValW, dstIdxH, dstIdxW, src0ValH, src0ValW, src0IdxH, src0IdxW,
                src1ValH, src1ValW, src1dxH, src1IdxW, vRows0, vCols0, vRows1, vCols1>
                ((TVal *)dstVal.getDevice(), (TIdx *)dstIdx.getDevice(), (TVal *)src0Val.getDevice(), (TIdx *)src0Idx.getDevice(),
                (TVal *)src1Val.getDevice(), (TIdx *)src1Idx.getDevice(), this->stream);
        }
        aclrtSynchronizeStream(this->stream);
        bool res = dstVal.checkOutput<TVal>("dst_val.bin", "dst_val_out.bin");
        res &= dstIdx.checkOutput<TIdx>("dst_idx.bin", "dst_idx_out.bin");
        dstVal.close();
        dstIdx.close();
        src0Val.close();
        src0Idx.close();
        src1Val.close();
        src1Idx.close();
        EXPECT_TRUE(res);
    }
};

TEST_F(TROWARGMAXTest, case_uint32_float_8x1_8x8_1x8_8x8)
{
    this->Launch<uint32_t, float, 8, 1, 8, 8, 1, 8, 8, 8>();
}
TEST_F(TROWARGMAXTest, case_uint32_float_1024x1_1024x8_1x8_1024x8)
{
    this->Launch<uint32_t, float, 1024, 1, 1024, 8, 1, 8, 1024, 8>();
}
TEST_F(TROWARGMAXTest, case_uint32_float_16x1_13x16_1x8_13x13)
{
    this->Launch<uint32_t, float, 16, 1, 13, 16, 1, 8, 13, 13>();
}
TEST_F(TROWARGMAXTest, case_uint32_float_1024x1_1023x24_1x8_1023x17)
{
    this->Launch<uint32_t, float, 1024, 1, 1023, 24, 1, 8, 1023, 17>();
}
TEST_F(TROWARGMAXTest, case_uint32_float_8x1_8x64_1x8_8x64)
{
    this->Launch<uint32_t, float, 8, 1, 8, 64, 1, 8, 8, 64>();
}
TEST_F(TROWARGMAXTest, case_uint32_float_264x1_260x64_1x8_260x64)
{
    this->Launch<uint32_t, float, 264, 1, 260, 64, 1, 8, 260, 64>();
}
TEST_F(TROWARGMAXTest, case_uint32_float_64x1_32x128_32x24_32x128)
{
    this->Launch<uint32_t, float, 64, 1, 32, 128, 32, 24, 32, 128>();
}
TEST_F(TROWARGMAXTest, case_uint32_float_8x1_3x4096_3x192_3x4095)
{
    this->Launch<uint32_t, float, 8, 1, 3, 4096, 3, 192, 3, 4095>();
}
TEST_F(TROWARGMAXTest, case_uint32_float_8x1_1x16384_1x768_1x16381)
{
    this->Launch<uint32_t, float, 8, 1, 1, 16384, 1, 768, 1, 16381>();
}
TEST_F(TROWARGMAXTest, case_uint32_half_16x1_2x16_2x16_2x16)
{
    this->Launch<uint32_t, aclFloat16, 16, 1, 2, 16, 2, 16, 2, 16, true>();
}
TEST_F(TROWARGMAXTest, case_uint32_half_16x1_13x16_1x16_13x13)
{
    this->Launch<uint32_t, aclFloat16, 16, 1, 13, 16, 1, 16, 13, 13, true>();
}
TEST_F(TROWARGMAXTest, case_uint32_half_272x1_260x64_1x16_260x64)
{
    this->Launch<uint32_t, aclFloat16, 272, 1, 260, 64, 1, 16, 260, 64, true>();
}
TEST_F(TROWARGMAXTest, case_uint32_half_272x1_260x128_1x16_260x128)
{
    this->Launch<uint32_t, aclFloat16, 272, 1, 260, 128, 1, 16, 260, 128, true>();
}
TEST_F(TROWARGMAXTest, case_uint32_half_16x1_3x8192_3x384_3x8191)
{
    this->Launch<uint32_t, aclFloat16, 16, 1, 3, 8192, 3, 384, 3, 8191, true>();
}
TEST_F(TROWARGMAXTest, case_uint32_half_16x1_1x16384_1x768_1x16381)
{
    this->Launch<uint32_t, aclFloat16, 16, 1, 1, 16384, 1, 768, 1, 16381, true>();
}
TEST_F(TROWARGMAXTest, case_uint32_half_16x1_1x32768_1x768_1x32761)
{
    this->Launch<uint32_t, aclFloat16, 16, 1, 1, 32768, 1, 768, 1, 32761, true>();
}
TEST_F(TROWARGMAXTest, case_int32_float_16x1_13x16_1x8_13x13)
{
    this->Launch<int32_t, float, 16, 1, 13, 16, 1, 8, 13, 13>();
}
TEST_F(TROWARGMAXTest, case_int32_half_16x1_13x16_1x16_13x13)
{
    this->Launch<int32_t, aclFloat16, 16, 1, 13, 16, 1, 16, 13, 13, true>();
}
TEST_F(TROWARGMAXTest, case_uint32_float_3x8_3x3480_3x168_3x3473)
{
    this->Launch<uint32_t, float, 3, 8, 3, 3480, 3, 168, 3, 3473>();
}
TEST_F(TROWARGMAXTest, case_uint32_float_260x8_260x64_1x8_260x64)
{
    this->Launch<uint32_t, float, 260, 8, 260, 64, 1, 8, 260, 64>();
}
TEST_F(TROWARGMAXTest, case_uint32_float_1023x8_1023x24_1x8_1023x17)
{
    this->Launch<uint32_t, float, 1023, 8, 1023, 24, 1, 8, 1023, 17>();
}
TEST_F(TROWARGMAXTest, case_uint32_float_2x8_2x16384_2x768_2x16381)
{
    this->Launch<uint32_t, float, 2, 8, 2, 16384, 2, 768, 2, 16381>();
}
TEST_F(TROWARGMAXTest, case_uint32_half_3x16_3x3488_3x768_3x3473)
{
    this->Launch<uint32_t, aclFloat16, 3, 16, 3, 3488, 3, 768, 3, 3473, true>();
}
TEST_F(TROWARGMAXTest, case_uint32_half_260x16_260x64_1x16_260x64)
{
    this->Launch<uint32_t, aclFloat16, 260, 16, 260, 64, 1, 16, 260, 64, true>();
}
TEST_F(TROWARGMAXTest, case_uint32_half_1023x16_1023x32_1x16_1023x17)
{
    this->Launch<uint32_t, aclFloat16, 1023, 16, 1023, 32, 1, 16, 1023, 17, true>();
}
TEST_F(TROWARGMAXTest, case_uint32_half_2x16_2x32768_2x768_2x32761)
{
    this->Launch<uint32_t, aclFloat16, 2, 16, 2, 32768, 2, 768, 2, 32761, true>();
}
TEST_F(TROWARGMAXTest, case_uint32_float_1024x1_1024x1_1023x24_1023x8_1023x17)
{
    this->Launch<uint32_t, float, 1024, 1, 1024, 1, 1023, 24, 1023, 8, 1023, 17>();
}
TEST_F(TROWARGMAXTest, case_uint32_float_264x1_264x1_260x64_260x8_260x64)
{
    this->Launch<uint32_t, float, 264, 1, 264, 1, 260, 64, 260, 8, 260, 64>();
}
TEST_F(TROWARGMAXTest, case_uint32_float_8x1_8x1_3x4096_3x192_3x4095)
{
    this->Launch<uint32_t, float, 8, 1, 8, 1, 3, 4096, 3, 192, 3, 4095>();
}
TEST_F(TROWARGMAXTest, case_uint32_float_8x1_8x1_1x16384_1x768_1x16381)
{
    this->Launch<uint32_t, float, 8, 1, 8, 1, 1, 16384, 1, 768, 1, 16381>();
}
TEST_F(TROWARGMAXTest, case_uint16_half_272x1_272x1_260x64_260x16_260x63)
{
    this->Launch<uint16_t, aclFloat16, 272, 1, 272, 1, 260, 64, 260, 16, 260, 63, true>();
}
TEST_F(TROWARGMAXTest, case_uint16_half_272x1_272x1_260x128_260x16_260x127)
{
    this->Launch<uint16_t, aclFloat16, 272, 1, 272, 1, 260, 128, 260, 16, 260, 127, true>();
}
TEST_F(TROWARGMAXTest, case_uint16_half_16x1_16x1_3x8192_3x384_3x8191)
{
    this->Launch<uint16_t, aclFloat16, 16, 1, 16, 1, 3, 8192, 3, 384, 3, 8191, true>();
}
TEST_F(TROWARGMAXTest, case_uint16_half_16x1_16x1_1x16384_1x768_1x16381)
{
    this->Launch<uint16_t, aclFloat16, 16, 1, 16, 1, 1, 16384, 1, 768, 1, 16381, true>();
}
TEST_F(TROWARGMAXTest, case_uint16_half_16x1_16x1_1x32768_1x768_1x32761)
{
    this->Launch<uint16_t, aclFloat16, 16, 1, 16, 1, 1, 32768, 1, 768, 1, 32761, true>();
}
TEST_F(TROWARGMAXTest, case_uint32_float_777x8_777x8_777x24_777x8_777x17)
{
    this->Launch<uint32_t, float, 777, 8, 777, 8, 777, 24, 777, 8, 777, 17>();
}
TEST_F(TROWARGMAXTest, case_uint32_float_784x1_777x8_777x24_777x8_777x17)
{
    this->Launch<uint32_t, float, 784, 1, 777, 8, 777, 24, 777, 8, 777, 17>();
}
TEST_F(TROWARGMAXTest, case_uint32_float_777x8_784x1_777x24_777x8_777x17)
{
    this->Launch<uint32_t, float, 777, 8, 784, 1, 777, 24, 777, 8, 777, 17>();
}
TEST_F(TROWARGMAXTest, case_uint32_float_3x8_3x8_3x4096_3x192_3x4095)
{
    this->Launch<uint32_t, float, 3, 8, 3, 8, 3, 4096, 3, 192, 3, 4095>();
}
TEST_F(TROWARGMAXTest, case_uint32_float_8x1_3x8_3x4096_3x192_3x4095)
{
    this->Launch<uint32_t, float, 8, 1, 3, 8, 3, 4096, 3, 192, 3, 4095>();
}
TEST_F(TROWARGMAXTest, case_uint32_float_3x8_8x1_3x4096_3x192_3x4095)
{
    this->Launch<uint32_t, float, 3, 8, 8, 1, 3, 4096, 3, 192, 3, 4095>();
}
TEST_F(TROWARGMAXTest, case_uint16_half_777x16_777x16_777x48_777x16_777x43)
{
    this->Launch<uint16_t, aclFloat16, 777, 16, 777, 16, 777, 48, 777, 16, 777, 43, true>();
}
TEST_F(TROWARGMAXTest, case_uint16_half_784x1_777x16_777x48_777x16_777x43)
{
    this->Launch<uint16_t, aclFloat16, 784, 1, 777, 16, 777, 48, 777, 16, 777, 43, true>();
}
TEST_F(TROWARGMAXTest, case_uint16_half_777x16_784x1_777x48_777x16_777x43)
{
    this->Launch<uint16_t, aclFloat16, 777, 16, 784, 1, 777, 48, 777, 16, 777, 43, true>();
}
TEST_F(TROWARGMAXTest, case_uint16_half_3x16_3x16_3x4096_3x192_3x4095)
{
    this->Launch<uint16_t, aclFloat16, 3, 16, 3, 16, 3, 4096, 3, 192, 3, 4095, true>();
}
TEST_F(TROWARGMAXTest, case_uint16_half_16x1_3x16_3x4096_3x192_3x4095)
{
    this->Launch<uint16_t, aclFloat16, 16, 1, 3, 16, 3, 4096, 3, 192, 3, 4095, true>();
}
TEST_F(TROWARGMAXTest, case_uint16_half_3x16_16x1_3x4096_3x192_3x4095)
{
    this->Launch<uint16_t, aclFloat16, 3, 16, 16, 1, 3, 4096, 3, 192, 3, 4095, true>();
}
TEST_F(TROWARGMAXTest, case_uint16_half_1x16_1x16_1x32768_1x768_1x32761)
{
    this->Launch<uint16_t, aclFloat16, 1, 16, 1, 16, 1, 32768, 1, 768, 1, 32761, true>();
}
TEST_F(TROWARGMAXTest, case_uint16_half_16x1_1x16_1x32768_1x768_1x32761)
{
    this->Launch<uint16_t, aclFloat16, 16, 1, 1, 16, 1, 32768, 1, 768, 1, 32761, true>();
}
TEST_F(TROWARGMAXTest, case_uint16_half_1x16_16x1_1x32768_1x768_1x32761)
{
    this->Launch<uint16_t, aclFloat16, 1, 16, 16, 1, 1, 32768, 1, 768, 1, 32761, true>();
}
