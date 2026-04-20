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

template <typename TVal, typename TIdx, int dstValH, int dstValW, int dstIdxH, int dstIdxW,
        int src0ValH, int src0ValW, int src0IdxH, int src0IdxW, int src1ValH, int src1ValW, int src1IdxH, int src1IdxW,
        int vRows0, int vCols0, int vRows1, int vCols1>
void LaunchTPartArgMax(TVal *outVal, TIdx *outIdx, TVal *src0Val, TIdx *src0Idx, TVal *src1Val, TIdx *src1Idx, void *stream);

template <typename TIdx, int dstValH, int dstValW, int dstIdxH, int dstIdxW,
        int src0ValH, int src0ValW, int src0IdxH, int src0IdxW, int src1ValH, int src1ValW, int src1IdxH, int src1IdxW,
        int vRows0, int vCols0, int vRows1, int vCols1>
void LaunchTPartArgMaxHalf(aclFloat16 *outVal, TIdx *outIdx, aclFloat16 *src0Val, TIdx *src0Idx, aclFloat16 *src1Val, TIdx *src1Idx, void *stream);

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
        this->device = nullptr;
    }
    ~TestResource()
    {
        this->close();
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
        ReadFile(fileName, size, host, size);
        aclrtMemcpy(this->device, size, host, size, ACL_MEMCPY_HOST_TO_DEVICE);
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
        aclrtMemcpy(output.data(), size, this->device, size, ACL_MEMCPY_DEVICE_TO_HOST);
        ReadFile(goldenFileName, size, golden.data(), size);
        bool res = ResultCmp<T>(golden, output, eps);
        if (!res) {
            WriteFile(outputFileName, output.data(), size);
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
    template <typename TVal, typename TIdx, int dstValH, int dstValW, int dstIdxH, int dstIdxW,
        int src0ValH, int src0ValW, int src0IdxH, int src0IdxW, int src1ValH, int src1ValW, int src1IdxH, int src1IdxW,
        int vRows0, int vCols0, int vRows1, int vCols1, bool isHalf = false>
    void Launch()
    {
        dstVal.init(sizeof(TVal) * dstValH * dstValW);
        dstIdx.init(sizeof(TIdx) * dstIdxH * dstIdxW);
        src0Val.init(sizeof(TVal) * src0ValH * src0ValW, GetGoldenDir() + "/src_val0.bin");
        src0Idx.init(sizeof(TIdx) * src0IdxH * src0IdxW, GetGoldenDir() + "/src_idx0.bin");
        src1Val.init(sizeof(TVal) * src1ValH * src1ValW, GetGoldenDir() + "/src_val1.bin");
        src1Idx.init(sizeof(TIdx) * src1IdxH * src1IdxW, GetGoldenDir() + "/src_idx1.bin");
        if constexpr (isHalf) {
            LaunchTPartArgMaxHalf<TIdx, dstValH, dstValW, dstIdxH, dstIdxW, src0ValH, src0ValW, src0IdxH, src0IdxW,
                src1ValH, src1ValW, src1IdxH, src1IdxW, vRows0, vCols0, vRows1, vCols1>
                ((TVal *)dstVal.getDevice(), (TIdx *)dstIdx.getDevice(), (TVal *)src0Val.getDevice(), (TIdx *)src0Idx.getDevice(),
                (TVal *)src1Val.getDevice(), (TIdx *)src1Idx.getDevice(), this->stream);
        } else {
            LaunchTPartArgMax<TVal, TIdx, dstValH, dstValW, dstIdxH, dstIdxW, src0ValH, src0ValW, src0IdxH, src0IdxW,
                src1ValH, src1ValW, src1IdxH, src1IdxW, vRows0, vCols0, vRows1, vCols1>
                ((TVal *)dstVal.getDevice(), (TIdx *)dstIdx.getDevice(), (TVal *)src0Val.getDevice(), (TIdx *)src0Idx.getDevice(),
                (TVal *)src1Val.getDevice(), (TIdx *)src1Idx.getDevice(), this->stream);
        }
        aclrtSynchronizeStream(this->stream);
        bool res = dstVal.checkOutput<TVal>(GetGoldenDir() + "/dst_val.bin", GetGoldenDir() + "/dst_val_out.bin");
        res &= dstIdx.checkOutput<TIdx>(GetGoldenDir() + "/dst_idx.bin", GetGoldenDir() + "/dst_idx_out.bin");
        dstVal.close();
        dstIdx.close();
        src0Val.close();
        src0Idx.close();
        src1Val.close();
        src1Idx.close();
        EXPECT_TRUE(res);
    }
};

TEST_F(TPARTARGMAXTest, case_float_uint32_same_size)
{
    this->Launch<float, uint32_t, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8>();
}
TEST_F(TPARTARGMAXTest, case_float_uint32_same_size_32k)
{
    this->Launch<float, uint32_t, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64>();
}
TEST_F(TPARTARGMAXTest, case_float_uint32_row_diff_0)
{
    this->Launch<float, uint32_t, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 111, 64, 128, 64>();
}
TEST_F(TPARTARGMAXTest, case_float_uint32_row_diff_1)
{
    this->Launch<float, uint32_t, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 111, 64, 128, 64>();
}
