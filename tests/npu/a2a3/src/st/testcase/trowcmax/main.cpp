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

template <typename T, int dstTileH, int dstTileW, int srcTileH, int srcTileW, int vRows, int vCols>
void LaunchTRowCMax(T *out, T *src, void *stream);
template <int dstTileH, int dstTileW, int srcTileH, int srcTileW, int vRows, int vCols>
void LaunchTRowCMaxHalf(aclFloat16 *out, aclFloat16 *src, void *stream);

class TROWCMAXTest : public testing::Test {
private:
    aclrtStream stream;
    void *dstHost;
    void *srcHost;
    void *dstDevice;
    void *srcDevice;
    size_t dstFileSize;
    size_t srcFileSize;

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
    template <typename T, int dstTileH, int dstTileW, int srcTileH, int srcTileW>
    void BeforeLaunch()
    {
        this->dstFileSize = sizeof(T) * dstTileH * dstTileW;
        this->srcFileSize = sizeof(T) * srcTileH * srcTileW;

        aclrtMallocHost(&this->dstHost, this->dstFileSize);
        aclrtMallocHost(&this->srcHost, this->srcFileSize);
        memset(this->dstHost, 0, this->dstFileSize);
        ReadFile(GetGoldenDir() + "/input.bin", this->srcFileSize, this->srcHost, this->srcFileSize);

        aclrtMalloc(&this->dstDevice, this->dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&this->srcDevice, this->srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(dstDevice, dstFileSize, dstHost, dstFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    template <typename T>
    bool AfterLaunch()
    {
        aclrtSynchronizeStream(this->stream);
        std::vector<T> golden(this->dstFileSize);
        std::vector<T> devFinal(this->dstFileSize);
        aclrtMemcpy(devFinal.data(), this->dstFileSize, this->dstDevice, this->dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

        aclrtFree(this->dstDevice);
        aclrtFree(this->srcDevice);
        aclrtFreeHost(this->srcHost);
        aclrtFreeHost(this->dstHost);

        ReadFile(GetGoldenDir() + "/golden.bin", this->dstFileSize, golden.data(), this->dstFileSize);
        bool res = ResultCmp<T>(golden, devFinal, 0.0001f);
        if (!res) {
            WriteFile(GetGoldenDir() + "/output.bin", devFinal.data(), this->dstFileSize);
        }
        return res;
    }

    template <typename T, int dstTileH, int dstTileW, int srcTileH, int srcTileW, int vRows, int vCols, bool isHalf = false>
    void Launch()
    {
        this->BeforeLaunch<T, dstTileH, dstTileW, srcTileH, srcTileW>();
        if constexpr (isHalf) {
            LaunchTRowCMaxHalf<dstTileH, dstTileW, srcTileH, srcTileW, vRows, vCols>(
                (T *)this->dstDevice, (T *)this->srcDevice, this->stream);
        } else {
            LaunchTRowCMax<T, dstTileH, dstTileW, srcTileH, srcTileW, vRows, vCols>(
                (T *)this->dstDevice, (T *)this->srcDevice, this->stream);
        }
        bool res = this->AfterLaunch<T>();
        EXPECT_TRUE(res);
    }
};

TEST_F(TROWCMAXTest, case_float_1x8_8x8_8x8)
{
    this->Launch<float, 1, 8, 8, 8, 8, 8>();
}
