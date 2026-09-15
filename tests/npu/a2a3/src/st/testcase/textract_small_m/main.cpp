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
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "test_common.h"
#include "textract_small_m_cases.h"

using namespace PtoTestCommon;

template <int Key>
void launchTEXTRACTSmallM(uint8_t* out, uint8_t* a, uint8_t* b, void* stream);

class TEXTRACTSmallMTest : public testing::Test {
protected:
    bool initialized = false;
    bool deviceSet = false;
    aclrtStream stream = nullptr;
    uint8_t* aDevice = nullptr;
    uint8_t* bDevice = nullptr;
    uint8_t* outDevice = nullptr;

    void SetUp() override
    {
        ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
        initialized = true;
        ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
        deviceSet = true;
        ASSERT_EQ(aclrtCreateStream(&stream), ACL_SUCCESS);
    }

    void TearDown() override
    {
        if (stream) {
            EXPECT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
        }
        for (auto* buffer : {aDevice, bDevice, outDevice}) {
            if (buffer) {
                EXPECT_EQ(aclrtFree(buffer), ACL_SUCCESS);
            }
        }
        if (stream) {
            EXPECT_EQ(aclrtDestroyStream(stream), ACL_SUCCESS);
        }
        if (deviceSet) {
            EXPECT_EQ(aclrtResetDevice(0), ACL_SUCCESS);
        }
        if (initialized) {
            EXPECT_EQ(aclFinalize(), ACL_SUCCESS);
        }
    }

    template <int Key>
    void testSmallM();
};

template <int Key>
void TEXTRACTSmallMTest::testSmallM()
{
    using C = TExtractSmallMCase<Key>;
    const size_t aBytes = C::Rows * (C::Cols + C::Col) * sizeof(uint16_t);
    const size_t bBytes = C::Cols * 32 * sizeof(uint16_t);
    const size_t outBytes = C::Groups * C::M * 32 * sizeof(float);
    std::vector<uint16_t> a(aBytes / 2), b(bBytes / 2);
    std::vector<float> out(outBytes / 4), golden(outBytes / 4);
    std::string path = "../TEXTRACTSmallMTest.case" + std::to_string(Key) + "/";
    size_t fileBytes = 0;
    ASSERT_TRUE(ReadFile(path + "a.bin", fileBytes, a.data(), aBytes));
    ASSERT_EQ(fileBytes, aBytes);
    ASSERT_TRUE(ReadFile(path + "b.bin", fileBytes, b.data(), bBytes));
    ASSERT_EQ(fileBytes, bBytes);
    ASSERT_TRUE(ReadFile(path + "golden.bin", fileBytes, golden.data(), outBytes));
    ASSERT_EQ(fileBytes, outBytes);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&aDevice), aBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&bDevice), bBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&outDevice), outBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(aDevice, aBytes, a.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(bDevice, bBytes, b.data(), bBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    launchTEXTRACTSmallM<Key>(outDevice, aDevice, bDevice, stream);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(out.data(), outBytes, outDevice, outBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_FLOAT_EQ(out[i], golden[i]) << "element " << i;
    }
}

TEST_F(TEXTRACTSmallMTest, case1) { testSmallM<1>(); }
TEST_F(TEXTRACTSmallMTest, case2) { testSmallM<2>(); }
TEST_F(TEXTRACTSmallMTest, case3) { testSmallM<3>(); }
TEST_F(TEXTRACTSmallMTest, case4) { testSmallM<4>(); }
TEST_F(TEXTRACTSmallMTest, case5) { testSmallM<5>(); }
TEST_F(TEXTRACTSmallMTest, case6) { testSmallM<6>(); }
TEST_F(TEXTRACTSmallMTest, case7) { testSmallM<7>(); }
TEST_F(TEXTRACTSmallMTest, case8) { testSmallM<8>(); }
TEST_F(TEXTRACTSmallMTest, case9) { testSmallM<9>(); }
TEST_F(TEXTRACTSmallMTest, case10) { testSmallM<10>(); }
TEST_F(TEXTRACTSmallMTest, case11) { testSmallM<11>(); }
TEST_F(TEXTRACTSmallMTest, case12) { testSmallM<12>(); }
TEST_F(TEXTRACTSmallMTest, case13) { testSmallM<13>(); }
