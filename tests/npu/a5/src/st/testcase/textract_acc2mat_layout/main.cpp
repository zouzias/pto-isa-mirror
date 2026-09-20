/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <acl/acl.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include "textract_acc2mat_layout_cases.h"

template <int Key>
void launchTExtractLayout(uint8_t*, uint8_t*, uint8_t*, uint8_t*, void*);

class TEXTRACTAcc2MatLayoutTest : public testing::Test {
protected:
    inline static aclrtStream stream = nullptr;
    inline static int device = 0;
    inline static bool initialized = false;
    inline static bool deviceSet = false;
    uint8_t *outDevice = nullptr, *aDevice = nullptr, *bDevice = nullptr, *biasDevice = nullptr;

    static void SetUpTestSuite()
    {
        if (const char* value = std::getenv("PTO_TEST_DEVICE"))
            device = std::atoi(value);
        ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
        initialized = true;
        ASSERT_EQ(aclrtSetDevice(device), ACL_SUCCESS);
        deviceSet = true;
        ASSERT_EQ(aclrtCreateStream(&stream), ACL_SUCCESS);
    }

    void TearDown() override
    {
        if (stream)
            EXPECT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
        for (auto* buffer : {outDevice, aDevice, bDevice, biasDevice}) {
            if (buffer)
                EXPECT_EQ(aclrtFree(buffer), ACL_SUCCESS);
        }
    }

    static void TearDownTestSuite()
    {
        if (stream)
            EXPECT_EQ(aclrtDestroyStream(stream), ACL_SUCCESS);
        if (deviceSet)
            EXPECT_EQ(aclrtResetDevice(device), ACL_SUCCESS);
        if (initialized)
            EXPECT_EQ(aclFinalize(), ACL_SUCCESS);
    }

    static std::vector<uint8_t> read(const std::string& path)
    {
        std::ifstream input(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

    template <int Key>
    void testLayout()
    {
        using C = TExtractLayoutCase<Key>;
        const std::string path = "../TEXTRACTAcc2MatLayoutTest.case" + std::to_string(Key) + "/";
        const auto a = read(path + "a.bin"), b = read(path + "b.bin"), bias = read(path + "bias.bin");
        const auto golden = read(path + "golden.bin");
        constexpr size_t inSize = C::KindValue == 2 ? 1 : 2;
        constexpr size_t outSize = C::KindValue >= 2 ? 4 : 2;
        constexpr size_t bytes = 256 + (C::PhaseValue == 2 ? 2 : 1) * C::RowsValue * C::ColsValue * outSize;
        ASSERT_EQ(a.size(), C::MValue * C::KValue * inSize);
        ASSERT_EQ(b.size(), C::KValue * C::NValue * inSize);
        ASSERT_EQ(bias.size(), C::NValue * sizeof(int32_t));
        ASSERT_EQ(golden.size(), bytes);
        std::vector<uint8_t> output(bytes, 0xA5);
        ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&aDevice), a.size(), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&bDevice), b.size(), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(
            aclrtMalloc(reinterpret_cast<void**>(&biasDevice), bias.size(), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&outDevice), bytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(aclrtMemcpy(aDevice, a.size(), a.data(), a.size(), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
        ASSERT_EQ(aclrtMemcpy(bDevice, b.size(), b.data(), b.size(), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
        ASSERT_EQ(
            aclrtMemcpy(biasDevice, bias.size(), bias.data(), bias.size(), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
        ASSERT_EQ(aclrtMemcpy(outDevice, bytes, output.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
        launchTExtractLayout<Key>(outDevice, aDevice, bDevice, biasDevice, stream);
        ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
        ASSERT_EQ(aclrtMemcpy(output.data(), bytes, outDevice, bytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
        size_t mismatches = 0;
        for (size_t i = 0; i < bytes; ++i) {
            if (output[i] != golden[i]) {
                if (mismatches < 8)
                    ADD_FAILURE() << "byte " << i << ": actual=" << int(output[i]) << " expected=" << int(golden[i]);
                ++mismatches;
            }
        }
        EXPECT_EQ(mismatches, 0u) << "Includes physical padding and 128-byte guards before/after L1 output.";
    }
};

TEST_F(TEXTRACTAcc2MatLayoutTest, case1) { testLayout<1>(); }
TEST_F(TEXTRACTAcc2MatLayoutTest, case2) { testLayout<2>(); }
TEST_F(TEXTRACTAcc2MatLayoutTest, case3) { testLayout<3>(); }
TEST_F(TEXTRACTAcc2MatLayoutTest, case4) { testLayout<4>(); }
TEST_F(TEXTRACTAcc2MatLayoutTest, case5) { testLayout<5>(); }
TEST_F(TEXTRACTAcc2MatLayoutTest, case6) { testLayout<6>(); }
TEST_F(TEXTRACTAcc2MatLayoutTest, case7) { testLayout<7>(); }
TEST_F(TEXTRACTAcc2MatLayoutTest, case8) { testLayout<8>(); }
TEST_F(TEXTRACTAcc2MatLayoutTest, case9) { testLayout<9>(); }
TEST_F(TEXTRACTAcc2MatLayoutTest, case10) { testLayout<10>(); }
TEST_F(TEXTRACTAcc2MatLayoutTest, case11) { testLayout<11>(); }
TEST_F(TEXTRACTAcc2MatLayoutTest, case12) { testLayout<12>(); }
TEST_F(TEXTRACTAcc2MatLayoutTest, case13) { testLayout<13>(); }
TEST_F(TEXTRACTAcc2MatLayoutTest, case14) { testLayout<14>(); }
TEST_F(TEXTRACTAcc2MatLayoutTest, case15) { testLayout<15>(); }
TEST_F(TEXTRACTAcc2MatLayoutTest, case16) { testLayout<16>(); }
TEST_F(TEXTRACTAcc2MatLayoutTest, case17) { testLayout<17>(); }
TEST_F(TEXTRACTAcc2MatLayoutTest, case18) { testLayout<18>(); }
TEST_F(TEXTRACTAcc2MatLayoutTest, case19) { testLayout<19>(); }
TEST_F(TEXTRACTAcc2MatLayoutTest, case20) { testLayout<20>(); }
TEST_F(TEXTRACTAcc2MatLayoutTest, case21) { testLayout<21>(); }
TEST_F(TEXTRACTAcc2MatLayoutTest, case22) { testLayout<22>(); }
TEST_F(TEXTRACTAcc2MatLayoutTest, case23) { testLayout<23>(); }
TEST_F(TEXTRACTAcc2MatLayoutTest, case24) { testLayout<24>(); }
TEST_F(TEXTRACTAcc2MatLayoutTest, case25) { testLayout<25>(); }
TEST_F(TEXTRACTAcc2MatLayoutTest, case26) { testLayout<26>(); }
