/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <pto/pto-inst.hpp>
#include "test_common.h"
#include <gtest/gtest.h>
#include <fstream>

using namespace std;
using namespace pto;
using namespace PtoTestCommon;

template <int32_t testKey>
void launchTVALIDSHAPE(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);

class TValidShapeTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// =============================================================================
// Test 1: Static validShape API — verify GetValidShape vs GetShape
// =============================================================================
TEST_F(TValidShapeTest, case_static_valid_api)
{
    using PhysShape = ConvTileShape<16, 4, 8, 8>;
    using ValidShape = ConvTileValidShape<10, 3, 5, 8>;
    using CT = ConvTile<TileType::Mat, half, 4096, Layout::NC1HWC0, PhysShape, ValidShape>;
    CT tile;

    EXPECT_EQ(tile.GetValidShape(0), 10);
    EXPECT_EQ(tile.GetShape(0), 16);

    EXPECT_EQ(tile.GetValidShape(1), 3);
    EXPECT_EQ(tile.GetShape(1), 4);

    EXPECT_EQ(tile.GetValidShape(2), 5);
    EXPECT_EQ(tile.GetShape(2), 8);

    EXPECT_EQ(tile.GetValidShape(3), 8);
    EXPECT_EQ(tile.GetShape(3), 8);
}

// =============================================================================
// Test 2: Dynamic validShape API — verify SetAllValidShape + SetValidShape
// =============================================================================
TEST_F(TValidShapeTest, case_dynamic_valid_api)
{
    using PhysShape = ConvTileShape<DYNAMIC, 4, DYNAMIC, 8>;
    using ValidShape = ConvTileValidShape<DYNAMIC, 4, DYNAMIC, 8>;
    using CT = ConvTile<TileType::Mat, half, 4096, Layout::NC1HWC0, PhysShape, ValidShape>;

    int64_t physN = 16;
    int64_t physH = 8;
    CT tile(physN, physH);

    EXPECT_EQ(tile.GetShape(0), 16);
    EXPECT_EQ(tile.GetShape(2), 8);

    EXPECT_EQ(tile.GetValidShape(0), 16);
    EXPECT_EQ(tile.GetValidShape(2), 8);

    int64_t validN = 10;
    int64_t validH = 5;
    tile.SetAllValidShape(validN, validH);
    EXPECT_EQ(tile.GetValidShape(0), 10);
    EXPECT_EQ(tile.GetValidShape(2), 5);

    EXPECT_EQ(tile.GetValidShape(1), 4);
    EXPECT_EQ(tile.GetValidShape(3), 8);

    tile.SetValidShape(0, int64_t(8));
    EXPECT_EQ(tile.GetValidShape(0), 8);
}

// =============================================================================
// Test 3: Default validShape = physical (backward compatibility)
// =============================================================================
TEST_F(TValidShapeTest, case_default_valid_eq_physical)
{
    using PhysShape = ConvTileShape<16, 4, 8, 8>;
    using CT = ConvTile<TileType::Mat, half, 4096, Layout::NC1HWC0, PhysShape>;
    CT tile;

    EXPECT_EQ(tile.GetValidShape(0), tile.GetShape(0));
    EXPECT_EQ(tile.GetValidShape(1), tile.GetShape(1));
    EXPECT_EQ(tile.GetValidShape(2), tile.GetShape(2));
    EXPECT_EQ(tile.GetValidShape(3), tile.GetShape(3));
}

// =============================================================================
// Test 4: TLOAD with validShape < physical (data-driven)
// Physical: N=4, C1=2, H=4, W=4, C0=16 (half)
// Valid:    N=2, C1=1, H=2, W=4
// Only valid portion is loaded; invalid positions remain zero.
// =============================================================================

inline size_t GetFileSize(const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    return in.is_open() ? static_cast<size_t>(in.tellg()) : 0;
}

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

TEST_F(TValidShapeTest, case_tload_nc1hwc0_valid)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    string path = GetGoldenDir();
    size_t in_byteSize = GetFileSize(path + "/input.bin");
    size_t gold_byteSize = GetFileSize(path + "/golden.bin");

    half *srcHost, *goldHost, *dstHost;
    void *srcDevice, *dstDevice;

    aclrtMallocHost((void**)&srcHost, in_byteSize);
    aclrtMallocHost((void**)&goldHost, gold_byteSize);
    aclrtMallocHost((void**)&dstHost, gold_byteSize);
    aclrtMalloc((void**)&srcDevice, in_byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&dstDevice, gold_byteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(path + "/input.bin", in_byteSize, srcHost, in_byteSize);
    ReadFile(path + "/golden.bin", gold_byteSize, goldHost, gold_byteSize);
    std::fill(dstHost, dstHost + (gold_byteSize / sizeof(*dstHost)), 0);

    aclrtMemcpy(srcDevice, in_byteSize, srcHost, in_byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dstDevice, gold_byteSize, dstHost, gold_byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launchTVALIDSHAPE<1>((uint8_t*)dstDevice, (uint8_t*)srcDevice, nullptr, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(dstHost, gold_byteSize, dstDevice, gold_byteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    int elements = gold_byteSize / sizeof(half);
    bool ret = ResultCmp(vector<half>(goldHost, goldHost + elements), vector<half>(dstHost, dstHost + elements), 0);

    aclrtFreeHost(srcHost);
    aclrtFreeHost(goldHost);
    aclrtFreeHost(dstHost);
    aclrtFree(srcDevice);
    aclrtFree(dstDevice);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    EXPECT_TRUE(ret);
}
