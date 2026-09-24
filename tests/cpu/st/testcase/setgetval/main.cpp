/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

#include "test_common.h"

using namespace std;
using namespace PtoTestCommon;

#if GTEST_HAS_DEATH_TEST && (defined(__unix__) || defined(__APPLE__))
namespace {

constexpr const char* CAPACITY_ENV_VARS[] = {
    "PTO_CPU_SIM_UB_BYTES", "PTO_CPU_SIM_L1_BYTES", "PTO_CPU_SIM_L0A_BYTES", "PTO_CPU_SIM_L0B_BYTES",
    "PTO_CPU_SIM_L0C_BYTES"};

using UbBoundaryTile = pto::Tile<pto::TileType::Vec, std::uint8_t, 1, 32>;

void expectUbCapacity(std::size_t capacityBytes)
{
    auto& memory = pto::NPUMemoryModel::Instance();
    const auto base = reinterpret_cast<std::uintptr_t>(memory.GetUBBase());
    EXPECT_TRUE(memory.ContainsAddress(base + capacityBytes - 1));
    EXPECT_FALSE(memory.ContainsAddress(base + capacityBytes));

    UbBoundaryTile tile;
    pto::TASSIGN(tile, capacityBytes - UbBoundaryTile::GetSizeInBytes());
    tile.data()[0] = 17;
    tile.data()[UbBoundaryTile::Numel - 1] = 29;
    EXPECT_EQ(static_cast<std::uint8_t>(memory.GetUBBase()[capacityBytes - UbBoundaryTile::GetSizeInBytes()]), 17);
    EXPECT_EQ(static_cast<std::uint8_t>(memory.GetUBBase()[capacityBytes - 1]), 29);
}

class MemoryCapacityDeathTest : public testing::TestWithParam<pto::NPUArch> {};

TEST_P(MemoryCapacityDeathTest, architecture_capacity)
{
    const auto arch = GetParam();
    const std::size_t capacityBytes = arch == pto::NPUArch::A5 ? 256 * 1024 : 192 * 1024;
    ASSERT_EXIT(
        {
            ASSERT_EQ(unsetenv("PTO_CPU_SIM_UB_BYTES"), 0);
            pto::NPU_MEMORY_INIT(pto::NPUArch::A5);
            expectUbCapacity(256 * 1024);
            pto::NPU_MEMORY_INIT(arch);
            expectUbCapacity(capacityBytes);
            std::_Exit(testing::Test::HasFailure() ? 1 : 0);
        },
        testing::ExitedWithCode(0), "");
}

TEST_P(MemoryCapacityDeathTest, explicit_capacity_override)
{
    const auto arch = GetParam();
    ASSERT_EXIT(
        {
            ASSERT_EQ(setenv("PTO_CPU_SIM_UB_BYTES", "131072", 1), 0);
            pto::NPU_MEMORY_INIT(arch);
            expectUbCapacity(128 * 1024);
            ASSERT_EQ(setenv("PTO_CPU_SIM_UB_BYTES", "524288", 1), 0);
            pto::NPU_MEMORY_INIT(arch);
            expectUbCapacity(512 * 1024);
            std::_Exit(testing::Test::HasFailure() ? 1 : 0);
        },
        testing::ExitedWithCode(0), "");
}

TEST_P(MemoryCapacityDeathTest, invalid_capacity_overrides_use_defaults)
{
    const auto arch = GetParam();
    for (const char* value :
         {"", "0", "0000", "abc", "123abc", "-1", "-18446744073709420544", "18446744073709551616", "+131072", " 131072",
          "131072 ", "131072\n", "1 2", "0x20000", "1.5", "1e5"}) {
        SCOPED_TRACE(value);
        EXPECT_EXIT(
            {
                for (const auto* name : CAPACITY_ENV_VARS) {
                    ASSERT_EQ(unsetenv(name), 0);
                }
                pto::NPU_MEMORY_INIT(arch);
                auto& memory = pto::NPUMemoryModel::Instance();
                const auto defaultSizes = std::to_array(memory.GetSizes());
                for (const auto* name : CAPACITY_ENV_VARS) {
                    ASSERT_EQ(setenv(name, value, 1), 0);
                }
                pto::NPU_MEMORY_INIT(arch);
                EXPECT_EQ(std::to_array(memory.GetSizes()), defaultSizes);
                std::_Exit(testing::Test::HasFailure() ? 1 : 0);
            },
            testing::ExitedWithCode(0), "");
    }
}

TEST_P(MemoryCapacityDeathTest, decimal_capacity_overrides)
{
    const auto arch = GetParam();
    constexpr std::size_t capacityBytes = 32 * 1024;
    for (const char* value : {"32768", "00032768"}) {
        SCOPED_TRACE(value);
        EXPECT_EXIT(
            {
                for (const auto* name : CAPACITY_ENV_VARS) {
                    ASSERT_EQ(setenv(name, value, 1), 0);
                }
                pto::NPU_MEMORY_INIT(arch);
                auto& memory = pto::NPUMemoryModel::Instance();
                for (const auto* base :
                     {memory.GetUBBase(), memory.GetL1Base(), memory.GetL0ABase(), memory.GetL0BBase(),
                      memory.GetL0CBase()}) {
                    const auto address = reinterpret_cast<std::uintptr_t>(base);
                    EXPECT_TRUE(memory.ContainsAddress(address + capacityBytes - 1));
                    EXPECT_FALSE(memory.ContainsAddress(address + capacityBytes));
                }
                std::_Exit(testing::Test::HasFailure() ? 1 : 0);
            },
            testing::ExitedWithCode(0), "");
    }
}

TEST_P(MemoryCapacityDeathTest, rejects_tile_past_architecture_capacity)
{
    const auto arch = GetParam();
    const std::size_t capacityBytes = arch == pto::NPUArch::A5 ? 256 * 1024 : 192 * 1024;
    ASSERT_DEATH(
        {
            ASSERT_EQ(unsetenv("PTO_CPU_SIM_UB_BYTES"), 0);
            pto::NPU_MEMORY_INIT(arch);
            UbBoundaryTile tile;
            pto::TASSIGN(tile, capacityBytes);
        },
        "Tile assignment exceeds memory region capacity");
}

TEST_P(MemoryCapacityDeathTest, rejects_tile_past_override_capacity)
{
    const auto arch = GetParam();
    ASSERT_DEATH(
        {
            ASSERT_EQ(setenv("PTO_CPU_SIM_UB_BYTES", "524288", 1), 0);
            pto::NPU_MEMORY_INIT(arch);
            UbBoundaryTile tile;
            pto::TASSIGN(tile, 512 * 1024);
        },
        "Tile assignment exceeds memory region capacity");
}

template <typename TileData>
void expectRegionBounds(pto::NPUArch arch, std::size_t capacityBytes, const char* capacityEnvVar)
{
    SCOPED_TRACE(static_cast<int>(TileData::Loc));
    constexpr std::size_t tileBytes = [] {
        if constexpr (pto::is_tile_data_v<TileData>) {
            return TileData::GetSizeInBytes();
        } else {
            return static_cast<std::size_t>(TileData::bufferSize);
        }
    }();
    ASSERT_EXIT(
        {
            if (capacityEnvVar != nullptr) {
                ASSERT_EQ(unsetenv(capacityEnvVar), 0);
            }
            pto::NPU_MEMORY_INIT(arch);
            TileData tile;
            pto::TASSIGN(tile, capacityBytes - tileBytes);
            TileData alias;
            pto::TASSIGN(alias, reinterpret_cast<std::uintptr_t>(tile.data()));
            EXPECT_EQ(alias.data(), tile.data());
            auto* aliasBytes = reinterpret_cast<unsigned char*>(alias.data());
            aliasBytes[0] = 17;
            aliasBytes[tileBytes - 1] = 29;
            const auto* originalBytes = reinterpret_cast<const unsigned char*>(tile.data());
            EXPECT_EQ(originalBytes[0], 17);
            EXPECT_EQ(originalBytes[tileBytes - 1], 29);
            std::_Exit(testing::Test::HasFailure() ? 1 : 0);
        },
        testing::ExitedWithCode(0), "");

    // Cover a tile crossing the end, an offset past the end, and addition overflow.
    for (const std::size_t offset :
         {capacityBytes - tileBytes + sizeof(typename TileData::DType), capacityBytes + 32,
          std::numeric_limits<std::size_t>::max() - tileBytes + 1}) {
        SCOPED_TRACE(offset);
        EXPECT_DEATH(
            {
                if (capacityEnvVar != nullptr) {
                    ASSERT_EQ(unsetenv(capacityEnvVar), 0);
                }
                pto::NPU_MEMORY_INIT(arch);
                TileData tile;
                pto::TASSIGN(tile, offset);
            },
            "Tile assignment exceeds memory region capacity");
    }

    EXPECT_DEATH(
        {
            if (capacityEnvVar != nullptr) {
                ASSERT_EQ(unsetenv(capacityEnvVar), 0);
            }
            pto::NPU_MEMORY_INIT(arch);
            TileData tile;
            pto::TASSIGN(tile, capacityBytes - tileBytes);
            TileData alias;
            pto::TASSIGN(alias, reinterpret_cast<std::uintptr_t>(tile.data() + 1));
        },
        "Tile assignment exceeds memory region capacity");
}

TEST_P(MemoryCapacityDeathTest, region_boundaries)
{
    const auto arch = GetParam();
    expectRegionBounds<UbBoundaryTile>(
        arch, arch == pto::NPUArch::A5 ? 256 * 1024 : 192 * 1024, "PTO_CPU_SIM_UB_BYTES");
    expectRegionBounds<pto::Tile<pto::TileType::Mat, float, 16, 16>>(arch, 512 * 1024, "PTO_CPU_SIM_L1_BYTES");
    expectRegionBounds<pto::TileLeft<float, 16, 16>>(arch, 64 * 1024, "PTO_CPU_SIM_L0A_BYTES");
    expectRegionBounds<pto::TileRight<float, 16, 16>>(arch, 64 * 1024, "PTO_CPU_SIM_L0B_BYTES");
    expectRegionBounds<pto::TileAcc<float, 16, 16>>(
        arch, arch == pto::NPUArch::A5 ? 256 * 1024 : 128 * 1024, "PTO_CPU_SIM_L0C_BYTES");
    expectRegionBounds<pto::TileLeftScale<std::uint8_t, 16, 32>>(arch, 4 * 1024, nullptr);
    expectRegionBounds<pto::TileRightScale<std::uint8_t, 16, 32>>(arch, 4 * 1024, nullptr);
}

TEST_P(MemoryCapacityDeathTest, packed_tile_boundaries)
{
    const auto arch = GetParam();
    const std::size_t capacityBytes = arch == pto::NPUArch::A5 ? 256 * 1024 : 192 * 1024;
    expectRegionBounds<pto::Tile<pto::TileType::Vec, float4_e2m1x2_t, 1, 64>>(
        arch, capacityBytes, "PTO_CPU_SIM_UB_BYTES");
    expectRegionBounds<pto::Tile<pto::TileType::Vec, float4_e1m2x2_t, 1, 64>>(
        arch, capacityBytes, "PTO_CPU_SIM_UB_BYTES");
}

TEST_P(MemoryCapacityDeathTest, conv_tile_boundaries)
{
    using ConvTile =
        pto::ConvTile<pto::TileType::Mat, float, 1024, pto::Layout::NCHW, pto::ConvTileShape<1, 1, 16, 16>>;
    expectRegionBounds<ConvTile>(GetParam(), 512 * 1024, "PTO_CPU_SIM_L1_BYTES");
}

INSTANTIATE_TEST_SUITE_P(
    Architectures, MemoryCapacityDeathTest, testing::Values(pto::NPUArch::A2A3, pto::NPUArch::A5),
    [](const testing::TestParamInfo<pto::NPUArch>& info) { return info.param == pto::NPUArch::A5 ? "A5" : "A2A3"; });

} // namespace
#endif

class SETGETVALTest : public testing::Test {
public:
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
template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchSetGetVal(T* src0, void* stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void test_setgetval()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t srcByteSize = kGRows_ * kGCols_ * sizeof(T);
    T* srcHost;
    T* srcDevice;

    aclrtMallocHost((void**)(&srcHost), srcByteSize);

    aclrtMalloc((void**)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    LaunchSetGetVal<T, kGRows_, kGCols_, kTRows_, kTCols_>(srcDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(srcHost, srcByteSize, srcDevice, srcByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(srcDevice);
    T value_4 = srcHost[4];
    T value_5 = srcHost[5];
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    bool res = false;
    if ((value_4 - 12.34 < 0.00001) && value_5 - 12.34 < 0.00001) {
        res = true;
    }
    EXPECT_TRUE(res);
}

TEST_F(SETGETVALTest, case1) { test_setgetval<float, 32, 32, 32, 32>(); }

template <typename TileData>
void VerifyKAlignedDefaultFalse()
{
    alignas(TileData) unsigned char zeroStorage[sizeof(TileData)] = {};
    TileData* zeroTile = new (zeroStorage) TileData;
    unsigned char beforeSet[sizeof(TileData)];
    std::memcpy(beforeSet, zeroStorage, sizeof(TileData));
    zeroTile->SetKAligned(true);

    int kAlignedOffset = -1;
    for (size_t i = 0; i < sizeof(TileData); ++i) {
        if (beforeSet[i] == 0 && zeroStorage[i] == 1) {
            kAlignedOffset = static_cast<int>(i);
            break;
        }
    }
    zeroTile->~TileData();
    ASSERT_GE(kAlignedOffset, 0);

    alignas(TileData) unsigned char filledStorage[sizeof(TileData)];
    std::memset(filledStorage, 0xff, sizeof(filledStorage));
    TileData* filledTile = new (filledStorage) TileData;
    EXPECT_EQ(filledStorage[kAlignedOffset], 0);
    filledTile->SetKAligned(true);
    EXPECT_EQ(static_cast<int>(filledTile->GetKAligned()), 1);
    filledTile->~TileData();
}

TEST_F(SETGETVALTest, k_aligned_default_false)
{
    using StaticTile = pto::TileLeft<float, 1, 128, 1, 128>;
    VerifyKAlignedDefaultFalse<StaticTile>();

    using DynamicTile = pto::Tile<pto::TileType::Vec, float, 32, 32, pto::BLayout::RowMajor, -1, -1>;
    alignas(DynamicTile) unsigned char zeroStorage[sizeof(DynamicTile)] = {};
    DynamicTile* zeroTile = new (zeroStorage) DynamicTile(16, 16);
    unsigned char beforeSet[sizeof(DynamicTile)];
    std::memcpy(beforeSet, zeroStorage, sizeof(DynamicTile));
    zeroTile->SetKAligned(true);

    int kAlignedOffset = -1;
    for (size_t i = 0; i < sizeof(DynamicTile); ++i) {
        if (beforeSet[i] == 0 && zeroStorage[i] == 1) {
            kAlignedOffset = static_cast<int>(i);
            break;
        }
    }
    zeroTile->~DynamicTile();
    ASSERT_GE(kAlignedOffset, 0);

    alignas(DynamicTile) unsigned char filledStorage[sizeof(DynamicTile)];
    std::memset(filledStorage, 0xff, sizeof(filledStorage));
    DynamicTile* filledTile = new (filledStorage) DynamicTile(16, 16);
    EXPECT_EQ(filledStorage[kAlignedOffset], 0);
    filledTile->SetKAligned(true);
    EXPECT_EQ(static_cast<int>(filledTile->GetKAligned()), 1);
    filledTile->~DynamicTile();
}
