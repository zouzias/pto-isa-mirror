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
#include <vector>
#include <algorithm>
#include <gtest/gtest.h>
#include "acl/acl.h"

template <bool Mat, int Format, uint64_t Gap, bool Dynamic, bool Outer, int Axis = 0, int AddressBits = 32>
void LaunchLargeStride(uint8_t* out, uint8_t* src, void* stream);

struct LargeStrideResources {
    bool initialized = false;
    bool deviceSet = false;
    aclrtStream stream = nullptr;
    uint8_t* input = nullptr;
    uint8_t* output = nullptr;
    void* reserved = nullptr;
    std::vector<void*> mappings;
    std::vector<aclrtDrvMemHandle> handles;

    void AllocateSparseInput(size_t size, const std::vector<size_t>& offsets)
    {
        aclrtPhysicalMemProp prop{};
        prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
        prop.memAttr = ACL_HBM_MEM_HUGE;
        prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = 0;
        size_t granularity = 0;
        ASSERT_EQ(
            aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, &granularity), ACL_SUCCESS);
        ASSERT_GT(granularity, 0U);
        size_t reservationSize = (size + granularity - 1) / granularity * granularity;
        ASSERT_EQ(aclrtReserveMemAddress(&reserved, reservationSize, 0, nullptr, 0), ACL_SUCCESS);
        input = static_cast<uint8_t*>(reserved);
        std::vector<size_t> pages;
        for (size_t offset : offsets) {
            pages.push_back(offset / granularity * granularity);
        }
        std::sort(pages.begin(), pages.end());
        pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
        for (size_t page : pages) {
            aclrtDrvMemHandle handle = nullptr;
            ASSERT_EQ(aclrtMallocPhysical(&handle, granularity, &prop, 0), ACL_SUCCESS);
            handles.push_back(handle);
            ASSERT_EQ(aclrtMapMem(input + page, granularity, 0, handle, 0), ACL_SUCCESS);
            mappings.push_back(input + page);
            ASSERT_EQ(aclrtMemset(input + page, granularity, 0x7e, granularity), ACL_SUCCESS);
        }
    }

    ~LargeStrideResources()
    {
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        for (void* mapping : mappings) {
            aclrtUnmapMem(mapping);
        }
        for (aclrtDrvMemHandle handle : handles) {
            aclrtFreePhysical(handle);
        }
        if (reserved != nullptr) {
            aclrtReleaseMemAddress(reserved);
        } else if (input != nullptr) {
            aclrtFree(input);
        }
        if (output != nullptr) {
            aclrtFree(output);
        }
        if (deviceSet) {
            aclrtResetDevice(0);
        }
        if (initialized) {
            aclFinalize();
        }
    }
};

template <bool Mat, int Format, uint64_t Gap, bool Dynamic, bool Outer, int Axis = 0, int AddressBits = 32>
void TestLargeStride()
{
    constexpr size_t bytes = Format == 2 ? 512 : 32;
    constexpr uint64_t burst = Gap + bytes;
    constexpr uint64_t batch = Outer ? (1ULL << AddressBits) + 4096 : 2 * bytes;
    constexpr size_t inputBytes = batch + burst + bytes;
    constexpr size_t outputBytes = 8 * bytes;
    LargeStrideResources resources;
    ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
    resources.initialized = true;
    ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
    resources.deviceSet = true;
    ASSERT_EQ(aclrtCreateStream(&resources.stream), ACL_SUCCESS);
    if constexpr (AddressBits == 40) {
        // Map each accessed block, including a possible page crossing.
        std::vector<size_t> offsets{0, 4 * bytes - 1};
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                offsets.push_back(i * batch + j * burst);
                offsets.push_back(i * batch + j * burst + bytes - 1);
            }
        }
        ASSERT_NO_FATAL_FAILURE(resources.AllocateSparseInput(inputBytes, offsets));
    } else {
        ASSERT_EQ(
            aclrtMalloc(reinterpret_cast<void**>(&resources.input), inputBytes, ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_SUCCESS);
    }
    ASSERT_EQ(
        aclrtMalloc(reinterpret_cast<void**>(&resources.output), outputBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    // Initialize only the accessed blocks and the low addresses reached by truncated strides.
    ASSERT_EQ(aclrtMemset(resources.input, inputBytes, 0x7e, 4 * bytes), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemset(resources.output, outputBytes, 0x7e, outputBytes), ACL_SUCCESS);
    std::vector<uint8_t> golden(4 * bytes);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            const size_t offset = i * batch + j * burst;
            auto* block = golden.data() + (i * 2 + j) * bytes;
            for (size_t k = 0; k < bytes; ++k) {
                block[k] = 1 + (i * 67 + j * 31 + k) % 113;
            }
            ASSERT_EQ(
                aclrtMemcpy(resources.input + offset, inputBytes - offset, block, bytes, ACL_MEMCPY_HOST_TO_DEVICE),
                ACL_SUCCESS);
        }
    }
    LaunchLargeStride<Mat, Format, Gap, Dynamic, Outer, Axis, AddressBits>(
        resources.output, resources.input, resources.stream);
    ASSERT_EQ(aclrtSynchronizeStreamWithTimeout(resources.stream, 30000), ACL_SUCCESS);
    std::vector<uint8_t> actual(outputBytes);
    ASSERT_EQ(
        aclrtMemcpy(actual.data(), outputBytes, resources.output, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    for (size_t block = 0; block < 4; ++block) {
        for (size_t k = 0; k < (Format == 3 ? 31 : bytes); ++k) {
            ASSERT_EQ(actual[block * (Format == 4 ? 1 : 2) * bytes + k], golden[block * bytes + k])
                << "block=" << block << " byte=" << k;
        }
        if constexpr (Format == 3) {
            ASSERT_EQ(actual[block * 2 * bytes + 31], 0);
        }
    }
}

#ifndef PTO_TLOAD_VEC_ONLY
TEST(TLoadLargeStrideTest, MatNdGapMax) { TestLargeStride<true, 0, 65535ULL * 32, false, false>(); }

TEST(TLoadLargeStrideTest, MatNdGapOverflow) { TestLargeStride<true, 0, 65536ULL * 32, true, false>(); }

TEST(TLoadLargeStrideTest, MatDnGapMax) { TestLargeStride<true, 1, 65535ULL * 32, false, false>(); }

TEST(TLoadLargeStrideTest, MatDnGapOverflow) { TestLargeStride<true, 1, 65536ULL * 32, true, false>(); }

TEST(TLoadLargeStrideTest, MatNzGapMax) { TestLargeStride<true, 2, 65535ULL * 32, false, false>(); }

TEST(TLoadLargeStrideTest, MatNzGapOverflow) { TestLargeStride<true, 2, 65536ULL * 32, true, false>(); }

TEST(TLoadLargeStrideTest, MatNdStride64) { TestLargeStride<true, 0, 1ULL << 32, false, false>(); }

TEST(TLoadLargeStrideTest, MatDnStride64) { TestLargeStride<true, 1, 1ULL << 32, true, false>(); }

TEST(TLoadLargeStrideTest, MatNzStride64) { TestLargeStride<true, 2, 1ULL << 32, false, false>(); }

TEST(TLoadLargeStrideTest, MatNdOuterStride64) { TestLargeStride<true, 0, 64, true, true>(); }

TEST(TLoadLargeStrideTest, MatDnOuterStride64) { TestLargeStride<true, 1, 64, false, true>(); }

TEST(TLoadLargeStrideTest, MatNzOuterStride64) { TestLargeStride<true, 2, 1024, true, true>(); }

#endif
TEST(TLoadLargeStrideTest, VecNdGapMax) { TestLargeStride<false, 0, (1ULL << 32) - 32, false, false>(); }

TEST(TLoadLargeStrideTest, VecNdGapOverflow) { TestLargeStride<false, 0, 1ULL << 32, true, false>(); }

TEST(TLoadLargeStrideTest, VecDnGapMax) { TestLargeStride<false, 1, (1ULL << 32) - 32, false, false>(); }

TEST(TLoadLargeStrideTest, VecDnGapOverflow) { TestLargeStride<false, 1, 1ULL << 32, true, false>(); }

TEST(TLoadLargeStrideTest, VecNzGapMax) { TestLargeStride<false, 2, (1ULL << 32) - 32, false, false>(); }

TEST(TLoadLargeStrideTest, VecNzGapOverflow) { TestLargeStride<false, 2, 1ULL << 32, true, false>(); }

TEST(TLoadLargeStrideTest, VecNdOuterStride64) { TestLargeStride<false, 0, 64, true, true>(); }

TEST(TLoadLargeStrideTest, VecDnOuterStride64) { TestLargeStride<false, 1, 64, false, true>(); }

TEST(TLoadLargeStrideTest, VecNzOuterStride64) { TestLargeStride<false, 2, 1024, true, true>(); }

TEST(TLoadLargeStrideTest, VecNdPartialGapSmall) { TestLargeStride<false, 3, 64, false, false>(); }

TEST(TLoadLargeStrideTest, VecNdPartialGapOverflow) { TestLargeStride<false, 3, 1ULL << 32, true, false>(); }

#ifndef PTO_TLOAD_VEC_ONLY
template <typename T, bool DN, int64_t RowStride, bool Dynamic>
void LaunchConversionStride(uint8_t* out, uint8_t* src, void* stream);

template <typename T, bool DN, int64_t RowStride, bool Dynamic>
void TestConversionStride()
{
    LargeStrideResources resources;
    ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
    resources.initialized = true;
    ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
    resources.deviceSet = true;
    ASSERT_EQ(aclrtCreateStream(&resources.stream), ACL_SUCCESS);
    std::vector<T> input(RowStride + 16, T(-7));
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 16; ++col) {
            input[row * RowStride + col] = T(row * 37 + col + 1);
        }
    }
    size_t inputBytes = input.size() * sizeof(T);
    constexpr size_t outputBytes = 16 * 16 * sizeof(T);
    ASSERT_EQ(
        aclrtMalloc(reinterpret_cast<void**>(&resources.input), inputBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(
        aclrtMalloc(reinterpret_cast<void**>(&resources.output), outputBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(
        aclrtMemcpy(resources.input, inputBytes, input.data(), inputBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    LaunchConversionStride<T, DN, RowStride, Dynamic>(resources.output, resources.input, resources.stream);
    ASSERT_EQ(aclrtSynchronizeStreamWithTimeout(resources.stream, 30000), ACL_SUCCESS);
    std::vector<T> actual(16 * 16);
    ASSERT_EQ(
        aclrtMemcpy(actual.data(), outputBytes, resources.output, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 16; ++col) {
            ASSERT_EQ(actual[(col / 8) * 16 * 8 + row * 8 + col % 8], input[row * RowStride + col])
                << "row=" << row << " col=" << col;
        }
    }
}
TEST(TLoadConversionStrideTest, Nd2Nz_float_32767_Static) { TestConversionStride<float, false, 32767, false>(); }

TEST(TLoadConversionStrideTest, Nd2Nz_float_32768_Static) { TestConversionStride<float, false, 32768, false>(); }

TEST(TLoadConversionStrideTest, Nd2Nz_float_32768_Dynamic) { TestConversionStride<float, false, 32768, true>(); }

TEST(TLoadConversionStrideTest, Nd2Nz_float_65535_Dynamic) { TestConversionStride<float, false, 65535, true>(); }

TEST(TLoadConversionStrideTest, Dn2Zn_float_32767_Static) { TestConversionStride<float, true, 32767, false>(); }

TEST(TLoadConversionStrideTest, Dn2Zn_float_32768_Static) { TestConversionStride<float, true, 32768, false>(); }

TEST(TLoadConversionStrideTest, Dn2Zn_float_32768_Dynamic) { TestConversionStride<float, true, 32768, true>(); }

TEST(TLoadConversionStrideTest, Dn2Zn_float_65535_Dynamic) { TestConversionStride<float, true, 65535, true>(); }

TEST(TLoadConversionStrideTest, Nd2Nz_int32_t_32767_Static) { TestConversionStride<int32_t, false, 32767, false>(); }

TEST(TLoadConversionStrideTest, Nd2Nz_int32_t_32768_Static) { TestConversionStride<int32_t, false, 32768, false>(); }

TEST(TLoadConversionStrideTest, Nd2Nz_int32_t_32768_Dynamic) { TestConversionStride<int32_t, false, 32768, true>(); }

TEST(TLoadConversionStrideTest, Nd2Nz_int32_t_65535_Dynamic) { TestConversionStride<int32_t, false, 65535, true>(); }

TEST(TLoadConversionStrideTest, Dn2Zn_int32_t_32767_Static) { TestConversionStride<int32_t, true, 32767, false>(); }

TEST(TLoadConversionStrideTest, Dn2Zn_int32_t_32768_Static) { TestConversionStride<int32_t, true, 32768, false>(); }

TEST(TLoadConversionStrideTest, Dn2Zn_int32_t_32768_Dynamic) { TestConversionStride<int32_t, true, 32768, true>(); }

TEST(TLoadConversionStrideTest, Dn2Zn_int32_t_65535_Dynamic) { TestConversionStride<int32_t, true, 65535, true>(); }

TEST(TLoadConversionStrideTest, Nd2Nz_uint32_t_32767_Static) { TestConversionStride<uint32_t, false, 32767, false>(); }

TEST(TLoadConversionStrideTest, Nd2Nz_uint32_t_32768_Static) { TestConversionStride<uint32_t, false, 32768, false>(); }

TEST(TLoadConversionStrideTest, Nd2Nz_uint32_t_32768_Dynamic) { TestConversionStride<uint32_t, false, 32768, true>(); }

TEST(TLoadConversionStrideTest, Nd2Nz_uint32_t_65535_Dynamic) { TestConversionStride<uint32_t, false, 65535, true>(); }

TEST(TLoadConversionStrideTest, Dn2Zn_uint32_t_32767_Static) { TestConversionStride<uint32_t, true, 32767, false>(); }

TEST(TLoadConversionStrideTest, Dn2Zn_uint32_t_32768_Static) { TestConversionStride<uint32_t, true, 32768, false>(); }

TEST(TLoadConversionStrideTest, Dn2Zn_uint32_t_32768_Dynamic) { TestConversionStride<uint32_t, true, 32768, true>(); }

TEST(TLoadConversionStrideTest, Dn2Zn_uint32_t_65535_Dynamic) { TestConversionStride<uint32_t, true, 65535, true>(); }

#endif

#ifdef PTO_TLOAD_WIDE_STRIDE
#ifndef PTO_TLOAD_VEC_ONLY
TEST(TLoadWideStrideTest, MatNdBurst40_Below)
{
    TestLargeStride<true, 0, ((1ULL << 40) - 32) - 32, false, false, 0, 40>();
}

TEST(TLoadWideStrideTest, MatNdBurst40_At) { TestLargeStride<true, 0, ((1ULL << 40) + 0) - 32, true, false, 0, 40>(); }

TEST(TLoadWideStrideTest, MatNdBurst40_Above)
{
    TestLargeStride<true, 0, ((1ULL << 40) + 64) - 32, false, false, 0, 40>();
}

TEST(TLoadWideStrideTest, MatDnBurst40_Below)
{
    TestLargeStride<true, 1, ((1ULL << 40) - 32) - 32, false, false, 0, 40>();
}

TEST(TLoadWideStrideTest, MatDnBurst40_At) { TestLargeStride<true, 1, ((1ULL << 40) + 0) - 32, true, false, 0, 40>(); }

TEST(TLoadWideStrideTest, MatDnBurst40_Above)
{
    TestLargeStride<true, 1, ((1ULL << 40) + 64) - 32, false, false, 0, 40>();
}

TEST(TLoadWideStrideTest, MatNzBurst40_Below)
{
    TestLargeStride<true, 2, ((1ULL << 40) - 32) - 512, false, false, 0, 40>();
}

TEST(TLoadWideStrideTest, MatNzBurst40_At) { TestLargeStride<true, 2, ((1ULL << 40) + 0) - 512, true, false, 0, 40>(); }

TEST(TLoadWideStrideTest, MatNzBurst40_Above)
{
    TestLargeStride<true, 2, ((1ULL << 40) + 64) - 512, false, false, 0, 40>();
}

TEST(TLoadWideStrideTest, MatNdLoop140) { TestLargeStride<true, 0, 64, true, true, 1, 40>(); }

TEST(TLoadWideStrideTest, MatNdLoop240) { TestLargeStride<true, 0, 64, true, true, 2, 40>(); }

TEST(TLoadWideStrideTest, MatDnLoop140) { TestLargeStride<true, 1, 64, true, true, 1, 40>(); }

TEST(TLoadWideStrideTest, MatDnLoop240) { TestLargeStride<true, 1, 64, true, true, 2, 40>(); }

#endif
TEST(TLoadWideStrideTest, VecNdBurst40_Below)
{
    TestLargeStride<false, 0, ((1ULL << 40) - 32) - 32, false, false, 0, 40>();
}

TEST(TLoadWideStrideTest, VecNdBurst40_At) { TestLargeStride<false, 0, ((1ULL << 40) + 0) - 32, true, false, 0, 40>(); }

TEST(TLoadWideStrideTest, VecNdBurst40_Above)
{
    TestLargeStride<false, 0, ((1ULL << 40) + 64) - 32, false, false, 0, 40>();
}

TEST(TLoadWideStrideTest, VecDnBurst40_Below)
{
    TestLargeStride<false, 1, ((1ULL << 40) - 32) - 32, false, false, 0, 40>();
}

TEST(TLoadWideStrideTest, VecDnBurst40_At) { TestLargeStride<false, 1, ((1ULL << 40) + 0) - 32, true, false, 0, 40>(); }

TEST(TLoadWideStrideTest, VecDnBurst40_Above)
{
    TestLargeStride<false, 1, ((1ULL << 40) + 64) - 32, false, false, 0, 40>();
}

TEST(TLoadWideStrideTest, VecNzBurst40_Below)
{
    TestLargeStride<false, 2, ((1ULL << 40) - 32) - 512, false, false, 0, 40>();
}

TEST(TLoadWideStrideTest, VecNzBurst40_At)
{
    TestLargeStride<false, 2, ((1ULL << 40) + 0) - 512, true, false, 0, 40>();
}

TEST(TLoadWideStrideTest, VecNzBurst40_Above)
{
    TestLargeStride<false, 2, ((1ULL << 40) + 64) - 512, false, false, 0, 40>();
}

TEST(TLoadWideStrideTest, VecNdLoop140) { TestLargeStride<false, 0, 64, true, true, 1, 40>(); }

TEST(TLoadWideStrideTest, VecNdLoop240) { TestLargeStride<false, 0, 64, true, true, 2, 40>(); }

TEST(TLoadWideStrideTest, VecDnLoop140) { TestLargeStride<false, 1, 64, true, true, 1, 40>(); }

TEST(TLoadWideStrideTest, VecDnLoop240) { TestLargeStride<false, 1, 64, true, true, 2, 40>(); }

#endif
#ifdef PTO_TLOAD_CONV_WIDE
TEST(TLoadWideStrideTest, MatConvLoop40) { TestLargeStride<true, 4, 64, true, true, 0, 40>(); }

#endif

#ifdef PTO_TLOAD_VECTOR_LENGTH
template <typename T, bool DN, int Length, bool Dynamic>
void LaunchVectorLength(uint8_t* out, uint8_t* src, void* stream);

template <typename T, bool DN, int Length, bool Dynamic, int SourceOffset>
void TestVectorLength()
{
    constexpr size_t alignedBytes = (Length * sizeof(T) + 31) / 32 * 32;
    constexpr size_t outputBytes = alignedBytes + 32;
    constexpr size_t inputBytes = outputBytes + SourceOffset * sizeof(T);
    LargeStrideResources resources;
    ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
    resources.initialized = true;
    ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
    resources.deviceSet = true;
    ASSERT_EQ(aclrtCreateStream(&resources.stream), ACL_SUCCESS);
    std::vector<T> input(inputBytes / sizeof(T), T(99));
    for (size_t i = 0; i < Length; ++i) {
        if constexpr (sizeof(T) == 8) {
            input[SourceOffset + i] = T((1ULL << 40) + i * 37 + 1);
        } else {
            input[SourceOffset + i] = T(i % 127 + 1);
        }
    }
    ASSERT_EQ(
        aclrtMalloc(reinterpret_cast<void**>(&resources.input), inputBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(
        aclrtMalloc(reinterpret_cast<void**>(&resources.output), outputBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(
        aclrtMemcpy(resources.input, inputBytes, input.data(), inputBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemset(resources.output, outputBytes, 0x5a, outputBytes), ACL_SUCCESS);
    LaunchVectorLength<T, DN, Length, Dynamic>(
        resources.output, resources.input + SourceOffset * sizeof(T), resources.stream);
    ASSERT_EQ(aclrtSynchronizeStreamWithTimeout(resources.stream, 30000), ACL_SUCCESS);
    std::vector<T> actual(outputBytes / sizeof(T));
    ASSERT_EQ(
        aclrtMemcpy(actual.data(), outputBytes, resources.output, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    for (size_t i = 0; i < Length; ++i) {
        ASSERT_EQ(actual[i], input[SourceOffset + i]) << "element=" << i;
    }
    auto bytes = reinterpret_cast<const uint8_t*>(actual.data());
    for (size_t i = Length * sizeof(T); i < alignedBytes; ++i) {
        ASSERT_EQ(bytes[i], 0) << "padding byte=" << i;
    }
    for (size_t i = alignedBytes; i < outputBytes; ++i) {
        ASSERT_EQ(bytes[i], 0x5a) << "guard byte=" << i;
    }
}
TEST(TLoadVectorLengthTest, Nd_uint8_t_65535_Static) { TestVectorLength<uint8_t, false, 65535, false, 0>(); }

TEST(TLoadVectorLengthTest, Nd_uint8_t_65536_Static) { TestVectorLength<uint8_t, false, 65536, false, 0>(); }

TEST(TLoadVectorLengthTest, Nd_uint8_t_65536_Dynamic) { TestVectorLength<uint8_t, false, 65536, true, 0>(); }

TEST(TLoadVectorLengthTest, Nd_uint8_t_65537_Dynamic) { TestVectorLength<uint8_t, false, 65537, true, 0>(); }

TEST(TLoadVectorLengthTest, Nd_uint16_t_65535_Static) { TestVectorLength<uint16_t, false, 65535, false, 0>(); }

TEST(TLoadVectorLengthTest, Nd_uint16_t_65536_Static) { TestVectorLength<uint16_t, false, 65536, false, 0>(); }

TEST(TLoadVectorLengthTest, Nd_uint16_t_65536_Dynamic) { TestVectorLength<uint16_t, false, 65536, true, 0>(); }

TEST(TLoadVectorLengthTest, Nd_uint16_t_65537_Dynamic) { TestVectorLength<uint16_t, false, 65537, true, 0>(); }

TEST(TLoadVectorLengthTest, Nd_float_65535_Static) { TestVectorLength<float, false, 65535, false, 0>(); }

TEST(TLoadVectorLengthTest, Nd_float_65536_Static) { TestVectorLength<float, false, 65536, false, 0>(); }

TEST(TLoadVectorLengthTest, Nd_float_65536_Dynamic) { TestVectorLength<float, false, 65536, true, 0>(); }

TEST(TLoadVectorLengthTest, Nd_float_65537_Dynamic) { TestVectorLength<float, false, 65537, true, 0>(); }

TEST(TLoadVectorLengthTest, Nd_int64_t_16383_Static) { TestVectorLength<int64_t, false, 16383, false, 0>(); }

TEST(TLoadVectorLengthTest, Nd_int64_t_16384_Static) { TestVectorLength<int64_t, false, 16384, false, 0>(); }

TEST(TLoadVectorLengthTest, Nd_int64_t_16384_Dynamic) { TestVectorLength<int64_t, false, 16384, true, 0>(); }

TEST(TLoadVectorLengthTest, Nd_int64_t_16385_Dynamic) { TestVectorLength<int64_t, false, 16385, true, 0>(); }

TEST(TLoadVectorLengthTest, Nd_int64_t_32767_Static) { TestVectorLength<int64_t, false, 32767, false, 0>(); }

TEST(TLoadVectorLengthTest, Nd_int64_t_32768_Static) { TestVectorLength<int64_t, false, 32768, false, 0>(); }

TEST(TLoadVectorLengthTest, Nd_int64_t_32768_Dynamic) { TestVectorLength<int64_t, false, 32768, true, 0>(); }

TEST(TLoadVectorLengthTest, Nd_int64_t_32769_Dynamic) { TestVectorLength<int64_t, false, 32769, true, 0>(); }

TEST(TLoadVectorLengthTest, Nd_uint64_t_32768_Dynamic) { TestVectorLength<uint64_t, false, 32768, true, 0>(); }

TEST(TLoadVectorLengthTest, Dn_uint8_t_31_Static) { TestVectorLength<uint8_t, true, 31, false, 0>(); }

TEST(TLoadVectorLengthTest, Dn_uint8_t_32_Dynamic) { TestVectorLength<uint8_t, true, 32, true, 0>(); }

TEST(TLoadVectorLengthTest, Dn_uint8_t_33_Dynamic) { TestVectorLength<uint8_t, true, 33, true, 0>(); }

TEST(TLoadVectorLengthTest, Dn_uint16_t_15_Static) { TestVectorLength<uint16_t, true, 15, false, 0>(); }

TEST(TLoadVectorLengthTest, Dn_uint16_t_16_Dynamic) { TestVectorLength<uint16_t, true, 16, true, 0>(); }

TEST(TLoadVectorLengthTest, Dn_uint16_t_17_Dynamic) { TestVectorLength<uint16_t, true, 17, true, 0>(); }

TEST(TLoadVectorLengthTest, Dn_float_7_Static) { TestVectorLength<float, true, 7, false, 0>(); }

TEST(TLoadVectorLengthTest, Dn_float_8_Dynamic) { TestVectorLength<float, true, 8, true, 0>(); }

TEST(TLoadVectorLengthTest, Dn_float_9_Dynamic) { TestVectorLength<float, true, 9, true, 0>(); }

TEST(TLoadVectorLengthTest, Dn_int64_t_3_Static) { TestVectorLength<int64_t, true, 3, false, 0>(); }

TEST(TLoadVectorLengthTest, Dn_int64_t_4_Dynamic) { TestVectorLength<int64_t, true, 4, true, 0>(); }

TEST(TLoadVectorLengthTest, Dn_int64_t_5_Dynamic) { TestVectorLength<int64_t, true, 5, true, 0>(); }

TEST(TLoadVectorLengthTest, Dn_int64_t_16383_Static) { TestVectorLength<int64_t, true, 16383, false, 0>(); }

TEST(TLoadVectorLengthTest, Dn_int64_t_16384_Static) { TestVectorLength<int64_t, true, 16384, false, 0>(); }

TEST(TLoadVectorLengthTest, Dn_int64_t_16384_Dynamic) { TestVectorLength<int64_t, true, 16384, true, 0>(); }

TEST(TLoadVectorLengthTest, Nd_int64_t_64_Static) { TestVectorLength<int64_t, false, 64, false, 0>(); }

TEST(TLoadVectorLengthTest, Nd_int64_t_3_Dynamic) { TestVectorLength<int64_t, false, 3, true, 0>(); }

TEST(TLoadVectorLengthTest, Nd_float_64_Static) { TestVectorLength<float, false, 64, false, 0>(); }

TEST(TLoadVectorLengthTest, Nd_float_7_Dynamic) { TestVectorLength<float, false, 7, true, 0>(); }

TEST(TLoadVectorLengthTest, Nd_uint8_t_65_Unaligned) { TestVectorLength<uint8_t, false, 65, true, 1>(); }

TEST(TLoadVectorLengthTest, Dn_uint8_t_65_Unaligned) { TestVectorLength<uint8_t, true, 65, true, 1>(); }

TEST(TLoadVectorLengthTest, Nd_uint16_t_65_Unaligned) { TestVectorLength<uint16_t, false, 65, true, 1>(); }

TEST(TLoadVectorLengthTest, Dn_uint16_t_65_Unaligned) { TestVectorLength<uint16_t, true, 65, true, 1>(); }

TEST(TLoadVectorLengthTest, Nd_float_65_Unaligned) { TestVectorLength<float, false, 65, true, 1>(); }

TEST(TLoadVectorLengthTest, Dn_float_65_Unaligned) { TestVectorLength<float, true, 65, true, 1>(); }

TEST(TLoadVectorLengthTest, Nd_int64_t_65_Unaligned) { TestVectorLength<int64_t, false, 65, true, 1>(); }

TEST(TLoadVectorLengthTest, Dn_int64_t_65_Unaligned) { TestVectorLength<int64_t, true, 65, true, 1>(); }
#endif
