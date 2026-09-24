/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <algorithm>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include "acl/acl.h"

template <bool Mat, int Format, uint64_t Gap, bool Dynamic, bool Outer, int Axis = 0, int AddressBits = 32>
void launchLargeStride(uint8_t* out, uint8_t* src, void* stream);

struct LargeStrideResources {
    bool initialized = false;
    bool deviceSet = false;
    aclrtStream stream = nullptr;
    uint8_t* input = nullptr;
    uint8_t* output = nullptr;
    void* reserved = nullptr;
    std::vector<void*> mappings;
    std::vector<aclrtDrvMemHandle> handles;

    void allocateSparseInput(size_t size, const std::vector<size_t>& offsets)
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
void testLargeStride()
{
    constexpr size_t BURST_BYTES = Format == 2 ? 512 : 32;
    constexpr uint64_t BURST_STRIDE_BYTES = Gap + BURST_BYTES;
    constexpr uint64_t BATCH_STRIDE_BYTES = Outer ? (1ULL << AddressBits) + 4096 : 2 * BURST_BYTES;
    constexpr size_t INPUT_BYTES = BATCH_STRIDE_BYTES + BURST_STRIDE_BYTES + BURST_BYTES;
    constexpr size_t OUTPUT_BYTES = 8 * BURST_BYTES;
    LargeStrideResources resources;
    ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
    resources.initialized = true;
    ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
    resources.deviceSet = true;
    ASSERT_EQ(aclrtCreateStream(&resources.stream), ACL_SUCCESS);
    if constexpr (AddressBits == 40) {
        // Map each accessed block, including a possible page crossing.
        std::vector<size_t> offsets{0, 4 * BURST_BYTES - 1};
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                offsets.push_back(i * BATCH_STRIDE_BYTES + j * BURST_STRIDE_BYTES);
                offsets.push_back(i * BATCH_STRIDE_BYTES + j * BURST_STRIDE_BYTES + BURST_BYTES - 1);
            }
        }
        ASSERT_NO_FATAL_FAILURE(resources.allocateSparseInput(INPUT_BYTES, offsets));
    } else {
        ASSERT_EQ(
            aclrtMalloc(reinterpret_cast<void**>(&resources.input), INPUT_BYTES, ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_SUCCESS);
    }
    ASSERT_EQ(
        aclrtMalloc(reinterpret_cast<void**>(&resources.output), OUTPUT_BYTES, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    // Initialize only the accessed blocks and the low addresses reached by truncated strides.
    ASSERT_EQ(aclrtMemset(resources.input, INPUT_BYTES, 0x7e, 4 * BURST_BYTES), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemset(resources.output, OUTPUT_BYTES, 0x7e, OUTPUT_BYTES), ACL_SUCCESS);
    std::vector<uint8_t> golden(4 * BURST_BYTES);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            const size_t offset = i * BATCH_STRIDE_BYTES + j * BURST_STRIDE_BYTES;
            auto* block = golden.data() + (i * 2 + j) * BURST_BYTES;
            for (size_t k = 0; k < BURST_BYTES; ++k) {
                block[k] = 1 + (i * 67 + j * 31 + k) % 113;
            }
            ASSERT_EQ(
                aclrtMemcpy(
                    resources.input + offset, INPUT_BYTES - offset, block, BURST_BYTES, ACL_MEMCPY_HOST_TO_DEVICE),
                ACL_SUCCESS);
        }
    }
    launchLargeStride<Mat, Format, Gap, Dynamic, Outer, Axis, AddressBits>(
        resources.output, resources.input, resources.stream);
    ASSERT_EQ(aclrtSynchronizeStreamWithTimeout(resources.stream, 30000), ACL_SUCCESS);
    std::vector<uint8_t> actual(OUTPUT_BYTES);
    ASSERT_EQ(
        aclrtMemcpy(actual.data(), OUTPUT_BYTES, resources.output, OUTPUT_BYTES, ACL_MEMCPY_DEVICE_TO_HOST),
        ACL_SUCCESS);
    for (size_t block = 0; block < 4; ++block) {
        for (size_t k = 0; k < (Format == 3 ? 31 : BURST_BYTES); ++k) {
            ASSERT_EQ(actual[block * (Format == 4 ? 1 : 2) * BURST_BYTES + k], golden[block * BURST_BYTES + k])
                << "block=" << block << " byte=" << k;
        }
        if constexpr (Format == 3) {
            ASSERT_EQ(actual[block * 2 * BURST_BYTES + 31], 0);
        }
    }
}

#ifndef PTO_TLOAD_VEC_ONLY
TEST(TLoadLargeStrideTest, MatNdGapMax) { testLargeStride<true, 0, 65535ULL * 32, false, false>(); }

TEST(TLoadLargeStrideTest, MatNdGapOverflow) { testLargeStride<true, 0, 65536ULL * 32, true, false>(); }

TEST(TLoadLargeStrideTest, MatDnGapMax) { testLargeStride<true, 1, 65535ULL * 32, false, false>(); }

TEST(TLoadLargeStrideTest, MatDnGapOverflow) { testLargeStride<true, 1, 65536ULL * 32, true, false>(); }

TEST(TLoadLargeStrideTest, MatNzGapMax) { testLargeStride<true, 2, 65535ULL * 32, false, false>(); }

TEST(TLoadLargeStrideTest, MatNzGapOverflow) { testLargeStride<true, 2, 65536ULL * 32, true, false>(); }

TEST(TLoadLargeStrideTest, MatNdStride64) { testLargeStride<true, 0, 1ULL << 32, false, false>(); }

TEST(TLoadLargeStrideTest, MatDnStride64) { testLargeStride<true, 1, 1ULL << 32, true, false>(); }

TEST(TLoadLargeStrideTest, MatNzStride64) { testLargeStride<true, 2, 1ULL << 32, false, false>(); }

TEST(TLoadLargeStrideTest, MatNdOuterStride64) { testLargeStride<true, 0, 64, true, true>(); }

TEST(TLoadLargeStrideTest, MatDnOuterStride64) { testLargeStride<true, 1, 64, false, true>(); }

TEST(TLoadLargeStrideTest, MatNzOuterStride64) { testLargeStride<true, 2, 1024, true, true>(); }

#endif
TEST(TLoadLargeStrideTest, VecNdGapMax) { testLargeStride<false, 0, (1ULL << 32) - 32, false, false>(); }

TEST(TLoadLargeStrideTest, VecNdGapOverflow) { testLargeStride<false, 0, 1ULL << 32, true, false>(); }

TEST(TLoadLargeStrideTest, VecDnGapMax) { testLargeStride<false, 1, (1ULL << 32) - 32, false, false>(); }

TEST(TLoadLargeStrideTest, VecDnGapOverflow) { testLargeStride<false, 1, 1ULL << 32, true, false>(); }

TEST(TLoadLargeStrideTest, VecNzGapMax) { testLargeStride<false, 2, (1ULL << 32) - 32, false, false>(); }

TEST(TLoadLargeStrideTest, VecNzGapOverflow) { testLargeStride<false, 2, 1ULL << 32, true, false>(); }

TEST(TLoadLargeStrideTest, VecNdOuterStride64) { testLargeStride<false, 0, 64, true, true>(); }

TEST(TLoadLargeStrideTest, VecDnOuterStride64) { testLargeStride<false, 1, 64, false, true>(); }

TEST(TLoadLargeStrideTest, VecNzOuterStride64) { testLargeStride<false, 2, 1024, true, true>(); }

TEST(TLoadLargeStrideTest, VecNdPartialGapSmall) { testLargeStride<false, 3, 64, false, false>(); }

TEST(TLoadLargeStrideTest, VecNdPartialGapOverflow) { testLargeStride<false, 3, 1ULL << 32, true, false>(); }

#ifndef PTO_TLOAD_VEC_ONLY
template <typename T, bool DN, int64_t RowStride, bool Dynamic>
void launchConversionStride(uint8_t* out, uint8_t* src, void* stream);

template <typename T, bool DN, int64_t RowStride, bool Dynamic>
void testConversionStride()
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
    constexpr size_t OUTPUT_BYTES = 16 * 16 * sizeof(T);
    ASSERT_EQ(
        aclrtMalloc(reinterpret_cast<void**>(&resources.input), inputBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(
        aclrtMalloc(reinterpret_cast<void**>(&resources.output), OUTPUT_BYTES, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(
        aclrtMemcpy(resources.input, inputBytes, input.data(), inputBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    launchConversionStride<T, DN, RowStride, Dynamic>(resources.output, resources.input, resources.stream);
    ASSERT_EQ(aclrtSynchronizeStreamWithTimeout(resources.stream, 30000), ACL_SUCCESS);
    std::vector<T> actual(16 * 16);
    ASSERT_EQ(
        aclrtMemcpy(actual.data(), OUTPUT_BYTES, resources.output, OUTPUT_BYTES, ACL_MEMCPY_DEVICE_TO_HOST),
        ACL_SUCCESS);
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 16; ++col) {
            ASSERT_EQ(actual[(col / 8) * 16 * 8 + row * 8 + col % 8], input[row * RowStride + col])
                << "row=" << row << " col=" << col;
        }
    }
}
TEST(TLoadConversionStrideTest, Nd2Nz_float_32767_Static) { testConversionStride<float, false, 32767, false>(); }

TEST(TLoadConversionStrideTest, Nd2Nz_float_32768_Static) { testConversionStride<float, false, 32768, false>(); }

TEST(TLoadConversionStrideTest, Nd2Nz_float_32768_Dynamic) { testConversionStride<float, false, 32768, true>(); }

TEST(TLoadConversionStrideTest, Nd2Nz_float_65535_Dynamic) { testConversionStride<float, false, 65535, true>(); }

TEST(TLoadConversionStrideTest, Dn2Zn_float_32767_Static) { testConversionStride<float, true, 32767, false>(); }

TEST(TLoadConversionStrideTest, Dn2Zn_float_32768_Static) { testConversionStride<float, true, 32768, false>(); }

TEST(TLoadConversionStrideTest, Dn2Zn_float_32768_Dynamic) { testConversionStride<float, true, 32768, true>(); }

TEST(TLoadConversionStrideTest, Dn2Zn_float_65535_Dynamic) { testConversionStride<float, true, 65535, true>(); }

TEST(TLoadConversionStrideTest, Nd2Nz_int32_t_32767_Static) { testConversionStride<int32_t, false, 32767, false>(); }

TEST(TLoadConversionStrideTest, Nd2Nz_int32_t_32768_Static) { testConversionStride<int32_t, false, 32768, false>(); }

TEST(TLoadConversionStrideTest, Nd2Nz_int32_t_32768_Dynamic) { testConversionStride<int32_t, false, 32768, true>(); }

TEST(TLoadConversionStrideTest, Nd2Nz_int32_t_65535_Dynamic) { testConversionStride<int32_t, false, 65535, true>(); }

TEST(TLoadConversionStrideTest, Dn2Zn_int32_t_32767_Static) { testConversionStride<int32_t, true, 32767, false>(); }

TEST(TLoadConversionStrideTest, Dn2Zn_int32_t_32768_Static) { testConversionStride<int32_t, true, 32768, false>(); }

TEST(TLoadConversionStrideTest, Dn2Zn_int32_t_32768_Dynamic) { testConversionStride<int32_t, true, 32768, true>(); }

TEST(TLoadConversionStrideTest, Dn2Zn_int32_t_65535_Dynamic) { testConversionStride<int32_t, true, 65535, true>(); }

TEST(TLoadConversionStrideTest, Nd2Nz_uint32_t_32767_Static) { testConversionStride<uint32_t, false, 32767, false>(); }

TEST(TLoadConversionStrideTest, Nd2Nz_uint32_t_32768_Static) { testConversionStride<uint32_t, false, 32768, false>(); }

TEST(TLoadConversionStrideTest, Nd2Nz_uint32_t_32768_Dynamic) { testConversionStride<uint32_t, false, 32768, true>(); }

TEST(TLoadConversionStrideTest, Nd2Nz_uint32_t_65535_Dynamic) { testConversionStride<uint32_t, false, 65535, true>(); }

TEST(TLoadConversionStrideTest, Dn2Zn_uint32_t_32767_Static) { testConversionStride<uint32_t, true, 32767, false>(); }

TEST(TLoadConversionStrideTest, Dn2Zn_uint32_t_32768_Static) { testConversionStride<uint32_t, true, 32768, false>(); }

TEST(TLoadConversionStrideTest, Dn2Zn_uint32_t_32768_Dynamic) { testConversionStride<uint32_t, true, 32768, true>(); }

TEST(TLoadConversionStrideTest, Dn2Zn_uint32_t_65535_Dynamic) { testConversionStride<uint32_t, true, 65535, true>(); }

#endif

#ifdef PTO_TLOAD_WIDE_STRIDE
#ifndef PTO_TLOAD_VEC_ONLY
TEST(TLoadWideStrideTest, MatNdBurst40_Below)
{
    testLargeStride<true, 0, ((1ULL << 40) - 32) - 32, false, false, 0, 40>();
}

TEST(TLoadWideStrideTest, MatNdBurst40_At) { testLargeStride<true, 0, ((1ULL << 40) + 0) - 32, true, false, 0, 40>(); }

TEST(TLoadWideStrideTest, MatNdBurst40_Above)
{
    testLargeStride<true, 0, ((1ULL << 40) + 64) - 32, false, false, 0, 40>();
}

TEST(TLoadWideStrideTest, MatDnBurst40_Below)
{
    testLargeStride<true, 1, ((1ULL << 40) - 32) - 32, false, false, 0, 40>();
}

TEST(TLoadWideStrideTest, MatDnBurst40_At) { testLargeStride<true, 1, ((1ULL << 40) + 0) - 32, true, false, 0, 40>(); }

TEST(TLoadWideStrideTest, MatDnBurst40_Above)
{
    testLargeStride<true, 1, ((1ULL << 40) + 64) - 32, false, false, 0, 40>();
}

TEST(TLoadWideStrideTest, MatNzBurst40_Below)
{
    testLargeStride<true, 2, ((1ULL << 40) - 32) - 512, false, false, 0, 40>();
}

TEST(TLoadWideStrideTest, MatNzBurst40_At) { testLargeStride<true, 2, ((1ULL << 40) + 0) - 512, true, false, 0, 40>(); }

TEST(TLoadWideStrideTest, MatNzBurst40_Above)
{
    testLargeStride<true, 2, ((1ULL << 40) + 64) - 512, false, false, 0, 40>();
}

TEST(TLoadWideStrideTest, MatNdLoop140) { testLargeStride<true, 0, 64, true, true, 1, 40>(); }

TEST(TLoadWideStrideTest, MatNdLoop240) { testLargeStride<true, 0, 64, true, true, 2, 40>(); }

TEST(TLoadWideStrideTest, MatDnLoop140) { testLargeStride<true, 1, 64, true, true, 1, 40>(); }

TEST(TLoadWideStrideTest, MatDnLoop240) { testLargeStride<true, 1, 64, true, true, 2, 40>(); }

#endif
TEST(TLoadWideStrideTest, VecNdBurst40_Below)
{
    testLargeStride<false, 0, ((1ULL << 40) - 32) - 32, false, false, 0, 40>();
}

TEST(TLoadWideStrideTest, VecNdBurst40_At) { testLargeStride<false, 0, ((1ULL << 40) + 0) - 32, true, false, 0, 40>(); }

TEST(TLoadWideStrideTest, VecNdBurst40_Above)
{
    testLargeStride<false, 0, ((1ULL << 40) + 64) - 32, false, false, 0, 40>();
}

TEST(TLoadWideStrideTest, VecDnBurst40_Below)
{
    testLargeStride<false, 1, ((1ULL << 40) - 32) - 32, false, false, 0, 40>();
}

TEST(TLoadWideStrideTest, VecDnBurst40_At) { testLargeStride<false, 1, ((1ULL << 40) + 0) - 32, true, false, 0, 40>(); }

TEST(TLoadWideStrideTest, VecDnBurst40_Above)
{
    testLargeStride<false, 1, ((1ULL << 40) + 64) - 32, false, false, 0, 40>();
}

TEST(TLoadWideStrideTest, VecNzBurst40_Below)
{
    testLargeStride<false, 2, ((1ULL << 40) - 32) - 512, false, false, 0, 40>();
}

TEST(TLoadWideStrideTest, VecNzBurst40_At)
{
    testLargeStride<false, 2, ((1ULL << 40) + 0) - 512, true, false, 0, 40>();
}

TEST(TLoadWideStrideTest, VecNzBurst40_Above)
{
    testLargeStride<false, 2, ((1ULL << 40) + 64) - 512, false, false, 0, 40>();
}

TEST(TLoadWideStrideTest, VecNdLoop140) { testLargeStride<false, 0, 64, true, true, 1, 40>(); }

TEST(TLoadWideStrideTest, VecNdLoop240) { testLargeStride<false, 0, 64, true, true, 2, 40>(); }

TEST(TLoadWideStrideTest, VecDnLoop140) { testLargeStride<false, 1, 64, true, true, 1, 40>(); }

TEST(TLoadWideStrideTest, VecDnLoop240) { testLargeStride<false, 1, 64, true, true, 2, 40>(); }

#endif
#ifdef PTO_TLOAD_CONV_WIDE
TEST(TLoadWideStrideTest, MatConvLoop40) { testLargeStride<true, 4, 64, true, true, 0, 40>(); }

#endif

#ifdef PTO_TLOAD_VECTOR_LENGTH
template <typename T, bool DN, int Length, bool Dynamic>
void launchVectorLength(uint8_t* out, uint8_t* src, void* stream);

template <typename T, bool DN, int Length, bool Dynamic, int SourceOffset>
void testVectorLength()
{
    constexpr size_t ALIGNED_BYTES = (Length * sizeof(T) + 31) / 32 * 32;
    constexpr size_t OUTPUT_BYTES = ALIGNED_BYTES + 32;
    constexpr size_t INPUT_BYTES = OUTPUT_BYTES + SourceOffset * sizeof(T);
    LargeStrideResources resources;
    ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
    resources.initialized = true;
    ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
    resources.deviceSet = true;
    ASSERT_EQ(aclrtCreateStream(&resources.stream), ACL_SUCCESS);
    std::vector<T> input(INPUT_BYTES / sizeof(T), T(99));
    for (size_t i = 0; i < Length; ++i) {
        if constexpr (sizeof(T) == 8) {
            input[SourceOffset + i] = T((1ULL << 40) + i * 37 + 1);
        } else {
            input[SourceOffset + i] = T(i % 127 + 1);
        }
    }
    ASSERT_EQ(
        aclrtMalloc(reinterpret_cast<void**>(&resources.input), INPUT_BYTES, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(
        aclrtMalloc(reinterpret_cast<void**>(&resources.output), OUTPUT_BYTES, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(
        aclrtMemcpy(resources.input, INPUT_BYTES, input.data(), INPUT_BYTES, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemset(resources.output, OUTPUT_BYTES, 0x5a, OUTPUT_BYTES), ACL_SUCCESS);
    launchVectorLength<T, DN, Length, Dynamic>(
        resources.output, resources.input + SourceOffset * sizeof(T), resources.stream);
    ASSERT_EQ(aclrtSynchronizeStreamWithTimeout(resources.stream, 30000), ACL_SUCCESS);
    std::vector<T> actual(OUTPUT_BYTES / sizeof(T));
    ASSERT_EQ(
        aclrtMemcpy(actual.data(), OUTPUT_BYTES, resources.output, OUTPUT_BYTES, ACL_MEMCPY_DEVICE_TO_HOST),
        ACL_SUCCESS);
    for (size_t i = 0; i < Length; ++i) {
        ASSERT_EQ(actual[i], input[SourceOffset + i]) << "element=" << i;
    }
    auto bytes = reinterpret_cast<const uint8_t*>(actual.data());
    for (size_t i = Length * sizeof(T); i < ALIGNED_BYTES; ++i) {
        ASSERT_EQ(bytes[i], 0) << "padding byte=" << i;
    }
    for (size_t i = ALIGNED_BYTES; i < OUTPUT_BYTES; ++i) {
        ASSERT_EQ(bytes[i], 0x5a) << "guard byte=" << i;
    }
}
TEST(TLoadVectorLengthTest, Nd_uint8_t_65535_Static) { testVectorLength<uint8_t, false, 65535, false, 0>(); }

TEST(TLoadVectorLengthTest, Nd_uint8_t_65536_Static) { testVectorLength<uint8_t, false, 65536, false, 0>(); }

TEST(TLoadVectorLengthTest, Nd_uint8_t_65536_Dynamic) { testVectorLength<uint8_t, false, 65536, true, 0>(); }

TEST(TLoadVectorLengthTest, Nd_uint8_t_65537_Dynamic) { testVectorLength<uint8_t, false, 65537, true, 0>(); }

TEST(TLoadVectorLengthTest, Nd_uint16_t_65535_Static) { testVectorLength<uint16_t, false, 65535, false, 0>(); }

TEST(TLoadVectorLengthTest, Nd_uint16_t_65536_Static) { testVectorLength<uint16_t, false, 65536, false, 0>(); }

TEST(TLoadVectorLengthTest, Nd_uint16_t_65536_Dynamic) { testVectorLength<uint16_t, false, 65536, true, 0>(); }

TEST(TLoadVectorLengthTest, Nd_uint16_t_65537_Dynamic) { testVectorLength<uint16_t, false, 65537, true, 0>(); }

TEST(TLoadVectorLengthTest, Nd_float_65535_Static) { testVectorLength<float, false, 65535, false, 0>(); }

TEST(TLoadVectorLengthTest, Nd_float_65536_Static) { testVectorLength<float, false, 65536, false, 0>(); }

TEST(TLoadVectorLengthTest, Nd_float_65536_Dynamic) { testVectorLength<float, false, 65536, true, 0>(); }

TEST(TLoadVectorLengthTest, Nd_float_65537_Dynamic) { testVectorLength<float, false, 65537, true, 0>(); }

TEST(TLoadVectorLengthTest, Nd_int64_t_16383_Static) { testVectorLength<int64_t, false, 16383, false, 0>(); }

TEST(TLoadVectorLengthTest, Nd_int64_t_16384_Static) { testVectorLength<int64_t, false, 16384, false, 0>(); }

TEST(TLoadVectorLengthTest, Nd_int64_t_16384_Dynamic) { testVectorLength<int64_t, false, 16384, true, 0>(); }

TEST(TLoadVectorLengthTest, Nd_int64_t_16385_Dynamic) { testVectorLength<int64_t, false, 16385, true, 0>(); }

TEST(TLoadVectorLengthTest, Nd_int64_t_32767_Static) { testVectorLength<int64_t, false, 32767, false, 0>(); }

TEST(TLoadVectorLengthTest, Nd_int64_t_32768_Static) { testVectorLength<int64_t, false, 32768, false, 0>(); }

TEST(TLoadVectorLengthTest, Nd_int64_t_32768_Dynamic) { testVectorLength<int64_t, false, 32768, true, 0>(); }

TEST(TLoadVectorLengthTest, Nd_int64_t_32769_Dynamic) { testVectorLength<int64_t, false, 32769, true, 0>(); }

TEST(TLoadVectorLengthTest, Nd_uint64_t_32768_Dynamic) { testVectorLength<uint64_t, false, 32768, true, 0>(); }

TEST(TLoadVectorLengthTest, Dn_uint8_t_31_Static) { testVectorLength<uint8_t, true, 31, false, 0>(); }

TEST(TLoadVectorLengthTest, Dn_uint8_t_32_Dynamic) { testVectorLength<uint8_t, true, 32, true, 0>(); }

TEST(TLoadVectorLengthTest, Dn_uint8_t_33_Dynamic) { testVectorLength<uint8_t, true, 33, true, 0>(); }

TEST(TLoadVectorLengthTest, Dn_uint16_t_15_Static) { testVectorLength<uint16_t, true, 15, false, 0>(); }

TEST(TLoadVectorLengthTest, Dn_uint16_t_16_Dynamic) { testVectorLength<uint16_t, true, 16, true, 0>(); }

TEST(TLoadVectorLengthTest, Dn_uint16_t_17_Dynamic) { testVectorLength<uint16_t, true, 17, true, 0>(); }

TEST(TLoadVectorLengthTest, Dn_float_7_Static) { testVectorLength<float, true, 7, false, 0>(); }

TEST(TLoadVectorLengthTest, Dn_float_8_Dynamic) { testVectorLength<float, true, 8, true, 0>(); }

TEST(TLoadVectorLengthTest, Dn_float_9_Dynamic) { testVectorLength<float, true, 9, true, 0>(); }

TEST(TLoadVectorLengthTest, Dn_int64_t_3_Static) { testVectorLength<int64_t, true, 3, false, 0>(); }

TEST(TLoadVectorLengthTest, Dn_int64_t_4_Dynamic) { testVectorLength<int64_t, true, 4, true, 0>(); }

TEST(TLoadVectorLengthTest, Dn_int64_t_5_Dynamic) { testVectorLength<int64_t, true, 5, true, 0>(); }

TEST(TLoadVectorLengthTest, Dn_int64_t_16383_Static) { testVectorLength<int64_t, true, 16383, false, 0>(); }

TEST(TLoadVectorLengthTest, Dn_int64_t_16384_Static) { testVectorLength<int64_t, true, 16384, false, 0>(); }

TEST(TLoadVectorLengthTest, Dn_int64_t_16384_Dynamic) { testVectorLength<int64_t, true, 16384, true, 0>(); }

TEST(TLoadVectorLengthTest, Nd_int64_t_64_Static) { testVectorLength<int64_t, false, 64, false, 0>(); }

TEST(TLoadVectorLengthTest, Nd_int64_t_3_Dynamic) { testVectorLength<int64_t, false, 3, true, 0>(); }

TEST(TLoadVectorLengthTest, Nd_float_64_Static) { testVectorLength<float, false, 64, false, 0>(); }

TEST(TLoadVectorLengthTest, Nd_float_7_Dynamic) { testVectorLength<float, false, 7, true, 0>(); }

TEST(TLoadVectorLengthTest, Nd_uint8_t_65_Unaligned) { testVectorLength<uint8_t, false, 65, true, 1>(); }

TEST(TLoadVectorLengthTest, Dn_uint8_t_65_Unaligned) { testVectorLength<uint8_t, true, 65, true, 1>(); }

TEST(TLoadVectorLengthTest, Nd_uint16_t_65_Unaligned) { testVectorLength<uint16_t, false, 65, true, 1>(); }

TEST(TLoadVectorLengthTest, Dn_uint16_t_65_Unaligned) { testVectorLength<uint16_t, true, 65, true, 1>(); }

TEST(TLoadVectorLengthTest, Nd_float_65_Unaligned) { testVectorLength<float, false, 65, true, 1>(); }

TEST(TLoadVectorLengthTest, Dn_float_65_Unaligned) { testVectorLength<float, true, 65, true, 1>(); }

TEST(TLoadVectorLengthTest, Nd_int64_t_65_Unaligned) { testVectorLength<int64_t, false, 65, true, 1>(); }

TEST(TLoadVectorLengthTest, Dn_int64_t_65_Unaligned) { testVectorLength<int64_t, true, 65, true, 1>(); }
#endif
