/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef UB2L1_ND2NZ_TEST_H_
#define UB2L1_ND2NZ_TEST_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <gtest/gtest.h>
#include "acl/acl.h"

#include "ub2l1_nd2nz_types.h"

namespace PtoTestCommon {

inline void testUbToL1Nd2Nz(
    void (*launch)(uint64_t*, uint64_t*, void*), int elementBits, int srcRows, int srcCols, int dstRows, int dstCols,
    int validRows, int validCols, Nd2NzOperation operation, int indexRow, int indexCol)
{
    const bool isInsert = operation == Nd2NzOperation::INSERT || operation == Nd2NzOperation::DUAL_INSERT;
    // Compare bytes, including padding sentinels, so packed and floating formats retain every bit.
    const size_t srcTileBytes = srcRows * srcCols * elementBits / 8;
    const int partCount = operation == Nd2NzOperation::DUAL_INSERT ? 2 : 1;
    const size_t srcBytes = partCount * srcTileBytes;
    const size_t dstBytes = dstRows * dstCols * elementBits / 8;
    std::vector<uint8_t> input(srcBytes);
    std::vector<uint8_t> output(dstBytes, 0xa5);
    std::vector<uint8_t> golden = output;
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<uint8_t>((i * 17 + i / 7) % 251);
    }
    for (int part = 0; part < partCount; ++part) {
        for (int row = 0; row < validRows; ++row) {
            for (int colByte = 0; colByte < validCols * elementBits / 8; ++colByte) {
                const int srcRow = row + (operation == Nd2NzOperation::EXTRACT ? indexRow : 0);
                const int srcColByte =
                    colByte + (operation == Nd2NzOperation::EXTRACT ? indexCol * elementBits / 8 : 0);
                const int dstRow = row + (isInsert ? indexRow : 0) + part * validRows;
                const int dstColByte = colByte + (isInsert ? indexCol * elementBits / 8 : 0);
                const int blockRow = dstRow / 16;
                const int blockCol = dstColByte / 32;
                const size_t offset =
                    ((blockCol * (dstRows / 16) + blockRow) * 16 + dstRow % 16) * 32 + dstColByte % 32;
                golden[offset] = input[part * srcTileBytes + srcRow * srcCols * elementBits / 8 + srcColByte];
            }
        }
    }
    ASSERT_TRUE(aclInit(nullptr) == ACL_SUCCESS);
    const char* deviceEnv = std::getenv("PTO_DEVICE_ID");
    const int deviceId = deviceEnv ? std::atoi(deviceEnv) : 0;
    ASSERT_TRUE(aclrtSetDevice(deviceId) == ACL_SUCCESS);
    aclrtStream stream;
    ASSERT_TRUE(aclrtCreateStream(&stream) == ACL_SUCCESS);
    void* srcDevice = nullptr;
    void* dstDevice = nullptr;
    ASSERT_TRUE(aclrtMalloc(&srcDevice, srcBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS);
    ASSERT_TRUE(aclrtMalloc(&dstDevice, dstBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS);
    ASSERT_TRUE(aclrtMemcpy(srcDevice, srcBytes, input.data(), srcBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS);
    ASSERT_TRUE(aclrtMemcpy(dstDevice, dstBytes, output.data(), dstBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS);
    launch(static_cast<uint64_t*>(dstDevice), static_cast<uint64_t*>(srcDevice), stream);
    EXPECT_TRUE(aclrtSynchronizeStream(stream) == ACL_SUCCESS);
    EXPECT_TRUE(aclrtMemcpy(output.data(), dstBytes, dstDevice, dstBytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS);
    EXPECT_TRUE(aclrtFree(srcDevice) == ACL_SUCCESS);
    EXPECT_TRUE(aclrtFree(dstDevice) == ACL_SUCCESS);
    EXPECT_TRUE(aclrtDestroyStream(stream) == ACL_SUCCESS);
    EXPECT_TRUE(aclrtResetDevice(deviceId) == ACL_SUCCESS);
    EXPECT_TRUE(aclFinalize() == ACL_SUCCESS);
    EXPECT_TRUE(output == golden);
}

} // namespace PtoTestCommon

#endif
