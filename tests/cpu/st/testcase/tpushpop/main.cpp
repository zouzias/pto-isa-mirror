/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <functional>
#include "test_common.h"
#include <gtest/gtest.h>
#include <pto/common/fifo.hpp>

using namespace std;
using namespace pto;
using namespace PtoTestCommon;

template <typename T, int rows, int cols, TileType srcLoc>
void testPushPop()
{
    constexpr int FiFoDepth = 8;
    constexpr int FiFoPeriod = 1;
    constexpr int LocalDepth = 2;
    using PPipe =
        TPipe<0, T, FIFOType::GM_FIFO, FiFoDepth, FiFoPeriod, LocalDepth, TSyncOpType::TSTORE_C2GM, TSyncOpType::TLOAD>;
    using PPTile = Tile<srcLoc, T, rows, cols>;
    using PPGlobal =
        GlobalTensor<T, Shape<1, 1, 1, rows, cols>, Stride<rows * cols, rows * cols, rows * cols, cols, 1>>;

    T *srcDevice;
    aclrtMalloc((void **)&srcDevice, PPTile::Numel * FiFoDepth, ACL_MEM_MALLOC_HUGE_FIRST);
    PPipe pipe(srcDevice, 0x0);
    PPTile src;
    PPTile dst;

    std::vector<T> srcData(PPTile::Numel, 0);
    std::vector<T> dstData(PPTile::Numel, 0);

    for (int i = 0; i < src.Numel; i++) {
        src.data()[i] = std::rand() / 1000.0;
    }
    for (int i = 0; i < dst.Numel; i++) {
        dst.data()[i] = 0;
    }

    PPGlobal srcTensor(srcData.data());
    PPGlobal dstTensor(dstData.data());

    TPUSH(src, pipe);
    TPOP(dst, pipe);

    EXPECT_TRUE(ResultCmp(srcData, dstData.data(), 0));

    aclrtFree(srcDevice);
}

class TPushPopTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

#define TPUSHPOP_TEST(T, rows, cols, srcLoc)             \
    TEST_F(TPushPopTest, T##_##rows##_##cols##_##srcLoc) \
    {                                                    \
        testPushPop<T, rows, cols, TileType::srcLoc>();  \
    }

TPUSHPOP_TEST(float, 64, 128, Vec)
TPUSHPOP_TEST(float, 128, 128, Vec)
TPUSHPOP_TEST(float, 64, 128, Mat)
TPUSHPOP_TEST(float, 128, 128, Mat)
TPUSHPOP_TEST(uint32_t, 64, 128, Vec)
TPUSHPOP_TEST(uint32_t, 128, 128, Vec)
TPUSHPOP_TEST(uint32_t, 64, 128, Mat)
TPUSHPOP_TEST(uint32_t, 128, 128, Mat)
