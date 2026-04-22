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
#include <pto/common/constants.hpp>
#include <gtest/gtest.h>

#include "cost_check.hpp"

using namespace pto;

namespace {

template <typename T>
constexpr T CeilAlign(T num_1, T num_2)
{
    return num_2 == 0 ? 0 : (num_1 + num_2 - 1) / num_2 * num_2;
}

template <typename outType, typename AType, typename BType, int validM, int validK, int validN, float profiling,
          float accuracy>
void runTMatmul()
{
    constexpr int blockAlign = (sizeof(AType) == 1) ? 32 : 16;
    constexpr int M = CeilAlign<int>(validM, blockAlign);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);

    using LeftTile = TileLeft<AType, M, K, validM, validK>;
    using RightTile = TileRight<BType, K, N, validK, validN>;
    using AccTile = TileAcc<outType, M, N, validM, validN>;

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x10000);
    TASSIGN(cTile, 0x20000);

    TMATMUL(cTile, aTile, bTile);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

template <typename outType, typename AType, typename BType, int M, int K, int N, int numRepeats, float profiling,
          float accuracy>
void runTMatmulSplitK()
{
    using LeftTile = TileLeft<AType, M, K, M, K>;
    using RightTile = TileRight<BType, K, N, K, N>;
    using AccTile = TileAcc<outType, M, N, M, N>;

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x10000);
    TASSIGN(cTile, 0x20000);

    for (uint32_t i = 0; i < numRepeats; i++) {
        if (i == 0) {
            TMATMUL(cTile, aTile, bTile);
        } else {
            TMATMUL_ACC(cTile, cTile, aTile, bTile);
        }
    }

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TMatmul, half_64x64x64)
{
    runTMatmul<float, half, half, 64, 64, 64, 107.0f, 0.0f>();
}

TEST(TMatmul, half_64x128x64)
{
    runTMatmul<float, half, half, 64, 128, 64, 171.0f, 0.0f>();
}

TEST(TMatmul, int32_t_64x64x64)
{
    runTMatmul<int32_t, int8_t, int8_t, 64, 64, 64, 75.0f, 0.0f>();
}


