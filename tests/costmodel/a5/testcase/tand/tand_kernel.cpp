/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>

#include <pto/pto-inst.hpp>

using namespace pto;

void LaunchTAnd()
{
    using TileData = Tile<TileType::Vec, uint16_t, 64, 64, BLayout::RowMajor, -1, -1>;

    TileData src0(63, 63);
    TileData src1(63, 63);
    TileData dst(63, 63);
    TASSIGN(src0, 0x0);
    TASSIGN(src1, 0x4000);
    TASSIGN(dst, 0x8000);

    TAND(dst, src0, src1);
}
