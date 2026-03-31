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

using namespace pto;

void LaunchTGemv()
{
    using LeftTile = TileLeft<float, 1, 16, 1, 16>;
    using RightTile = TileRight<float, 16, 16, 16, 16>;
    using AccTile = TileAcc<float, 16, 16, 1, 16>;

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;

    aTile.SetKAligned(false);
    bTile.SetKAligned(false);

    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x20000);
    TASSIGN(cTile, 0x40000);

    TGEMV(cTile, aTile, bTile);
}
