/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef FFN_GRID_TILE_ALIASES_INL_HPP
#define FFN_GRID_TILE_ALIASES_INL_HPP

using pto::BLayout;
using pto::Direction;
using pto::GlobalTensor;
using pto::GridCoord;
using pto::GridDirection;
using pto::GridPipe;
using pto::GridShape;
using pto::Layout;
using pto::SLayout;
using pto::Shape;
using pto::Stride;
using pto::Tile;
using pto::TileAcc;
using pto::TileLeft;
using pto::TileRight;
using pto::TileType;
using pto::TPipe;

using GateF32Tile = Tile<TileType::Vec, float, FFN_TOKEN_TILE, FFN_FFN_TILE, BLayout::RowMajor>;
using UpF32Tile = GateF32Tile;
using HiddenF32Tile = GateF32Tile;
using HiddenF16Tile = Tile<TileType::Vec, half, FFN_TOKEN_TILE, FFN_FFN_TILE, BLayout::RowMajor>;

#endif
