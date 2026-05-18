/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef FA_QK_TILE_TRAITS_H
#define FA_QK_TILE_TRAITS_H

#include <cstdint>

template <int CUBE_S0, int CUBE_S1, int TILE_S1, int QKP_CV_FIFO, int CV_FIFO_CONS_SYNC_PERIOD, int HEAD_SIZE>
struct QkTileTraits {
    static constexpr uint32_t CubeS0 = CUBE_S0;
    static constexpr uint32_t CubeS1 = CUBE_S1;
    static constexpr uint32_t TileS1 = TILE_S1;
    static constexpr uint32_t kTileFactor = TileS1 / CubeS1;
    static constexpr uint32_t CubeHead = HEAD_SIZE;

    static_assert(QKP_CV_FIFO >= 1, "QKP_CV_FIFO must be >= 1");
    static_assert(TileS1 % CubeS1 == 0, "TILE_S1 must be divisible by CUBE_S1");
};

#endif // FA_QK_TILE_TRAITS_H
