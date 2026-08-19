/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Host driver for the GridPipe routed K-hop unicast smoke kernel.

#include "khop_smoke_config.hpp"
#include "khop_smoke_launch.hpp"

#define GRID_SMOKE_ROWS KHOP_ROWS
#define GRID_SMOKE_COLS KHOP_COLS
#define GRID_SMOKE_WINDOW_BYTES KHOP_WINDOW_BYTES
#define GRID_SMOKE_TILE_BYTES KHOP_TILE_BYTES
#define GRID_SMOKE_TILE_ELEMS KHOP_TILE_ELEMS
#define GRID_SMOKE_FLAGS_BYTES KHOP_GRID_FLAGS_BYTES
#define GRID_SMOKE_T KHOP_T
#define GRID_SMOKE_W KHOP_W
#define GRID_SMOKE_LAUNCH launchKHopSmokeKernel
#define GRID_SMOKE_TITLE "GridPipe routed K-hop unicast smoke test"
#define GRID_SMOKE_RESULT_LABEL "K-hop smoke:"
#define GRID_SMOKE_SUCCESS "[SUCCESS] GridPipe K-hop unicast smoke PASS."
#define GRID_SMOKE_FAILED "[FAILED] GridPipe K-hop unicast smoke FAILED."
#define GRID_SMOKE_EXPECTED_VALUE(row, col, cols) \
    ((col) < KHOP_DIST ? 0.0f : static_cast<float>(static_cast<size_t>(row) * (cols) + ((col) - KHOP_DIST) + 1))
#define GRID_SMOKE_PRINT_CONFIG()                                                                         \
    do {                                                                                                  \
        std::cout << "  grid=" << KHOP_ROWS << "x" << KHOP_COLS << " DIST=" << KHOP_DIST << " tile="     \
                  << KHOP_T << "x" << KHOP_W << std::endl;                                               \
    } while (0)

#include "grid_smoke_driver_inl.hpp"
