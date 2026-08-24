/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Host driver for the GridPipe single-source broadcast smoke kernel.

#include "bcast_smoke_config.hpp"
#include "bcast_smoke_launch.hpp"

#define GRID_SMOKE_ROWS BCAST_ROWS
#define GRID_SMOKE_COLS BCAST_COLS
#define GRID_SMOKE_WINDOW_BYTES BCAST_WINDOW_BYTES
#define GRID_SMOKE_TILE_BYTES BCAST_TILE_BYTES
#define GRID_SMOKE_TILE_ELEMS BCAST_TILE_ELEMS
#define GRID_SMOKE_FLAGS_BYTES BCAST_GRID_FLAGS_BYTES
#define GRID_SMOKE_T BCAST_T
#define GRID_SMOKE_W BCAST_W
#define GRID_SMOKE_LAUNCH launchBcastSmokeKernel
#define GRID_SMOKE_TITLE "GridPipe single-source broadcast smoke test"
#define GRID_SMOKE_RESULT_LABEL (BCAST_SPAN_COL != 0 ? "broadcast smoke: span=COL" : "broadcast smoke: span=ROW")
#define GRID_SMOKE_SUCCESS "[SUCCESS] GridPipe single-source broadcast smoke PASS."
#define GRID_SMOKE_FAILED "[FAILED] GridPipe single-source broadcast smoke FAILED."
#define GRID_SMOKE_EXPECTED_VALUE(row, col, cols)         \
    ((BCAST_SPAN_COL != 0 ? (row) : (col)) == BCAST_SRC ? \
         0.0f :                                           \
         static_cast<float>(                              \
             ((BCAST_SPAN_COL != 0 ? BCAST_SRC : (row)) * (cols) + (BCAST_SPAN_COL != 0 ? (col) : BCAST_SRC)) + 1))
#define GRID_SMOKE_PRINT_CONFIG()                                                                         \
    do {                                                                                                  \
        std::cout << "  span=" << (BCAST_SPAN_COL != 0 ? "COL" : "ROW") << " src=" << BCAST_SRC           \
                  << " grid=" << BCAST_ROWS << "x" << BCAST_COLS << " tile=" << BCAST_T << "x" << BCAST_W \
                  << std::endl;                                                                           \
    } while (0)

#include "grid_smoke_driver_inl.hpp"
