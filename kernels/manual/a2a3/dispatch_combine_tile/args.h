/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_COMBINE_TILE_ARGS_H_
#define DISPATCH_COMBINE_TILE_ARGS_H_

#include "common.h"

namespace dispatch_combine_tile {

struct DispatchCombineTileArgs {
    DispatchCombineTileShape shape;
    DispatchCombineTileRuntimeConfig runtime;
};

// Task 2 implements explicit argument parsing, validation, and run summaries.
DispatchCombineTileArgs ParseArgs(int argc, char **argv);
void ValidateArgs(const DispatchCombineTileArgs &args);
void PrintRunSummary(const DispatchCombineTileArgs &args);

} // namespace dispatch_combine_tile

#endif // DISPATCH_COMBINE_TILE_ARGS_H_
