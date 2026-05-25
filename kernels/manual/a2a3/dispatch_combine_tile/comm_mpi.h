/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_COMBINE_TILE_COMM_MPI_H_
#define DISPATCH_COMBINE_TILE_COMM_MPI_H_

#include <cstddef>
#include <cstdint>

namespace dispatch_combine_tile {

struct MpiContext {
    int rank;
    int size;
};

// Task 3 adapts kernels/manual/a2a3/gemm_ar dlopen MPI wrapper for
// multi-process rank/size discovery, broadcast, barrier, and finalize.
MpiContext InitMpiAndRank(int *argc, char ***argv);
void MpiBroadcast(MpiContext *context, void *data, size_t bytes, int root);
void MpiBarrier(MpiContext *context);
void FinalizeMpi(MpiContext *context);

} // namespace dispatch_combine_tile

#endif // DISPATCH_COMBINE_TILE_COMM_MPI_H_
