/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/
#include <cstdint>

#include "kernel_operator.h"
#define MEMORY_BASE
#define NUM_BLOCKS 20 // number of AICs

#if defined __CCE_AICORE__ == 220 && \
    defined(__DAV_C220_VEC__) // Placeholder for VEC compilation (the real
                              // kernel is CUBE-only).
#include <pto/common/type.hpp>

extern "C" __global__ AICORE void batch_matrix_square_fp16(GM_ADDR x, GM_ADDR z,
                                                           uint32_t matrix_size,
                                                           uint32_t block_dim)
{
}

#elif (__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__)) // CUBE compilation

#include <pto/pto-inst.hpp>

using namespace pto;

template <pipe_t SrcPipe, pipe_t DstPipe>
AICORE inline void SetFlag(uint32_t id)
{
    set_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
}
template <pipe_t SrcPipe, pipe_t DstPipe>
AICORE inline void WaitFlag(uint32_t id)
{
    wait_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
}

template <typename InputT, typename OutputT, uint32_t MatrixSize>
AICORE void runKernelBatchMatrixSquare(__gm__ OutputT *z, __gm__ InputT *x,
                                       uint32_t block_dim)
{
    if (get_block_idx() < block_dim)
    {
        constexpr uint32_t tile_len = MatrixSize * MatrixSize;
        const uint32_t global_index = get_block_idx() * tile_len;

        /* Global Memory / Tensors */
        using TensorShapeIn =
            TileShape2D<InputT, MatrixSize, MatrixSize, Layout::ND>;
        using TensorStridesIn =
            BaseShape2D<InputT, MatrixSize, MatrixSize, Layout::ND>;
        using GlobalTensorIn =
            GlobalTensor<InputT, TensorShapeIn, TensorStridesIn, Layout::ND>;

        using TensorShapeOut =
            TileShape2D<OutputT, MatrixSize, MatrixSize, Layout::ND>;
        using TensorStridesOut =
            BaseShape2D<OutputT, MatrixSize, MatrixSize, Layout::ND>;
        using GlobalTensorOut =
            GlobalTensor<OutputT, TensorShapeOut, TensorStridesOut, Layout::ND>;

        /* L1 Memory */
        using TileL1AB =
            Tile<TileType::Mat, InputT, MatrixSize, MatrixSize, BLayout::ColMajor,
                 MatrixSize, MatrixSize, SLayout::RowMajor, 512>;

        /* L0 Memory */
        using TileL0A = TileLeft<InputT, MatrixSize, MatrixSize>;
        using TileL0B = TileRight<InputT, MatrixSize, MatrixSize>;
        using TileL0C = TileAcc<OutputT, MatrixSize, MatrixSize>;

        GlobalTensorIn x_global_in(x + global_index);
        GlobalTensorOut z_global_out(z + global_index);
        TileL1AB ab_l1_tile;
        TileL0A a_l0_tile;
        TileL0B b_l0_tile;
        TileL0C c_l0_tile;

        TASSIGN(ab_l1_tile, 0x0);

        TASSIGN(a_l0_tile, 0x0);
        TASSIGN(b_l0_tile, 0x0);
        TASSIGN(c_l0_tile, 0x0);

        // LOAD GM -> L1 (MTE2)
        TLOAD(ab_l1_tile, x_global_in);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(0);  // MTE2 pipe sets flag for MTE1 pipe
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(0); // MTE1 pipe waits for MTE2 to set flag

        // LOAD L1 -> L0 (MTE1)
        TEXTRACT(a_l0_tile, ab_l1_tile, 0, 0);
        TEXTRACT(b_l0_tile, ab_l1_tile, 0, 0);
        SetFlag<PIPE_MTE1, PIPE_M>(0);  // MTE1 pipe sets flag for MM pipe
        WaitFlag<PIPE_MTE1, PIPE_M>(0); // MM pipe waits for MTE1 pipe to set flag

        // MATMUL (M)
        TMATMUL(c_l0_tile, a_l0_tile, b_l0_tile);
        SetFlag<PIPE_M, PIPE_FIX>(0);  // M pipe sets flag for FIX pipe
        WaitFlag<PIPE_M, PIPE_FIX>(0); // FIX pipe waits for M pipe to set flag
        TSTORE(z_global_out, c_l0_tile);
    }
}

extern "C" __global__ AICORE void batch_matrix_square_fp16(GM_ADDR x, GM_ADDR z,
                                                           uint32_t matrix_size,
                                                           uint32_t block_dim)
{
    switch (matrix_size)
    {
    case 16:
        runKernelBatchMatrixSquare<half, float, 16>((__gm__ float *)z,
                                                    (__gm__ half *)x, block_dim);
        break;
    case 32:
        runKernelBatchMatrixSquare<half, float, 32>((__gm__ float *)z,
                                                    (__gm__ half *)x, block_dim);
        break;
    case 64:
        runKernelBatchMatrixSquare<half, float, 64>((__gm__ float *)z,
                                                    (__gm__ half *)x, block_dim);
        break;
    case 96:
        runKernelBatchMatrixSquare<half, float, 96>((__gm__ float *)z,
                                                    (__gm__ half *)x, block_dim);
        break;
    case 128:
        runKernelBatchMatrixSquare<half, float, 128>((__gm__ float *)z,
                                                     (__gm__ half *)x, block_dim);
        break;
    }
}
#else

#include <pto/common/type.hpp>
extern "C" __global__ AICORE void batch_matrix_square_fp16(GM_ADDR x, GM_ADDR z,
                                                           uint32_t matrix_size,
                                                           uint32_t block_dim)
{
}

#endif
extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *out,
                            uint8_t *src, uint32_t matrix_size)
{
    batch_matrix_square_fp16<<<NUM_BLOCKS, nullptr, stream>>>(src, out, matrix_size,
                                                              blockDim);
}
