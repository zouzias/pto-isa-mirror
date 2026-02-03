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
#include <pto/pto-inst.hpp>
using namespace pto;

template <typename InputT, typename OutputT, uint32_t MatrixSize>
AICORE void runKernelBatchMatrixSquare(__gm__ OutputT *z, __gm__ InputT *x)
{
// #if defined __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
// // Placeholder for AIV -- nothing to do on vector unit.
// #el
#if (__CHECK_FEATURE_AT_PRECOMPILE) || (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__)) // CUBE compilation
    constexpr uint32_t TileLen = MatrixSize * MatrixSize;
    const uint32_t global_index = get_block_idx() * TileLen;

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
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);  // MTE2 pipe sets flag for MTE1 pipe
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0); // MTE1 pipe waits for MTE2 to set flag

    // LOAD L1 -> L0 (MTE1)
    TEXTRACT(a_l0_tile, ab_l1_tile, 0, 0);
    TEXTRACT(b_l0_tile, ab_l1_tile, 0, 0);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);  // MTE1 pipe sets flag for M pipe
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0); // M pipe waits for MTE1 pipe to set flag

    // MATMUL (M)
    TMATMUL(c_l0_tile, a_l0_tile, b_l0_tile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);  // M pipe sets flag for FIX pipe
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0); // FIX pipe waits for M pipe to set flag
    TSTORE(z_global_out, c_l0_tile);
#else
// Nothing to do.
#endif
}

__global__ AICORE void batch_matrix_square_fp16(__gm__ void *x, __gm__ void *z,
                                                uint32_t matrix_size)
{
    switch (matrix_size)
    {
    case 16:
        runKernelBatchMatrixSquare<half, float, 16>((__gm__ float *)z,
                                                    (__gm__ half *)x);
        break;
    case 32:
        runKernelBatchMatrixSquare<half, float, 32>((__gm__ float *)z,
                                                    (__gm__ half *)x);
        break;
    case 64:
        runKernelBatchMatrixSquare<half, float, 64>((__gm__ float *)z,
                                                    (__gm__ half *)x);
        break;
    case 96:
        runKernelBatchMatrixSquare<half, float, 96>((__gm__ float *)z,
                                                    (__gm__ half *)x);
        break;
    case 128:
        runKernelBatchMatrixSquare<half, float, 128>((__gm__ float *)z,
                                                     (__gm__ half *)x);
        break;
    }
}

extern "C" void call_kernel(uint32_t block_dim, void *stream, uint8_t *out,
                            uint8_t *src, uint32_t matrix_size)
{
    batch_matrix_square_fp16<<<block_dim, nullptr, stream>>>(src, out, matrix_size);
}
