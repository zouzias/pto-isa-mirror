/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
/**
 * AscendC Eltwise Add Kernel - Baseline comparison with PTO TADD
 *
 * Pattern: TPipe + TQue CopyIn→Compute→CopyOut
 * Architecture: A5 (dav-3510 / Ascend950PR_9599)
 *
 * Two variants measured:
 *  1. FLAT  - single DataCopy of whole tile, Add(count) - uses VF to process all rows at once
 *  2. ROWLOOP - DataCopy whole tile, row-by-row Add loop - no VF row fusion, explicit for-loop
 *
 * Purpose: Show whether AscendC row-loop matches PTO VF EPC, or if row fusion is needed.
 */
#include "kernel_operator.h"

using namespace AscendC;

// ============================================================================
// Config: match 32KB tile_perf shapes
// ============================================================================
constexpr uint32_t ITERATIONS = 10;

// ============================================================================
// Variant 1: FLAT - single Add over entire tile
// ============================================================================
template <typename T, uint32_t ROWS, uint32_t COLS>
class KernelEltwiseFlat {
public:
    static constexpr uint32_t ELEMENTS = ROWS * COLS;

    __aicore__ inline KernelEltwiseFlat() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
        xGm.SetGlobalBuffer((__gm__ T*)x, ELEMENTS);
        yGm.SetGlobalBuffer((__gm__ T*)y, ELEMENTS);
        zGm.SetGlobalBuffer((__gm__ T*)z, ELEMENTS);
        pipe.InitBuffer(inQueueX, 1, ELEMENTS * sizeof(T));
        pipe.InitBuffer(inQueueY, 1, ELEMENTS * sizeof(T));
        pipe.InitBuffer(outQueueZ, 1, ELEMENTS * sizeof(T));
    }

    __aicore__ inline void Process() {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        for (uint32_t iter = 0; iter < ITERATIONS; iter++) {
            DataCopy(xLocal, xGm, ELEMENTS);
            DataCopy(yLocal, yGm, ELEMENTS);
            // Single Add over entire flat tile - lets hardware fuse rows
            Add(zLocal, xLocal, yLocal, ELEMENTS);
            DataCopy(zGm, zLocal, ELEMENTS);
        }
        outQueueZ.FreeTensor(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, 1> outQueueZ;
    GlobalTensor<T> xGm, yGm, zGm;
};

// ============================================================================
// Variant 2: ROWLOOP - row-by-row Add loop
// ============================================================================
template <typename T, uint32_t ROWS, uint32_t COLS>
class KernelEltwiseRowLoop {
public:
    static constexpr uint32_t ELEMENTS = ROWS * COLS;

    __aicore__ inline KernelEltwiseRowLoop() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
        xGm.SetGlobalBuffer((__gm__ T*)x, ELEMENTS);
        yGm.SetGlobalBuffer((__gm__ T*)y, ELEMENTS);
        zGm.SetGlobalBuffer((__gm__ T*)z, ELEMENTS);
        pipe.InitBuffer(inQueueX, 1, ELEMENTS * sizeof(T));
        pipe.InitBuffer(inQueueY, 1, ELEMENTS * sizeof(T));
        pipe.InitBuffer(outQueueZ, 1, ELEMENTS * sizeof(T));
    }

    __aicore__ inline void Process() {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        for (uint32_t iter = 0; iter < ITERATIONS; iter++) {
            DataCopy(xLocal, xGm, ELEMENTS);
            DataCopy(yLocal, yGm, ELEMENTS);
            // Row-by-row loop: each Add handles one row (no VF row fusion)
            for (uint32_t row = 0; row < ROWS; row++) {
                uint32_t off = row * COLS;
                Add(zLocal[off], xLocal[off], yLocal[off], COLS);
            }
            DataCopy(zGm, zLocal, ELEMENTS);
        }
        outQueueZ.FreeTensor(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, 1> outQueueZ;
    GlobalTensor<T> xGm, yGm, zGm;
};

// ============================================================================
// Kernel entry points - shapes matching tile_perf 32KB cases
// ============================================================================

// --- FLAT variants ---
extern "C" __global__ __aicore__ void asc_add_flat_fp32_1x8192(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
    KernelEltwiseFlat<float, 1, 8192> op; op.Init(x, y, z); op.Process();
}
extern "C" __global__ __aicore__ void asc_add_flat_fp32_16x512(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
    KernelEltwiseFlat<float, 16, 512> op; op.Init(x, y, z); op.Process();
}
extern "C" __global__ __aicore__ void asc_add_flat_fp32_32x256(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
    KernelEltwiseFlat<float, 32, 256> op; op.Init(x, y, z); op.Process();
}
extern "C" __global__ __aicore__ void asc_add_flat_fp16_1x16384(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
    KernelEltwiseFlat<half, 1, 16384> op; op.Init(x, y, z); op.Process();
}
extern "C" __global__ __aicore__ void asc_add_flat_fp16_32x512(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
    KernelEltwiseFlat<half, 32, 512> op; op.Init(x, y, z); op.Process();
}
extern "C" __global__ __aicore__ void asc_add_flat_fp16_64x256(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
    KernelEltwiseFlat<half, 64, 256> op; op.Init(x, y, z); op.Process();
}

// --- ROWLOOP variants ---
extern "C" __global__ __aicore__ void asc_add_rowloop_fp32_16x512(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
    KernelEltwiseRowLoop<float, 16, 512> op; op.Init(x, y, z); op.Process();
}
extern "C" __global__ __aicore__ void asc_add_rowloop_fp32_32x256(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
    KernelEltwiseRowLoop<float, 32, 256> op; op.Init(x, y, z); op.Process();
}
extern "C" __global__ __aicore__ void asc_add_rowloop_fp16_32x512(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
    KernelEltwiseRowLoop<half, 32, 512> op; op.Init(x, y, z); op.Process();
}
extern "C" __global__ __aicore__ void asc_add_rowloop_fp16_64x256(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
    KernelEltwiseRowLoop<half, 64, 256> op; op.Init(x, y, z); op.Process();
}
