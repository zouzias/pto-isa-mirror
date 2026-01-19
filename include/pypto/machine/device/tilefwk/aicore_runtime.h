/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aicore_runtime.h
 * \brief
 */

#ifndef AICORE_RUNTIME_H
#define AICORE_RUNTIME_H

#include <cstdint>

#include "aicore_data.h"
#include <pypto/tileop/distributed/hccl_context.h>

#define CACHELINE_SIZE_FOR_B32 128
#define CACHELINE_SIZE_FOR_B64 64
#define DEFAULT_TOTAL_BLOCK_NUM 75
#define DEBUG_OFFSET_FOR_B64 DEFAULT_TOTAL_BLOCK_NUM *CACHELINE_SIZE_FOR_B64 / sizeof(int64_t)
#define DEBUG_SIZE_PER_CORE (1 * 1024 * 1024)
#define PAD_LIMIT 512

const int SHAKE_SAY_HELLO = 100;
const int SHAKE_HELLO_ACK = 200;

struct RealizedVar {
    __gm__ void *Addr;
    int64_t offset0; // NEXTNEXT: should divide by TILESIZE or not? E.g . 128 or 128/128
    int64_t offset1;
};

struct GMTensorInfo {
    uint64_t Addr;
};

template <typename T>
struct IOCastInfo {
    __gm__ T *Addr;
    int64_t Size;
};

// | GmCount | IncastCount | OutcastCount | GmArrary | IncastArray | OutcastArray |

struct InvokeEntry {
    int64_t SubGraphProgramId;
    uint64_t gmCount;
    uint64_t incastCount;
    uint64_t outcastCount;
};

template <unsigned GRAPH_INVOKE_COUNT>
struct GraphInvokeInfo {
    uint64_t GraphInvokeCount{GRAPH_INVOKE_COUNT};
    uint64_t GraphInvokeOffset[GRAPH_INVOKE_COUNT];
};

template <typename T, unsigned SIZE>
struct RingBuffer {
    T elements[SIZE];
    uint64_t MAX_SIZE{SIZE};
    char pad1[PAD_LIMIT - SIZE * sizeof(T) - 1 * sizeof(uint64_t)];
    int64_t front{0};
    char pad2[PAD_LIMIT - 1 * sizeof(int64_t)];
    int64_t rear{0};
    char pad3[PAD_LIMIT - 1 * sizeof(int64_t)];
};

template <typename T, unsigned SIZE>
INLINE void InitRingBuffer(RingBuffer<T, SIZE> *Q) {
    Q->front = 0;
    Q->rear = 0;
    Q->MAX_SIZE = SIZE;
}

// Enqueue:
template <typename T, unsigned SIZE>
INLINE bool EnQueue(volatile __gm__ RingBuffer<T, SIZE> *Q, T value) {
    dcci(Q, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    if (Q->rear - Q->front == Q->MAX_SIZE) {
        return false;
    }
    Q->elements[Q->rear % Q->MAX_SIZE] = value;
    dsb((mem_dsb_t)0);
    Q->rear = Q->rear + 1;
    dcci(Q, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    return true;
}

// Enqueue:
template <typename T, unsigned SIZE>
INLINE bool EnQueuePrivate(volatile RingBuffer<T, SIZE> *Q, T value) {
    if ((Q->rear + 1) % Q->MAX_SIZE == Q->front) {
        return false;
    }
    Q->elements[Q->rear] = value;
    dsb((mem_dsb_t)0);
    Q->rear = (Q->rear + 1) % Q->MAX_SIZE;
    return true;
}

// EnQueueLocalToGM:
template <typename T, unsigned SIZE>
INLINE bool EnQueueLocalToGM(volatile __gm__ RingBuffer<T, SIZE> *Q, T value) {
    if ((Q->rear + 1) % Q->MAX_SIZE == Q->front) {
        return false;
    }
    Q->elements[Q->rear] = value;
    dsb((mem_dsb_t)0);
    Q->rear = (Q->rear + 1) % Q->MAX_SIZE;
    dcci(Q, 0);
    return true;
}

// dequeue:
template <typename T, unsigned SIZE>
INLINE bool DeQueue(volatile __gm__ RingBuffer<T, SIZE> *Q, T &value) {
    dcci(Q, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    if (Q->front == Q->rear) {
        return false;
    }
    value = Q->elements[Q->front % Q->MAX_SIZE];
    dsb((mem_dsb_t)0);
    Q->front = Q->front + 1;
    dcci(Q, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    return true;
}

// dequeue:
template <typename T, unsigned SIZE>
INLINE volatile T *DeQueuePrivate(volatile RingBuffer<T, SIZE> *Q) {
    if (Q->front == Q->rear) {
        return nullptr;
    }
    volatile T *ret = &(Q->elements[Q->front]);
    Q->front = (Q->front + 1) % Q->MAX_SIZE;
    return ret;
}

// Peek:
template <typename T, unsigned SIZE>
INLINE __gm__ T *Peek(volatile __gm__ RingBuffer<T, SIZE> *Q) {
    if (Q->front == Q->rear) {
        return nullptr;
    }
    __gm__ T *ret = &(Q->elements[Q->front]);
    return ret;
}

// dequeue:
template <typename T, unsigned SIZE>
INLINE T *PeekPrivate(volatile RingBuffer<T, SIZE> *Q) {
    if (Q->front == Q->rear) {
        return nullptr;
    }
    T *ret = &(Q->elements[Q->front]);
    return ret;
}

template <typename T, unsigned SIZE>
INLINE bool IsFull(volatile __gm__ RingBuffer<T, SIZE> *Q) {
    dcci(Q, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    return (Q->rear - Q->front) == Q->MAX_SIZE;
}

template <typename T, unsigned SIZE>
INLINE bool IsFullPrivate(volatile RingBuffer<T, SIZE> *Q) {
    return (Q->rear + 1) % Q->MAX_SIZE == Q->front;
}

template <typename T, unsigned SIZE>
INLINE bool IsEmpty(volatile __gm__ RingBuffer<T, SIZE> *Q) {
    dcci(Q, 0);
    return Q->front == Q->rear;
}

template <typename T, unsigned SIZE>
INLINE bool IsEmptyPrivate(volatile RingBuffer<T, SIZE> *Q) {
    return Q->front == Q->rear;
}

template <typename T, unsigned SIZE>
INLINE uint64_t GetLength(volatile __gm__ RingBuffer<T, SIZE> *Q) {
    dcci(Q, 0);
    return (Q->rear - Q->front + Q->MAX_SIZE) % Q->MAX_SIZE;
}

template <typename T, unsigned SIZE>
INLINE uint64_t GetLengthPrivate(volatile RingBuffer<T, SIZE> *Q) {
    return (Q->rear - Q->front + Q->MAX_SIZE) % Q->MAX_SIZE;
}

struct LogContext;

struct CoreFuncParam {
    __gm__ npu::tile_fwk::DynFuncData *funcData;
    __gm__ uint64_t *opAttrs;
    __gm__ uint64_t *exprTbl;
    uint32_t taskId;
    LogContext *ctx;
};

#define TASKID_TASK_BITS 20
#define FuncID(id)       (id >> TASKID_TASK_BITS)
#define TaskID(id)       (id & ((1 << TASKID_TASK_BITS) - 1))

#define SYM_VALUE_LEN 63
#define SYM_VALUE_MASK ((1UL << SYM_VALUE_LEN) - 1)
#define SYM_IS_EXPR(val) (val & (1UL << SYM_VALUE_LEN))
#define SYM_VALUE(val) (val & SYM_VALUE_MASK)

#define RAW_TENSOR_ADDR_MASK ((1UL << 63) - 1)

INLINE __gm__ npu::tile_fwk::DevStartArgsBase *RuntimeGetStartArgs(CoreFuncParam *param) {
    auto func = param->funcData;
    auto startArgs = func->startArgs;
    return startArgs;
}

INLINE uint64_t GetTensorAddr(CoreFuncParam *ctx, int idx) {
    auto func = ctx->funcData;
    auto desc = &func->rawTensorDesc[ctx->opAttrs[idx]];
    if (desc->location == npu::tile_fwk::RAW_TENSOR_LOCATION_LOCAL)
        return func->workspaceAddr + desc->offsetOrIndex;
    else
        return func->rawTensorAddr[desc->offsetOrIndex] & RAW_TENSOR_ADDR_MASK ;
}

INLINE uint64_t GetCoa(CoreFuncParam *ctx, int idx) {
    uint64_t val = ctx->opAttrs[idx];
    if (SYM_IS_EXPR(val))
        return ctx->exprTbl[SYM_VALUE(val)];
    else
        return SYM_VALUE(val);
}

INLINE
int64_t RuntimeGetViewValidShapeDim(int64_t validshape, int64_t viewOffset, int64_t viewshape) {
    validshape -= viewOffset;
    if (validshape > viewshape)
        validshape = viewshape;
    else if (validshape < 0)
        validshape = 0;
    return validshape;
}

INLINE uint64_t GetShmemTensorAddr(CoreFuncParam *ctx, int idx, int groupIndex, uint64_t offset) {
    auto dstRankId = GetCoa(ctx, idx + 1);
    return ((__gm__ TileOp::HcclCombinOpParam *)ctx->funcData->hcclContext[groupIndex])->windowsIn[dstRankId] + offset;
}

#define RUNTIME_GetViewValidShapeDim(validShape, viewOffset, viewShape) RuntimeGetViewValidShapeDim(validShape, viewOffset, viewShape)

#define GET_PARAM_ADDR(param, n, base) GetTensorAddr(param, base)
#define GET_SHMEM_ADDR(param, n, base, group, offset) GetShmemTensorAddr(param, base, group, offset)

#define GET_PARAM_OFFSET_BY_IDX(param, n, base, dim, idx)         GetCoa(param, ((base) + 1) + 0 * (dim) + idx)
#define GET_PARAM_SHAPE_BY_IDX(param, n, base, dim, idx)          GetCoa(param, ((base) + 1) + 1 * (dim) + idx)
#define GET_PARAM_RAWSHAPE_BY_IDX(param, n, base, dim, idx)       GetCoa(param, ((base) + 1) + 2 * (dim) + idx)
#define GET_PARAM_VALID_SHAPE_BY_IDX(param, n, base, dim, idx)    GetCoa(param, ((base) + 1) + 3 * (dim) + idx)

#define GET_PARAM_ATTR_1(name, param, n, base)  GET_PARAM_##name##_BY_IDX(param, n, base, 1, 0)
#define GET_PARAM_ATTR_2(name, param, n, base)  GET_PARAM_##name##_BY_IDX(param, n, base, 2, 0), GET_PARAM_##name##_BY_IDX(param, n, base, 2, 1)
#define GET_PARAM_ATTR_3(name, param, n, base)  GET_PARAM_##name##_BY_IDX(param, n, base, 3, 0), GET_PARAM_##name##_BY_IDX(param, n, base, 3, 1), \
                                                  GET_PARAM_##name##_BY_IDX(param, n, base, 3, 2)
#define GET_PARAM_ATTR_4(name, param, n, base)  GET_PARAM_##name##_BY_IDX(param, n, base, 4, 0), GET_PARAM_##name##_BY_IDX(param, n, base, 4, 1), \
                                                  GET_PARAM_##name##_BY_IDX(param, n, base, 4, 2), GET_PARAM_##name##_BY_IDX(param, n, base, 4, 3)
#define GET_PARAM_ATTR_5(name, param, n, base)  GET_PARAM_##name##_BY_IDX(param, n, base, 5, 0), GET_PARAM_##name##_BY_IDX(param, n, base, 5, 1), \
                                                  GET_PARAM_##name##_BY_IDX(param, n, base, 5, 2), GET_PARAM_##name##_BY_IDX(param, n, base, 5, 3), \
                                                  GET_PARAM_##name##_BY_IDX(param, n, base, 5, 4)

#define GET_PARAM_ATTR_2_STRIDE(name, param, n, base) GET_PARAM_##name##_BY_IDX(param, n, base, 2, 1), 1
#define GET_PARAM_ATTR_3_STRIDE(name, param, n, base)                                                  \
    GET_PARAM_##name##_BY_IDX(param, n, base, 3, 1) * GET_PARAM_##name##_BY_IDX(param, n, base, 3, 2), \
        GET_PARAM_##name##_BY_IDX(param, n, base, 3, 2), 1
#define GET_PARAM_ATTR_4_STRIDE(name, param, n, base)                                                      \
    GET_PARAM_##name##_BY_IDX(param, n, base, 4, 1) * GET_PARAM_##name##_BY_IDX(param, n, base, 4, 2) *    \
        GET_PARAM_##name##_BY_IDX(param, n, base, 4, 3),                                                   \
        GET_PARAM_##name##_BY_IDX(param, n, base, 4, 2) * GET_PARAM_##name##_BY_IDX(param, n, base, 4, 3), \
        GET_PARAM_##name##_BY_IDX(param, n, base, 4, 3), 1
#define GET_PARAM_ATTR_5_STRIDE(name, param, n, base)                                                       \
    GET_PARAM_##name##_BY_IDX(param, n, base, 5, 1) * GET_PARAM_##name##_BY_IDX(param, n, base, 5, 2) *     \
        GET_PARAM_##name##_BY_IDX(param, n, base, 5, 3) * GET_PARAM_##name##_BY_IDX(param, n, base, 5, 4),  \
        GET_PARAM_##name##_BY_IDX(param, n, base, 5, 2) * GET_PARAM_##name##_BY_IDX(param, n, base, 5, 3) * \
            GET_PARAM_##name##_BY_IDX(param, n, base, 5, 4),                                                \
        GET_PARAM_##name##_BY_IDX(param, n, base, 5, 3) * GET_PARAM_##name##_BY_IDX(param, n, base, 5, 4),  \
        GET_PARAM_##name##_BY_IDX(param, n, base, 5, 4), 1

#define GET_PARAM_OFFSET_1(param, n, base) GET_PARAM_ATTR_1(OFFSET, param, n, base)
#define GET_PARAM_SHAPE_1(param, n, base)  GET_PARAM_ATTR_1(SHAPE, param, n, base)
#define GET_PARAM_RAWSHAPE_1(param, n, base) GET_PARAM_ATTR_1(RAWSHAPE, param, n, base)
#define GET_PARAM_STRIDE_1(param, n, base) 1

#define GET_PARAM_OFFSET_2(param, n, base) GET_PARAM_ATTR_2(OFFSET, param, n, base)
#define GET_PARAM_SHAPE_2(param, n, base)  GET_PARAM_ATTR_2(SHAPE, param, n, base)
#define GET_PARAM_RAWSHAPE_2(param, n, base) GET_PARAM_ATTR_2(RAWSHAPE, param, n, base)
#define GET_PARAM_STRIDE_2(param, n, base) GET_PARAM_ATTR_2_STRIDE(RAWSHAPE, param, n, base)

#define GET_PARAM_OFFSET_3(param, n, base) GET_PARAM_ATTR_3(OFFSET, param, n, base)
#define GET_PARAM_SHAPE_3(param, n, base)  GET_PARAM_ATTR_3(SHAPE, param, n, base)
#define GET_PARAM_RAWSHAPE_3(param, n, base) GET_PARAM_ATTR_3(RAWSHAPE, param, n, base)
#define GET_PARAM_STRIDE_3(param, n, base) GET_PARAM_ATTR_3_STRIDE(RAWSHAPE, param, n, base)

#define GET_PARAM_OFFSET_4(param, n, base) GET_PARAM_ATTR_4(OFFSET, param, n, base)
#define GET_PARAM_SHAPE_4(param, n, base)  GET_PARAM_ATTR_4(SHAPE, param, n, base)
#define GET_PARAM_RAWSHAPE_4(param, n, base) GET_PARAM_ATTR_4(RAWSHAPE, param, n, base)
#define GET_PARAM_STRIDE_4(param, n, base) GET_PARAM_ATTR_4_STRIDE(RAWSHAPE, param, n, base)

#define GET_PARAM_OFFSET_5(param, n, base) GET_PARAM_ATTR_5(OFFSET, param, n, base)
#define GET_PARAM_SHAPE_5(param, n, base)  GET_PARAM_ATTR_5(SHAPE, param, n, base)
#define GET_PARAM_RAWSHAPE_5(param, n, base) GET_PARAM_ATTR_5(RAWSHAPE, param, n, base)
#define GET_PARAM_STRIDE_5(param, n, base) GET_PARAM_ATTR_5_STRIDE(RAWSHAPE, param, n, base)

INLINE uint64_t RUNTIME_Min(uint64_t input1, uint64_t input2) {
    return input1 < input2 ? input1 : input2;
}

INLINE uint64_t RUNTIME_Max(uint64_t input1, uint64_t input2) {
    return input1 > input2 ? input1 : input2;
}

INLINE uint64_t RUNTIME_Eq(uint64_t input1, uint64_t input2) {
    return input1 == input2;
}

INLINE uint64_t RUNTIME_Ne(uint64_t input1, uint64_t input2) {
    return input1 != input2;
}

INLINE uint32_t GetTensorDataInt32(CoreFuncParam *ctx, uint64_t address) {
    dcci((__gm__ uint32_t *)address, ENTIRE_DATA_CACHE, CACHELINE_OUT); 
    return *(__gm__ uint32_t *)(address);
}
#define RUNTIME_GetTensorDataInt32Dim1(index, ioType, ioTypeIndex, address, ...)    GetTensorDataInt32(param, address)
#define RUNTIME_GetTensorDataInt32Dim2(index, ioType, ioTypeIndex, address, ...)    GetTensorDataInt32(param, address)
#define RUNTIME_GetTensorDataInt32Dim3(index, ioType, ioTypeIndex, address, ...)    GetTensorDataInt32(param, address)
#define RUNTIME_GetTensorDataInt32Dim4(index, ioType, ioTypeIndex, address, ...)    GetTensorDataInt32(param, address)
#define RUNTIME_GetTensorDataInt32Dim5(index, ioType, ioTypeIndex, address, ...)    GetTensorDataInt32(param, address)

#define RuntimeGetInputShapeDim(input, n) ((input)->shape.dim[(n)])
#define RUNTIME_GetInputShapeDim(inputIndex, n) RuntimeGetInputShapeDim(&(RuntimeGetStartArgs(param))->devTensorList[(inputIndex)], (n))

#define RUNTIME_COA_GET_PARAM_OFFSET(dim, base, idx)                                GET_PARAM_OFFSET_BY_IDX(param, 0, base, dim, idx)
#define RUNTIME_COA_GET_PARAM_VALID_SHAPE(dim, base, idx)                           GET_PARAM_VALID_SHAPE_BY_IDX(param, 0, base, dim, idx)
#define RUNTIME_COA_GET_PARAM_ADDR(_, idx)                                          GET_PARAM_ADDR(param, _, idx)
#define RUNTIME_COA_GET_PARAM(idx)                                                  GetCoa(param, idx)

#define RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST_0(value, dim, base, idx)           RUNTIME_COA_GET_PARAM_OFFSET(dim, base, idx)
#define RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST_1(value, dim, base, idx)           value
#define RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(isConst, value, dim, base, idx)    RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST_##isConst(value, dim, base, idx)

#define RUNTIME_COA_GET_PARAM_VALID_SHAPE_MAYBE_CONST_0(value, dim, base, idx)           RUNTIME_COA_GET_PARAM_VALID_SHAPE(dim, base, idx)
#define RUNTIME_COA_GET_PARAM_VALID_SHAPE_MAYBE_CONST_1(value, dim, base, idx)           value
#define RUNTIME_COA_GET_PARAM_VALID_SHAPE_MAYBE_CONST(isConst, value, dim, base, idx)    RUNTIME_COA_GET_PARAM_VALID_SHAPE_MAYBE_CONST_##isConst(value, dim, base, idx)

#define RUNTIME_COA_GET_PARAM_MAYBE_CONST_0(value, idx)           RUNTIME_COA_GET_PARAM(idx)
#define RUNTIME_COA_GET_PARAM_MAYBE_CONST_1(value, idx)           value
#define RUNTIME_COA_GET_PARAM_MAYBE_CONST(isConst, value, idx)    RUNTIME_COA_GET_PARAM_MAYBE_CONST_##isConst(value, idx)

#define RuntimeGetInputDataInt32Dim1(input, off0) (((int32_t *)(input)->address)[(off0)])
#define RuntimeGetInputDataInt32Dim2(input, off0, off1) \
    (((int32_t *)(input)->address)[(off0) * (input)->shape.dim[1] + (off1)])
#define RuntimeGetInputDataInt32Dim3(input, off0, off1, off2) \
    (((int32_t *)(input)->address)[(off0) * (input)->shape.dim[1] * (input)->shape.dim[2] + (off1) * (input)->shape.dim[2] + (off2)])
#define RuntimeGetInputDataInt32Dim4(input, off0, off1, off2, off3) \
    (((int32_t *)(input)->address)[(((off0) * (input)->shape.dim[1] + (off1)) * (input)->shape.dim[2] + (off2)) * (input)->shape.dim[3] + (off3)])

#define RUNTIME_GetInputDataInt32Dim1(inputIndex, off0) \
    RuntimeGetInputDataInt32Dim1(&(RuntimeGetStartArgs(param))->devTensorList[(inputIndex)], (off0))
#define RUNTIME_GetInputDataInt32Dim2(inputIndex, off0, off1) \
    RuntimeGetInputDataInt32Dim2(&(RuntimeGetStartArgs(param))->devTensorList[(inputIndex)], (off0), (off1))
#define RUNTIME_GetInputDataInt32Dim3(inputIndex, off0, off1, off2) \
    RuntimeGetInputDataInt32Dim3(&(RuntimeGetStartArgs(param))->devTensorList[(inputIndex)], (off0), (off1), (off2))
#define RUNTIME_GetInputDataInt32Dim4(inputIndex, off0, off1, off2, off3) \
    RuntimeGetInputDataInt32Dim4(&(RuntimeGetStartArgs(param))->devTensorList[(inputIndex)], (off0), (off1), (off2), (off3))

#define RUNTIME_TensorExtract(type, mem, dst, src) \
    do { \
        pipe_barrier(PIPE_ALL); \
        *(mem type *)(dst) = *(mem type *)(src); \
        pipe_barrier(PIPE_ALL); \
    } while(0)

#define RUNTIME_GetSymbol(idx)          (param->exprTbl[idx])

#endif // AST_RUNTIME_H
