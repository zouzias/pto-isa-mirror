/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aikernel_data.h
 * \brief
 */

#ifndef AIKERNEL_DATA_H
#define AIKERNEL_DATA_H

#include "tilefwk/aikernel_define.h"

struct LogContext;

namespace npu::tile_fwk {

constexpr uint32_t HCCL_GROUP_NUM = 2;
const uint32_t RAW_TENSOR_LOCATION_LOCAL = 0;
const uint32_t RAW_TENSOR_LOCATION_INCAST = 1;
const uint32_t RAW_TENSOR_LOCATION_OUTCAST = 2;
constexpr int32_t DEV_SHAPE_DIM_MAX = 5;
constexpr uint32_t TENSOR_INFO_OFFSET = 2;

struct DevShape {
    int dimSize{0};
    int dim[DEV_SHAPE_DIM_MAX];

#ifdef __TILE_FWK_HOST__
    int64_t GetSize() const {
        int64_t size = 1;
        for (int idx = 0; idx < dimSize; idx++) {
            size *= dim[idx];
        }
        return size;
    }

    bool Equal(const DevShape &s) const {
        if (dimSize != s.dimSize) {
            return false;
        }
        for (int i = 0; i < dimSize; i++) {
            if (dim[i] != s.dim[i]) {
                return false;
            }
        }
        return true;
    }
#endif
};

struct DevTensorData {
    uint64_t address{0};
    DevShape shape;
};

struct DevStartArgsBase {
    __gm__ DevTensorData *devTensorList;
    uint64_t inputTensorSize;
    uint64_t outputTensorSize;
    __gm__ int64_t *commContexts;
    uint64_t commGroupNum;

#ifdef __TILE_FWK_HOST__
    int GetInputTensorSize() const { return inputTensorSize; }
    const DevTensorData &GetInputTensor(int index) const { return devTensorList[index]; }
    DevTensorData &GetInputTensor(int index) { return devTensorList[index]; }

    int GetOutputTensorSize() const { return outputTensorSize; }
    const DevTensorData &GetOutputTensor(int index) const { return devTensorList[index + inputTensorSize]; }
    DevTensorData &GetOutputTensor(int index) { return devTensorList[index + inputTensorSize]; }
#endif
};

struct DevRawTensorDesc {
    uint32_t location;
    uint32_t offsetOrIndex;
};

struct DynFuncData {
    uint64_t exprNum;               // static
    __gm__ uint64_t *opAttrs;       // static
    __gm__ int32_t *opAtrrOffsets;  // static
    __gm__ uint64_t *exprTbl;       // dyn
    __gm__ DevRawTensorDesc *rawTensorDesc;
    __gm__ uint64_t *rawTensorAddr;
    uint64_t opAttrSize;
    uint64_t rawTensorDescSize;
    uint64_t rawTensorAddrSize;
    uint64_t workspaceAddr;
    uint64_t stackWorkSpaceAddr;
    uint64_t stackWorkSpaceSize;
    __gm__ DevStartArgsBase *startArgs;
};

struct DynFuncBin {
    uint32_t coreType;
    uint32_t psgId;
    uint64_t funcHash;
    int32_t wrapVecId {-1};
    uint32_t mixResourceType {0};
};

struct DynFuncHeader {
    uint64_t seqNo;
    uint32_t funcNum;
    uint32_t funcSize;
    __gm__ DynFuncBin *cceBinary;

    INLINE uint64_t GetIndex() {
        return seqNo;
    }

    INLINE DynFuncData &At(int index) {
        return (reinterpret_cast<DynFuncData *>(this + 1))[index];
    }
    INLINE uint32_t Size() {
        return funcNum;
    }
};

struct CoreFuncParam {
    __gm__ npu::tile_fwk::DynFuncData *funcData;
    __gm__ uint64_t *opAttrs;
    __gm__ uint64_t *exprTbl;
    uint32_t taskId;
    LogContext *ctx;
};

#define TASKID_TASK_BITS                        20
#define TASKID_TASK_MASK                        ((1 << TASKID_TASK_BITS) - 1)

#define TASKID_FUNC_BITS                        11
#define TASKID_FUNC_MASK                        ((1 << TASKID_FUNC_BITS) - 1)

#define TASKID_SHIFT32                          32

INLINE uint32_t FuncID(uint32_t taskId) {
    return taskId >> TASKID_TASK_BITS;
}

INLINE uint32_t TaskID(uint32_t taskId) {
    return taskId & TASKID_TASK_MASK;
}

INLINE uint32_t MakeTaskID(uint32_t rootId, uint32_t leafId) {
    return (rootId << TASKID_TASK_BITS) | leafId;
}

}

#endif
