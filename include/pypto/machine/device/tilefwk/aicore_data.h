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
 * \file aicore_data.h
 * \brief
 */

#ifndef AICORE_DATA_H
#define AICORE_DATA_H

#ifndef __gm__
#define __gm__
#define __aicore__
#define INLINE inline
#define __TILE_FWK_HOST__
#else
#define __aicore__ [aicore]
#define INLINE __attribute__((always_inline)) inline __aicore__
#endif

namespace npu::tile_fwk {

const uint32_t HCCL_GROUP_NUM = 2;
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
    uint64_t *hcclContextAddr;

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
    uint64_t hcclContext[HCCL_GROUP_NUM]{0, 0};
    uint64_t commGroupNum{0};
    __gm__ DevStartArgsBase *startArgs;
};
}

#endif
