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
 * \file aikernel_runtime.h
 * \brief
 */

#ifndef AIKERNEL_RUNTIME_H
#define AIKERNEL_RUNTIME_H

#include "tilefwk/aikernel_data.h"

constexpr int MAIN_BLOCK_INDEX = 1;

#ifdef __TILE_FWK_AICPU__

#define RuntimeGetStartArgs()                   startArgs
#define RuntimeGetSymbol(idx)                   (symbolTable[idx])

#else

#define RuntimeGetStartArgs()                   AiCoreRuntimeGetStartArgs(param)
#define RuntimeGetSymbol(idx)                   (param->exprTbl[(idx) + MAIN_BLOCK_INDEX])  // inserted the mainBlock expression at position 0 of the expressionSet

#endif

#define RuntimeGetInputShapeDimSize(input) \
    ((input)->shape.dimSize)
#define RuntimeGetInputShapeDim(input, n) \
    ((input)->shape.dim[(n)])
#define RuntimeGetInputDataInt32Dim1(input, off0) \
    (((int32_t *)(input)->address)[(off0)])
#define RuntimeGetInputDataInt32Dim2(input, off0, off1) \
    (((int32_t *)(input)->address)[(off0) * (input)->shape.dim[1] + (off1)])
#define RuntimeGetInputDataInt32Dim3(input, off0, off1, off2) \
    (((int32_t *)(input)->address)[(off0) * (input)->shape.dim[1] * (input)->shape.dim[2] + (off1) * (input)->shape.dim[2] + (off2)])
#define RuntimeGetInputDataInt32Dim4(input, off0, off1, off2, off3) \
    (((int32_t *)(input)->address)[(((off0) * (input)->shape.dim[1] + (off1)) * (input)->shape.dim[2] + (off2)) * (input)->shape.dim[3] + (off3)])

#define RUNTIME_GetInputShapeDimSize(inputIndex) \
    RuntimeGetInputShapeDimSize(&(RuntimeGetStartArgs())->devTensorList[(inputIndex)])
#define RUNTIME_GetInputShapeDim(inputIndex, n) \
    RuntimeGetInputShapeDim(&(RuntimeGetStartArgs())->devTensorList[(inputIndex)], (n))
#define RUNTIME_GetInputDataInt32Dim1(inputIndex, off0) \
    RuntimeGetInputDataInt32Dim1(&(RuntimeGetStartArgs())->devTensorList[(inputIndex)], (off0))
#define RUNTIME_GetInputDataInt32Dim2(inputIndex, off0, off1) \
    RuntimeGetInputDataInt32Dim2(&(RuntimeGetStartArgs())->devTensorList[(inputIndex)], (off0), (off1))
#define RUNTIME_GetInputDataInt32Dim3(inputIndex, off0, off1, off2) \
    RuntimeGetInputDataInt32Dim3(&(RuntimeGetStartArgs())->devTensorList[(inputIndex)], (off0), (off1), (off2))
#define RUNTIME_GetInputDataInt32Dim4(inputIndex, off0, off1, off2, off3) \
    RuntimeGetInputDataInt32Dim4(&(RuntimeGetStartArgs())->devTensorList[(inputIndex)], (off0), (off1), (off2), (off3))
#define RUNTIME_GetSymbol(idx) \
    RuntimeGetSymbol(idx)

#endif
