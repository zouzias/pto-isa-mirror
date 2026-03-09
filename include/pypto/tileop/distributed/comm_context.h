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
 * \file comm_context.h
 * \brief
 */

#ifndef __COMM_CONTEXT__
#define __COMM_CONTEXT__

#include <type_traits>

namespace TileOp {
struct CommContext {
    uint64_t rankId = 0; // 当前卡rankId
    uint64_t rankNum = 0;
    int64_t startIndex = 0; // 每个win区起始Index
    int64_t statusIndex = -1; 
    int64_t debugIndex = -1; 
    uint64_t winDataSize = 0; // 每个win区大小
    uint64_t winStatusSize = 0; 
    uint64_t winDebugSize = 0; 
    uint64_t totalWinNum = 0; 
    uint64_t winAddr[0]; // size大小rankNum*3，内存排布windata[0~rankNum-1], winStatus[0~rankNum-1], winDebug[0~rankNum-1]
};
} // namespace TileOp

#endif