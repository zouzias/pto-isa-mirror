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
 * \file common_type.h
 * \brief
 */

#ifndef TILEOP_UTILS_COMMON_TYPE_H
#define TILEOP_UTILS_COMMON_TYPE_H

enum class Hardware : uint8_t { GM = 0, UB, L1, L0A, L0B, L0C, BIAS, FIXBUF, MAX };

enum class UnaryOp : uint8_t { ABS = 0, EXP, NEG, REC, RSQRT, SQRT, BRCB };

enum class BinaryOp : uint8_t { ADD = 0, SUB, MUL, DIV, AND, OR, MAX, MIN, SUM, AMAX};

enum class BroadcastOperand : uint8_t { NONE = 0, LEFT, RIGHT };

enum class PairBinaryOp : uint8_t { ADD = 0, MAX, MIN };

enum class ReduceOp : uint8_t { SUM = 0, MAX, MIN};

enum class BinaryScalarOp : uint8_t { ADD = 0, SUB, MUL, DIV };
#endif // TILEOP_UTILS_COMMON_TYPE_H
