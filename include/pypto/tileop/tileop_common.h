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
 * \file tileop_common.h
 * \brief
 */

#ifndef __LOGICALTENSOR_TILEOP_COMMON__
#define __LOGICALTENSOR_TILEOP_COMMON__

#ifdef SUPPORT_TILE_TENSOR
#include "pto/pto-inst.hpp"
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#ifndef __aicore_host__
#define __aicore_host__ [host, aicore]
#endif

#ifndef TILEOP
#define TILEOP static __attribute__((always_inline))[aicore]
#endif

#ifndef INLINE
#define INLINE __attribute__((always_inline)) inline[aicore]
#endif

#ifndef CORELOG
#define CORELOG(x...)
#endif

#define SUBKERNEL_PHASE1
#define SUBKERNEL_PHASE2

#if defined(__DAV_C220_VEC__)
#define WAIT_TASK_FIN                       \
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7); \
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7)
#else
#define WAIT_TASK_FIN                      \
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7); \
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7)
#endif

#if defined(__DAV_C220_VEC__)
#define WAIT_PRE_TASK                          \
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7); \
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7)
#else
#define WAIT_PRE_TASK                          \
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7); \
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7)
#endif

enum class CopyInMode : int64_t
{
    ND2ND = 0,
    ND2NZ = 1,
    NZ2NZ = 2
};

enum class CopyOutMode : int64_t
{
    NZ2ND = 0,
    NZ2NZ = 1
};

enum class ReLuType : int64_t
{
    NoReLu = 0,
    ReLu = 1
};

namespace TileOp {
enum CastMode {
    CAST_NONE = 0,
    CAST_RINT = 1,  // round to nearest, tie to even
    CAST_ROUND = 2, // round to nearest, tie away from zero
    CAST_FLOOR = 3, // round to minus infinity
    CAST_CEIL = 4,  // round to positive infinity
    CAST_TRUNC = 5, // round to zero
    CAST_ODD = 6,   // round to odd (Von Neumann rounding)
};

enum BroadcastOperand : int64_t {
    NONE          = 0,
    LEFT_OPERAND  = 1,
    RIGHT_OPERAND = 2,
};

constexpr uint64_t MASK_LEN = 64;
constexpr uint64_t BLOCK_NELEM_B16 = 16;
constexpr uint64_t BLOCK_NELEM_B32 = 8;
constexpr uint64_t NBLOCK_PER_MASK_B16 = 4;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t REPEAT_MAX = 255;
constexpr uint64_t REPEAT_BYTE = 256;
constexpr uint64_t REPEAT_STRIDE_MAX = 255;
constexpr uint64_t DUP_REPEAT_STRIDE_MAX = 4095;
constexpr uint64_t BLOCK_NUM_ONE_REPEAT = 8;

inline TILEOP void SetContinuousMask(unsigned n) {
    set_vector_mask(static_cast<uint64_t>(
                        (n > MASK_LEN) ? (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(n - MASK_LEN)) - 1) : 0),
        static_cast<uint64_t>(
            (n >= MASK_LEN) ? 0xffffffffffffffff : (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(n)) - 1)));
}

// Calculation linear offset for multi-dimension tensor
INLINE unsigned CalcLinearOffset(unsigned GmShape1, unsigned GmShape2, unsigned GmShape3, unsigned GmShape4,
    unsigned Offset0, unsigned Offset1, unsigned Offset2, unsigned Offset3, unsigned Offset4) {
    return Offset4 + Offset3 * GmShape4 + Offset2 * (GmShape3 * GmShape4) + Offset1 * (GmShape2 * GmShape3 * GmShape4) +
           Offset0 * (GmShape1 * GmShape2 * GmShape3 * GmShape4);
}

// Calculation linear offset for multi-dimension tensor
INLINE unsigned CalcLinearOffset(unsigned GmShape1, unsigned GmShape2, unsigned GmShape3, unsigned Offset0,
    unsigned Offset1, unsigned Offset2, unsigned Offset3) {
    return Offset3 + Offset2 * GmShape3 + Offset1 * (GmShape2 * GmShape3) + Offset0 * (GmShape1 * GmShape2 * GmShape3);
}

// Calculation linear offset for multi-dimension tensor
INLINE unsigned CalcLinearOffset(
    unsigned GmShape1, unsigned GmShape2, unsigned Offset0, unsigned Offset1, unsigned Offset2) {
    return Offset2 + Offset1 * GmShape2 + Offset0 * (GmShape1 * GmShape2);
}

// Calculation linear offset for multi-dimension tensor
INLINE unsigned CalcLinearOffset(unsigned GmShape1, unsigned Offset0, unsigned Offset1) {
    return Offset1 + Offset0 * GmShape1;
}
}

#endif
