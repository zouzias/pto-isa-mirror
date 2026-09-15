/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_COSTMODEL_TILEOP_DISPATCH_HPP
#define PTO_COSTMODEL_TILEOP_DISPATCH_HPP

#include "pto/costmodel/common/tileop_runtime.hpp"

#define TSTORE_FP_IMPL TSTORE_IMPL
#define TEXTRACT_FP_IMPL TEXTRACT_IMPL
#define TINSERT_FP_IMPL TINSERT_IMPL
#define TMOV_FP_IMPL TMOV_IMPL

#define PTO_FIRST_ARG(first, ...) first
#define PTO_SECOND_ARG(_first, second, ...) second
#define PTO_TEMPLATE_ARGS(...) <__VA_ARGS__>

#if defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRINX90) || defined(PTO_NPU_ARCH_KIRINDEV0000)
#define PTO_FORWARD_L2HINT_TO_IMPL 0
#else
#define PTO_FORWARD_L2HINT_TO_IMPL 1
#endif
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 3510))
#include "pto/costmodel/a5/tileop_dispatch.hpp"
#else
#define MAP_INSTR_IMPL(API, ...)                                     \
    do {                                                             \
        ::pto::mocker::PtoInstrScope _scope(#API);                   \
        API##_IMPL(__VA_ARGS__);                                     \
        _scope.Finish();                                             \
        ::pto::mocker::InjectTileCycles(PTO_FIRST_ARG(__VA_ARGS__)); \
        ::RecordInstrFromFirst(#API, __VA_ARGS__);                   \
    } while (0)
// Template calls use a dedicated macro because the preprocessor does not parse
// template commas in a generic `_IMPL(...)` wrapper reliably.
#define MAP_INSTR_IMPL_T(API, TEMPLATE_ARGS, ...)                    \
    do {                                                             \
        ::pto::mocker::PtoInstrScope _scope(#API);                   \
        API##_IMPL TEMPLATE_ARGS(__VA_ARGS__);                       \
        _scope.Finish();                                             \
        ::pto::mocker::InjectTileCycles(PTO_FIRST_ARG(__VA_ARGS__)); \
        ::RecordInstrFromFirst(#API, __VA_ARGS__);                   \
    } while (0)
#define RECORD_INSTR_ONLY(API, ...)                                  \
    do {                                                             \
        ::pto::mocker::PtoInstrScope _scope(#API);                   \
        _scope.Finish();                                             \
        ::pto::mocker::InjectTileCycles(PTO_FIRST_ARG(__VA_ARGS__)); \
        ::RecordInstrFromFirst(#API, __VA_ARGS__);                   \
    } while (0)
#define RECORD_A5_VF_INSTR(API, OP_PARAMS, ...) MAP_INSTR_IMPL(API, __VA_ARGS__)
#define MAP_A5_VF_PRECISION_IMPL(API, TEMPLATE_ARGS, IS_HIGH_PRECISION, ...) \
    MAP_INSTR_IMPL_T(API, TEMPLATE_ARGS, __VA_ARGS__)

// First arg is Pipe, second is Tile, third is tile index.
#define MAP_INSTR_IMPL_TPUSH_POP(IS_TPUSH, API, TEMPLATE_ARGS, ...)                                                  \
    do {                                                                                                             \
        ::pto::mocker::PtoInstrScope _scope(#API);                                                                   \
        API##_IMPL TEMPLATE_ARGS(__VA_ARGS__);                                                                       \
        _scope.Finish();                                                                                             \
        ::pto::mocker::InjectTileCycles(PTO_FIRST_ARG(__VA_ARGS__));                                                 \
        /* Call RecordTPushSync or RecordTPopSync for FFTS sync */                                                   \
        if constexpr (IS_TPUSH) {                                                                                    \
            ::RecordTPushSync(PTO_FIRST_ARG(__VA_ARGS__), PTO_NTH_ARG(__VA_ARGS__, 2), PTO_NTH_ARG(__VA_ARGS__, 3)); \
        } else {                                                                                                     \
            ::RecordTPopSync(PTO_FIRST_ARG(__VA_ARGS__), PTO_NTH_ARG(__VA_ARGS__, 2), PTO_NTH_ARG(__VA_ARGS__, 3));  \
        }                                                                                                            \
        ::RecordInstrFromFirst(#API, __VA_ARGS__);                                                                   \
    } while (0)

// Special macro for TPUSH: includes CV ring buffer FFTS sync recording
// Args: API, TEMPLATE_ARGS, pipe, tile, [tile_index]
#define MAP_INSTR_IMPL_T_TPUSH(API, TEMPLATE_ARGS, ...)                                                          \
    do {                                                                                                         \
        ::pto::mocker::PtoInstrScope _scope(#API);                                                               \
        API##_IMPL TEMPLATE_ARGS(__VA_ARGS__);                                                                   \
        _scope.Finish();                                                                                         \
        ::pto::mocker::InjectTileCycles(PTO_FIRST_ARG(__VA_ARGS__));                                             \
        ::RecordTPushSync(                                                                                       \
            PTO_FIRST_ARG(__VA_ARGS__), PTO_SECOND_ARG(__VA_ARGS__), PTO_FIRST_ARG(__VA_ARGS__).prod.tileIndex); \
        ::RecordInstrFromFirst(#API, __VA_ARGS__);                                                               \
    } while (0)

// Special macro for TPOP: includes CV ring buffer FFTS sync recording
#define MAP_INSTR_IMPL_T_TPOP(API, TEMPLATE_ARGS, ...)                                                           \
    do {                                                                                                         \
        ::pto::mocker::PtoInstrScope _scope(#API);                                                               \
        API##_IMPL TEMPLATE_ARGS(__VA_ARGS__);                                                                   \
        _scope.Finish();                                                                                         \
        ::pto::mocker::InjectTileCycles(PTO_FIRST_ARG(__VA_ARGS__));                                             \
        ::RecordTPopSync(                                                                                        \
            PTO_FIRST_ARG(__VA_ARGS__), PTO_SECOND_ARG(__VA_ARGS__), PTO_FIRST_ARG(__VA_ARGS__).cons.tileIndex); \
        ::RecordInstrFromFirst(#API, __VA_ARGS__);                                                               \
    } while (0)
#endif

#endif
