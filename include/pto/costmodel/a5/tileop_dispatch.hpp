/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_COSTMODEL_A5_TILEOP_DISPATCH_HPP
#define PTO_COSTMODEL_A5_TILEOP_DISPATCH_HPP

// Included by common/tileop_dispatch.hpp after the shared helpers and macros.
#define MAP_INSTR_IMPL(API, ...)                                      \
    do {                                                              \
        ::pto::mocker::PtoInstrScope _scope(#API);                    \
        _scope.Finish();                                              \
        ::pto::mocker::RecordInstrWithOptions(#API, {}, __VA_ARGS__);             \
        ::pto::mocker::InjectTileCycles(PTO_FIRST_ARG(__VA_ARGS__));  \
    } while (0)
#define MAP_INSTR_IMPL_T(API, TEMPLATE_ARGS, ...) MAP_INSTR_IMPL(API, __VA_ARGS__)
#define RECORD_INSTR_ONLY(API, ...) MAP_INSTR_IMPL(API, __VA_ARGS__)
#define RECORD_A5_VF_INSTR(API, OP_PARAMS, ...)                                      \
    do {                                                                             \
        ::pto::mocker::PtoInstrScope _scope(#API);                                   \
        _scope.Finish();                                                             \
        ::pto::mocker::RecordInstrWithOptions(                                                    \
            #API, ::pto::mocker::A5TileOpOptions{OP_PARAMS}, __VA_ARGS__);            \
        ::pto::mocker::InjectTileCycles(PTO_FIRST_ARG(__VA_ARGS__));                 \
    } while (0)
#define MAP_A5_VF_PRECISION_IMPL(API, TEMPLATE_ARGS, IS_HIGH_PRECISION, ...) \
    RECORD_A5_VF_INSTR(API, (IS_HIGH_PRECISION) ? "high_precision" : "", __VA_ARGS__)

// Record FIFO synchronization without executing the NPU implementation.
#define MAP_INSTR_IMPL_TPUSH_POP(IS_TPUSH, API, TEMPLATE_ARGS, ...)                                  \
    do {                                                                                             \
        ::pto::mocker::PtoInstrScope _scope(#API);                                                   \
        _scope.Finish();                                                                             \
        if constexpr (IS_TPUSH) {                                                                    \
            ::RecordTPushSync(PTO_FIRST_ARG(__VA_ARGS__), PTO_NTH_ARG(__VA_ARGS__, 2), 0);           \
        } else {                                                                                     \
            ::RecordTPopSync(PTO_FIRST_ARG(__VA_ARGS__), PTO_NTH_ARG(__VA_ARGS__, 2), 0);            \
        }                                                                                            \
        ::pto::mocker::RecordInstrWithOptions(#API, {}, __VA_ARGS__);                                            \
        ::pto::mocker::InjectTileCycles(PTO_FIRST_ARG(__VA_ARGS__));                                 \
    } while (0)

#define MAP_INSTR_IMPL_T_TPUSH(API, TEMPLATE_ARGS, ...)                                              \
    do {                                                                                             \
        ::pto::mocker::PtoInstrScope _scope(#API);                                                   \
        _scope.Finish();                                                                             \
        ::RecordTPushSync(PTO_FIRST_ARG(__VA_ARGS__), PTO_SECOND_ARG(__VA_ARGS__), 0);               \
        ::pto::mocker::RecordInstrWithOptions(#API, {}, __VA_ARGS__);                                            \
        ::pto::mocker::InjectTileCycles(PTO_FIRST_ARG(__VA_ARGS__));                                 \
    } while (0)

#define MAP_INSTR_IMPL_T_TPOP(API, TEMPLATE_ARGS, ...)                                               \
    do {                                                                                             \
        ::pto::mocker::PtoInstrScope _scope(#API);                                                   \
        _scope.Finish();                                                                             \
        ::RecordTPopSync(PTO_FIRST_ARG(__VA_ARGS__), PTO_SECOND_ARG(__VA_ARGS__), 0);                \
        ::pto::mocker::RecordInstrWithOptions(#API, {}, __VA_ARGS__);                                            \
        ::pto::mocker::InjectTileCycles(PTO_FIRST_ARG(__VA_ARGS__));                                 \
    } while (0)

#endif
