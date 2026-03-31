/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MOCKER_EVALUATOR_ARG_READER_HPP
#define PTO_MOCKER_EVALUATOR_ARG_READER_HPP

#include <initializer_list>
#include <cstdint>
#include <string_view>

#include <pto/costmodel/trace.hpp>

namespace pto::mocker::evaluator {

inline const TraceArgRecord *FindArg(const CceCallRecord &call, std::string_view name)
{
    for (const auto &arg : call.params) {
        if (arg.name == name) {
            return &arg;
        }
    }
    return nullptr;
}

inline bool HasArg(const CceCallRecord &call, std::string_view name)
{
    return FindArg(call, name) != nullptr;
}

inline bool TryGetArg(const CceCallRecord &call, std::string_view name, uint64_t &value)
{
    const TraceArgRecord *arg = FindArg(call, name);
    if (arg == nullptr) {
        return false;
    }
    value = arg->value;
    return true;
}

inline bool TryGetAnyArg(const CceCallRecord &call, std::initializer_list<std::string_view> names, uint64_t &value)
{
    for (std::string_view name : names) {
        if (TryGetArg(call, name, value)) {
            return true;
        }
    }
    return false;
}

inline bool TryGetRepeatArg(const CceCallRecord &call, uint64_t &value)
{
    return TryGetAnyArg(call, {"repeats", "repeat"}, value);
}

inline uint64_t GetArgOr(const CceCallRecord &call, std::string_view name, uint64_t fallback)
{
    uint64_t value = fallback;
    (void)TryGetArg(call, name, value);
    return value;
}

} // namespace pto::mocker::evaluator

#endif
