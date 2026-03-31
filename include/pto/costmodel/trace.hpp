/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MOCKER_TRACE_HPP
#define PTO_MOCKER_TRACE_HPP

#include <bit>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pto::mocker {

struct TraceArgRecord {
    std::string name;
    uint64_t value;
};

struct CceCallRecord {
    std::string name;
    std::vector<TraceArgRecord> params;
};

struct PtoInstrRecord {
    std::string name;
    std::vector<CceCallRecord> cce_calls;
};

struct TraceState {
    std::vector<PtoInstrRecord> executed_pto;
    std::vector<CceCallRecord> raw_cce_calls;
    std::vector<std::size_t> active_pto_stack;
};

inline thread_local TraceState g_trace_state;

inline void ResetTrace()
{
    g_trace_state = {};
}

inline TraceState &GetMutableTrace()
{
    return g_trace_state;
}

inline const TraceState &GetTrace()
{
    return g_trace_state;
}

template <typename T>
inline constexpr bool kUnsupportedTraceType = false;

template <typename T>
inline uint64_t ToTraceValue(T value)
{
    using Decayed = std::remove_cv_t<std::remove_reference_t<T>>;
    if constexpr (std::is_pointer_v<Decayed>) {
        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(value));
    } else if constexpr (std::is_enum_v<Decayed>) {
        using Underlying = std::underlying_type_t<Decayed>;
        return static_cast<uint64_t>(static_cast<Underlying>(value));
    } else if constexpr (std::is_integral_v<Decayed>) {
        return static_cast<uint64_t>(value);
    } else if constexpr (std::is_floating_point_v<Decayed> && sizeof(Decayed) == sizeof(uint32_t)) {
        return static_cast<uint64_t>(std::bit_cast<uint32_t>(value));
    } else if constexpr (std::is_floating_point_v<Decayed> && sizeof(Decayed) == sizeof(uint64_t)) {
        return std::bit_cast<uint64_t>(value);
    } else {
        static_assert(kUnsupportedTraceType<Decayed>, "Unsupported trace argument type.");
        return 0;
    }
}

template <typename T>
inline TraceArgRecord MakeTraceArg(std::string_view name, T value)
{
    return {std::string(name), ToTraceValue(value)};
}

inline void BeginPtoInstr(std::string_view name)
{
    auto &trace = g_trace_state;
    if (trace.active_pto_stack.empty()) {
        trace.executed_pto.push_back(PtoInstrRecord{std::string(name), {}});
        trace.active_pto_stack.push_back(trace.executed_pto.size() - 1);
    } else {
        // Collapse nested PTO helper calls into the current top-level PTO record.
        trace.active_pto_stack.push_back(trace.active_pto_stack.back());
    }
}

inline void EndPtoInstr()
{
    auto &stack = g_trace_state.active_pto_stack;
    if (!stack.empty()) {
        stack.pop_back();
    }
}

inline void RecordCceCall(std::string_view name, std::initializer_list<TraceArgRecord> args = {})
{
    CceCallRecord call{std::string(name), std::vector<TraceArgRecord>(args)};
    auto &trace = g_trace_state;
    if (!trace.active_pto_stack.empty()) {
        trace.executed_pto[trace.active_pto_stack.back()].cce_calls.push_back(std::move(call));
    } else {
        trace.raw_cce_calls.push_back(std::move(call));
    }
}

template <typename... Args>
inline void RecordCceCallVariadic(std::string_view name, Args &&... args)
{
    CceCallRecord call;
    call.name = std::string(name);
    std::size_t index = 0;
    (call.params.push_back({std::string("arg") + std::to_string(index++), ToTraceValue(std::forward<Args>(args))}), ...);

    auto &trace = g_trace_state;
    if (!trace.active_pto_stack.empty()) {
        trace.executed_pto[trace.active_pto_stack.back()].cce_calls.push_back(std::move(call));
    } else {
        trace.raw_cce_calls.push_back(std::move(call));
    }
}

class PtoInstrScope {
public:
    explicit PtoInstrScope(std::string_view name)
    {
        BeginPtoInstr(name);
    }

    ~PtoInstrScope()
    {
        EndPtoInstr();
    }

    PtoInstrScope(const PtoInstrScope &) = delete;
    PtoInstrScope &operator=(const PtoInstrScope &) = delete;
};

} // namespace pto::mocker

#endif
