/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_TRACE_HPP
#define PTO_CPU_TRACE_HPP

#include <array>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <mutex>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <pto/common/pto_tile.hpp>
#include <pto/common/type.hpp>

namespace pto::cpu_sim {

#if defined(PTO_CPU_SIM_TRACE_MODE) && PTO_CPU_SIM_TRACE_MODE
inline constexpr bool kInstructionTraceEnabled = true;
#else
inline constexpr bool kInstructionTraceEnabled = false;
#endif

struct TileOperandTrace {
    std::uintptr_t address = 0;
    std::vector<int64_t> shape;
    std::string layout;
    std::string dtype;
};

struct ScalarOperandTrace {
    std::string dtype;
    std::string value;
};

struct InstructionTraceRecord {
    uint32_t block_idx = 0;
    uint64_t sequence_id = 0;
    std::string opcode;
    std::vector<TileOperandTrace> input_tiles;
    std::vector<ScalarOperandTrace> scalar_inputs;
    std::vector<TileOperandTrace> output_tiles;
};

struct InstructionTraceState {
    uint64_t next_sequence_id = 0;
    std::vector<InstructionTraceRecord> records;
    mutable std::mutex mutex;
};

inline thread_local InstructionTraceState g_instruction_trace_state;
inline thread_local InstructionTraceState* g_instruction_trace_override = nullptr;

inline InstructionTraceState& CurrentInstructionTraceState()
{
    return g_instruction_trace_override == nullptr ? g_instruction_trace_state : *g_instruction_trace_override;
}

inline void ResetInstructionTrace()
{
    auto& trace = CurrentInstructionTraceState();
    std::scoped_lock lock(trace.mutex);
    trace.next_sequence_id = 0;
    trace.records.clear();
}

inline InstructionTraceState& GetMutableInstructionTrace() { return CurrentInstructionTraceState(); }

// Returns the active trace state for advanced consumers that need direct access
// to the state object. Callers must lock InstructionTraceState::mutex before
// reading records or mutating shared members.
inline const InstructionTraceState& GetInstructionTrace() { return CurrentInstructionTraceState(); }

inline std::vector<InstructionTraceRecord> CopyInstructionTraceRecords()
{
    const auto& trace = GetInstructionTrace();
    std::scoped_lock lock(trace.mutex);
    return trace.records;
}

class ScopedInstructionTraceState {
public:
    explicit ScopedInstructionTraceState(InstructionTraceState& state) : saved_(g_instruction_trace_override)
    {
        g_instruction_trace_override = &state;
    }

    ~ScopedInstructionTraceState() { g_instruction_trace_override = saved_; }

    ScopedInstructionTraceState(const ScopedInstructionTraceState&) = delete;
    ScopedInstructionTraceState& operator=(const ScopedInstructionTraceState&) = delete;

private:
    InstructionTraceState* saved_ = nullptr;
};

inline uint64_t ReserveInstructionTraceSequenceId()
{
    auto& trace = GetMutableInstructionTrace();
    std::scoped_lock lock(trace.mutex);
    return trace.next_sequence_id++;
}

inline void AppendInstructionTraceRecord(InstructionTraceRecord record)
{
    auto& trace = GetMutableInstructionTrace();
    std::scoped_lock lock(trace.mutex);
    trace.records.push_back(std::move(record));
}

#include <pto/cpu/trace_detail_inl.hpp>
} // namespace pto::cpu_sim

#endif
