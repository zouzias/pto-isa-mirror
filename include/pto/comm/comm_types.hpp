/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_COMM_TYPES_HPP
#define PTO_COMM_COMM_TYPES_HPP

#include <cstdint>
#include <type_traits>

#include "pto/common/type.hpp"
#include "pto/common/pto_tile.hpp"

namespace pto {
namespace comm {

// ============================================================================
// ParallelGroup: Groups GlobalTensors participating in collective communication
//
// Notes:
// - This is a lightweight "view" wrapper: no dynamic memory allocation on 
//   device side, avoiding unsupported containers like std::vector.
// - Each element in the group typically represents "the GlobalTensor view 
//   for that team rank" (usually mapped to world rank via SetRank).
// ============================================================================

template <typename GlobalData>
struct ParallelGroup {
    using value_type = GlobalData;  // Type alias for type traits
    
    GlobalData **tensors {nullptr}; // Points to external array: tensors[teamRank] -> GlobalTensor*
    int nranks {0};
    int my_rank {-1};

    constexpr ParallelGroup() = default;
    AICORE constexpr ParallelGroup(GlobalData **tensorPtrs, int size, int rank_id) 
        : tensors(tensorPtrs), nranks(size), my_rank(rank_id) {}

    AICORE constexpr int size() const { return nranks; }
    AICORE constexpr bool empty() const { return nranks == 0; }

    AICORE constexpr int GetRank() const { return my_rank; }
    AICORE constexpr int GetSize() const { return nranks; }

    AICORE constexpr GlobalData &operator[](int teamRank) { return *tensors[teamRank]; }
    AICORE constexpr const GlobalData &operator[](int teamRank) const { return *tensors[teamRank]; }
};

// Type traits: Extract GlobalData type from ParallelGroup<GlobalData>
template <typename T>
struct ParallelGroupTraits {
    static_assert(std::is_same_v<T, void>, 
                  "ParallelGroupTraits: T must be ParallelGroup<GlobalData>");
};

template <typename GlobalData>
struct ParallelGroupTraits<ParallelGroup<GlobalData>> {
    using GlobalDataType = GlobalData;
};

// ============================================================================
// NotifyOp: Notification operation type for TNOTIFY
// ============================================================================

enum class NotifyOp : uint8_t {
    AtomicAdd = 0,  // Atomic add operation
    Set = 1,        // Direct set operation
};

// ============================================================================
// WaitCmp: Comparison operators for signal wait/test operations
// ============================================================================

enum class WaitCmp : uint8_t {
    EQ = 0,  // Equal
    NE = 1,  // Not equal
    GT = 2,  // Greater than
    GE = 3,  // Greater than or equal to
    LT = 4,  // Less than
    LE = 5,  // Less than or equal to
};

// ============================================================================
// SdmaEvent: Event handle for SDMA asynchronous operations
//
// Used to track and synchronize SDMA transfer operations.
// The event can be used with TWAIT_SDMA to wait for completion.
// ============================================================================

struct SdmaEvent {
    uint64_t event_id;  // SDMA event identifier
    
    AICORE constexpr SdmaEvent() : event_id(0) {}
    AICORE constexpr explicit SdmaEvent(uint64_t id) : event_id(id) {}
    
    AICORE constexpr bool operator==(const SdmaEvent& other) const {
        return event_id == other.event_id;
    }
    
    AICORE constexpr bool operator!=(const SdmaEvent& other) const {
        return event_id != other.event_id;
    }
};

} // namespace comm
} // namespace pto

#endif // PTO_COMM_COMM_TYPES_HPP
