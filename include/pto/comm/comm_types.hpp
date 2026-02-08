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
// - `tensors` points to an array of GlobalData objects (not pointers).
// ============================================================================

template <typename GlobalData>
struct ParallelGroup {
    using value_type = GlobalData;  // Type alias for type traits
    
    GlobalData *tensors {nullptr};  // Points to external array of GlobalData objects
    int nranks {0};
    int my_rank {-1};

    constexpr ParallelGroup() = default;
    
    // Constructor: takes array of GlobalData objects
    AICORE constexpr ParallelGroup(GlobalData *tensorArray, int size, int rank_id) 
        : tensors(tensorArray), nranks(size), my_rank(rank_id) {}

    // Factory function (recommended)
    AICORE static constexpr ParallelGroup Create(GlobalData *tensorArray, int size, int rank_id) {
        return ParallelGroup(tensorArray, size, rank_id);
    }

    AICORE constexpr int size() const { return nranks; }
    AICORE constexpr bool empty() const { return nranks == 0; }

    AICORE constexpr int GetRank() const { return my_rank; }
    AICORE constexpr int GetSize() const { return nranks; }

    AICORE constexpr GlobalData &operator[](int teamRank) { return tensors[teamRank]; }
    AICORE constexpr const GlobalData &operator[](int teamRank) const { return tensors[teamRank]; }
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
// DmaEngine: DMA engine type for asynchronous data transfer
// ============================================================================

enum class DmaEngine : uint8_t {
    SDMA = 0,  // System DMA - high bandwidth, for large transfers
    URMA = 1,  // User-space RMA - low latency, for small transfers
};

// ============================================================================
// AsyncEvent: Event handle for asynchronous operations
// ============================================================================

struct AsyncEvent {
    uint64_t handle {0};
    DmaEngine engine {DmaEngine::SDMA};
    
    constexpr AsyncEvent() = default;
    AICORE constexpr AsyncEvent(uint64_t h, DmaEngine e) : handle(h), engine(e) {}
    AICORE constexpr bool valid() const { return handle != 0; }
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
// ReduceOp: Reduction operators for TREDUCE
// ============================================================================

enum class ReduceOp : uint8_t {
    Sum = 0,  // Element-wise sum
    Max = 1,  // Element-wise maximum
    Min = 2,  // Element-wise minimum
};

// ============================================================================
// Signal: Scalar signal (1 element, fully static)
//
// Equivalent to:
//   GlobalTensor<int32_t, Shape<1,1,1,1,1>, Stride<1,1,1,1,1>, Layout::ND>
//
// Usage:
//   comm::Signal sig(ptr);
// ============================================================================

using Signal = GlobalTensor<int32_t, Shape<1,1,1,1,1>, Stride<1,1,1,1,1>, Layout::ND>;
// GlobalSignal: alias for GlobalTensor, used by signal-related instructions
template <typename Element_, typename Shape_, typename Stride_, Layout Layout_ = Layout::ND>
using GlobalSignal = GlobalTensor<Element_, Shape_, Stride_, Layout_>;
// ============================================================================
// Signal2D: 2D signal matrix with compile-time shape
//
// Dense case:     stride auto-derived from shape (DIM_3 stride = Cols).
// Sub-region:     pass a custom stride for views into a larger signal grid.
//
// Usage:
//   // Dense 4x8 grid (stride auto = 8)
//   comm::Signal2D<4, 8> grid(ptr);
//
//   // Sub-region 4x8 from a 128-col grid (stride = 128)
//   comm::Signal2D<4, 8> sub(ptr + offset, 128);
// ============================================================================

template <int Rows, int Cols>
struct Signal2D : public GlobalTensor<int32_t,
    Shape<1, 1, 1, Rows, Cols>, Stride<1, 1, 1, DYNAMIC, 1>, Layout::ND> {
private:
    using Base = GlobalTensor<int32_t,
        Shape<1, 1, 1, Rows, Cols>, Stride<1, 1, 1, DYNAMIC, 1>, Layout::ND>;

public:
    // Dense constructor: stride = Cols (contiguous layout)
    PTO_INTERNAL Signal2D(typename Base::DType *ptr)
        : Base(ptr, typename Base::Shape{}, typename Base::Stride{Cols}) {}

    // Strided constructor: custom DIM_3 stride for sub-region views
    PTO_INTERNAL Signal2D(typename Base::DType *ptr, int stride)
        : Base(ptr, typename Base::Shape{}, typename Base::Stride{stride}) {}
};

} // namespace comm
} // namespace pto

#endif // PTO_COMM_COMM_TYPES_HPP
