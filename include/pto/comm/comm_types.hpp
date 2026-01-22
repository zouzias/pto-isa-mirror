#ifndef PTO_COMM_COMM_TYPES_HPP
#define PTO_COMM_COMM_TYPES_HPP

#include <cstdint>
#include <type_traits>

#include "pto/common/type.hpp"

namespace pto {
namespace comm {

// 2D copy descriptor, all units are in element count
struct Copy2DParams {
    uint32_t repeat {1};         // Number of rows or repeat count
    uint32_t lenElems {0};       // Number of elements per copy
    uint32_t srcStrideElems {0}; // Source stride (in elements)
    uint32_t dstStrideElems {0}; // Destination stride (in elements)

    static Copy2DParams Contiguous(uint32_t elemCount, uint32_t repeatCount = 1)
    {
        Copy2DParams p;
        p.repeat = repeatCount;
        p.lenElems = elemCount;
        p.srcStrideElems = elemCount;
        p.dstStrideElems = elemCount;
        return p;
    }
};

// ParallelGroup: Used to group GlobalTensors participating in collective communication into a "team".
//
// Notes:
// - This is a lightweight "view" wrapper: no dynamic memory allocation on device side, avoiding unsupported containers like `std::vector`.
// - Each element in the group typically represents "the GlobalTensor view for that team rank" (usually mapped to world rank via SetRank).
template <typename GlobalData>
struct ParallelGroup {
    using value_type = GlobalData;  // Type alias for type traits
    
    GlobalData **tensors {nullptr}; // Points to external array: tensors[teamRank] -> GlobalTensor*
    int nranks {0};
    int my_rank {-1};

    constexpr ParallelGroup() = default;
    AICORE constexpr ParallelGroup(GlobalData **tensorPtrs, int size, int rank_id) : tensors(tensorPtrs), nranks(size), my_rank(rank_id) {}

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
    // If not ParallelGroup, this will trigger a compile error
    static_assert(std::is_same_v<T, void>, 
                  "TALLREDUCE: T must be ParallelGroup<GlobalData>");
};

template <typename GlobalData>
struct ParallelGroupTraits<ParallelGroup<GlobalData>> {
    using GlobalDataType = GlobalData;
};

// Backend enum, for future extension
enum class BackendKind : uint8_t {
    Shmem = 0,
};

// NotifyOp: Notification operation type
enum class NotifyOp : uint8_t {
    AtomicAdd = 0,  // Atomic add operation
    Set = 1,        // Direct set operation
};

// WaitCmp: Comparison operators for signal wait operations
// Compatible with shmem comparison constants
enum class WaitCmp : uint8_t {
    EQ = 0,  // Equal
    NE = 1,  // Not equal
    GT = 2,  // Greater than
    GE = 3,  // Greater than or equal to
    LT = 4,  // Less than
    LE = 5,  // Less than or equal to
};

} // namespace comm
} // namespace pto

#endif // PTO_COMM_COMM_TYPES_HPP

