/**
 * Common enums for gather/scatter operations shared across backends.
 */
#ifndef PTO_GATHER_SCATTER_HPP
#define PTO_GATHER_SCATTER_HPP

#include <cstdint>

namespace pto {

// Coalesce mode for gather/scatter kernels
enum class Coalesce : uint8_t
{
    Row = 0,
    Elem = 1,
};

// Out-of-bounds handling for gather
enum class GatherOOB : uint8_t
{
    Undefined = 0, // No bounds check
    Clamp = 1,
    Wrap = 2,
    Zero = 3,
};

// Scatter atomic operation kinds
enum class ScatterAtomicOp : uint8_t
{
    None = 0,
    Add = 1,
    Max = 2,
    Min = 3,
};

// Scatter out-of-bounds behavior
enum class ScatterOOB : uint8_t
{
    Undefined = 0,
    Skip = 1,
    Clamp = 2,
    Wrap = 3,
};

// Collision policy for non-atomic scatter writes
enum class ScatterConflict : uint8_t
{
    First = 0,
    Last = 1,
};

} // namespace pto

#endif // PTO_GATHER_SCATTER_HPP
