/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
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
