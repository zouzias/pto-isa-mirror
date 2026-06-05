/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMMON_NPU_DEDUP_SYNC_COMMON_HPP
#define PTO_COMMON_NPU_DEDUP_SYNC_COMMON_HPP

#include <pto/common/type.hpp>
#include <pto/common/event.hpp>

namespace pto {

template <Op OpCode>
PTO_INTERNAL static constexpr pipe_t GetPipeByOp()
{
    if constexpr ((OpCode >= static_cast<Op>(0)) && (OpCode <= Op::OP_COUNT)) {
        return opPipeList[static_cast<int>(OpCode)];
    }
    return PIPE_ALL;
}

template <typename T>
struct is_event : std::false_type {
};

template <typename... Ts>
inline constexpr bool all_events_v = (is_event<Ts>::value && ...);

} // namespace pto

#endif
