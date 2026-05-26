/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_DISPATCH_CONST_ARGS_HPP_
#define MOE_DISPATCH_CONST_ARGS_HPP_

#include <cstdint>

constexpr uint64_t KB_SIZE = 1024ULL;
constexpr uint64_t MB_SIZE = 1024ULL * 1024ULL;
constexpr uint64_t MOE_DISPATCH_WINDOW_HEAD_GUARD_BYTES = 4ULL * KB_SIZE;
constexpr uint64_t MOE_DISPATCH_WINDOW_ALIGN_BYTES = 512ULL;
constexpr uint64_t MOE_DISPATCH_TAIL_CONTROL_BYTES = 2ULL * MB_SIZE;
constexpr uint64_t MOE_DISPATCH_SIGNAL_BYTES = 1ULL * MB_SIZE;
constexpr uint32_t MOE_DISPATCH_SIGNAL_STRIDE_I32 = 16U;
constexpr uint32_t MOE_DISPATCH_TOKEN_READY_BASE_INDEX = 4096U;

#endif // MOE_DISPATCH_CONST_ARGS_HPP_
