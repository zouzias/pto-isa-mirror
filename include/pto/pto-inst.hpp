/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_INST_HPP
#define PTO_INST_HPP

#include "common/type.hpp"
#ifdef __CPU_SIM
#include "common/cpu_stub.hpp"
#endif
#include "common/memory.hpp"

#if defined(__CPU_SIM) || defined(__CCE_AICORE__)
#include "common/arch_macro.hpp"
#include "common/pto_tile.hpp"
#include "common/pto_instr.hpp"
#endif
#endif
