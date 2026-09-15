/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#pragma once

namespace pto {
enum QuantMode_t {
    NoQuant = 0,
    F322F16 = 1,
    F322BF16 = 16,
    DEQF16 = 5,
    VDEQF16 = 4,
    QF322B8_PRE = 24,
    QF322HIF8_PRE = 25,
    QF322FP8_PRE = 26,
    QF322F32_PRE = 27,
    QF322F16_PRE = 32,
    QF322BF16_PRE = 34,
    QS322BF16_PRE = 35,
    VQF322B8_PRE = 23,
    VQF322HIF8_PRE = 28,
    VQF322F16_PRE = 33,
    VQF322BF16_PRE = 36,
    VQF322FP8_PRE = 37,
    VQF322F32_PRE = 38,
    REQ8 = 3,
    VREQ8 = 2,
    VQS322BF16_PRE = 39
};
} // namespace pto

#include "cce_costmodel_sync.hpp"
