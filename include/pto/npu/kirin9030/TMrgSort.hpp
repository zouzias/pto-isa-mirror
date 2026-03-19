/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#define DEVICE_UB_SIZE 128
#define COPY_UBUF_TO_UBUF(dst, src, nBrust, lenBrust) copy_ubuf_to_ubuf((dst), (src), (nBrust), (lenBrust), 0, 0);
#include "pto/npu/a5/TMrgSort.hpp"
#undef DEVICE_UB_SIZE
#define COPY_UBUF_TO_UBUF