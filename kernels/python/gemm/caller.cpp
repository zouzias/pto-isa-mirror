/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Host-side shim compiled into gemm.so.
//
// compile.sh defines KERNEL_CPP as the generated ptoas C++ file. The exported
// call_kernel function is loaded from Python with ctypes and launches Gemm on
// the caller-provided stream.

#ifndef KERNEL_CPP
#error "KERNEL_CPP must be defined at compile time (see compile.sh)."
#endif

#include <cstdint>

#include KERNEL_CPP

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *c, uint8_t *a, uint8_t *b)
{
    Gemm<<<blockDim, nullptr, stream>>>(reinterpret_cast<half *>(c), reinterpret_cast<half *>(a),
                                        reinterpret_cast<half *>(b));
}
