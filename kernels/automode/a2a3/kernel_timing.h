/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef KERNELS_AUTOMODE_A2A3_KERNEL_TIMING_H_
#define KERNELS_AUTOMODE_A2A3_KERNEL_TIMING_H_

#include "acl/acl.h"

#include <chrono>
#include <cstdio>

namespace PtoTiming {

template <typename LaunchFn>
inline aclError TimeKernelCallUs(const char *label, aclrtStream stream, LaunchFn launch)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    launch();
    aclError ret = aclrtSynchronizeStream(stream);
    auto t1 = std::chrono::high_resolution_clock::now();

    double timeUs = std::chrono::duration<double, std::micro>(t1 - t0).count();
    printf("[time] %s: %.3f us\n", label, timeUs);
    return ret;
}

}  // namespace PtoTiming

#endif  // KERNELS_AUTOMODE_A2A3_KERNEL_TIMING_H_
