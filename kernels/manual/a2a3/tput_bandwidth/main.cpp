/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "comm_mpi.h"
#include "tput_bandwidth_kernel.h"

int main(int argc, char** argv)
{
    CommMpiInit(&argc, &argv);
    const char* mode = std::getenv("TPUT_BENCH_MODE");
    bool ok = false;
    if (mode == nullptr || std::strcmp(mode, "device_baseline") == 0) {
        const char* firstDeviceValue = std::getenv("TPUT_DEVICE_BASELINE_FIRST_DEVICE_ID");
        const int firstDeviceId = firstDeviceValue == nullptr ? 0 : std::atoi(firstDeviceValue);
        ok = RunTPutDeviceBaseline(2, 2, 0, firstDeviceId);
    } else {
        std::fprintf(stderr, "[ERROR] Unsupported TPUT_BENCH_MODE=%s\n", mode);
    }
    CommMpiFinalize();
    std::printf(ok ? "test success\n" : "test failed\n");
    return ok ? 0 : 1;
}
