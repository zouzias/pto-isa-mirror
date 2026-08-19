/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef FFN_GRID_DEVICE_ID_INL_HPP
#define FFN_GRID_DEVICE_ID_INL_HPP

#include <climits>
#include <cstdlib>
#include <cstring>
#include <iostream>

static bool ParseDeviceIdValue(const char *value, int &deviceId)
{
    if (value == nullptr || value[0] == '\0') {
        return false;
    }

    char *end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    deviceId = static_cast<int>(parsed);
    return true;
}

static bool ParseDeviceIdArg(int argc, char **argv, int &deviceId)
{
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device-id") == 0 || std::strcmp(argv[i], "-d") == 0) {
            if (i + 1 >= argc || !ParseDeviceIdValue(argv[i + 1], deviceId)) {
                std::cerr << "[ERROR] invalid --device-id value" << std::endl;
                std::exit(1);
            }
            return true;
        }

        constexpr const char *kPrefix = "--device-id=";
        constexpr size_t kPrefixLen = 12;
        if (std::strncmp(argv[i], kPrefix, kPrefixLen) == 0) {
            if (!ParseDeviceIdValue(argv[i] + kPrefixLen, deviceId)) {
                std::cerr << "[ERROR] invalid --device-id value" << std::endl;
                std::exit(1);
            }
            return true;
        }
    }
    return false;
}

#endif
