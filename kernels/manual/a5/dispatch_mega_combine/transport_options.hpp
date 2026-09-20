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

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

struct MegaMoeTransportOptions {
    bool urma = false;
    uint32_t ranksPerServer = 0U;
    bool localDeviceMapping = false;
};

inline uint32_t MegaMoeEnvInteger(const char* suffix, uint32_t maximum, uint32_t defaultValue = 0U, int base = 0)
{
    const std::string name = std::string("DISPATCH_MEGA_COMBINE_") + suffix;
    const char* value = std::getenv(name.c_str());
    if (value == nullptr || *value == '\0')
        return defaultValue;
    const std::string text(value);
    if (text.find_first_of(" \t\n\r\v\f") != std::string::npos)
        throw std::runtime_error("invalid whitespace in " + name);
    size_t parsed = 0U;
    long long number = -1;
    try {
        number = std::stoll(text, &parsed, base);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid integer in " + name);
    }
    if (parsed != text.size() || number < 0 || static_cast<unsigned long long>(number) > maximum)
        throw std::runtime_error("out-of-range integer in " + name);
    return static_cast<uint32_t>(number);
}

inline MegaMoeTransportOptions LoadMegaMoeTransportOptions()
{
    MegaMoeTransportOptions options;
    options.urma = MegaMoeEnvInteger("URMA", 1U) != 0U;
    options.ranksPerServer = MegaMoeEnvInteger("RANKS_PER_SERVER", 64U);
    options.urma = options.urma || options.ranksPerServer != 0U;
    options.localDeviceMapping = MegaMoeEnvInteger("LOCAL_DEVICE_MAPPING", 1U) != 0U;
    if (options.localDeviceMapping && (!options.urma || options.ranksPerServer == 0U))
        throw std::runtime_error("server-local device mapping requires URMA and explicit RANKS_PER_SERVER");
    return options;
}

inline uint32_t MegaMoeRanksPerServer(const MegaMoeTransportOptions& options, uint32_t ranks)
{
    if (ranks == 0U)
        throw std::runtime_error("world size must be positive");
    const uint32_t perServer = options.ranksPerServer == 0U ? ranks : options.ranksPerServer;
    if (perServer > ranks || ranks % perServer != 0U)
        throw std::runtime_error("RANKS_PER_SERVER must be positive, <= world size and divide world size");
    return perServer;
}

inline uint32_t MegaMoeDeviceForRank(
    const MegaMoeTransportOptions& options, uint32_t rank, uint32_t ranks, uint32_t first)
{
    const uint32_t perServer = MegaMoeRanksPerServer(options, ranks);
    if (rank >= ranks)
        throw std::runtime_error("invalid global rank");
    const uint32_t local = options.localDeviceMapping ? rank % perServer : rank;
    if (first > static_cast<uint32_t>(INT32_MAX) - local)
        throw std::runtime_error("physical device overflow");
    return first + local;
}
