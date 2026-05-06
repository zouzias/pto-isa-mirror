/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0.
 * Please refer to the License for details. You may not use this file except in
 * compliance with the License.
 */

#include "pto/ccu/pto_gate_registry.hpp"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace pto {
namespace ccu {

bool IsPtoGateEnabledFromEnv()
{
    const char *v = std::getenv(PTO_GATE_ENV);
    return v != nullptr && std::strcmp(v, "1") == 0;
}

namespace {

std::mutex &MapMutex()
{
    static std::mutex m;
    return m;
}

std::unordered_map<uint32_t, pto::host::PtoGateDescriptor> &Map()
{
    static std::unordered_map<uint32_t, pto::host::PtoGateDescriptor> m;
    return m;
}

} // namespace

void Publish(uint32_t rankId, uint32_t dieId, uint32_t ckeId, uint32_t mask)
{
    pto::host::PtoGateDescriptor desc{};
    desc.dieId    = dieId;
    desc.ckeId    = ckeId;
    desc.mask     = mask;
    desc.mmioAddr = 0;

    std::lock_guard<std::mutex> lk(MapMutex());
    Map()[rankId] = desc;
}

bool TryGet(uint32_t rankId, pto::host::PtoGateDescriptor &out)
{
    std::lock_guard<std::mutex> lk(MapMutex());
    auto it = Map().find(rankId);
    if (it == Map().end()) {
        return false;
    }
    out = it->second;
    return true;
}

} // namespace ccu
} // namespace pto

// dlsym entry points — visibility forced to default so a parent build that
// sets `-fvisibility=hidden` (this CMakeLists does) still publishes these in
// `nm -D libpto_ccu_host.so`. Same defensive attribute as the legacy hccl
// `HcclPtoTryGetLastReduceGateDescriptor`.

extern "C" __attribute__((visibility("default"))) bool
PtoCcuTryGetLastReduceScatterGateDescriptor(uint32_t rankId,
                                            pto::host::PtoGateDescriptor *out)
{
    if (out == nullptr) {
        return false;
    }
    return pto::ccu::TryGet(rankId, *out);
}

extern "C" __attribute__((visibility("default"))) void
PtoCcuPublish_C(uint32_t rankId, uint32_t dieId, uint32_t ckeId, uint32_t mask)
{
    pto::ccu::Publish(rankId, dieId, ckeId, mask);
}
