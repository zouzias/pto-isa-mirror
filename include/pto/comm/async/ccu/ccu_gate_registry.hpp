/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_CCU_CCU_GATE_REGISTRY_HPP
#define PTO_COMM_ASYNC_CCU_CCU_GATE_REGISTRY_HPP

// Host-only header — must NOT be included from device (bisheng -xcce) code.
#if defined(__CCE_KT_TEST__)
#error "ccu_gate_registry.hpp is a host-only header and cannot be included in device code."
#endif

// Process-local descriptor registry for PTO gated CCU kernels. Header-only.
// Each rank holds one gate descriptor and a progress slot vector (e.g. ping-pong).

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pto/comm/async/ccu/ccu_types.hpp"

namespace pto {
namespace comm {
namespace ccu {

namespace detail {

inline CcuGateDescriptor MakeDesc(uint32_t dieId, uint32_t ckeId, uint32_t mask)
{
    return CcuGateDescriptor{dieId, ckeId, mask, 0};
}

struct RankEntry {
    CcuGateDescriptor gate{};
    bool hasGate{false};
    std::vector<CcuGateDescriptor> progress;
};

struct Registry {
    std::mutex mu;
    std::unordered_map<uint32_t, RankEntry> ranks;

    static Registry &Instance()
    {
        static Registry r;
        return r;
    }

    template <typename Fn>
    auto WithLock(Fn &&fn) -> decltype(fn(ranks))
    {
        std::lock_guard<std::mutex> lk(mu);
        return std::forward<Fn>(fn)(ranks);
    }
};

} // namespace detail

inline bool IsCcuGateEnabledFromEnv()
{
    const char *v = std::getenv(CCU_GATE_ENV);
    return v != nullptr && std::strcmp(v, "1") == 0;
}

inline void Publish(uint32_t rankId, uint32_t dieId, uint32_t ckeId, uint32_t mask)
{
    detail::Registry::Instance().WithLock([&](auto &ranks) {
        auto &e = ranks[rankId];
        e.gate = detail::MakeDesc(dieId, ckeId, mask);
        e.hasGate = true;
    });
}

inline bool TryGet(uint32_t rankId, CcuGateDescriptor &out)
{
    return detail::Registry::Instance().WithLock([&](auto &ranks) {
        auto it = ranks.find(rankId);
        if (it == ranks.end() || !it->second.hasGate) {
            return false;
        }
        out = it->second.gate;
        return true;
    });
}

inline void PublishProgress(uint32_t rankId, uint32_t slotIndex, uint32_t dieId, uint32_t ckeId, uint32_t mask)
{
    detail::Registry::Instance().WithLock([&](auto &ranks) {
        auto &slots = ranks[rankId].progress;
        if (slots.size() <= slotIndex) {
            slots.resize(slotIndex + 1);
        }
        slots[slotIndex] = detail::MakeDesc(dieId, ckeId, mask);
    });
}

inline bool TryGetProgress(uint32_t rankId, std::vector<CcuGateDescriptor> &out)
{
    return detail::Registry::Instance().WithLock([&](auto &ranks) {
        auto it = ranks.find(rankId);
        if (it == ranks.end() || it->second.progress.empty()) {
            return false;
        }
        out = it->second.progress;
        return true;
    });
}

inline void ClearProgress(uint32_t rankId)
{
    detail::Registry::Instance().WithLock([&](auto &ranks) {
        auto it = ranks.find(rankId);
        if (it != ranks.end()) {
            it->second.progress.clear();
        }
    });
}

} // namespace ccu
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_CCU_CCU_GATE_REGISTRY_HPP
