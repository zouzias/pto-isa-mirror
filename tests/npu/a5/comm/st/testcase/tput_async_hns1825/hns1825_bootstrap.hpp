/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Host-only bootstrap helpers for the RoCE (HNS_1825) control plane.
//
//   1. ResolvePhyId — global physical device id via aclrt/runtime (HCCP/topo use phyId, not ACL id).
//   2. ResolveLocalRdmaIp — CLOS IPv4 for that phyId from /etc/hccl_rootinfo.json.
//   3. ResolveLocalRdmaIpFromVirtualTopology — RoCE IPv4 selected from topologyd's virtualTopology.xml.
//
// Optional symbols are resolved with dlsym(RTLD_DEFAULT, ...); missing symbols degrade gracefully
// so callers can fall back to environment overrides.

#ifndef PTO_TESTS_NPU_A5_COMM_ST_HNS1825_BOOTSTRAP_HPP
#define PTO_TESTS_NPU_A5_COMM_ST_HNS1825_BOOTSTRAP_HPP

#if defined(__CCE_KT_TEST__)
#error "hns_1825_bootstrap.hpp is a host-only header and cannot be included in device code."
#endif

#include <dlfcn.h>

#include <cctype>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

namespace pto {
namespace comm {
namespace rdma {
namespace hns_1825 {
namespace bootstrap {

constexpr const char* kDefaultRootInfoPath = "/etc/hccl_rootinfo.json";
constexpr const char* kDefaultVirtualTopologyPath = "/var/run/ascend-topologyd/virtualTopology.xml";

// Resolve the global physical device id used by HCOMM and the topology data.
// Returns false (and leaves phyId untouched) if the runtime symbols are unavailable or fail.
inline bool ResolvePhyId(uint32_t& phyId)
{
    using GetDevFn = int (*)(int32_t*);
    using MapFn = int (*)(int32_t, int32_t*);

    auto getDev = reinterpret_cast<GetDevFn>(dlsym(RTLD_DEFAULT, "aclrtGetDevice"));
    auto byLogic = reinterpret_cast<MapFn>(dlsym(RTLD_DEFAULT, "aclrtGetPhyDevIdByLogicDevId"));
    auto byUser = reinterpret_cast<MapFn>(dlsym(RTLD_DEFAULT, "aclrtGetPhyDevIdByUserDevId"));
    auto byIndex = reinterpret_cast<MapFn>(dlsym(RTLD_DEFAULT, "rtGetDevicePhyIdByIndex"));
    if (getDev == nullptr) {
        return false;
    }

    int32_t userId = -1;
    if (getDev(&userId) != 0 || userId < 0) {
        return false;
    }

    int32_t phy = -1;
    // Prefer ByUserDevId; deprecated ByLogicDevId still expects userId (not logic id).
    if (byUser != nullptr && byUser(userId, &phy) == 0 && phy >= 0) {
        phyId = static_cast<uint32_t>(phy);
        return true;
    }
    if (byLogic != nullptr && byLogic(userId, &phy) == 0 && phy >= 0) {
        phyId = static_cast<uint32_t>(phy);
        return true;
    }
    if (byIndex != nullptr && byIndex(userId, &phy) == 0 && phy >= 0) {
        phyId = static_cast<uint32_t>(phy);
        return true;
    }
    return false;
}

namespace detail {

// Read an unsigned integer that follows `"key"` (value may be a bare number or a quoted number).
// Returns false if the key is not present at/after `from`.
inline bool ExtractUintAfterKey(
    const std::string& s, const std::string& key, size_t from, uint32_t& value, size_t& keyPos)
{
    size_t p = s.find(key, from);
    if (p == std::string::npos) {
        return false;
    }
    keyPos = p;
    p += key.size();
    // skip ':' whitespace and an optional opening quote
    while (p < s.size() && (s[p] == ':' || std::isspace(static_cast<unsigned char>(s[p])) || s[p] == '"')) {
        ++p;
    }
    size_t start = p;
    while (p < s.size() && std::isdigit(static_cast<unsigned char>(s[p]))) {
        ++p;
    }
    if (p == start) {
        return false;
    }
    value = static_cast<uint32_t>(std::stoul(s.substr(start, p - start)));
    return true;
}

// Extract the quoted string value that follows `"key"` (e.g. "addr":"10.0.0.1").
inline bool ExtractQuotedAfterKey(
    const std::string& s, const std::string& key, size_t from, std::string& out, size_t limit)
{
    size_t p = s.find(key, from);
    if (p == std::string::npos || p >= limit) {
        return false;
    }
    p = s.find('"', p + key.size());
    if (p == std::string::npos || p >= limit) {
        return false;
    }
    size_t start = p + 1;
    size_t end = s.find('"', start);
    if (end == std::string::npos || end > limit) {
        return false;
    }
    out = s.substr(start, end - start);
    return true;
}

inline bool LooksLikeIpv4(const std::string& v)
{
    int dots = 0;
    for (char c : v) {
        if (c == '.') {
            ++dots;
        } else if (!std::isdigit(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return dots == 3 && !v.empty();
}

} // namespace detail

// Parse the local RDMA NIC IPv4 for `phyId` from a rootinfo JSON file (default /etc/hccl_rootinfo.json).
// Locate the rank whose "device_id" equals phyId, then take the CLOS-level
// ("net_type":"CLOS") IPv4 address from that rank's block. This dependency-free
// targeted scan bounds each rank block by the next "device_id" occurrence;
// level_list entries do not contain "device_id".
inline bool ResolveLocalRdmaIp(uint32_t phyId, std::string& ip)
{
    std::ifstream f(kDefaultRootInfoPath);
    if (!f.is_open()) {
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string s = ss.str();
    if (s.empty()) {
        return false;
    }

    const std::string kDeviceKey = "\"device_id\"";
    size_t scan = 0;
    while (scan < s.size()) {
        uint32_t dev = 0;
        size_t devKeyPos = 0;
        if (!detail::ExtractUintAfterKey(s, kDeviceKey, scan, dev, devKeyPos)) {
            break;
        }
        // Bound this rank's block by the next "device_id" (or EOF).
        size_t nextDev = s.find(kDeviceKey, devKeyPos + kDeviceKey.size());
        size_t blockEnd = (nextDev == std::string::npos) ? s.size() : nextDev;

        if (dev == phyId) {
            size_t closPos = s.find("\"CLOS\"", devKeyPos);
            if (closPos != std::string::npos && closPos < blockEnd) {
                // Scan "addr" entries within the CLOS region and take the first IPv4-looking value.
                size_t cur = closPos;
                std::string cand;
                while (detail::ExtractQuotedAfterKey(s, "\"addr\"", cur, cand, blockEnd)) {
                    if (detail::LooksLikeIpv4(cand)) {
                        ip = cand;
                        return true;
                    }
                    size_t adv = s.find("\"addr\"", cur);
                    if (adv == std::string::npos) {
                        break;
                    }
                    cur = adv + 6; // len("\"addr\"")
                }
            }
            return false;
        }
        scan = blockEnd;
    }
    return false;
}

// Resolve the RoCE IPv4 selected for `phyId` by HCOMM's virtual-topology parser.
// The parser consumes /var/run/ascend-topologyd/virtualTopology.xml directly; it
// does not generate or modify rootinfo. The symbol belongs to HCOMM's topology
// component, so older installations may not provide it and callers must retain
// another bootstrap fallback.
inline bool ResolveLocalRdmaIpFromVirtualTopology(uint32_t phyId, std::string& ip)
{
    if (phyId > static_cast<uint32_t>(INT_MAX)) {
        return false;
    }

    using ResolveFn = int (*)(int, char*, size_t);
    void* topoHandle = nullptr;
    auto resolve = reinterpret_cast<ResolveFn>(dlsym(RTLD_DEFAULT, "GetRoceIpFromXml"));
    if (resolve == nullptr) {
        topoHandle = dlopen("libtopoaddrinfo.so", RTLD_NOW | RTLD_LOCAL);
        if (topoHandle == nullptr) {
            return false;
        }
        resolve = reinterpret_cast<ResolveFn>(dlsym(topoHandle, "GetRoceIpFromXml"));
    }

    char ipBuffer[64] = {0};
    const bool resolved = resolve != nullptr && resolve(static_cast<int>(phyId), ipBuffer, sizeof(ipBuffer)) == 0 &&
                          detail::LooksLikeIpv4(ipBuffer);
    if (resolved) {
        ip = ipBuffer;
    }
    if (topoHandle != nullptr) {
        dlclose(topoHandle);
    }
    return resolved;
}

} // namespace bootstrap
} // namespace hns_1825
} // namespace rdma
} // namespace comm
} // namespace pto

#endif // PTO_TESTS_NPU_A5_COMM_ST_HNS1825_BOOTSTRAP_HPP
