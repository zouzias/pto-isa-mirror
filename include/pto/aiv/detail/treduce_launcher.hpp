/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_AIV_DETAIL_TREDUCE_LAUNCHER_HPP
#define PTO_AIV_DETAIL_TREDUCE_LAUNCHER_HPP

// Inline host-side launcher for the AIV trigger kernel.
//
// The kernel itself is a separate translation unit compiled with the AscendC
// toolchain (`--cce-aicore-arch=dav-c310-vec`) into `libpto_aiv_treduce.so`,
// exporting one C symbol:
//
//     extern "C" int pto_aiv_treduce_launch(
//         void *stream,
//         uint64_t mmioAddr, uint32_t dieId, uint32_t ckeId, uint32_t mask,
//         uint64_t stride, uint64_t byte_off);
//
// We `dlopen` that .so on first call (RTLD_LAZY|RTLD_GLOBAL, with the same
// candidate-search-list pattern as `LibHccp` in host_trigger_cke_impl.hpp),
// `dlsym` the entry point, and forward the call. Stride/byte-off come from
// env (PTO_AIV_TRIGGER_STRIDE, PTO_AIV_TRIGGER_BYTE_OFF) with safe defaults
// matching host PTO_CKE_LAYOUT=B (stride=0x40, byte_off=6).
//
// Defaults are bake-in candidates; production should pin them once empirical
// sweep confirms the right values for the target driver/CANN version.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <dlfcn.h>

#include "pto/aiv/treduce.hpp"
#include "pto/host/pto_gate_descriptor.hpp"

namespace pto {
namespace aiv {
namespace detail {

// Mirror of the AIV kernel's exported entry point.
using LaunchFn = int (*)(void * /*stream*/,
                          uint64_t /*mmioAddr*/, uint32_t /*dieId*/,
                          uint32_t /*ckeId*/, uint32_t /*mask*/,
                          uint64_t /*stride*/, uint64_t /*byte_off*/);

class LibAivTreduce {
public:
    static LibAivTreduce &Instance() {
        static LibAivTreduce inst;
        std::call_once(inst.init_flag_, [&]() { inst.Load(); });
        return inst;
    }

    bool Ok() const { return ok_; }
    LaunchFn Launch() const { return launch_; }
    const std::string &ErrorMessage() const { return error_; }

private:
    LibAivTreduce() = default;

    void Load() {
        // Candidate search order (override via PTO_AIV_TREDUCE_LIB env):
        //   - libpto_aiv_treduce.so   ← canonical name
        //   - libtreduce.so           ← legacy alt
        std::vector<std::string> candidates;
        const char *override_env = std::getenv("PTO_AIV_TREDUCE_LIB");
        if (override_env != nullptr && override_env[0] != '\0') {
            candidates.emplace_back(override_env);
        } else {
            candidates = {"libpto_aiv_treduce.so", "libtreduce.so"};
        }

        std::string acc_err;
        for (const std::string &name : candidates) {
            handle_ = dlopen(name.c_str(), RTLD_LAZY | RTLD_GLOBAL);
            if (handle_ != nullptr) {
                tried_name_ = name;
                break;
            }
            const char *e = dlerror();
            acc_err += "dlopen " + name + " failed: " +
                       (e == nullptr ? "(no detail)" : e) + "; ";
        }
        if (handle_ == nullptr) {
            error_ = acc_err.empty() ? "no candidate" : acc_err;
            return;
        }

        void *sym = dlsym(handle_, "pto_aiv_treduce_launch");
        if (sym == nullptr) {
            const char *e = dlerror();
            error_ = "dlsym pto_aiv_treduce_launch failed in " + tried_name_ + ": " +
                     (e == nullptr ? "(no detail)" : e);
            return;
        }
        launch_ = reinterpret_cast<LaunchFn>(sym);
        ok_ = true;
    }

    std::once_flag init_flag_;
    void *handle_ = nullptr;
    LaunchFn launch_ = nullptr;
    std::string error_;
    std::string tried_name_;
    bool ok_ = false;
};

// Resolve stride / byte-off from env, with defaults matching host PTO_CKE_LAYOUT=B.
//
// Initial sweep candidates (empirical):
//   STRIDE          BYTE_OFF
//   ─────────────────────────────────────────────────────────────
//   0x40 (= 64)     6        (host layout B; ccu_data_type_union slot)
//   0x40            0        (slot start, layout A position)
//   0x08            0        (compact 8-byte stride; ccu_data_byte8 slot)
//   0x02            0        (16-bit packed; minimum)
//   0x10            0        (32-byte stride; ccu_data_byte32)
//
// Override via `PTO_AIV_TRIGGER_STRIDE=0x40` etc.
inline uint64_t ResolveStride() {
    const char *env = std::getenv("PTO_AIV_TRIGGER_STRIDE");
    if (env == nullptr || env[0] == '\0') return 0x40ULL;
    return std::strtoull(env, nullptr, 0);  // base=0 → auto-detect 0x prefix
}
inline uint64_t ResolveByteOff() {
    const char *env = std::getenv("PTO_AIV_TRIGGER_BYTE_OFF");
    if (env == nullptr || env[0] == '\0') return 6ULL;
    return std::strtoull(env, nullptr, 0);
}

} // namespace detail

inline int32_t launch_treduce(void *stream, const host::PtoGateDescriptor &desc) {
    if (stream == nullptr || desc.mask == 0 || desc.ckeId > 4096) {
        std::fprintf(stderr,
            "[PTO_AIV_TREDUCE] BadArg stream=%p mask=0x%x ckeId=%u\n",
            stream, desc.mask, desc.ckeId);
        return kLaunchTReduceBadArg;
    }
    if (desc.mmioAddr == 0) {
        std::fprintf(stderr,
            "[PTO_AIV_TREDUCE] mmioAddr is 0 — caller must call "
            "pto::host::QueryCcuBaseInfo and fill desc.mmioAddr first.\n");
        return kLaunchTReduceMmioAddrZero;
    }

    detail::LibAivTreduce &lib = detail::LibAivTreduce::Instance();
    if (!lib.Ok()) {
        std::fprintf(stderr,
            "[PTO_AIV_TREDUCE] kernel .so load failed: %s\n",
            lib.ErrorMessage().c_str());
        return kLaunchTReduceLoadFailed;
    }

    const uint64_t stride   = detail::ResolveStride();
    const uint64_t byte_off = detail::ResolveByteOff();

    std::fprintf(stderr,
        "[PTO_AIV_TREDUCE] launching: mmioAddr=0x%lx dieId=%u ckeId=%u "
        "mask=0x%x stride=0x%lx byte_off=%lu (target=0x%lx)\n",
        static_cast<unsigned long>(desc.mmioAddr),
        desc.dieId, desc.ckeId, desc.mask,
        static_cast<unsigned long>(stride),
        static_cast<unsigned long>(byte_off),
        static_cast<unsigned long>(desc.mmioAddr + desc.ckeId * stride + byte_off));

    int rc = lib.Launch()(stream, desc.mmioAddr, desc.dieId, desc.ckeId,
                           desc.mask, stride, byte_off);
    if (rc != 0) {
        std::fprintf(stderr,
            "[PTO_AIV_TREDUCE] aclrtLaunchKernel returned %d\n", rc);
        return kLaunchTReduceLaunchFailed;
    }
    return kLaunchTReduceOk;
}

} // namespace aiv
} // namespace pto

#endif // PTO_AIV_DETAIL_TREDUCE_LAUNCHER_HPP
