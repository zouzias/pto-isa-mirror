/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_HOST_DETAIL_HOST_TRIGGER_CKE_IMPL_HPP
#define PTO_HOST_DETAIL_HOST_TRIGGER_CKE_IMPL_HPP

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <dlfcn.h>

#include "pto/host/pto_gate_descriptor.hpp"

namespace pto {
namespace host {
namespace detail {

// =============================================================================
// Driver ABI mirror
// =============================================================================
//
// We deliberately do NOT `#include <hccp_ctx.h>` here, for two reasons:
//   1. Keeps `include/pto/host/` self-contained — the header tree compiles
//      without CANN driver headers being on the include path.
//   2. CANN driver headers transitively pull in dozens of unrelated symbols.
//
// The mirror below covers exactly the subset we call. Layout matches the
// driver's `struct RaInfo` / `struct CustomChanInfoIn` / `struct CustomChanInfoOut`
// in `hcomm/src/platform/hccp/inc/network/hccp_common.h:151` and
// `hccp_ctx.h:506`. If the driver ABI ever changes, the static_asserts in
// `BuildCkePayload()` will fire on size mismatch.

constexpr int kDriverNetworkOffline = 1;          // enum NetworkMode::NETWORK_OFFLINE
constexpr uint32_t kDriverOpSetCke = 254;         // ccu_u_opcode_t::CCU_U_OP_SET_CKE
constexpr std::size_t kDriverChanDataMax = 2048;  // CUSTOM_CHAN_DATA_MAX_SIZE

// `struct RaInfo` from hccp_common.h
struct DriverRaInfo {
    int mode;
    unsigned int phyId;
};
static_assert(sizeof(DriverRaInfo) == 8, "DriverRaInfo size mismatch");

// `struct CustomChanInfoIn` from hccp_ctx.h
struct DriverCustomChanInfoIn {
    char data[kDriverChanDataMax];
    unsigned int offsetStart;
    unsigned int op;
};
static_assert(sizeof(DriverCustomChanInfoIn) == kDriverChanDataMax + 8,
              "DriverCustomChanInfoIn size mismatch");

// `struct CustomChanInfoOut` from hccp_ctx.h
struct DriverCustomChanInfoOut {
    char data[kDriverChanDataMax];
    unsigned int offsetNext;
    int opRet;
};
static_assert(sizeof(DriverCustomChanInfoOut) == kDriverChanDataMax + 8,
              "DriverCustomChanInfoOut size mismatch");

// `int RaCustomChannel(struct RaInfo info, struct CustomChanInfoIn *in,
//                      struct CustomChanInfoOut *out);`
// Note: the first arg is passed by value, not by pointer.
using RaCustomChannelFn = int (*)(DriverRaInfo, DriverCustomChanInfoIn *, DriverCustomChanInfoOut *);

// =============================================================================
// libhccp.so lazy loader
// =============================================================================

class LibHccp {
public:
    static LibHccp &Instance() {
        static LibHccp inst;
        std::call_once(inst.init_flag_, [&]() { inst.Load(); });
        return inst;
    }

    bool Ok() const { return ok_; }
    const std::string &ErrorMessage() const { return error_; }
    RaCustomChannelFn RaCustomChannel() const { return ra_custom_channel_; }

private:
    LibHccp() = default;

    void Load() {
        // RaCustomChannel is provided by HCCP. The library it lives in has been
        // renamed across CANN/driver versions, so we try a fixed list of known
        // names (newest first) before giving up. The user can override the list
        // entirely via PTO_HCCP_LIBNAME (single name; `:` separated for multiple).
        //
        //   - libra.so   : CANN ≥ 9.0.0 (driver 7.0.t9.0.B806 verified 2026-04-27)
        //   - libhccp.so : legacy alias used in older driver packages
        //
        // RTLD_GLOBAL so that downstream loads of libraries that themselves
        // depend on libra/libhccp (rare here, but defensive) resolve cleanly.
        std::vector<std::string> candidates;
        const char *override_env = std::getenv("PTO_HCCP_LIBNAME");
        if (override_env != nullptr && override_env[0] != '\0') {
            std::string spec = override_env;
            std::size_t start = 0;
            for (;;) {
                std::size_t end = spec.find(':', start);
                if (end == std::string::npos) {
                    candidates.emplace_back(spec.substr(start));
                    break;
                }
                candidates.emplace_back(spec.substr(start, end - start));
                start = end + 1;
            }
        } else {
            candidates = {"libra.so", "libhccp.so"};
        }

        std::string accumulated_error;
        for (const std::string &name : candidates) {
            handle_ = dlopen(name.c_str(), RTLD_LAZY | RTLD_GLOBAL);
            if (handle_ != nullptr) {
                tried_name_ = name;
                break;
            }
            const char *err = dlerror();
            accumulated_error += "dlopen " + name + " failed: " +
                (err == nullptr ? "(no detail)" : err) + "; ";
        }
        if (handle_ == nullptr) {
            error_ = accumulated_error.empty()
                ? "dlopen failed for all candidates (no candidates configured)"
                : accumulated_error;
            return;
        }

        void *sym = dlsym(handle_, "RaCustomChannel");
        if (sym == nullptr) {
            const char *err = dlerror();
            error_ = "dlsym RaCustomChannel failed in " + tried_name_ + ": " +
                     (err == nullptr ? "(no detail)" : err);
            return;
        }
        ra_custom_channel_ = reinterpret_cast<RaCustomChannelFn>(sym);
        ok_ = true;
    }

    std::once_flag init_flag_;
    void *handle_ = nullptr;
    RaCustomChannelFn ra_custom_channel_ = nullptr;
    std::string error_;
    std::string tried_name_;
    bool ok_ = false;
};

// =============================================================================
// CKE payload builder
// =============================================================================
//
// The driver interprets `CustomChanInfoIn::data[2048]` as a `union ccu_data_union`,
// which casts to `struct ccu_data` (`ccu_u_comm.h:208`):
//
//     struct ccu_data {
//         unsigned int udie_idx;                      // [0..3]
//         unsigned int data_array_len;                // [4..7]   = 8 * dataArraySize
//         unsigned int data_array_size;               // [8..11]
//         union ccu_data_type_union data_array[8];    // [12..]
//     };
//
// `union ccu_data_type_union`'s largest member is `ccu_data_byte64 { char raw[64]; }`,
// and one of its members is `ccu_data_byte8 { char raw[8]; }`. For SET_CKE the
// payload-of-interest is the `byte8` view of each slot, but the slot stride is
// the full union width (= 64 B on 64-bit ABIs, alignment driven by `void *` in
// `ccu_baseinfo`). This matches what `CcuComponent::CleanDieCkes()` does in
// `hcomm/src/legacy/.../ccu_component.cpp:789`: it sets `dataArraySize = N`,
// `dataLen = 8*N`, and writes nothing else (relying on the zero-init).
//
// Mask layout inside the 8-byte slot — empirically determined on Ascend 950 with
// CANN 9.0.0 + driver 7.0.t9.0.B806 (2026-04-27, hccl reduce_scatter Mesh1D gate
// run). The four candidates were each driven through pto::host::HostTriggerCke
// against a real reduce_scatter kernel parked at WaitCKE; layout B was the only
// one that woke the kernel up. Default switched A → B accordingly.
//
//   PTO_CKE_LAYOUT=A            raw[0..1] = uint16 LE       <- doesn't release CKE on 950
//   PTO_CKE_LAYOUT=B  (default) raw[6..7] = uint16 LE       <- VERIFIED on 950
//   PTO_CKE_LAYOUT=C            raw[0..3] = uint32 LE       <- doesn't release CKE on 950
//   PTO_CKE_LAYOUT=D            raw[0..1] = uint16 BE       <- doesn't release CKE on 950
//
// The other three layouts are kept around as escape hatches for future chip
// revisions / driver ABI bumps; if a new SoC turns out to need a different byte
// position, switching `PTO_CKE_LAYOUT` lets us re-pin without a code change.
//
// All four overwrite only the first 8 bytes of the slot; the remaining 56 are
// kept zero from `memset`, so wrong guesses are harmless on the wire (driver
// sees ill-formed mask, CCU stays gated, kernel hangs — easy to spot).

enum class CkeLayout : int {
    kALowLE = 0,
    kBHighLE = 1,
    kCWideLE = 2,
    kDLowBE = 3,
};

inline CkeLayout ResolveCkeLayout() {
    const char *env = std::getenv("PTO_CKE_LAYOUT");
    if (env == nullptr) return CkeLayout::kBHighLE;
    if (env[0] == '\0' || env[1] != '\0') return CkeLayout::kBHighLE;
    switch (env[0]) {
        case 'A': case 'a': return CkeLayout::kALowLE;
        case 'B': case 'b': return CkeLayout::kBHighLE;
        case 'C': case 'c': return CkeLayout::kCWideLE;
        case 'D': case 'd': return CkeLayout::kDLowBE;
        default: return CkeLayout::kBHighLE;
    }
}

inline void WriteMaskInSlot(char *slot, uint16_t mask, CkeLayout layout) {
    const uint8_t lo = static_cast<uint8_t>(mask & 0xff);
    const uint8_t hi = static_cast<uint8_t>((mask >> 8) & 0xff);
    switch (layout) {
        case CkeLayout::kALowLE:
            slot[0] = static_cast<char>(lo);
            slot[1] = static_cast<char>(hi);
            break;
        case CkeLayout::kBHighLE:
            slot[6] = static_cast<char>(lo);
            slot[7] = static_cast<char>(hi);
            break;
        case CkeLayout::kCWideLE:
            slot[0] = static_cast<char>(lo);
            slot[1] = static_cast<char>(hi);
            slot[2] = 0;
            slot[3] = 0;
            break;
        case CkeLayout::kDLowBE:
            slot[0] = static_cast<char>(hi);
            slot[1] = static_cast<char>(lo);
            break;
    }
}

// Layout constants for `struct ccu_data` inside CustomChanInfoIn::data[].
constexpr std::size_t kCcuDataUdieIdxOffset = 0;
constexpr std::size_t kCcuDataLenOffset = 4;
constexpr std::size_t kCcuDataSizeOffset = 8;
constexpr std::size_t kCcuDataArrayOffset = 12;
constexpr std::size_t kCcuUnionSlotSize = 64;   // sizeof(union ccu_data_type_union)
constexpr std::size_t kCcuByte8Width = 8;       // sizeof(struct ccu_data_byte8)
constexpr std::size_t kCcuMaxArraySize = 8;     // CCU_DATA_TYPE_UNION_ARRAY_SIZE

// BuildCkePayload: zero-init `in` then populate it for one SET_CKE call.
// If `mask_or_null == nullptr`, builds the all-zero payload (chip-mandated
// pre-clear); otherwise writes the 16-bit mask into the first slot.
inline void BuildCkePayload(DriverCustomChanInfoIn *in, uint32_t dieId, uint32_t ckeIdx,
                            const uint16_t *mask_or_null, CkeLayout layout) {
    static_assert(kCcuDataArrayOffset + kCcuUnionSlotSize * kCcuMaxArraySize <= kDriverChanDataMax,
                  "ccu_data layout exceeds CustomChanInfoIn::data buffer");

    std::memset(in, 0, sizeof(*in));
    in->op = kDriverOpSetCke;
    in->offsetStart = ckeIdx;

    auto write_u32 = [&](std::size_t off, uint32_t v) {
        std::memcpy(in->data + off, &v, sizeof(v));
    };
    write_u32(kCcuDataUdieIdxOffset, dieId);
    write_u32(kCcuDataSizeOffset, 1u);                             // dataArraySize = 1
    write_u32(kCcuDataLenOffset, static_cast<uint32_t>(kCcuByte8Width));  // dataLen = 8

    if (mask_or_null != nullptr) {
        char *slot0 = in->data + kCcuDataArrayOffset;
        WriteMaskInSlot(slot0, *mask_or_null, layout);
    }
}

inline int32_t HostTriggerCkeImpl(uint32_t devPhyId, const PtoGateDescriptor &desc) {
    // Sanity check: kernels that haven't been Translate()'d yet leave dieId in
    // an out-of-range state; reject early to give a clean error instead of a
    // silent driver hang.
    if (desc.mask == 0) {
        return -4;  // kHostTriggerCkeBadArg — degenerate gate
    }

    LibHccp &lib = LibHccp::Instance();
    if (!lib.Ok()) {
        return -1;  // kHostTriggerCkeDlopenFailed
    }

    static const CkeLayout kLayout = ResolveCkeLayout();

    DriverRaInfo info{kDriverNetworkOffline, devPhyId};
    DriverCustomChanInfoIn in{};
    DriverCustomChanInfoOut out{};

    // Step 1: clear (chip rule: SET_CKE requires a prior write of zeros).
    BuildCkePayload(&in, desc.dieId, desc.ckeId, /*mask_or_null=*/nullptr, kLayout);
    int rc = lib.RaCustomChannel()(info, &in, &out);
    if (rc != 0 || out.opRet != 0) {
        return -2;  // kHostTriggerCkeClearFailed
    }

    // Step 2: write the actual mask.
    const uint16_t mask16 = static_cast<uint16_t>(desc.mask & 0xffff);
    BuildCkePayload(&in, desc.dieId, desc.ckeId, &mask16, kLayout);
    rc = lib.RaCustomChannel()(info, &in, &out);
    if (rc != 0 || out.opRet != 0) {
        return -3;  // kHostTriggerCkeSetFailed
    }

    return 0;  // kHostTriggerCkeOk
}

} // namespace detail

// Definition of the public API — declared inline in `host_trigger_cke.hpp`.
inline int32_t HostTriggerCke(uint32_t devPhyId, const PtoGateDescriptor &desc) {
    return detail::HostTriggerCkeImpl(devPhyId, desc);
}

} // namespace host
} // namespace pto

#endif // PTO_HOST_DETAIL_HOST_TRIGGER_CKE_IMPL_HPP
