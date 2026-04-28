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

// RaTlvInit / RaTlvRequest / RaTlvDeinit — TLV (Type-Length-Value) protocol
// for richer device queries that don't fit the CustomChannel ABI. Used by
// hccp_test.so's test_ra_get_mem_info to query CCU mem regions with non-null
// `mem_va`. Signatures inferred from libra.so disassembly + hccp_test.so call sites.
//
// RaTlvRequest argument layout (from `objdump -d libra.so` at +0x247a7..):
//   rdi: void *tlv_handle  (returned by RaTlvInit)
//   esi: uint32_t type     (request type; test_ra_get_mem_info uses 1)
//   rdx: void *in_data     (per-request input struct)
//   rcx: void *out_data    (per-request output struct)
using RaTlvInitFn    = int (*)(void **out_handle);
using RaTlvRequestFn = int (*)(void *handle, uint32_t type, void *in_data, void *out_data);
using RaTlvDeinitFn  = int (*)(void *handle);

// ccu_mem_info / ccu_mem_rsp from hcomm/.../ccu_u_comm.h:282
//   struct ccu_mem_info { uint64_t mem_va; uint32_t mem_size; uint32_t resv[1]; }; // 16B
//   struct ccu_mem_rsp  { uint32_t die_id; uint32_t num; ccu_mem_info list[64]; }; // 8 + 64*16 = 1032B
struct DiagCcuMemInfo {
    uint64_t mem_va;       // ← THE region VA we want
    uint32_t mem_size;
    uint32_t resv[1];
};
struct DiagCcuMemRsp {
    uint32_t die_id;
    uint32_t num;
    DiagCcuMemInfo list[64];
};
static_assert(sizeof(DiagCcuMemInfo) == 16, "ccu_mem_info ABI size mismatch");
static_assert(sizeof(DiagCcuMemRsp)  == 8 + 64 * 16, "ccu_mem_rsp ABI size mismatch (= 0x408)");

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
    // Lazy lookup of the TLV trio — only resolved if the diag path actually
    // touches them. Failure is non-fatal; nullptr just means the diag print
    // skips the TLV branch.
    RaTlvInitFn RaTlvInit() const {
        if (handle_ != nullptr && ra_tlv_init_ == nullptr) {
            ra_tlv_init_ = (RaTlvInitFn)dlsym(handle_, "RaTlvInit");
        }
        return ra_tlv_init_;
    }
    RaTlvRequestFn RaTlvRequest() const {
        if (handle_ != nullptr && ra_tlv_request_ == nullptr) {
            ra_tlv_request_ = (RaTlvRequestFn)dlsym(handle_, "RaTlvRequest");
        }
        return ra_tlv_request_;
    }
    RaTlvDeinitFn RaTlvDeinit() const {
        if (handle_ != nullptr && ra_tlv_deinit_ == nullptr) {
            ra_tlv_deinit_ = (RaTlvDeinitFn)dlsym(handle_, "RaTlvDeinit");
        }
        return ra_tlv_deinit_;
    }

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
    mutable RaTlvInitFn    ra_tlv_init_    = nullptr;
    mutable RaTlvRequestFn ra_tlv_request_ = nullptr;
    mutable RaTlvDeinitFn  ra_tlv_deinit_  = nullptr;
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

// Diag opcode: same custom-channel ABI, but request CCU resource base info instead
// of writing a CKE. The driver returns a `ccu_u_info`-shaped struct in `out.data`,
// whose `resourceAddr` field carries the CCU resource MMIO mapping VA we need for
// Step 3 AIV-trigger prototyping. Triggered once per (devPhyId, dieId) on first
// HostTriggerCke call.
constexpr uint32_t kDriverOpGetBasicInfo = 11;  // CCU_U_OP_GET_BASIC_INFO

// Mirror of struct ccu_u_info from hcomm/src/platform/hccp/external_depends/ccu/ccu_u_comm.h.
// Only the prefix up to resourceAddr matters; we cast the returned data buffer to this.
struct DiagCcuUInfo {
    unsigned int uent_num;
    unsigned int ccu_flag;
    unsigned int eid;
    unsigned int ms_id;
    unsigned int missionKey;
    void *resourceAddr;   // ← CCU resource VA
    // remaining fields (caps, version) intentionally elided
};

inline void DiagPrintCcuResourceAddrOnce(uint32_t devPhyId, uint32_t dieId,
                                          RaCustomChannelFn ra_fn) {
    static std::mutex diag_mu;
    static std::vector<std::pair<uint32_t, uint32_t>> already_dumped;
    std::lock_guard<std::mutex> lk(diag_mu);
    for (auto &kv : already_dumped) {
        if (kv.first == devPhyId && kv.second == dieId) return;
    }
    already_dumped.emplace_back(devPhyId, dieId);

    DriverRaInfo info{kDriverNetworkOffline, devPhyId};
    DriverCustomChanInfoIn in{};
    DriverCustomChanInfoOut out{};

    in.op = kDriverOpGetBasicInfo;
    in.offsetStart = 0;
    // ccu_data layout: [0..3]=udie_idx, [4..7]=dataLen, [8..11]=dataArraySize
    auto write_u32 = [&](std::size_t off, uint32_t v) {
        std::memcpy(in.data + off, &v, sizeof(v));
    };
    write_u32(kCcuDataUdieIdxOffset, dieId);
    write_u32(kCcuDataLenOffset, static_cast<uint32_t>(sizeof(DiagCcuUInfo)));
    write_u32(kCcuDataSizeOffset, 1u);

    int rc = ra_fn(info, &in, &out);
    std::fprintf(stderr,
        "[DIAG/ccu_baseinfo] dev=%u die=%u: RaCustomChannel(op=GET_BASIC_INFO=11) rc=%d opRet=%d\n",
        devPhyId, dieId, rc, out.opRet);
    if (rc == 0 && out.opRet == 0) {
        DiagCcuUInfo *u = reinterpret_cast<DiagCcuUInfo*>(out.data);
        std::fprintf(stderr,
            "[DIAG/ccu_baseinfo] dev=%u die=%u: uent_num=%u ccu_flag=%u eid=%u ms_id=%u missionKey=%u\n"
            "[DIAG/ccu_baseinfo] dev=%u die=%u: resourceAddr=%p  ← CCU resource VA (Step 3 candidate)\n",
            devPhyId, dieId, u->uent_num, u->ccu_flag, u->eid, u->ms_id, u->missionKey,
            devPhyId, dieId, u->resourceAddr);
    }
}

// Step 3 R&D diag #2: try the TLV path that hccp_test.so's test_ra_get_mem_info
// uses (RaTlvInit + RaTlvRequest(type=1) + RaTlvDeinit). The driver returns a
// `ccu_mem_rsp` populated with `ccu_mem_info[]`, where each entry has a
// non-null `mem_va` — the actual mmap'd CCU resource VAs we need for the AIV
// trigger path. Like the GET_BASIC_INFO diag above, only fires once per
// (devPhyId, dieId), and only when PTO_DIAG_CCU_RESOURCE_ADDR is set.
inline void DiagPrintCcuMemInfoOnce(uint32_t devPhyId, uint32_t dieId, LibHccp &lib) {
    static std::mutex tlv_diag_mu;
    static std::vector<std::pair<uint32_t, uint32_t>> tlv_already_dumped;
    std::lock_guard<std::mutex> lk(tlv_diag_mu);
    for (auto &kv : tlv_already_dumped) {
        if (kv.first == devPhyId && kv.second == dieId) return;
    }
    tlv_already_dumped.emplace_back(devPhyId, dieId);

    auto init_fn    = lib.RaTlvInit();
    auto request_fn = lib.RaTlvRequest();
    auto deinit_fn  = lib.RaTlvDeinit();
    if (init_fn == nullptr || request_fn == nullptr || deinit_fn == nullptr) {
        std::fprintf(stderr,
            "[DIAG/ccu_mem_info] dev=%u die=%u: TLV symbols missing (init=%p req=%p deinit=%p)\n",
            devPhyId, dieId, (void*)init_fn, (void*)request_fn, (void*)deinit_fn);
        return;
    }

    void *tlv_handle = nullptr;
    int rc_init = init_fn(&tlv_handle);
    std::fprintf(stderr,
        "[DIAG/ccu_mem_info] dev=%u die=%u: RaTlvInit rc=%d handle=%p\n",
        devPhyId, dieId, rc_init, tlv_handle);
    if (rc_init != 0 || tlv_handle == nullptr) {
        return;
    }

    // ccu_mem_rsp output buffer (1032 bytes). Driver fills die_id, num, list[].
    DiagCcuMemRsp *rsp = (DiagCcuMemRsp *)std::malloc(sizeof(DiagCcuMemRsp));
    std::memset(rsp, 0, sizeof(*rsp));

    // Input struct from test_ra_get_mem_info disassembly: 16 bytes,
    //   offset 0: uint32_t (mem_type / die_id?)
    //   offset 8: void*    (out buffer pointer)
    struct DiagTlvInData {
        uint32_t arg0;  // we'll put dieId here
        uint32_t pad;
        void    *out_ptr;
    };
    DiagTlvInData *in_data = (DiagTlvInData *)std::malloc(sizeof(DiagTlvInData));
    in_data->arg0    = dieId;
    in_data->pad     = 0;
    in_data->out_ptr = rsp;

    // type=1 matches what test_ra_get_mem_info passes
    int rc_req = request_fn(tlv_handle, /*type=*/1, in_data, /*out=*/nullptr);
    std::fprintf(stderr,
        "[DIAG/ccu_mem_info] dev=%u die=%u: RaTlvRequest(type=1) rc=%d, rsp.die_id=%u rsp.num=%u\n",
        devPhyId, dieId, rc_req, rsp->die_id, rsp->num);

    if (rc_req == 0 && rsp->num > 0 && rsp->num <= 64) {
        for (uint32_t i = 0; i < rsp->num; ++i) {
            std::fprintf(stderr,
                "[DIAG/ccu_mem_info]   list[%u]: mem_va=0x%016llx mem_size=%u\n",
                i,
                (unsigned long long)rsp->list[i].mem_va,
                rsp->list[i].mem_size);
        }
    }

    std::free(in_data);
    std::free(rsp);
    deinit_fn(tlv_handle);
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

    // Step 3 prototyping diag: dump CCU resourceAddr once per (devPhyId, dieId).
    // This piggybacks on the already-initialized HDC session that's about to do
    // the SET_CKE — RaCustomChannel will succeed without separate ACL/HCCL setup.
    if (std::getenv("PTO_DIAG_CCU_RESOURCE_ADDR") != nullptr) {
        DiagPrintCcuResourceAddrOnce(devPhyId, desc.dieId, lib.RaCustomChannel());
        // Try the TLV path too — likely returns non-null mem_va that we need.
        DiagPrintCcuMemInfoOnce(devPhyId, desc.dieId, lib);
    }

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
