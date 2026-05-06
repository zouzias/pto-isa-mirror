/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0.
 * Please refer to the License for details. You may not use this file except in
 * compliance with the License.
 */

// pto-isa gated reduce-scatter standalone ST.
// =============================================================================
//
// Demonstrates the pto-native gated CCU kernel pilot end-to-end:
//
//   ┌── Per-rank flow ──────────────────────────────────────────────────────┐
//   │                                                                       │
//   │  acl/hccl init  →  HcclThreadAcquire(CCU)  →  alloc dev buffers       │
//   │                                                                       │
//   │  HcclCcuKernelRegister(creator=MakeGatedKernelCreator,                │
//   │                        arg=PtoGatedKernelArg{...})                    │
//   │  HcclCcuKernelRegisterFinish  ← hcomm translates IR → CCU microcode   │
//   │  HcclCcuKernelLaunch(taskArg=PtoGatedTaskArg{addr/len/token})         │
//   │      → CCU stream parks at WaitEvent(gateEvent_)                      │
//   │                                                                       │
//   │  pto::ccu::TryGet(rankId, &desc)  ← read published (dieId,ckeId,mask) │
//   │  pto::host::QueryCcuBaseInfo(devId, dieId)  ← MMIO base VA            │
//   │  desc.mmioAddr = probe.resourceAddr                                   │
//   │                                                                       │
//   │  pto::aiv::launch_treduce(aivStream, desc, marker)                    │
//   │      → AIV-side MMIO store releases CKE gate                          │
//   │      → CCU resumes, runs LocalCopyNb(input → output), RecordEvent      │
//   │                                                                       │
//   │  aclrtSynchronizeStream(stream)                                       │
//   │  verify output == input                                               │
//   │                                                                       │
//   └───────────────────────────────────────────────────────────────────────┘
//
// Pilot scope (matches `include/pto/ccu/pto_gated_kernel.hpp`):
//   - The kernel performs a placeholder identity copy (correct for nranks=1,
//     stand-in for nranks>1). Real GroupReduce data path is phase 3.
//   - hccl repo is not used by this ST except via its public C ABI symbols
//     in libhccl.so / libhcomm.so. No hccl-internal headers (`OpParam`,
//     `CcuKernelAlgBase`, ...) are required.
//
// Build & run (server side):
//
//   # 1. build registry + kernel libs (once per pto-isa checkout)
//   cmake -S kernels/host/pto_ccu_host -B build/pto_ccu_host
//   cmake --build build/pto_ccu_host -j
//
//   cmake -S kernels/host/gated_reduce_scatter
//         -B build/pto_gated_reduce_scatter
//         -DPTO_CCU_HOST_LIB=$PWD/build/pto_ccu_host/dist/libpto_ccu_host.so
//   cmake --build build/pto_gated_reduce_scatter -j
//
//   # 2. build this ST
//   cmake -S tests/host/gated_reduce_scatter
//         -B build/gated_rs_st
//         -DPTO_GATED_RS_LIB_DIR=$PWD/build/pto_gated_reduce_scatter/dist
//         -DPTO_CCU_HOST_LIB_DIR=$PWD/build/pto_ccu_host/dist
//   cmake --build build/gated_rs_st -j
//
//   # 3. run (1-rank smoke)
//   export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
//   export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:
//     $PWD/build/pto_ccu_host/dist:
//     $PWD/build/pto_gated_reduce_scatter/dist:
//     <libpto_aiv_treduce.so dir>:$LD_LIBRARY_PATH
//   timeout 60 ./build/gated_rs_st/gated_rs_st
//
//   # expected: "[GATED_RS_ST] PASS rank=0" + rc=0
//   # if hang  : timeout 124 — gate not released, check pto::aiv::launch_treduce
//   #            return code in stderr

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ACL / HCCL public C APIs (from CANN install, ${ASCEND_HOME_PATH}/include).
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_res.h"             // HcclThreadAcquire / CommEngine enum
#include "hcomm/ccu/hccl_ccu_res.h"    // HcclCcuKernelRegister / Launch
#include "hcomm/ccu/ccu_assist_pub.h"  // GetTokenInfo

// pto-isa public APIs.
#include "pto/aiv/treduce.hpp"           // pto::aiv::launch_treduce
#include "pto/ccu/pto_gate_registry.hpp" // pto::ccu::TryGet
#include "pto/ccu/pto_gated_kernel.hpp"  // pto::ccu::PtoGatedKernelArg / TaskArg / MakeGatedKernelCreator
#include "pto/host/pto_gate_descriptor.hpp"
#include "pto/host/query_ccu_baseinfo.hpp"

// =============================================================================
// Helpers
// =============================================================================
namespace {

#define ACL_OK(expr)                                                       \
    do {                                                                   \
        aclError _r = (expr);                                              \
        if (_r != ACL_SUCCESS) {                                           \
            std::fprintf(stderr,                                           \
                "[GATED_RS_ST] ACL FAIL %s = %d at %s:%d\n",               \
                #expr, static_cast<int>(_r), __FILE__, __LINE__);          \
            std::exit(2);                                                  \
        }                                                                  \
    } while (0)

#define HCCL_OK(expr)                                                      \
    do {                                                                   \
        HcclResult _r = (expr);                                            \
        if (_r != HCCL_SUCCESS) {                                          \
            std::fprintf(stderr,                                           \
                "[GATED_RS_ST] HCCL FAIL %s = %d at %s:%d\n",              \
                #expr, static_cast<int>(_r), __FILE__, __LINE__);          \
            std::exit(2);                                                  \
        }                                                                  \
    } while (0)

constexpr int    kDeviceId    = 0;        // single-rank pilot uses device 0
constexpr size_t kPayloadSize = 4096;     // 4 KB placeholder payload
constexpr size_t kAivMarkerBytes = 64;    // 8×u64 diagnostic buffer for AIV trigger

// Single-rank HcclRootInfo bootstrap. For multi-rank, use MPI_Bcast as
// existing pto-isa ST framework does (see `tests/npu/a5/comm/st/testcase/
// common.hpp::ForkAndRunWithHcclRootInfo`).
void InitHcclSingleRank(HcclComm *outComm)
{
    HcclRootInfo rootInfo{};
    HCCL_OK(HcclGetRootInfo(&rootInfo));
    HCCL_OK(HcclCommInitRootInfo(/*nRanks=*/1, &rootInfo, /*rank=*/0, outComm));
}

}  // namespace

// =============================================================================
// Main
// =============================================================================
int main(int /*argc*/, char ** /*argv*/)
{
    std::fprintf(stderr, "[GATED_RS_ST] start, payloadBytes=%zu\n", kPayloadSize);

    // -------------------------------------------------------------------------
    // 1. ACL init + device + ACL stream (= the "main" stream the gated CCU
    //    kernel runs on).
    // -------------------------------------------------------------------------
    ACL_OK(aclInit(nullptr));
    ACL_OK(aclrtSetDevice(kDeviceId));

    aclrtStream stream = nullptr;
    ACL_OK(aclrtCreateStream(&stream));

    // Separate AIV stream for the trigger kernel — CRITICAL: enqueueing on
    // the same stream as the gated CCU kernel deadlocks (the gated kernel
    // must be in flight before the trigger fires; same-stream serialization
    // makes that impossible).
    aclrtStream aivStream = nullptr;
    ACL_OK(aclrtCreateStream(&aivStream));

    // -------------------------------------------------------------------------
    // 2. HcclComm bootstrap (single rank).
    // -------------------------------------------------------------------------
    HcclComm comm = nullptr;
    InitHcclSingleRank(&comm);
    std::fprintf(stderr, "[GATED_RS_ST] HcclComm init OK (single rank)\n");

    // -------------------------------------------------------------------------
    // 3. Acquire CCU thread.
    //
    // notifyNumPerThread is a hint for how many notify slots the CCU thread
    // pre-allocates. The placeholder kernel uses none (no NotifyRecord/Wait
    // in our minimal Algorithm()), so 1 is sufficient. Bump if the kernel
    // grows real channel sync logic.
    // -------------------------------------------------------------------------
    constexpr uint32_t kThreadNum            = 1;
    constexpr uint32_t kNotifyNumPerThread   = 1;
    ThreadHandle threadHandle = 0;
    HCCL_OK(HcclThreadAcquire(comm, COMM_ENGINE_CCU, kThreadNum,
                              kNotifyNumPerThread, &threadHandle));
    std::fprintf(stderr, "[GATED_RS_ST] HcclThreadAcquire OK threadHandle=0x%llx\n",
                 static_cast<unsigned long long>(threadHandle));

    // -------------------------------------------------------------------------
    // 4. Allocate device input/output + diagnostic AIV marker buffer.
    //
    // For single-rank pilot we just use raw `aclrtMalloc` GM allocations.
    // Multi-rank version would register into the comm window via
    // `HcclCommMemReg` so peer ranks can address remote buffers.
    // -------------------------------------------------------------------------
    void *inputDev  = nullptr;
    void *outputDev = nullptr;
    ACL_OK(aclrtMalloc(&inputDev,  kPayloadSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_OK(aclrtMalloc(&outputDev, kPayloadSize, ACL_MEM_MALLOC_HUGE_ONLY));

    // Initialize input with a deterministic byte pattern; output to known-bad.
    std::vector<uint8_t> inputHost(kPayloadSize);
    std::vector<uint8_t> outputHost(kPayloadSize, 0xFF);
    for (size_t i = 0; i < kPayloadSize; ++i) {
        inputHost[i] = static_cast<uint8_t>(i & 0xFF);
    }
    ACL_OK(aclrtMemcpy(inputDev,  kPayloadSize, inputHost.data(),  kPayloadSize,
                       ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemcpy(outputDev, kPayloadSize, outputHost.data(), kPayloadSize,
                       ACL_MEMCPY_HOST_TO_DEVICE));

    void *aivMarkerDev = nullptr;
    ACL_OK(aclrtMalloc(&aivMarkerDev, kAivMarkerBytes, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_OK(aclrtMemset(aivMarkerDev,  kAivMarkerBytes, 0, kAivMarkerBytes));

    const uint64_t inputVa  = reinterpret_cast<uint64_t>(inputDev);
    const uint64_t outputVa = reinterpret_cast<uint64_t>(outputDev);

    // -------------------------------------------------------------------------
    // 5. Register + translate the gated kernel.
    //
    // ORDERING INVARIANT (learned 2026-05-06):
    //   `hcomm::CcuRep::GetTokenInfo(va, size)` MUST be called AFTER
    //   `HcclCcuKernelRegisterFinish` returns. Calling it earlier throws
    //   `Hccl::CcuApiException: failed to query tokenInfo` because the CCU
    //   translation context that maps GM VA → CCU token is only built up
    //   during `RegisterFinish` (when hcomm runs `Translate()` on each
    //   registered kernel and stages the microcode + mem aperture).
    //
    //   This matches hccl's own internal ordering — see
    //   `hccl/src/ops/op_common/op_common.cc:1201-1206` (Register/Finish)
    //   then `op_common.cc:474-496` (Orchestrate → KernelRun) then
    //   `hccl/src/ops/reduce_scatter/template/ccu/ccu_temp_reduce_scatter_mesh_1D.cc:99-138`
    //   where `GetToken(buffInfo_, token)` is called inside `KernelRun`,
    //   strictly after the resource path has run `RegisterFinish`. hccl never
    //   exposes this ordering as a public contract — it's enforced
    //   structurally by the framework. Standalone CCU users (us) have to
    //   replicate it manually.
    //
    //   Since `token` is a task-level (per-launch) parameter, not a
    //   kernel-level (per-register) parameter, deferring its query until
    //   after RegisterFinish is fine — `PtoGatedTaskArg` carries it into
    //   `GeneArgs()` which runs during `HcclCcuKernelLaunch`.
    // -------------------------------------------------------------------------
    pto::ccu::PtoGatedKernelArg karg{
        /*rankId=*/   0,
        /*rankSize=*/ 1,
        /*payloadBytes=*/ kPayloadSize,
        /*gateMask=*/ 1u << 0,
        /*doneMask=*/ 1u << 0,
    };

    hcomm::KernelCreator creator = pto::ccu::MakeGatedKernelCreator();
    CcuKernelHandle      kHandle = 0;

    // The C ABI of HcclCcuKernelRegister is `void *kernelCreator, void *kernelArg`
    // — we pass pointers to our typed objects. The hcomm impl reinterprets
    // `kernelCreator` as `hcomm::KernelCreator*` and `kernelArg` as
    // `hcomm::CcuKernelArg*` (see `coll_comm_res_c_adpt.cc:315-317`).
    HCCL_OK(HcclCcuKernelRegister(comm, &kHandle, &creator, &karg));
    std::fprintf(stderr,
        "[GATED_RS_ST] HcclCcuKernelRegister OK kHandle=0x%llx\n",
        static_cast<unsigned long long>(kHandle));

    HCCL_OK(HcclCcuKernelRegisterFinish(comm));
    std::fprintf(stderr,
        "[GATED_RS_ST] HcclCcuKernelRegisterFinish OK — microcode translated\n");

    // -------------------------------------------------------------------------
    // 6. Query CCU token AFTER translate.
    //
    // Token covers BOTH input + output regions. `GetTokenInfo` returns a
    // 64-bit handle the CCU IR uses to address GM through the kernel's
    // `LocalAddr{addr=Var, token=Var}` plumbing. We span both buffers with a
    // single token by passing the lower address as `va` and the contiguous
    // total bytes as `size` — same trick hccl mesh1d uses for RS
    // (`ccu_temp_reduce_scatter_mesh_1D.cc:120-126`).
    // -------------------------------------------------------------------------
    const uint64_t spanBase = (inputVa < outputVa) ? inputVa : outputVa;
    const uint64_t spanEnd  = (inputVa < outputVa)
                              ? (outputVa + kPayloadSize)
                              : (inputVa  + kPayloadSize);
    const uint64_t spanSize = spanEnd - spanBase;
    const uint64_t token    = hcomm::CcuRep::GetTokenInfo(spanBase, spanSize);
    std::fprintf(stderr,
        "[GATED_RS_ST] tokens: inputVa=0x%llx outputVa=0x%llx spanBase=0x%llx "
        "spanSize=%llu token=0x%llx\n",
        static_cast<unsigned long long>(inputVa),
        static_cast<unsigned long long>(outputVa),
        static_cast<unsigned long long>(spanBase),
        static_cast<unsigned long long>(spanSize),
        static_cast<unsigned long long>(token));

    // -------------------------------------------------------------------------
    // 7. Launch the gated kernel.
    //
    // CCU stream is now pushed and parks at `WaitEvent(gateEvent_)`. The
    // kernel's `GeneArgs()` ran during launch and published the gate
    // descriptor into `pto::ccu::Map()` for us.
    // -------------------------------------------------------------------------
    pto::ccu::PtoGatedTaskArg targ{
        /*inputAddr=*/  inputVa,
        /*outputAddr=*/ outputVa,
        /*length=*/     kPayloadSize,
        /*token=*/      token,
    };
    HCCL_OK(HcclCcuKernelLaunch(comm, threadHandle, kHandle, &targ));
    std::fprintf(stderr,
        "[GATED_RS_ST] HcclCcuKernelLaunch OK — CCU stream parked at gate\n");

    // -------------------------------------------------------------------------
    // 8. Read back the published gate descriptor.
    // -------------------------------------------------------------------------
    pto::host::PtoGateDescriptor desc{};
    if (!pto::ccu::TryGet(/*rankId=*/0, desc)) {
        std::fprintf(stderr,
            "[GATED_RS_ST] FAIL: pto::ccu::TryGet returned false — kernel did "
            "not Publish during GeneArgs(). Possible causes:\n"
            "  - PtoGatedKernelArg signature mismatch caused dynamic_cast to fail\n"
            "  - libpto_gated_reduce_scatter.so and libpto_ccu_host.so loaded\n"
            "    different copies of the registry static map\n"
            "  - HcclCcuKernelLaunch failed silently before GeneArgs ran\n");
        return 3;
    }
    std::fprintf(stderr,
        "[GATED_RS_ST] descriptor: dieId=%u ckeId=%u mask=0x%x\n",
        desc.dieId, desc.ckeId, desc.mask);

    // -------------------------------------------------------------------------
    // 9. Resolve CCU MMIO base address.
    //
    // QueryCcuBaseInfo returns a struct with `resourceAddr` = base of the CKE
    // chunk for the given die. The AIV trigger computes the per-CKE offset
    // (`base + ckeId * stride + byte_off`) using `PTO_AIV_TRIGGER_STRIDE` /
    // `PTO_AIV_TRIGGER_BYTE_OFF` env vars. Defaults that match the
    // empirically validated layout (Exp3, 2026-05-06): stride=0x40, byte_off=6.
    // -------------------------------------------------------------------------
    setenv("PTO_AIV_TRIGGER_STRIDE",   "0x40", /*overwrite=*/0);
    setenv("PTO_AIV_TRIGGER_BYTE_OFF", "6",    /*overwrite=*/0);

    auto probe = pto::host::QueryCcuBaseInfo(/*devPhyId=*/kDeviceId, desc.dieId);
    if (probe.rc != 0 || probe.resourceAddr == nullptr) {
        std::fprintf(stderr,
            "[GATED_RS_ST] FAIL: QueryCcuBaseInfo rc=%d resourceAddr=%p — "
            "cannot compute MMIO target for AIV trigger. Verify the test rig "
            "has libhccp.so + driver 7.0.t9.0 or newer.\n",
            probe.rc, probe.resourceAddr);
        return 4;
    }
    desc.mmioAddr = reinterpret_cast<uint64_t>(probe.resourceAddr);
    std::fprintf(stderr,
        "[GATED_RS_ST] mmioAddr=0x%llx (from QueryCcuBaseInfo)\n",
        static_cast<unsigned long long>(desc.mmioAddr));

    // -------------------------------------------------------------------------
    // 10. AIV trigger — release the gate.
    // -------------------------------------------------------------------------
    int32_t trigRc = pto::aiv::launch_treduce(aivStream, desc, aivMarkerDev);
    if (trigRc != 0) {
        std::fprintf(stderr,
            "[GATED_RS_ST] FAIL: pto::aiv::launch_treduce rc=%d "
            "(dieId=%u ckeId=%u mask=0x%x mmioAddr=0x%llx)\n",
            trigRc, desc.dieId, desc.ckeId, desc.mask,
            static_cast<unsigned long long>(desc.mmioAddr));
        return 5;
    }
    std::fprintf(stderr, "[GATED_RS_ST] AIV launch_treduce OK\n");

    // -------------------------------------------------------------------------
    // 11. Sync streams. Order matters: AIV must drain first so the gate
    //     write is in flight before we try to wait for CCU completion.
    // -------------------------------------------------------------------------
    ACL_OK(aclrtSynchronizeStream(aivStream));
    std::fprintf(stderr, "[GATED_RS_ST] aivStream synced\n");

    aclError ccuSync = aclrtSynchronizeStream(stream);
    if (ccuSync != ACL_SUCCESS) {
        std::fprintf(stderr,
            "[GATED_RS_ST] FAIL: ccu stream sync rc=%d — gate not released. "
            "Inspect AIV marker buffer for triage:\n",
            static_cast<int>(ccuSync));
        uint64_t marker[8] = {};
        aclrtMemcpy(marker, sizeof(marker), aivMarkerDev, sizeof(marker),
                    ACL_MEMCPY_DEVICE_TO_HOST);
        for (int i = 0; i < 8; ++i) {
            std::fprintf(stderr,
                "[GATED_RS_ST] aivMarker[%d] = 0x%016llx\n",
                i, static_cast<unsigned long long>(marker[i]));
        }
        return 6;
    }
    std::fprintf(stderr, "[GATED_RS_ST] ccu stream synced — gate released\n");

    // -------------------------------------------------------------------------
    // 12. Verify output == input (placeholder identity copy).
    // -------------------------------------------------------------------------
    std::vector<uint8_t> outputBack(kPayloadSize);
    ACL_OK(aclrtMemcpy(outputBack.data(), kPayloadSize, outputDev, kPayloadSize,
                       ACL_MEMCPY_DEVICE_TO_HOST));

    int mismatch = 0;
    for (size_t i = 0; i < kPayloadSize; ++i) {
        if (outputBack[i] != inputHost[i]) {
            if (mismatch < 16) {
                std::fprintf(stderr,
                    "[GATED_RS_ST] mismatch byte[%zu]: expected=0x%02x got=0x%02x\n",
                    i, inputHost[i], outputBack[i]);
            }
            ++mismatch;
        }
    }
    if (mismatch != 0) {
        std::fprintf(stderr,
            "[GATED_RS_ST] FAIL %d/%zu bytes mismatch — placeholder identity "
            "copy did not run after gate released. Verify Algorithm() emits "
            "LocalCopyNb after WaitEvent.\n",
            mismatch, kPayloadSize);
        return 7;
    }

    std::fprintf(stderr, "[GATED_RS_ST] PASS rank=0 — %zu bytes verified\n",
                 kPayloadSize);

    // -------------------------------------------------------------------------
    // 13. Cleanup. Best-effort — exit code already set by check above.
    // -------------------------------------------------------------------------
    aclrtFree(aivMarkerDev);
    aclrtFree(outputDev);
    aclrtFree(inputDev);

    HcclCommDestroy(comm);
    aclrtDestroyStream(aivStream);
    aclrtDestroyStream(stream);
    aclrtResetDevice(kDeviceId);
    aclFinalize();

    return 0;
}
