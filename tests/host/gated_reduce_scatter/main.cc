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
//   │  acl init  →  HcclCommInitRootInfo  →  HcclThreadAcquireWithStream    │
//   │                                                                       │
//   │  HcclCcuKernelRegister(creator=MakeGatedKernelCreator,                │
//   │                        arg=PtoGatedKernelArg{...})                    │
//   │  HcclCcuKernelRegisterFinish  ← hcomm translates IR → CCU microcode   │
//   │  hcomm::CcuRep::GetTokenInfo(span)  ← AFTER RegisterFinish            │
//   │  HcclCcuKernelLaunch(taskArg=PtoGatedTaskArg{addr/len/token})         │
//   │      → CCU stream parks at WaitEvent(gateEvent_)                      │
//   │                                                                       │
//   │  pto::ccu::TryGet(rankId, &desc)  ← read published (dieId,ckeId,mask) │
//   │  pto::host::QueryCcuBaseInfo(devId, dieId)  ← MMIO base VA            │
//   │  desc.mmioAddr = probe.resourceAddr                                   │
//   │                                                                       │
//   │  pto::aiv::launch_treduce(aivStream, desc, marker)                    │
//   │      → AIV-side MMIO store releases CKE gate                          │
//   │      → CCU resumes, runs LocalCopyNb(input → output), RecordEvent     │
//   │                                                                       │
//   │  aclrtSynchronizeStream(stream)                                       │
//   │  verify output == input                                               │
//   │                                                                       │
//   └───────────────────────────────────────────────────────────────────────┘
//
// MULTI-RANK BOOTSTRAP (2026-05-06 pivot)
// =============================================================================
//
// `HcclCcuKernelRegister` requires the comm's internal `ccuContainer`
// (`MyRank::ccuContainer`, hcomm-private member) to be non-null. Disassembly
// of `libhcomm.so::HcclCcuKernelRegister` (offset 0x8163fd-816407) shows the
// sanity check:
//     mov  0x150(%rax), %rax    ; collComm->myRank
//     mov  0xd8(%rax),  %r14    ; myRank->ccuContainer
//     test %r14, %r14
//     je   <fail "ccuContainer is nullptr">  → returns HCCL_E_PTR (=2)
//
// `MyRank::TryInitCcuInstance` (offset 0x803b40) only ChangeMode()'s an
// EXISTING container — it does not create one. Empirically (2026-05-06)
// `ccuContainer` only gets allocated when hccl runs through its multi-rank
// CCU init path (driven via `HcclCommInitRootInfo` with `nRanks >= 2`).
// Single-rank comms shortcut through `SingleRankProc` and never enter that
// code, so `HcclCcuKernelRegister` always fails with HCCL_E_PTR.
//
// → ST has to drive ≥ 2 ranks. We launch via MPI (mpirun -n N) and use the
//   pto-isa `comm_mpi.h` shim (dlopen-based; no link-time MPI dependency).
//   The comm bootstrap mirrors `tests/npu/a5/comm/st/testcase/common.hpp::
//   ForkAndRunWithHcclRootInfo` but is inlined here to avoid pulling the
//   workspace_manager / hccl_context dependency chain.
//
// Pilot scope (matches `include/pto/ccu/pto_gated_kernel.hpp`):
//   - The kernel performs a placeholder identity copy (no actual reduce
//     across ranks; data path is phase 3). Each rank still performs its
//     `LocalCopyNb(output ← input)` independently → `output == input` is
//     a valid check on every rank for nRanks ≥ 1.
//   - hccl repo is not used by this ST except via its public C ABI symbols
//     in libhccl.so / libhcomm.so. No hccl-internal headers are required.
//
// Build & run (server side):
//
//   cmake -S kernels/host/pto_ccu_host -B build/pto_ccu_host
//   cmake --build build/pto_ccu_host -j
//   cmake -S kernels/host/gated_reduce_scatter \
//         -B build/pto_gated_reduce_scatter \
//         -DPTO_CCU_HOST_LIB=$PWD/build/pto_ccu_host/dist/libpto_ccu_host.so
//   cmake --build build/pto_gated_reduce_scatter -j
//   cmake -S tests/host/gated_reduce_scatter \
//         -B build/gated_rs_st \
//         -DPTO_GATED_RS_LIB_DIR=$PWD/build/pto_gated_reduce_scatter/dist \
//         -DPTO_CCU_HOST_LIB_DIR=$PWD/build/pto_ccu_host/dist
//   cmake --build build/gated_rs_st -j
//
//   export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
//   export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:\
//     $PWD/build/pto_ccu_host/dist:\
//     $PWD/build/pto_gated_reduce_scatter/dist:\
//     <libpto_aiv_treduce.so dir>:$LD_LIBRARY_PATH
//   # If mpich is not on default ld path, also export MPI_LIB_PATH.
//
//   # Multi-rank (REQUIRED for non-trivial smoke):
//   mpirun -n 2 ./build/gated_rs_st/gated_rs_st
//
//   # expected: each rank prints "[GATED_RS_ST] PASS rank=R" + rc=0

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_res.h"
#include "hcomm/ccu/hccl_ccu_res.h"
#include "hcomm/ccu/ccu_assist_pub.h"

#include "pto/aiv/treduce.hpp"
#include "pto/ccu/pto_gate_registry.hpp"
#include "pto/ccu/pto_gated_kernel.hpp"
#include "pto/host/pto_gate_descriptor.hpp"
#include "pto/host/query_ccu_baseinfo.hpp"

// Lightweight MPI shim (dlopen-based). Lives in pto-isa under
// `tests/npu/a5/comm/st/testcase/comm_mpi.h`; the CMakeLists adds that
// directory to the include path so we can pick it up here without copying.
#include "comm_mpi.h"

// Low-level CANN runtime device-set API (from libruntime.so). Required IN
// ADDITION to `aclrtSetDevice` because hcomm's CCU subsystem caches
// `g_deviceLogicId` keyed off the **runtime layer** state, which `aclrtSetDevice`
// (ACL wrap layer) does NOT synchronise. Without this call, when the consumer
// is launched as a separate process (mpirun) instead of a same-process thread
// (e.g. hccl example which spawns ranks via std::thread), some hcomm-internal
// worker thread spawned during `HcclCommInitRootInfo` reads
// `g_deviceLogicId == INVALID_INT`, propagates that into
// `CcuKernelMgr::GetInstance(devLogicId=invalid)`, hits the backup-device path
// (devLogicId=MAX_MODULE_DEVICE_NUM=65), and the entire CcuResPack is
// allocated against the minimal-fallback budget. Symptom: subsequent
// `HcclCcuKernelRegister` returns HCCL_E_UNAVAIL (=7) with an
// `[CcuKernelMgr][CheckResIfAvailable] dieId[0] not enough, ckeReq[3] gsaReq[2]
// xnReq[6] missionReq[1]` ERROR log right after the kernel ctor traces.
//
// Cure: rank 0 must call `rtSetDevice` *between* `aclInit` and the first
// `aclrtSetDevice`, mirroring what pto-isa's
// `tests/npu/a5/comm/st/testcase/common.hpp:344-347 ForkAndRunWithHcclRootInfo`
// does. Empirically only rank 0 needs to call it (the runtime cache is keyed
// per-process, not per-thread, on the side that matters for the
// HcclCommInitRootInfo bootstrap path).
//
// Refer to plan §3.13.G for the full diagnosis chain (hcomm
// `adapter_rts.cc::__hrtGetDevice` → cached `g_deviceLogicId` → `op_base.cc::
// HcclGetThreadDeviceId` → `coll_comm_res_c_adpt.cc:319 HcclCcuKernelRegister`).
extern "C" int32_t rtSetDevice(int32_t deviceId);

namespace {

constexpr size_t kPayloadSize    = 4096;
constexpr size_t kAivMarkerBytes = 64;

#define ACL_OK(expr)                                                       \
    do {                                                                   \
        aclError _r = (expr);                                              \
        if (_r != ACL_SUCCESS) {                                           \
            std::fprintf(stderr,                                           \
                "[GATED_RS_ST] ACL FAIL %s = %d at %s:%d (rank=%d)\n",     \
                #expr, static_cast<int>(_r), __FILE__, __LINE__,           \
                g_rankId);                                                 \
            return false;                                                  \
        }                                                                  \
    } while (0)

#define HCCL_OK(expr)                                                      \
    do {                                                                   \
        HcclResult _r = (expr);                                            \
        if (_r != HCCL_SUCCESS) {                                          \
            std::fprintf(stderr,                                           \
                "[GATED_RS_ST] HCCL FAIL %s = %d at %s:%d (rank=%d)\n",    \
                #expr, static_cast<int>(_r), __FILE__, __LINE__,           \
                g_rankId);                                                 \
            return false;                                                  \
        }                                                                  \
    } while (0)

// Used by ACL_OK / HCCL_OK macros for nicer log lines. Set per-rank in main()
// before RunOneRank() runs. Read-only for the duration of RunOneRank().
thread_local int g_rankId = 0;

bool IsLaunchedByMpi()
{
    // Standard env vars set by mpirun launchers. We probe a few to cover
    // OpenMPI, MPICH/Hydra, MVAPICH, SLURM srun.
    return std::getenv("OMPI_COMM_WORLD_RANK") != nullptr ||
           std::getenv("PMI_RANK")             != nullptr ||
           std::getenv("MV2_COMM_WORLD_RANK")  != nullptr ||
           std::getenv("SLURM_PROCID")         != nullptr;
}

bool RunOneRank(int rankId, int nRanks, int firstDeviceId,
                const HcclRootInfo *rootInfo)
{
    g_rankId = rankId;
    const int deviceId = rankId + firstDeviceId;

    std::fprintf(stderr,
        "[GATED_RS_ST] rank=%d/%d device=%d payloadBytes=%zu — start\n",
        rankId, nRanks, deviceId, kPayloadSize);

    // -------------------------------------------------------------------------
    // 1. ACL device + streams.
    // -------------------------------------------------------------------------
    ACL_OK(aclrtSetDevice(deviceId));

    aclrtStream stream = nullptr;
    ACL_OK(aclrtCreateStream(&stream));

    // Separate stream for the AIV trigger — same-stream serialization with
    // the gated CCU kernel deadlocks (gate must arm before trigger fires).
    aclrtStream aivStream = nullptr;
    ACL_OK(aclrtCreateStream(&aivStream));

    // -------------------------------------------------------------------------
    // 2. HcclComm bootstrap.
    //
    // `nRanks >= 2` is required for `HcclCcuKernelRegister` to succeed —
    // see file header for the disassembly-derived rationale.
    // -------------------------------------------------------------------------
    HcclComm comm = nullptr;
    HCCL_OK(HcclCommInitRootInfo(static_cast<uint32_t>(nRanks), rootInfo,
                                  static_cast<uint32_t>(rankId), &comm));
    std::fprintf(stderr,
        "[GATED_RS_ST] rank=%d HcclCommInitRootInfo OK (nRanks=%d)\n",
        rankId, nRanks);

    // -------------------------------------------------------------------------
    // 3. CCU thread acquire — *WithStream variant required.
    //
    // Plain `HcclThreadAcquire` returns a thread NOT bound to any aclrtStream.
    // Subsequent `HcclCcuKernelRegister` then fails because hcomm cannot
    // resolve which user stream to emit microcode onto. hccl's own internal
    // CCU host path uses `*WithStream` (`op_common.cc:881-885`); we must too.
    // Symptom of using the wrong variant: `HcclCcuKernelRegister` returns
    // HCCL_E_PTR (=2) — same code as a NULL creator/arg, hence misleading.
    // -------------------------------------------------------------------------
    constexpr uint32_t kNotifyNum = 1;
    ThreadHandle threadHandle = 0;
    HCCL_OK(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CCU, stream,
                                         kNotifyNum, &threadHandle));
    std::fprintf(stderr,
        "[GATED_RS_ST] rank=%d HcclThreadAcquireWithStream OK "
        "threadHandle=0x%llx\n",
        rankId, static_cast<unsigned long long>(threadHandle));

    // -------------------------------------------------------------------------
    // 4. Allocate device input/output + diagnostic AIV marker buffer.
    //
    // Each rank holds an independent payload (no peer addressability needed
    // for the placeholder identity copy). For real ReduceScatter (phase 3)
    // these would be registered via `HcclCommMemReg` so peers can address
    // remote buffers.
    // -------------------------------------------------------------------------
    void *inputDev  = nullptr;
    void *outputDev = nullptr;
    // ─────────────────────────────────────────────────────────────────────────
    // CRITICAL — must use `aclrtMallocWithCfg` + `ACL_MEM_TYPE_HIGH_BAND_WIDTH`
    // and the cfg MUST have a non-empty `attrs` array (driver rejects
    // `numAttrs=0` with errno 107000 invalid value — verified empirically
    // 2026-05-07).
    //
    // Rationale (verified against mpi9/mpi10 log diffs, not猜测):
    //   1. `aclrtMalloc(HUGE_ONLY)` (mpi9) → driver alloc普通 device mem,
    //      NOT entered into driver UB token table → `GetTokenInfo` throws
    //      `Va is not alloced` (errno 3, RT 107000).
    //   2. `aclrtMallocWithCfg(... HIGH_BAND_WIDTH, cfg{nullptr,0})` (first
    //      mpi10 attempt) → driver also fails 107000 — empty cfg attrs
    //      rejected.
    //   3. hcomm self-allocs via `aclrtMallocWithCfg(... HIGH_BAND_WIDTH,
    //      cfg{&attr,1})` with one `ACL_RT_MEM_ATTR_MODULE_ID` attribute
    //      (`adapter_rts.cc:570-593` `HrtDevMalloc`, `tester.cc:414-419`).
    //      The 400 MB `selfOwned=1 DevBuffer[addr=0x12004ca00000]` seen in
    //      mpi9 log (line 3293) came from this path — UB-tagged, in driver
    //      UB table, `GetTokenInfo` succeeds.
    //   4. `moduleId` field is `uint16_t` (acl/acl_rt.h:166); slog
    //      `log_types.h:52` defines `HCCL = 3` and we use that literal here
    //      to mirror hcomm exactly — value is a diagnostic label only,
    //      `aclrtMallocWithCfg` only requires `numAttrs >= 1` to succeed.
    // ─────────────────────────────────────────────────────────────────────────
    aclrtMallocAttrValue kModuleIdValue{};
    kModuleIdValue.moduleId = 3;  // HCCL slog module id (mirrors hcomm internal alloc path)
    aclrtMallocAttribute kAttrs{ACL_RT_MEM_ATTR_MODULE_ID, kModuleIdValue};
    aclrtMallocConfig kCfg{&kAttrs, 1};
    ACL_OK(aclrtMallocWithCfg(&inputDev,  kPayloadSize,
        ACL_MEM_TYPE_HIGH_BAND_WIDTH, &kCfg));
    ACL_OK(aclrtMallocWithCfg(&outputDev, kPayloadSize,
        ACL_MEM_TYPE_HIGH_BAND_WIDTH, &kCfg));

    // Initialise input with a per-rank deterministic byte pattern; output
    // to known-bad. Rank-tagging the input lets us spot accidental cross-rank
    // overwrites if real reduce semantics ever leak in.
    std::vector<uint8_t> inputHost(kPayloadSize);
    std::vector<uint8_t> outputHost(kPayloadSize, 0xFF);
    for (size_t i = 0; i < kPayloadSize; ++i) {
        inputHost[i] = static_cast<uint8_t>((i + rankId) & 0xFF);
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
    // ORDERING INVARIANT: `hcomm::CcuRep::GetTokenInfo(va, size)` MUST be
    // called AFTER `HcclCcuKernelRegisterFinish` returns. Calling it earlier
    // throws `Hccl::CcuApiException: failed to query tokenInfo` because the
    // CCU translation context (GM VA → CCU token map) is built up DURING
    // `RegisterFinish` when hcomm runs `Translate()` and stages microcode.
    //
    // hccl's own internal ordering matches — see `op_common.cc:1201-1206`
    // (Register/Finish) → `op_common.cc:474-496` (Orchestrate → KernelRun) →
    // `ccu_temp_reduce_scatter_mesh_1D.cc:99-138` where `GetToken(buffInfo_,
    // token)` is called inside `KernelRun`, strictly after `RegisterFinish`.
    // hccl never exposes this ordering as a public contract — it's enforced
    // structurally by the framework. Standalone CCU users replicate it
    // manually.
    // -------------------------------------------------------------------------
    pto::ccu::PtoGatedKernelArg karg{
        /*rankId=*/   static_cast<uint32_t>(rankId),
        /*rankSize=*/ static_cast<uint32_t>(nRanks),
        /*payloadBytes=*/ kPayloadSize,
        /*gateMask=*/ 1u << 0,
        /*doneMask=*/ 1u << 0,
    };

    hcomm::KernelCreator creator = pto::ccu::MakeGatedKernelCreator();
    CcuKernelHandle      kHandle = 0;

    HCCL_OK(HcclCcuKernelRegister(comm, &kHandle, &creator, &karg));
    std::fprintf(stderr,
        "[GATED_RS_ST] rank=%d HcclCcuKernelRegister OK kHandle=0x%llx\n",
        rankId, static_cast<unsigned long long>(kHandle));

    HCCL_OK(HcclCcuKernelRegisterFinish(comm));
    std::fprintf(stderr,
        "[GATED_RS_ST] rank=%d HcclCcuKernelRegisterFinish OK\n", rankId);

    // -------------------------------------------------------------------------
    // 6. Query CCU token AFTER translate.
    //
    // Token covers BOTH input + output regions in a single span so
    // LocalCopyNb's source/dest reuse it. This mirrors what hccl mesh1d does
    // (`ccu_temp_reduce_scatter_mesh_1D.cc:120-126`).
    // -------------------------------------------------------------------------
    const uint64_t spanBase = (inputVa < outputVa) ? inputVa : outputVa;
    const uint64_t spanEnd  = (inputVa < outputVa)
                              ? (outputVa + kPayloadSize)
                              : (inputVa  + kPayloadSize);
    const uint64_t spanSize = spanEnd - spanBase;
    const uint64_t token    = hcomm::CcuRep::GetTokenInfo(spanBase, spanSize);
    std::fprintf(stderr,
        "[GATED_RS_ST] rank=%d tokens: inputVa=0x%llx outputVa=0x%llx "
        "spanBase=0x%llx spanSize=%llu token=0x%llx\n",
        rankId,
        static_cast<unsigned long long>(inputVa),
        static_cast<unsigned long long>(outputVa),
        static_cast<unsigned long long>(spanBase),
        static_cast<unsigned long long>(spanSize),
        static_cast<unsigned long long>(token));

    // -------------------------------------------------------------------------
    // 7. Launch the gated kernel — CCU stream parks at WaitEvent(gateEvent_).
    // -------------------------------------------------------------------------
    pto::ccu::PtoGatedTaskArg targ{
        /*inputAddr=*/  inputVa,
        /*outputAddr=*/ outputVa,
        /*length=*/     kPayloadSize,
        /*token=*/      token,
    };
    HCCL_OK(HcclCcuKernelLaunch(comm, threadHandle, kHandle, &targ));
    std::fprintf(stderr,
        "[GATED_RS_ST] rank=%d HcclCcuKernelLaunch OK — parked at gate\n",
        rankId);

    // -------------------------------------------------------------------------
    // 8. Read back the published gate descriptor.
    // -------------------------------------------------------------------------
    pto::host::PtoGateDescriptor desc{};
    if (!pto::ccu::TryGet(static_cast<uint32_t>(rankId), desc)) {
        std::fprintf(stderr,
            "[GATED_RS_ST] rank=%d FAIL: pto::ccu::TryGet returned false — "
            "kernel did not Publish during GeneArgs(). Possible causes:\n"
            "  - PtoGatedKernelArg signature mismatch caused dynamic_cast "
            "    to fail\n"
            "  - libpto_gated_reduce_scatter.so and libpto_ccu_host.so loaded "
            "    different copies of the registry static map\n"
            "  - HcclCcuKernelLaunch failed silently before GeneArgs ran\n",
            rankId);
        return false;
    }
    std::fprintf(stderr,
        "[GATED_RS_ST] rank=%d descriptor: dieId=%u ckeId=%u mask=0x%x\n",
        rankId, desc.dieId, desc.ckeId, desc.mask);

    // -------------------------------------------------------------------------
    // 9. Resolve CCU MMIO base for the AIV trigger.
    //
    // Defaults match the empirically validated layout (Exp3, 2026-05-06):
    // stride=0x40, byte_off=6.
    // -------------------------------------------------------------------------
    setenv("PTO_AIV_TRIGGER_STRIDE",   "0x40", /*overwrite=*/0);
    setenv("PTO_AIV_TRIGGER_BYTE_OFF", "6",    /*overwrite=*/0);

    auto probe = pto::host::QueryCcuBaseInfo(deviceId, desc.dieId);
    if (probe.rc != 0 || probe.resourceAddr == nullptr) {
        std::fprintf(stderr,
            "[GATED_RS_ST] rank=%d FAIL: QueryCcuBaseInfo rc=%d "
            "resourceAddr=%p\n",
            rankId, probe.rc, probe.resourceAddr);
        return false;
    }
    desc.mmioAddr = reinterpret_cast<uint64_t>(probe.resourceAddr);
    std::fprintf(stderr,
        "[GATED_RS_ST] rank=%d mmioAddr=0x%llx (from QueryCcuBaseInfo)\n",
        rankId, static_cast<unsigned long long>(desc.mmioAddr));

    // -------------------------------------------------------------------------
    // 10. AIV trigger — release this rank's gate.
    // -------------------------------------------------------------------------
    int32_t trigRc = pto::aiv::launch_treduce(aivStream, desc, aivMarkerDev);
    if (trigRc != 0) {
        std::fprintf(stderr,
            "[GATED_RS_ST] rank=%d FAIL: pto::aiv::launch_treduce rc=%d "
            "(dieId=%u ckeId=%u mask=0x%x mmioAddr=0x%llx)\n",
            rankId, trigRc, desc.dieId, desc.ckeId, desc.mask,
            static_cast<unsigned long long>(desc.mmioAddr));
        return false;
    }
    std::fprintf(stderr,
        "[GATED_RS_ST] rank=%d AIV launch_treduce OK\n", rankId);

    // -------------------------------------------------------------------------
    // 11. Sync streams. Order matters: AIV must drain first so the gate
    //     write is in flight before we wait for CCU completion.
    //
    // Inline aivStream sync error handling: if the AIV kernel traps (RT
    // 507035 = ACL_ERROR_RT_VECTOR_CORE_EXCEPTION), dump aivMarker before
    // returning so we can triage which decision-tree branch fired (markers
    // documented in `examples/02_collectives/04_reduce_scatter/main.cc:228-244`).
    // -------------------------------------------------------------------------
    aclError aivSync = aclrtSynchronizeStream(aivStream);
    if (aivSync != ACL_SUCCESS) {
        std::fprintf(stderr,
            "[GATED_RS_ST] rank=%d FAIL: aivStream sync rc=%d (likely AIV "
            "trap — descriptor: dieId=%u ckeId=%u mask=0x%x mmioAddr=0x%llx, "
            "trigger target=0x%llx). AIV marker dump:\n",
            rankId, static_cast<int>(aivSync),
            desc.dieId, desc.ckeId, desc.mask,
            static_cast<unsigned long long>(desc.mmioAddr),
            static_cast<unsigned long long>(desc.mmioAddr + desc.ckeId * 0x40ULL + 6ULL));
        // Pre-poison the host marker buffer so we can distinguish two
        // failure modes:
        //   - host buffer == 0xDEAD... → memcpy DEVICE→HOST failed silently
        //     (device fault left the stream / buffer in unreadable state)
        //   - host buffer == 0          → memcpy succeeded, kernel really
        //     never wrote marker[0] (entry trap before first GM store)
        uint64_t marker[8];
        for (int i = 0; i < 8; ++i) marker[i] = 0xDEADBEEF00000000ULL | static_cast<uint64_t>(i);
        aclError mcpyRc = aclrtMemcpy(marker, sizeof(marker), aivMarkerDev,
                                       sizeof(marker), ACL_MEMCPY_DEVICE_TO_HOST);
        std::fprintf(stderr,
            "[GATED_RS_ST] rank=%d aivMarker memcpy rc=%d (0=ok)\n",
            rankId, static_cast<int>(mcpyRc));
        for (int i = 0; i < 8; ++i) {
            std::fprintf(stderr,
                "[GATED_RS_ST] rank=%d aivMarker[%d] = 0x%016llx\n",
                rankId, i, static_cast<unsigned long long>(marker[i]));
        }
        return false;
    }
    std::fprintf(stderr,
        "[GATED_RS_ST] rank=%d aivStream synced\n", rankId);

    aclError ccuSync = aclrtSynchronizeStream(stream);
    if (ccuSync != ACL_SUCCESS) {
        std::fprintf(stderr,
            "[GATED_RS_ST] rank=%d FAIL: ccu stream sync rc=%d — gate not "
            "released. AIV marker dump:\n",
            rankId, static_cast<int>(ccuSync));
        uint64_t marker[8] = {};
        aclrtMemcpy(marker, sizeof(marker), aivMarkerDev, sizeof(marker),
                    ACL_MEMCPY_DEVICE_TO_HOST);
        for (int i = 0; i < 8; ++i) {
            std::fprintf(stderr,
                "[GATED_RS_ST] rank=%d aivMarker[%d] = 0x%016llx\n",
                rankId, i, static_cast<unsigned long long>(marker[i]));
        }
        return false;
    }
    std::fprintf(stderr,
        "[GATED_RS_ST] rank=%d ccu stream synced — gate released\n", rankId);

    // -------------------------------------------------------------------------
    // 12. Verify output == input (placeholder identity copy is rank-local,
    //     so this is correct for any nRanks).
    // -------------------------------------------------------------------------
    std::vector<uint8_t> outputBack(kPayloadSize);
    ACL_OK(aclrtMemcpy(outputBack.data(), kPayloadSize, outputDev, kPayloadSize,
                       ACL_MEMCPY_DEVICE_TO_HOST));

    int mismatch = 0;
    for (size_t i = 0; i < kPayloadSize; ++i) {
        if (outputBack[i] != inputHost[i]) {
            if (mismatch < 16) {
                std::fprintf(stderr,
                    "[GATED_RS_ST] rank=%d mismatch byte[%zu]: "
                    "expected=0x%02x got=0x%02x\n",
                    rankId, i, inputHost[i], outputBack[i]);
            }
            ++mismatch;
        }
    }
    if (mismatch != 0) {
        std::fprintf(stderr,
            "[GATED_RS_ST] rank=%d FAIL %d/%zu bytes mismatch — placeholder "
            "identity copy did not run after gate released.\n",
            rankId, mismatch, kPayloadSize);
        return false;
    }

    std::fprintf(stderr,
        "[GATED_RS_ST] rank=%d PASS — %zu bytes verified\n",
        rankId, kPayloadSize);

    // -------------------------------------------------------------------------
    // 13. Cleanup.
    // -------------------------------------------------------------------------
    aclrtFree(aivMarkerDev);
    aclrtFree(outputDev);
    aclrtFree(inputDev);

    HcclCommDestroy(comm);
    aclrtDestroyStream(aivStream);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    return true;
}

}  // namespace

// =============================================================================
// Main — MPI bootstrap (auto-degrades to single-rank when not launched by
// mpirun, useful for local sanity check though the kernel will fail at
// `HcclCcuKernelRegister` due to the ccuContainer-null invariant — see file
// header for details).
// =============================================================================
int main(int argc, char **argv)
{
    // Force stderr unbuffered. mpirun's tee/output-collection layer can otherwise
    // swallow our prints if the process aborts (SEGV) before flush.
    setvbuf(stderr, nullptr, _IONBF, 0);

    const bool mpiLaunched = IsLaunchedByMpi();

    int rankId        = 0;
    int nRanks        = 1;
    int firstDeviceId = 0;

    if (mpiLaunched) {
        if (!CommMpiInit(&argc, &argv)) {
            std::fprintf(stderr,
                "[GATED_RS_ST] FATAL: launched by mpirun but MPI_Init failed. "
                "Verify libmpi.so / libmpich.so on LD_LIBRARY_PATH or set "
                "MPI_LIB_PATH.\n");
            return 2;
        }
        rankId = CommMpiRank();
        nRanks = CommMpiSize();
        std::fprintf(stderr,
            "[GATED_RS_ST] MPI launched: rank=%d / nRanks=%d\n",
            rankId, nRanks);
    } else {
        std::fprintf(stderr,
            "[GATED_RS_ST] standalone (single-rank) mode — note: this WILL "
            "fail at HcclCcuKernelRegister with HCCL_E_PTR because hcomm's "
            "internal ccuContainer is only initialised when nRanks >= 2. "
            "Use `mpirun -n 2 ./gated_rs_st` for a real smoke test.\n");
    }

    // ACL init must precede aclrtSetDevice / HcclGetRootInfo.
    constexpr int kAclRepeatInit = 100002;
    aclError aclRet = aclInit(nullptr);
    if (aclRet != ACL_SUCCESS && static_cast<int>(aclRet) != kAclRepeatInit) {
        std::fprintf(stderr,
            "[GATED_RS_ST] FATAL: aclInit failed: %d\n",
            static_cast<int>(aclRet));
        if (mpiLaunched) CommMpiFinalize();
        return 2;
    }

    // Rank 0 generates root info; broadcast to peers.
    HcclRootInfo rootInfo{};
    if (rankId == 0) {
        // CRITICAL: rtSetDevice must precede aclrtSetDevice on the root rank
        // (mirrors common.hpp:344-347 ForkAndRunWithHcclRootInfo). aclrtSetDevice
        // alone is NOT enough — hcomm's CCU subsystem caches `g_deviceLogicId`
        // off the runtime-layer state, so a missing rtSetDevice here will let
        // `HcclCommInitRootInfo` walk through `CcuKernelMgr::GetInstance` with an
        // INVALID device id, fall through to the backup-device path
        // (`devLogicId=MAX_MODULE_DEVICE_NUM=65`), and provision a minimal-budget
        // CcuResPack — which then makes any later `HcclCcuKernelRegister` for a
        // 3-cke / 6-xn / 2-gsa kernel return HCCL_E_UNAVAIL (=7). See plan §3.13.G.
        int32_t rtRet = rtSetDevice(firstDeviceId);
        std::fprintf(stderr,
            "[GATED_RS_ST] rank=0 rtSetDevice(%d) -> %d\n",
            firstDeviceId, static_cast<int>(rtRet));
        // aclrtSetDevice is required before HcclGetRootInfo on the root rank
        // — same convention as ForkAndRunWithHcclRootInfo.
        aclrtSetDevice(firstDeviceId);
        HcclResult hret = HcclGetRootInfo(&rootInfo);
        if (hret != HCCL_SUCCESS) {
            std::fprintf(stderr,
                "[GATED_RS_ST] FATAL: HcclGetRootInfo rc=%d\n",
                static_cast<int>(hret));
            if (mpiLaunched) CommMpiFinalize();
            return 2;
        }
    }

    if (mpiLaunched && nRanks > 1) {
        CommMpiBcast(&rootInfo, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0);
        CommMpiBarrier();
        std::fprintf(stderr,
            "[GATED_RS_ST] rank=%d rootInfo bcast complete\n", rankId);
    }

    const bool ok = RunOneRank(rankId, nRanks, firstDeviceId, &rootInfo);

    aclFinalize();
    if (mpiLaunched) {
        // Final barrier before MPI_Finalize so all ranks' logs flush in order.
        CommMpiBarrier();
        CommMpiFinalize();
    }

    if (!ok) {
        std::fprintf(stderr,
            "[GATED_RS_ST] rank=%d OVERALL FAIL\n", rankId);
        return 1;
    }
    std::fprintf(stderr,
        "[GATED_RS_ST] rank=%d OVERALL PASS\n", rankId);
    return 0;
}
