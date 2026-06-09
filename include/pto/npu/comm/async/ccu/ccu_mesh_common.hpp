/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_CCU_CCU_MESH_COMMON_HPP
#define PTO_COMM_ASYNC_CCU_CCU_MESH_COMMON_HPP

// Host-only header — must NOT be included from device (bisheng -xcce) code.
#if defined(__CCE_KT_TEST__)
#error "ccu_mesh_common.hpp is a host-only header and cannot be included in device code."
#endif

// Shared scaffolding for the 1D-mesh CCU collective kernels (reduce / gather /
// scatter / broadcast).  These kernels differ only in their data path
// (DoReduce / DoGather / ...); the argument-loading, notify barriers and
// GeneArgs packing are identical and live here once.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "hcomm/ccu/ccu_kernel.h"

namespace pto {
namespace comm {
namespace ccu {

// Maximum peer-rank count supported by the host-side address exchange path.
//
// WHAT 16 IS NOT
// --------------
// It is NOT a hardware SQE limit.  hcomm's per-SQE args array has
// CCU_SQE_ARGS_LEN = 13 (see hcomm/pkg_inc/hcomm/ccu/ccu_task_param_v1.h),
// but the framework auto-fragments long GeneArgs vectors across
// ceil(agrsNum / 13) SQEs — the first seqNum-1 SQEs carry args only, the
// last one also carries the mission instructions.  Each microcode
// Load(var) is wired to `loadArgIndex_ % CCU_SQE_ARGS_LEN`, i.e. the
// fragmentation is transparent to kernel authors.  So at N=16 our largest
// packing (Reduce/Gather, 3N+1 = 49 args) ships as 4 SQEs — the 13-slot
// physical limit is not the bottleneck.
//
// WHAT 16 ACTUALLY IS
// -------------------
// A conservative algorithmic / resource heuristic, in this order:
//   1. Mesh1D is fundamentally O(N) fan-out from root — beyond ~16 peers
//      the per-launch latency dominates and the algorithm itself stops
//      being the right tool (ring / tree / halving-doubling become
//      preferable; see SCALING BEYOND below).
//   2. The auto-fragmented SQE chain consumes `missionInstrCount` budget
//      (the kernel's compiled instruction pool, hcomm-side); each extra
//      SQE adds (CCU_SQE_ARGS_LEN-1) instructions.  This is bounded by
//      the kernel compilation params, not the SQE descriptor layout, but
//      it grows linearly with N.
//   3. Per-rank ChannelAcquire / notify-bit / completedEvent-mask slots
//      grow O(N) too; these are tied to physical topology and have not
//      been characterized beyond N=4 in this codebase.
//
// At N=16 we have not yet hit any of (2) or (3) experimentally, but we
// also have not tried.  16 was picked to keep one byte of comfort margin
// vs. the historical AIV-side ceiling (N=8 in tbroadcast/treduce STs).
//
// HONESTY NOTE
// ------------
// 16 is inherited from the previous commit as a guess; the CCU mesh ST
// suite has only been exercised at N=2 and N=4 prior to this change.
// The N=12 / N=16 boundary tests added alongside this constant are the
// FIRST runs that actually push the mesh path that far — until they
// pass, treat "16" as an aspirational ceiling, not empirical evidence.
//
// SCALING BEYOND ~16
// ------------------
// The right answer is NOT to bump this constant but to switch topology.
// All of these decouple Load count from N (the SQE auto-fragmentation
// machinery never becomes a concern):
//   - Ring        : per-rank Loads O(1) (left+right neighbor only),
//                   latency O(N)
//   - Binary tree : per-rank Loads O(1) (parent + <=2 children),
//                   latency O(log N)
//   - Halving-doubling: per-launch Loads O(1), spread across log(N)
//                       kernel launches (1 partner per round)
//
// All four CCU collective kernels MUST use this single constant for their
// peer-address arrays so that a future bump lands in one place.
static constexpr uint32_t kCcuMeshMaxRanks = 16;

// Returns false (with stderr trace) when `rankSize` exceeds kCcuMeshMaxRanks.
// Callers SetPeerAddrs / SetWritePeers MUST honor the return value rather
// than silently truncating — silent truncation produces peer-address arrays
// that look zero-initialized to the kernel and lead to writes/reads against
// VA=0 on the rank just past the cutoff (extremely painful to debug).
inline bool EnsurePeerCapacity(const char *who, uint32_t rankSize)
{
    if (rankSize > kCcuMeshMaxRanks) {
        std::fprintf(stderr,
                     "[CCU/%s] rankSize=%u exceeds kCcuMeshMaxRanks=%u; "
                     "raise the bound (see ccu_mesh_common.hpp) before scaling further.\n",
                     who, rankSize, kCcuMeshMaxRanks);
        return false;
    }
    return true;
}

namespace detail {

class CcuMeshKernelBase : public hcomm::CcuKernel {
public:
    using hcomm::CcuKernel::CcuKernel;

protected:
    // Load every rank's input/output/token (packed by GeneArgs) then length.
    // Mirrors the GeneArgs layout [input_0..N-1, output_0..N-1, token_0..N-1, length].
    // Used by Reduce/Gather (root reads peer inputs) and Scatter (root needs
    // per-rank slice VAs that live in its own buffer but are addressed by index).
    template <typename VarVec>
    inline void LoadPeerArgs(const VarVec &input, const VarVec &output, const VarVec &token,
                             const hcomm::CcuRep::Variable &length)
    {
        for (const auto &v : input) {
            Load(v);
        }
        for (const auto &v : output) {
            Load(v);
        }
        for (const auto &v : token) {
            Load(v);
        }
        Load(length);
    }

    // Lighter loader for write-only collectives (Broadcast): only this rank's
    // own input plus every peer's output/token are needed (root writes; peers
    // contribute nothing input-side beyond the gate barrier).  Mirrors the
    // GeneArgs layout [ownInput, output_0..N-1, token_0..N-1, length] for a
    // total of 2*N+2 Loads instead of 3*N+1.
    template <typename VarVec>
    inline void LoadPeerArgsWriteOnly(const hcomm::CcuRep::Variable &ownInput, const VarVec &output,
                                      const VarVec &token, const hcomm::CcuRep::Variable &length)
    {
        Load(ownInput);
        for (const auto &v : output) {
            Load(v);
        }
        for (const auto &v : token) {
            Load(v);
        }
        Load(length);
    }

    // Scatter loader (CCU v100): mirrors PackScatterArgs above.  Layout
    // [rootInputBase, sliceStep, output_0..N-1, token_0..N-1, length]
    // = 2*N+3 Loads.  See PackScatterArgs for the rationale on keeping
    // `sliceStep` separate from `length`.
    template <typename VarVec>
    inline void LoadScatterArgs(const hcomm::CcuRep::Variable &rootInputBase,
                                const hcomm::CcuRep::Variable &sliceStep, const VarVec &output, const VarVec &token,
                                const hcomm::CcuRep::Variable &length)
    {
        Load(rootInputBase);
        Load(sliceStep);
        for (const auto &v : output) {
            Load(v);
        }
        for (const auto &v : token) {
            Load(v);
        }
        Load(length);
    }

    // Symmetric notify barrier across all channels on a single notify bit.
    // Used for both the readiness PreSync and the data-landing PostSync; the
    // two are kept distinct by passing different syncBit values.
    template <typename ChannelVec>
    inline void NotifyBarrier(const ChannelVec &channels, uint32_t notifyIdx, uint32_t syncBit)
    {
        for (const auto &ch : channels) {
            (void)NotifyRecord(ch, notifyIdx, 1u << syncBit);
        }
        for (const auto &ch : channels) {
            (void)NotifyWait(ch, notifyIdx, 1u << syncBit);
        }
    }
};

// Pack all ranks' peer addresses into the GeneArgs uint64 vector:
// [input_0..N-1, output_0..N-1, token_0..N-1, length].  Addresses are
// exchanged on the host (MPI AllGather) and shipped to the CCU microcode
// through this packing, so no runtime address-exchange PreSync is needed.
inline std::vector<uint64_t> PackPeerArgs(uint32_t rankSize, const uint64_t *peerInput, const uint64_t *peerOutput,
                                          const uint64_t *peerToken, uint64_t length)
{
    std::vector<uint64_t> args;
    args.reserve(3 * rankSize + 1);
    for (uint32_t i = 0; i < rankSize; ++i) {
        args.push_back(peerInput[i]);
    }
    for (uint32_t i = 0; i < rankSize; ++i) {
        args.push_back(peerOutput[i]);
    }
    for (uint32_t i = 0; i < rankSize; ++i) {
        args.push_back(peerToken[i]);
    }
    args.push_back(length);
    return args;
}

// Pack for write-only collectives (Broadcast): peer input is dead, only
// this rank's own input is needed plus every peer's output/token.  Layout:
// [ownInput, output_0..N-1, token_0..N-1, length] = 2*N+2 entries.
//
// MUST match LoadPeerArgsWriteOnly()'s Load() order exactly (ccu-pitfalls
// rule #5: GeneArgs vector length == Load() count).
inline std::vector<uint64_t> PackPeerArgsWriteOnly(uint32_t rankSize, uint64_t ownInput, const uint64_t *peerOutput,
                                                   const uint64_t *peerToken, uint64_t length)
{
    std::vector<uint64_t> args;
    args.reserve(2 * rankSize + 2);
    args.push_back(ownInput);
    for (uint32_t i = 0; i < rankSize; ++i) {
        args.push_back(peerOutput[i]);
    }
    for (uint32_t i = 0; i < rankSize; ++i) {
        args.push_back(peerToken[i]);
    }
    args.push_back(length);
    return args;
}

// Pack for Scatter on CCU v100: the per-slice source VA is derived inside
// the microcode via Address::operator+=(Variable) (the only arithmetic
// CcuRep exposes on v100 — see hcomm/ccu/ccu_datatype_v1.h), so we only
// need to ship the root buffer's base + step (instead of N individual
// slice VAs).  Layout:
//   [rootInputBase, sliceStep, output_0..N-1, token_0..N-1, length]
// = 2*N + 3 entries.
//
// `sliceStep` is kept distinct from `length` even though Scatter happens
// to use the same byte count for both today — a future variable-sized
// scatter (a la all-to-all-v) would diverge them, and conflating them in
// the wire format would make that change a silent ABI break.
//
// MUST match LoadScatterArgs()'s Load() order exactly.
inline std::vector<uint64_t> PackScatterArgs(uint32_t rankSize, uint64_t rootInputBase, uint64_t sliceStep,
                                             const uint64_t *peerOutput, const uint64_t *peerToken, uint64_t length)
{
    std::vector<uint64_t> args;
    args.reserve(2 * rankSize + 3);
    args.push_back(rootInputBase);
    args.push_back(sliceStep);
    for (uint32_t i = 0; i < rankSize; ++i) {
        args.push_back(peerOutput[i]);
    }
    for (uint32_t i = 0; i < rankSize; ++i) {
        args.push_back(peerToken[i]);
    }
    args.push_back(length);
    return args;
}

} // namespace detail
} // namespace ccu
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_CCU_CCU_MESH_COMMON_HPP
