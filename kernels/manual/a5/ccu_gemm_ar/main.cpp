/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/**
 * CCU GEMM AllReduce — CCU RS+AG (Pull Reduce @ owner + Push Broadcast)
 * HCCL backend — launched via mpirun
 *
 * Unlike v3 (full-buffer GroupReduce on every rank), this demo scopes communication
 * to owner shards for RS+AG-equivalent bandwidth ~ 2*(P-1)/P:
 *   1. AIC: swizzle + groupDone AtomicAdd; gemm_output packed by owner-contiguous layout
 *      (each owner shard padded to a multiple of comm-group-tiles for residual groups).
 *   2. AIV progress: local groupDone → peer TNOTIFY(groupReady) → owner TriggerProgressCke.
 *   3. CCU pipelined: HcommCcuKernelRegister fused RS+AG (pull reduce then push broadcast per group).
 *   4. CCU Sequential baseline: dedicated one-shot fused RS+AG (gate CKE only).
 *      Launch parks CCU on WaitEvent(seqGate); after full GEMM, AIV peer-sync Triggers once.
 *
 * Usage:
 *   mpirun -n <NRANKS> ./ccu_gemm_allreduce [--first-device ID]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <random>
#include <thread>
#include <chrono>
#include <algorithm>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

#include <sys/stat.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>

#include "acl/acl.h"
#include "acl/error_codes/rt_error_codes.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_comm.h"
#include "comm_mpi.h"
#include "comm_context.h"
#include "config.h"
#include "kernel_launchers.h"

#include "host_comm.hpp"

// cann-9.2 libtiling_api.a may pull platform_ascendc.o which needs C CheckLogLevel.
extern "C" int32_t CheckLogLevel(int32_t moduleId, int32_t logLevel)
{
    (void)moduleId;
    (void)logLevel;
    return 0;
}

using SchedTiming = ProgressCtx::Timing;
static SchedTiming g_lastPipeRsTiming{};
static bool g_lastPipeSchedTimingValid = false;

// =============================================================================
// Host-side helpers
// ============================================================================
inline void HcclHostBarrier(HcclComm comm, aclrtStream stream)
{
    HcclBarrier(comm, stream);
    aclrtSynchronizeStream(stream);
}

inline void* WindowAlloc(uint64_t windowBase, size_t& offset, size_t bytes)
{
    void* ptr = reinterpret_cast<void*>(windowBase + offset);
    offset += bytes;
    return ptr;
}

// ============================================================================
// Helpers
// ============================================================================

struct PerfStats {
    double avg;
    double med;
    double min_val;
    double max_val;
    double std_dev;
};

// Headline metric is avg (matches ccu_gemm_ar_v3 / HCCL all_reduce_test);
// median is reported in brackets.
static PerfStats calcStats(const std::vector<double>& times)
{
    if (times.empty())
        return {0.0, 0.0, 0.0, 0.0, 0.0};

    double sum = 0.0;
    double mn = times[0];
    double mx = times[0];
    for (double t : times) {
        sum += t;
        if (t < mn)
            mn = t;
        if (t > mx)
            mx = t;
    }
    double avg = sum / times.size();
    double var = 0.0;
    for (double t : times)
        var += (t - avg) * (t - avg);

    std::vector<double> sorted = times;
    std::sort(sorted.begin(), sorted.end());
    size_t n = sorted.size();
    double med = (n % 2 == 0) ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 : sorted[n / 2];

    return {avg, med, mn, mx, std::sqrt(var / times.size())};
}

// ============================================================================
// Per-rank execution: sub-functions
// ============================================================================

static float halfToFloat(uint16_t h)
{
    uint32_t sign = ((uint32_t)h & 0x8000) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1F;
    uint32_t mant = (uint32_t)h & 0x03FF;
    if (exp == 0) {
        if (mant == 0) {
            union {
                uint32_t u;
                float f;
            } r;
            r.u = sign;
            return r.f;
        }
        while (!(mant & 0x0400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x0400;
    } else if (exp == 31) {
        union {
            uint32_t u;
            float f;
        } r;
        r.u = sign | 0x7F800000 | (mant << 13);
        return r.f;
    }
    exp = exp + (127 - 15);
    uint32_t bits = sign | (exp << 23) | (mant << 13);
    union {
        uint32_t u;
        float f;
    } r;
    r.u = bits;
    return r.f;
}

static bool VerifyOutput(const uint16_t* output_fp16, const float* golden, int nRanks)
{
    const float atol = 1.0f;
    const float rtol = 0.01f;
    const size_t valid_elements = (size_t)G_ORIG_M * G_ORIG_N;
    float max_diff = 0.0f;
    float max_diff_ratio = 0.0f;
    size_t err_count = 0;
    const size_t err_threshold = static_cast<size_t>(valid_elements * rtol);
    constexpr size_t MAX_ERR_PRINT = 4;

    for (size_t row = 0; row < G_ORIG_M; ++row) {
        for (size_t col = 0; col < G_ORIG_N; ++col) {
            size_t idx = row * G_N + col;
            float exp_val = golden[idx];
            float act_val = halfToFloat(output_fp16[idx]);
            float diff = std::abs(exp_val - act_val);
            float rel = (std::abs(exp_val) > 1e-3f) ? (diff / std::abs(exp_val)) : 0.0f;
            if (diff > max_diff)
                max_diff = diff;
            if (rel > max_diff_ratio)
                max_diff_ratio = rel;
            if (diff > atol + rtol * std::abs(exp_val)) {
                err_count++;
                if (err_count <= MAX_ERR_PRINT) {
                    uint32_t ti = (uint32_t)(row / G_BASE_M);
                    uint32_t tj = (uint32_t)(col / G_BASE_N);
                    int tile_idx = ti * G_N_TILES + tj;
                    int owner = (nRanks > 0) ? (tile_idx % nRanks) : 0;
                    printf(
                        "  ERR[%zu] row=%zu col=%zu tile=(%u,%u) tile_idx=%d owner=%d "
                        "exp=%.4f act=%.4f diff=%.4f rel=%.4f\n",
                        err_count, row, col, ti, tj, tile_idx, owner, exp_val, act_val, diff, rel);
                }
            }
        }
    }

    bool ok = (err_count <= err_threshold);
    std::cout << "[VERIFY] valid_region=" << G_ORIG_M << "x" << G_ORIG_N << " max_diff=" << max_diff
              << " max_ratio=" << max_diff_ratio << " err=" << err_count << "/" << err_threshold << " -> "
              << (ok ? "PASS" : "FAIL") << std::endl;
    return ok;
}

static void DumpVerifySamples(const DeviceBuffers& buf, int rank_id, const float* golden)
{
    constexpr size_t kSample = 16;
    uint16_t packed[kSample]{};
    uint16_t row[kSample]{};
    aclrtMemcpy(packed, sizeof(packed), buf.reduced_output, sizeof(packed), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(row, sizeof(row), buf.row_output, sizeof(row), ACL_MEMCPY_DEVICE_TO_HOST);

    std::cerr << "[VERIFY-DIAG] rank=" << rank_id << " tile0 row0 sample";
    for (size_t i = 0; i < kSample; ++i) {
        std::cerr << " c" << i << "{gold=" << golden[i] << ",packed=" << halfToFloat(packed[i])
                  << ",row=" << halfToFloat(row[i]) << "}";
    }
    std::cerr << std::endl;
}

static void PrintTimingDetails(
    const PerfStats& comp_s, const PerfStats& seq_s, const PerfStats& pipe_s, const PerfStats& seq_comp_s,
    const PerfStats& seq_comm_s, const PerfStats& pipe_comp_s, const PerfStats& pipe_comm_s, double flops_per_rank,
    double flops_total, double rs_bytes, double ag_bytes)
{
    auto gflops = [](double flops, double us) { return (us > 0) ? (flops / (us * 1e-6) / 1e9) : 0.0; };
    auto bw_gbs = [&](double us) {
        return (us > 0) ? ((rs_bytes + ag_bytes) / (us * 1e-6) / (1024.0 * 1024.0 * 1024.0)) : 0.0;
    };

    std::cout << "\n  (avg over iterations; brackets: med=median, std=stddev vs avg)" << std::endl;
    std::cout << "  Compute-only:   " << std::setprecision(1) << comp_s.avg << " us"
              << "  (" << std::setprecision(0) << gflops(flops_per_rank, comp_s.avg) << " GFLOPS)"
              << "  [med=" << std::setprecision(1) << comp_s.med << ", std=" << comp_s.std_dev << "]" << std::endl;
    std::cout << "\n  Sequential:     " << std::setprecision(1) << seq_s.avg << " us"
              << "  [med=" << seq_s.med << ", std=" << seq_s.std_dev << "]  (one-shot fused RS→AG)" << std::endl;
    std::cout << "    compute:      " << seq_comp_s.avg << " us"
              << "  (" << std::setprecision(0) << gflops(flops_per_rank, seq_comp_s.avg) << " GFLOPS)" << std::endl;
    std::cout << "    comm:         " << std::setprecision(1) << seq_comm_s.avg << " us"
              << "  (" << std::setprecision(1) << bw_gbs(seq_comm_s.avg) << " GB/s)"
              << "  [med=" << seq_comm_s.med << ", std=" << seq_comm_s.std_dev << "]" << std::endl;
    std::cout << "\n  Pipelined:      " << std::setprecision(1) << pipe_s.avg << " us (end-to-end)"
              << "  [med=" << pipe_s.med << ", std=" << pipe_s.std_dev << "]" << std::endl;
    std::cout << "    compute done: " << pipe_comp_s.avg << " us (kernel)"
              << "  (" << std::setprecision(0) << gflops(flops_per_rank, pipe_comp_s.avg) << " GFLOPS, "
              << std::setprecision(1)
              << (gflops(flops_per_rank, pipe_comp_s.avg) / gflops(flops_per_rank, comp_s.avg) * 100.0) << "% of pure)"
              << std::endl;
    std::cout << "    comm done:    " << std::setprecision(1) << pipe_comm_s.avg << " us"
              << "  (" << std::setprecision(1) << bw_gbs(pipe_comm_s.avg) << " GB/s)" << std::endl;

    double speedup = (pipe_s.avg > 0) ? (seq_s.avg / pipe_s.avg) : 0.0;
    // Overlap uses medians so a few sequential/pipe outliers do not inflate overlap >100%.
    double overlap_time = (seq_comp_s.med + seq_comm_s.med) - pipe_s.med;
    double overlap_eff = (overlap_time > 0) ? (overlap_time / std::min(seq_comp_s.med, seq_comm_s.med) * 100.0) : 0.0;

    std::cout << "\n  Speedup:        " << std::setprecision(3) << speedup << "x" << std::endl;
    std::cout << "  Time saved:     " << std::setprecision(1) << (seq_s.avg - pipe_s.avg) << " us"
              << " (" << std::setprecision(1)
              << ((seq_s.avg > 0) ? ((seq_s.avg - pipe_s.avg) / seq_s.avg * 100.0) : 0.0) << "%)" << std::endl;
    std::cout << "  Overlap eff:    " << std::setprecision(1) << overlap_eff << "%" << std::endl;
    std::cout << "  Throughput:     " << std::setprecision(0) << gflops(flops_total, pipe_s.avg) << " GFLOPS (total)"
              << std::endl;
    std::cout << "================================================================\n" << std::endl;
}

static bool ReadSchedTimingFromDevice(const DeviceBuffers& buf, SchedTiming& rsTiming)
{
    if (buf.progressCtx_dev == nullptr)
        return false;

    const uint8_t* base = reinterpret_cast<const uint8_t*>(buf.progressCtx_dev);
    aclError rsRet = aclrtMemcpy(
        &rsTiming, sizeof(rsTiming), base + offsetof(ProgressDeviceCtx, header) + offsetof(ProgressCtx, rsSchedTiming),
        sizeof(rsTiming), ACL_MEMCPY_DEVICE_TO_HOST);
    return rsRet == ACL_SUCCESS;
}

static void PrintSchedTimingReport(int rank_id, double pipe_aiv_wall_us)
{
    if (!SchedTimingEnabled() || !g_lastPipeSchedTimingValid)
        return;

    // All ranks print (stderr) so 2/4-rank critical-path owner can be compared.
    std::cerr << "\n  --- Progress wait breakdown (PTO_CCU_GEMM_AR_SCHED_TIMING=1, last pipelined iter, rank" << rank_id
              << ") ---" << std::endl;
    if (rank_id == 0) {
        std::cerr << "  SYS_CNT tick: " << SchedTickNs() << " ns (override via PTO_CCU_GEMM_AR_SCHED_TICK_NS)"
                  << std::endl;
        std::cerr << "  Default path: localGD → peer TNOTIFY(groupReady) → owner (peerReady||BP||ring) → fire; "
                     "peerReady samples → waitPeer; BP/ring attribute to first still-pending"
                  << std::endl;
    }
    const uint64_t total = g_lastPipeRsTiming.totalCycles;
    std::cerr << "    [progress_kernel] total=" << SchedCyclesToUs(total)
              << " us  triggerCount=" << g_lastPipeRsTiming.triggerCount
              << "  waitLocalGD=" << SchedCyclesToUs(g_lastPipeRsTiming.groupDoneCycles)
              << " us  waitPeer=" << SchedCyclesToUs(g_lastPipeRsTiming.notifyCycles)
              << " us  waitBP=" << SchedCyclesToUs(g_lastPipeRsTiming.backpressureCycles)
              << " us  finalWait=" << SchedCyclesToUs(g_lastPipeRsTiming.finalWaitCycles)
              << " us  vs aiv_wall=" << std::setprecision(1) << pipe_aiv_wall_us << " us" << std::endl;
    std::cerr.flush();
}

static void PrintCommPhaseStats(const char* label, const std::vector<CommPhaseTiming>& records)
{
    const PerfStats comm_wall = calcStats(ExtractCommField(records, &CommPhaseTiming::comm_wall_us));
    const PerfStats aiv_wall = calcStats(ExtractCommField(records, &CommPhaseTiming::aiv_wall_us));
    const PerfStats ccu_wall = calcStats(ExtractCommField(records, &CommPhaseTiming::ccu_wall_us));
    const std::vector<double> aiv_event_vals = ExtractValidEventField(records, &CommPhaseTiming::aiv_event_us);
    const std::vector<double> comm_event_vals = ExtractValidEventField(records, &CommPhaseTiming::comm_event_us);
    const uint32_t outliers =
        CountCommOutliers(ExtractCommField(records, &CommPhaseTiming::comm_wall_us), comm_wall.med, comm_wall.std_dev);

    std::cout << "  [" << label << "] comm wall med=" << std::setprecision(1) << comm_wall.med << " us"
              << " [" << comm_wall.avg << " ± " << comm_wall.std_dev << ", min=" << comm_wall.min_val
              << ", max=" << comm_wall.max_val << "]" << std::endl;
    std::cout << "    aiv wall:  med=" << aiv_wall.med << " us (" << std::setprecision(1)
              << (comm_wall.med > 0.0 ? (aiv_wall.med / comm_wall.med * 100.0) : 0.0)
              << "% of comm)  ccu wall: med=" << ccu_wall.med << " us" << std::endl;
    if (!aiv_event_vals.empty()) {
        const PerfStats aiv_event = calcStats(aiv_event_vals);
        const PerfStats comm_event = comm_event_vals.empty() ? aiv_event : calcStats(comm_event_vals);
        std::cout << "    aiv event: med=" << aiv_event.med << " us  comm~event: med=" << comm_event.med << " us"
                  << "  (aivStream only; ccu tail = wall)" << std::endl;
    }
    std::cout << "    outliers:  " << outliers << "/" << records.size() << "  (>1.25x median or > median+2σ)"
              << std::endl;
}

static void PrintCommIterSamples(
    const std::vector<CommPhaseTiming>& seq_comm_diag, const std::vector<CommPhaseTiming>& pipe_comm_diag)
{
    std::cout << "\n  --- Per-iteration comm samples ---" << std::endl;
    std::cout << "  iter  seq_wall  seq_aiv  seq_ccu  seq_aiv_ev  pipe_wall pipe_aiv pipe_ccu pipe_aiv_ev" << std::endl;
    const size_t n = std::max(seq_comm_diag.size(), pipe_comm_diag.size());
    for (size_t i = 0; i < n; ++i) {
        std::cout << "  " << std::setw(4) << i;
        if (i < seq_comm_diag.size()) {
            const CommPhaseTiming& s = seq_comm_diag[i];
            std::cout << "  " << std::setw(8) << std::setprecision(1) << s.comm_wall_us << "  " << std::setw(8)
                      << s.aiv_wall_us << "  " << std::setw(8) << s.ccu_wall_us << "  " << std::setw(9)
                      << FormatEventUs(s.aiv_event_us);
        } else {
            std::cout << "  " << std::setw(8) << "-" << "  " << std::setw(8) << "-" << "  " << std::setw(8) << "-"
                      << "  " << std::setw(9) << "-";
        }
        if (i < pipe_comm_diag.size()) {
            const CommPhaseTiming& p = pipe_comm_diag[i];
            std::cout << "  " << std::setw(9) << p.comm_wall_us << "  " << std::setw(8) << p.aiv_wall_us << "  "
                      << std::setw(8) << p.ccu_wall_us << "  " << std::setw(9) << FormatEventUs(p.aiv_event_us);
        }
        std::cout << std::endl;
    }
}

static void PrintCommDiagnosticReport(
    int rank_id, const std::vector<CommPhaseTiming>& seq_comm_diag, const std::vector<CommPhaseTiming>& pipe_comm_diag)
{
    if (!CommDiagEnabled() || rank_id != 0 || seq_comm_diag.empty() || pipe_comm_diag.empty()) {
        return;
    }
    std::cout << "\n  --- Comm timing breakdown (aiv scheduler vs ccu stream sync) ---" << std::endl;
    std::cout << "  Note: ccu wall is sync(ccuStream) after scheduler done; ~0 us means CCU"
              << " finished before/with scheduler exit." << std::endl;
    PrintCommPhaseStats("sequential comm", seq_comm_diag);
    PrintCommPhaseStats("pipelined comm", pipe_comm_diag);
    PrintCommIterSamples(seq_comm_diag, pipe_comm_diag);
}

static void PrintPerfReport(
    bool is_ok, int n_ranks, const std::vector<double>& compute_times_us,
    const std::vector<double>& sequential_times_us, const std::vector<double>& pipelined_times_us,
    const std::vector<double>& seq_compute_us, const std::vector<double>& seq_comm_us,
    const std::vector<double>& pipe_compute_us, const std::vector<double>& pipe_comm_us, int rank_id,
    const std::vector<CommPhaseTiming>& seq_comm_diag, const std::vector<CommPhaseTiming>& pipe_comm_diag)
{
    PerfStats comp_s = calcStats(compute_times_us);
    PerfStats seq_s = calcStats(sequential_times_us);
    PerfStats pipe_s = calcStats(pipelined_times_us);
    PerfStats seq_comp_s = calcStats(seq_compute_us);
    PerfStats seq_comm_s = calcStats(seq_comm_us);
    PerfStats pipe_comp_s = calcStats(pipe_compute_us);
    PerfStats pipe_comm_s = calcStats(pipe_comm_us);

    double flops_per_rank = 2.0 * G_ORIG_M * (double)G_K * G_ORIG_N;
    double flops_total = flops_per_rank * ((n_ranks > 0) ? n_ranks : 1);

    size_t tileBytes = static_cast<size_t>(G_BASE_M) * G_BASE_N * sizeof(uint16_t);
    int tiles_per_owner = (n_ranks > 0) ? ((G_NUM_TILES + n_ranks - 1) / n_ranks) : G_NUM_TILES;
    double rs_bytes = static_cast<double>(G_NUM_TILES - tiles_per_owner) * tileBytes;
    int safe_remotes = (n_ranks > 1) ? (n_ranks - 1) : 0;
    double ag_bytes = static_cast<double>(tiles_per_owner) * safe_remotes * tileBytes;
    double data_gb = (rs_bytes + ag_bytes) / (1024.0 * 1024.0 * 1024.0);

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "\n================================================================" << std::endl;
    std::cout << (is_ok ? "[SUCCESS]" : "[FAILED]") << " CCU GEMM AllReduce A5 FP16 (HCCL)" << std::endl;
    std::cout << "  M=" << G_ORIG_M << " K=" << G_K << " N=" << G_ORIG_N;
    if (G_M != G_ORIG_M || G_N != G_ORIG_N)
        std::cout << "  (padded " << G_M << "x" << G_K << "x" << G_N << ")";
    std::cout << "  ranks=" << n_ranks << "  compute_blocks=" << COMPUTE_BLOCK_NUM
              << "  ccu_group_tiles=" << G_COMM_GROUP_TILES << std::endl;
    std::cout << "  tiles=" << G_NUM_TILES << " (" << G_M_TILES << "x" << G_N_TILES << ")"
              << "  comm_data=" << std::setprecision(3) << data_gb << " GB/rank" << std::endl;

    PrintTimingDetails(
        comp_s, seq_s, pipe_s, seq_comp_s, seq_comm_s, pipe_comp_s, pipe_comm_s, flops_per_rank, flops_total, rs_bytes,
        ag_bytes);
    PrintCommDiagnosticReport(rank_id, seq_comm_diag, pipe_comm_diag);
}

template <typename ResetState, typename LaunchCompute, typename SyncAll>
static void RunComputeOnlyBenchmark(
    int rank_id, int n_ranks, const DeviceBuffers& buf, ResetState& resetState, LaunchCompute& launchComp,
    SyncAll& syncAll, aclrtStream computeStream, aclrtStream commStream, HcclComm comm,
    std::vector<double>& compute_times_us)
{
    for (int iter = 0; iter < COMPUTE_ONLY_ITERS; ++iter) {
        resetState();
        aclrtSynchronizeStream(computeStream);
        HcclHostBarrier(comm, commStream);
        auto t0 = std::chrono::high_resolution_clock::now();
        launchComp(computeStream);
        aclrtSynchronizeStream(computeStream);
        auto t1 = std::chrono::high_resolution_clock::now();
        compute_times_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        HcclHostBarrier(comm, commStream);
    }
}

// Measured iterations for avg/median under CCU re-launch jitter.
static constexpr int CCU_MEASURE_ITERS = 20;

// True if ccuStream still has outstanding work (blocked in WaitEvent(seqGate) as expected).
static bool SeqCcuStillWaitingOnGate(const CcuState& ccu)
{
    if (ccu.ccuStream == nullptr) {
        return false;
    }
    aclrtStreamStatus status = ACL_STREAM_STATUS_RESERVED;
    const aclError q = aclrtStreamQuery(ccu.ccuStream, &status);
    if (q != ACL_SUCCESS) {
        return false;
    }
    // NOT_READY ⇒ kernel still running (good: WaitEvent holding).
    // COMPLETE ⇒ already drained without Trigger (gate failed).
    return status == ACL_STREAM_STATUS_NOT_READY;
}

// Shared seq one-shot body (warmup + measure): Launch → GEMM → assert WaitEvent holds → peer-sync Trigger → sync.
// Launch-before-GEMM matches Pipelined; CCU must stay in WaitEvent(seqGate) until post-GEMM Trigger.
// timingOut == nullptr skips recording (warmup). Shared peer-ready TNOTIFY walk; no progress CKE / Host barrier.
template <typename LaunchCompute>
static bool RunSeqOneShotCore(
    int rank_id, int n_ranks, LaunchCompute& launchComp, aclrtStream computeStream, CcuState& ccu,
    const DeviceBuffers& buf, const char* syncLabel, const char* fatalMsg, std::vector<double>* seq_us,
    std::vector<double>* seq_comp_us, std::vector<double>* seq_comm_us, std::vector<CommPhaseTiming>* seq_comm_diag)
{
    if (!LaunchStoredSeqOneShotCcuGemmAr(ccu)) {
        return false;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    launchComp(computeStream);
    aclrtSynchronizeStream(computeStream);
    auto t1 = std::chrono::high_resolution_clock::now();

    // Gate must still hold after full GEMM — otherwise one-shot ran overlapped with compute.
    if (!SeqCcuStillWaitingOnGate(ccu)) {
        std::cerr << "[CCU-AR] rank=" << rank_id << ": FATAL " << fatalMsg << std::endl;
        return false;
    }

    if (!TriggerSeqOneShotAfterPeerSync(ccu, buf, rank_id, n_ranks)) {
        return false;
    }
    auto tAivDone = std::chrono::high_resolution_clock::now();
    if (!SyncCcuStreams(ccu, rank_id, syncLabel)) {
        return false;
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    if (seq_us != nullptr && seq_comp_us != nullptr && seq_comm_us != nullptr && seq_comm_diag != nullptr) {
        CommPhaseTiming commDiag{};
        commDiag.comm_wall_us = ChronoMicros(t1, t2);
        commDiag.aiv_wall_us = ChronoMicros(t1, tAivDone);
        commDiag.ccu_wall_us = ChronoMicros(tAivDone, t2);
        seq_comm_diag->push_back(commDiag);
        seq_comp_us->push_back(ChronoMicros(t0, t1));
        seq_comm_us->push_back(commDiag.comm_wall_us);
        seq_us->push_back(ChronoMicros(t0, t2));
    }
    return true;
}

// Warm up the dedicated seq one-shot CCU handle before measuring sequential comm.
template <typename ResetState, typename LaunchCompute, typename SyncAll>
static bool WarmupSeqOneShotBaseline(
    int rank_id, int n_ranks, ResetState& resetState, LaunchCompute& launchComp, SyncAll& syncAll,
    aclrtStream computeStream, CcuState& ccu, const DeviceBuffers& buf)
{
    if (!ccu.seqKernelRegistered || std::getenv("PTO_VERIFY_ONLY") != nullptr)
        return true;

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        resetState();
        syncAll();
        const std::string fatal =
            "seq one-shot warmup: CCU finished before seqGate Trigger (iter=" + std::to_string(i) + ")";
        if (!RunSeqOneShotCore(
                rank_id, n_ranks, launchComp, computeStream, ccu, buf, "seq one-shot warmup", fatal.c_str(), nullptr,
                nullptr, nullptr, nullptr)) {
            return false;
        }
        syncAll();
    }
    return true;
}

template <typename LaunchCompute, typename SyncAll>
static bool RunSeqOneShotIter(
    int rank_id, int n_ranks, LaunchCompute& launchComp, SyncAll& syncAll, aclrtStream computeStream, CcuState& ccu,
    const DeviceBuffers& buf, std::vector<double>& seq_us, std::vector<double>& seq_comp_us,
    std::vector<double>& seq_comm_us, std::vector<CommPhaseTiming>& seq_comm_diag)
{
    if (!RunSeqOneShotCore(
            rank_id, n_ranks, launchComp, computeStream, ccu, buf, "sequential one-shot",
            "seq one-shot CCU finished before seqGate Trigger (WaitEvent(seqGate) did not block across GEMM)", &seq_us,
            &seq_comp_us, &seq_comm_us, &seq_comm_diag)) {
        return false;
    }
    syncAll();
    return true;
}

// Fallback only when seq one-shot handle is unavailable.
template <typename LaunchCompute, typename PrepareCcu, typename LaunchProgress, typename SyncAll>
static bool RunSeqProgressFallbackIter(
    int rank_id, LaunchCompute& launchComp, PrepareCcu& prepareCcu, LaunchProgress& launchProgress, SyncAll& syncAll,
    aclrtStream computeStream, CcuState& ccu, std::vector<double>& seq_us, std::vector<double>& seq_comp_us,
    std::vector<double>& seq_comm_us, std::vector<CommPhaseTiming>& seq_comm_diag)
{
    if (!prepareCcu()) {
        return false;
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    launchComp(computeStream);
    aclrtSynchronizeStream(computeStream);
    auto t1 = std::chrono::high_resolution_clock::now();
    launchProgress();
    aclrtSynchronizeStream(ccu.aivStream);
    if (!SyncCcuStreams(ccu, rank_id, "sequential fallback ccu")) {
        return false;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    CommPhaseTiming commDiag{};
    commDiag.comm_wall_us = ChronoMicros(t1, t2);
    seq_comm_diag.push_back(commDiag);
    seq_comp_us.push_back(ChronoMicros(t0, t1));
    seq_comm_us.push_back(commDiag.comm_wall_us);
    seq_us.push_back(ChronoMicros(t0, t2));
    syncAll();
    return true;
}

// Sequential benchmark: one-shot fused RS→AG after full GEMM (peer TNOTIFY, no progress CKE).
template <typename ResetState, typename LaunchCompute, typename PrepareCcu, typename LaunchProgress, typename SyncAll>
static bool RunSequentialBenchmark(
    int rank_id, int n_ranks, const DeviceBuffers& buf, ResetState& resetState, LaunchCompute& launchComp,
    PrepareCcu& prepareCcu, LaunchProgress& launchProgress, SyncAll& syncAll, aclrtStream computeStream, CcuState& ccu,
    std::vector<double>& seq_us, std::vector<double>& seq_comp_us, std::vector<double>& seq_comm_us,
    std::vector<CommPhaseTiming>& seq_comm_diag)
{
    for (int iter = 0; iter < CCU_MEASURE_ITERS; ++iter) {
        resetState();
        syncAll();
        if (!ccu.seqKernelRegistered) {
            if (!RunSeqProgressFallbackIter(
                    rank_id, launchComp, prepareCcu, launchProgress, syncAll, computeStream, ccu, seq_us, seq_comp_us,
                    seq_comm_us, seq_comm_diag)) {
                return false;
            }
            continue;
        }
        if (!RunSeqOneShotIter(
                rank_id, n_ranks, launchComp, syncAll, computeStream, ccu, buf, seq_us, seq_comp_us, seq_comm_us,
                seq_comm_diag)) {
            std::cerr << "[CCU-AR] rank=" << rank_id << ": sequential benchmark aborted at iter=" << iter << std::endl;
            return false;
        }
    }
    return true;
}

// Pipelined one iter: prepareCcu → launchProgress(AIV) → launchComp (overlapped).
template <typename LaunchCompute, typename PrepareCcu, typename LaunchProgress, typename SyncAll>
static bool RunPipelinedIter(
    int rank_id, int iter, const DeviceBuffers& buf, LaunchCompute& launchComp, PrepareCcu& prepareCcu,
    LaunchProgress& launchProgress, SyncAll& syncAll, aclrtStream computeStream, CcuState& ccu, aclrtEvent evStart,
    aclrtEvent evEnd, aclrtEvent evCommStart, aclrtEvent evAivDone, aclrtEvent evCcuDone, std::vector<double>& pipe_us,
    std::vector<double>& pipe_comp_us, std::vector<double>& pipe_comm_us, std::vector<CommPhaseTiming>& pipe_comm_diag)
{
    if (!prepareCcu()) {
        return false;
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    aclrtRecordEvent(evCommStart, ccu.aivStream);
    launchProgress();
    aclrtRecordEvent(evStart, computeStream);
    launchComp(computeStream);
    aclrtRecordEvent(evEnd, computeStream);

    aclrtSynchronizeStream(ccu.aivStream);
    aclrtRecordEvent(evAivDone, ccu.aivStream);
    auto tAivDone = std::chrono::high_resolution_clock::now();
    if (!SyncCcuStreams(ccu, rank_id, "pipelined benchmark ccu")) {
        return false;
    }
    aclrtRecordEvent(evCcuDone, ccu.ccuStream);
    auto tCommDone = std::chrono::high_resolution_clock::now();
    aclrtSynchronizeStream(computeStream);
    auto t1 = std::chrono::high_resolution_clock::now();

    float compute_ms = 0.0f;
    aclrtEventElapsedTime(&compute_ms, evStart, evEnd);
    CommPhaseTiming commDiag{};
    commDiag.comm_wall_us = ChronoMicros(t0, tCommDone);
    commDiag.aiv_wall_us = ChronoMicros(t0, tAivDone);
    commDiag.ccu_wall_us = ChronoMicros(tAivDone, tCommDone);
    FillCommEventElapsed(commDiag, evCommStart, evAivDone, evCcuDone);
    pipe_comm_diag.push_back(commDiag);

    if (SchedTimingEnabled() && iter == CCU_MEASURE_ITERS - 1) {
        if (ReadSchedTimingFromDevice(buf, g_lastPipeRsTiming)) {
            g_lastPipeSchedTimingValid = true;
        }
    }
    pipe_comp_us.push_back(static_cast<double>(compute_ms) * 1000.0);
    pipe_comm_us.push_back(commDiag.comm_wall_us);
    pipe_us.push_back(ChronoMicros(t0, t1));
    syncAll();
    return true;
}

// Pipelined benchmark: overlap via window groupDone + owner remote poll + CKE.
template <typename ResetState, typename LaunchCompute, typename PrepareCcu, typename LaunchProgress, typename SyncAll>
static void RunPipelinedBenchmark(
    int rank_id, int n_ranks, const DeviceBuffers& buf, ResetState& resetState, LaunchCompute& launchComp,
    PrepareCcu& prepareCcu, LaunchProgress& launchProgress, SyncAll& syncAll, aclrtStream computeStream, CcuState& ccu,
    std::vector<double>& pipe_us, std::vector<double>& pipe_comp_us, std::vector<double>& pipe_comm_us,
    std::vector<CommPhaseTiming>& pipe_comm_diag)
{
    (void)n_ranks;
    aclrtEvent evStart = nullptr;
    aclrtEvent evEnd = nullptr;
    aclrtEvent evCommStart = nullptr;
    aclrtEvent evAivDone = nullptr;
    aclrtEvent evCcuDone = nullptr;
    aclrtCreateEvent(&evStart);
    aclrtCreateEvent(&evEnd);
    aclrtCreateEvent(&evCommStart);
    aclrtCreateEvent(&evAivDone);
    aclrtCreateEvent(&evCcuDone);

    for (int iter = 0; iter < CCU_MEASURE_ITERS; ++iter) {
        resetState();
        syncAll();
        (void)RunPipelinedIter(
            rank_id, iter, buf, launchComp, prepareCcu, launchProgress, syncAll, computeStream, ccu, evStart, evEnd,
            evCommStart, evAivDone, evCcuDone, pipe_us, pipe_comp_us, pipe_comm_us, pipe_comm_diag);
    }

    DestroyAclEvents(evCommStart, evAivDone, evCcuDone);
    aclrtDestroyEvent(evStart);
    aclrtDestroyEvent(evEnd);
}

// ============================================================================
// Per-rank device buffer management
// ============================================================================

static size_t AlignUpSize(size_t v, size_t a)
{
    if (a == 0) {
        return v;
    }
    return (v + a - 1) / a * a;
}

static void FreePartialDeviceBuffers(DeviceBuffers& buf)
{
    if (buf.ccuBlock) {
        aclrtFree(buf.ccuBlock);
        buf.ccuBlock = nullptr;
    }
    if (buf.src0_dev) {
        aclrtFree(buf.src0_dev);
        buf.src0_dev = nullptr;
    }
    if (buf.src1_dev) {
        aclrtFree(buf.src1_dev);
        buf.src1_dev = nullptr;
    }
    if (buf.progressCtx_dev) {
        aclrtFree(buf.progressCtx_dev);
        buf.progressCtx_dev = nullptr;
    }
}

static bool AllocCcuBlockAndSignals(
    DeviceBuffers& buf, int rank_id, const CommDeviceContext& windowHostCtx, size_t ccuBlockSize)
{
    aclrtMallocAttrValue modVal{};
    modVal.moduleId = 3;
    aclrtMallocAttribute attr{ACL_RT_MEM_ATTR_MODULE_ID, modVal};
    aclrtMallocConfig cfg{&attr, 1};
    aclError aRet = aclrtMallocWithCfg(&buf.ccuBlock, ccuBlockSize, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);
    if (aRet != ACL_SUCCESS || !buf.ccuBlock) {
        std::cerr << "[ERROR] Rank " << rank_id << ": alloc CCU HBM block failed: " << static_cast<int>(aRet)
                  << std::endl;
        return false;
    }

    buf.gemm_output = buf.ccuBlock;
    buf.reduced_output = static_cast<char*>(buf.ccuBlock) + buf.packedSize;
    buf.row_output = static_cast<char*>(buf.ccuBlock) + 2 * buf.packedSize;
    buf.ccuToken = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(buf.ccuBlock), ccuBlockSize);

    uint64_t windowBase = windowHostCtx.windowsIn[windowHostCtx.rankId];
    size_t winOffset = 0;
    void* guardPad = WindowAlloc(windowBase, winOffset, WINDOW_GUARD_BYTES);
    buf.signal_matrix = WindowAlloc(windowBase, winOffset, buf.signalBytes);
    if (winOffset > windowHostCtx.winSize) {
        std::cerr << "[ERROR] Rank " << rank_id << ": HCCL window too small for signal_matrix (need " << winOffset
                  << ", have " << windowHostCtx.winSize << ")" << std::endl;
        FreePartialDeviceBuffers(buf);
        return false;
    }
    buf.groupDoneSize = static_cast<size_t>(G_SIGNAL_GROUP_DONE_SLOTS) * sizeof(int32_t);
    buf.groupDone_dev = reinterpret_cast<int32_t*>(buf.signal_matrix) + G_SIGNAL_GROUP_DONE_OFFSET;

    if (VerboseLog()) {
        std::cerr << "[INFO] Rank " << rank_id << ": CCU block VA=0x" << std::hex
                  << reinterpret_cast<uint64_t>(buf.ccuBlock) << " size=" << std::dec << ccuBlockSize << " token=0x"
                  << std::hex << buf.ccuToken << std::dec << " signal_matrix(window)=0x" << std::hex
                  << reinterpret_cast<uint64_t>(buf.signal_matrix) << std::dec << std::endl;
    }

    aclrtMemset(buf.gemm_output, buf.packedSize, 0, buf.packedSize);
    aclrtMemset(buf.reduced_output, buf.packedSize, 0, buf.packedSize);
    aclrtMemset(buf.row_output, buf.rowOutputSize, 0, buf.rowOutputSize);
    aclrtMemset(guardPad, WINDOW_GUARD_BYTES, 0, WINDOW_GUARD_BYTES);
    aclrtMemset(buf.signal_matrix, buf.signalBytes, 0, buf.signalBytes);
    return true;
}

static bool AllocSrcAndProgressCtx(DeviceBuffers& buf, int rank_id, size_t aSize, size_t bSize)
{
    aclError aRet = aclrtMalloc(&buf.src0_dev, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aRet != ACL_SUCCESS || !buf.src0_dev) {
        std::cerr << "[ERROR] Rank " << rank_id << ": alloc src0_dev failed: " << static_cast<int>(aRet) << std::endl;
        FreePartialDeviceBuffers(buf);
        return false;
    }
    aRet = aclrtMalloc(&buf.src1_dev, bSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aRet != ACL_SUCCESS || !buf.src1_dev) {
        std::cerr << "[ERROR] Rank " << rank_id << ": alloc src1_dev failed: " << static_cast<int>(aRet) << std::endl;
        FreePartialDeviceBuffers(buf);
        return false;
    }
    aRet = aclrtMalloc(&buf.progressCtx_dev, sizeof(ProgressDeviceCtx), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aRet != ACL_SUCCESS || !buf.progressCtx_dev) {
        std::cerr << "[ERROR] Rank " << rank_id << ": alloc progressCtx_dev failed: " << static_cast<int>(aRet)
                  << std::endl;
        FreePartialDeviceBuffers(buf);
        return false;
    }
    WriteProgressCtxToDevice(buf);
    return true;
}

static bool AllocDeviceBuffers(
    DeviceBuffers& buf, int rank_id, int n_ranks, const CommDeviceContext& windowHostCtx, const uint16_t* a_data,
    size_t a_bytes, const uint16_t* b_data, size_t b_bytes)
{
    const uint32_t safeRanks = (n_ranks > 0) ? static_cast<uint32_t>(n_ranks) : 1;
    buf.packedSize =
        static_cast<size_t>(CcuTotalPaddedTiles(G_NUM_TILES, safeRanks)) * static_cast<size_t>(G_TILE_BYTES);
    buf.rowOutputSize = static_cast<size_t>(G_M) * G_N * sizeof(uint16_t);
    // Layout: [gemm_output_packed | reduced_output_packed | row_output]
    // signal_matrix lives in the HCCL window (groupDone + groupReady).
    buf.signalBytes = AlignUpSize(static_cast<size_t>(G_SIGNAL_MATRIX_SLOTS) * sizeof(int32_t), 512);
    const size_t ccuBlockSize = 2 * buf.packedSize + buf.rowOutputSize;
    if (!AllocCcuBlockAndSignals(buf, rank_id, windowHostCtx, ccuBlockSize)) {
        return false;
    }

    const size_t aSize = static_cast<size_t>(G_M) * G_K * sizeof(uint16_t);
    const size_t bSize = static_cast<size_t>(G_K) * G_N * sizeof(uint16_t);
    if (!AllocSrcAndProgressCtx(buf, rank_id, aSize, bSize)) {
        return false;
    }

    aclrtMemcpy(buf.src0_dev, aSize, a_data, a_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(buf.src1_dev, bSize, b_data, b_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    return true;
}

static void FreeDeviceBuffers(DeviceBuffers& buf)
{
    if (buf.ccuBlock)
        aclrtFree(buf.ccuBlock);
    if (buf.src0_dev)
        aclrtFree(buf.src0_dev);
    if (buf.src1_dev)
        aclrtFree(buf.src1_dev);
    if (buf.progressCtx_dev)
        aclrtFree(buf.progressCtx_dev);
    // groupDone_dev aliases signal_matrix (HCCL window) — not separately freed.
    buf = {};
}

// ============================================================================
// Per-rank execution logic
// ============================================================================
// Warmup: pipelined ordering — progress monitor then overlapped compute.
template <typename ResetState, typename LaunchCompute, typename PrepareCcu, typename LaunchProgress, typename SyncAll>
static void RunWarmupPasses(
    int rank_id, int n_ranks, ResetState& resetState, LaunchCompute& launchComp, PrepareCcu& prepareCcu,
    LaunchProgress& launchProgress, SyncAll& syncAll, aclrtStream computeStream, CcuState& ccu,
    const DeviceBuffers& buf)
{
    (void)n_ranks;
    (void)buf;
    const int warmupIters = (std::getenv("PTO_VERIFY_ONLY") != nullptr) ? 0 : WARMUP_ITERS;
    for (int i = 0; i < warmupIters; ++i) {
        resetState();
        syncAll();
        if (!prepareCcu())
            continue;
        if (VerboseLog()) {
            std::cerr << "[CCU-AR] rank=" << rank_id << ": prepare done, fire progress+compute (warmup " << i << ")"
                      << std::endl;
            std::cerr.flush();
        }
        launchProgress();
        launchComp(computeStream);
        if (aclrtSynchronizeStream(computeStream) != ACL_SUCCESS)
            return;
        if (!SyncCcuStreams(ccu, rank_id, "warmup-ccu"))
            return;
        aclrtSynchronizeStream(ccu.aivStream);
        syncAll();
    }
}

// Final verification pass.
template <typename ResetState, typename LaunchCompute, typename PrepareCcu, typename LaunchProgress, typename SyncAll>
static bool RunFinalVerificationPass(
    int rank_id, int n_ranks, ResetState& resetState, LaunchCompute& launchComp, PrepareCcu& prepareCcu,
    LaunchProgress& launchProgress, SyncAll& syncAll, aclrtStream computeStream, CcuState& ccu,
    const DeviceBuffers& buf, const float* golden)
{
    resetState();
    syncAll();
    if (!prepareCcu())
        return false;
    launchProgress();
    launchComp(computeStream);
    if (aclrtSynchronizeStream(computeStream) != ACL_SUCCESS)
        return false;
    if (!SyncCcuStreams(ccu, rank_id, "verify-ccu"))
        return false;
    if (aclrtSynchronizeStream(ccu.aivStream) != ACL_SUCCESS)
        return false;

    // AG writes into this rank's reduced_output are issued by peer CCU streams.
    // Local stream completion only proves this rank has finished its own work,
    // so wait for every rank before unpacking the peer-written packed buffer.
    syncAll();
    launchCcuGemmArUnpack(
        reinterpret_cast<uint8_t*>(buf.reduced_output), reinterpret_cast<uint8_t*>(buf.row_output), ccu.aivStream,
        kCcuSubtilesPerTile, static_cast<uint32_t>(n_ranks));
    aclrtSynchronizeStream(ccu.aivStream);
    syncAll();
    if (VerboseLog()) {
        DumpVerifySamples(buf, rank_id, golden);
    }

    uint16_t* output_host_fp16 = nullptr;
    aclrtMallocHost(reinterpret_cast<void**>(&output_host_fp16), buf.rowOutputSize);
    aclrtMemcpy(output_host_fp16, buf.rowOutputSize, buf.row_output, buf.rowOutputSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // row_output is [M, G_N] row-major; CCU communication uses packed buffers.
    bool is_ok = VerifyOutput(output_host_fp16, golden, n_ranks);
    std::cerr << "[VERIFY] rank=" << rank_id << " result=" << (is_ok ? "PASS" : "FAIL") << std::endl;
    aclrtFreeHost(output_host_fp16);
    return is_ok;
}

template <typename ResetState, typename LaunchCompute, typename PrepareCcu, typename LaunchProgress, typename SyncAll>
static bool RunBenchmarkAndVerify(
    int rank_id, int n_ranks, ResetState& resetState, LaunchCompute& launchComp, PrepareCcu& prepareCcu,
    LaunchProgress& launchProgress, SyncAll& syncAll, aclrtStream computeStream, aclrtStream commStream, HcclComm comm,
    CcuState& ccu, const DeviceBuffers& buf, const float* golden)
{
    RunWarmupPasses(
        rank_id, n_ranks, resetState, launchComp, prepareCcu, launchProgress, syncAll, computeStream, ccu, buf);
    if (!WarmupSeqOneShotBaseline(rank_id, n_ranks, resetState, launchComp, syncAll, computeStream, ccu, buf)) {
        return false;
    }

    std::vector<double> compute_us;
    std::vector<double> seq_us;
    std::vector<double> seq_comp_us;
    std::vector<double> seq_comm_us;
    std::vector<double> pipe_us;
    std::vector<double> pipe_comp_us;
    std::vector<double> pipe_comm_us;
    std::vector<CommPhaseTiming> seq_comm_diag;
    std::vector<CommPhaseTiming> pipe_comm_diag;
    // PTO_VERIFY_ONLY=1: skip benchmarks so verify runs against the first and only
    // CCU pass (after warmup) — isolates correctness from repeated-launch state.
    const bool verifyOnly = std::getenv("PTO_VERIFY_ONLY") != nullptr;
    if (!verifyOnly) {
        g_lastPipeSchedTimingValid = false;
        RunComputeOnlyBenchmark(
            rank_id, n_ranks, buf, resetState, launchComp, syncAll, computeStream, commStream, comm, compute_us);
        if (!RunSequentialBenchmark(
                rank_id, n_ranks, buf, resetState, launchComp, prepareCcu, launchProgress, syncAll, computeStream, ccu,
                seq_us, seq_comp_us, seq_comm_us, seq_comm_diag)) {
            return false;
        }
        RunPipelinedBenchmark(
            rank_id, n_ranks, buf, resetState, launchComp, prepareCcu, launchProgress, syncAll, computeStream, ccu,
            pipe_us, pipe_comp_us, pipe_comm_us, pipe_comm_diag);
    }

    const bool is_ok = RunFinalVerificationPass(
        rank_id, n_ranks, resetState, launchComp, prepareCcu, launchProgress, syncAll, computeStream, ccu, buf, golden);

    if (!verifyOnly) {
        if (rank_id == 0) {
            PrintPerfReport(
                is_ok, n_ranks, compute_us, seq_us, pipe_us, seq_comp_us, seq_comm_us, pipe_comp_us, pipe_comm_us,
                rank_id, seq_comm_diag, pipe_comm_diag);
        }
        // Every rank reports its own progress wait mix (critical-path owner may not be rank0).
        syncAll();
        const double last_pipe_aiv_us = pipe_comm_diag.empty() ? 0.0 : pipe_comm_diag.back().aiv_wall_us;
        PrintSchedTimingReport(rank_id, last_pipe_aiv_us);
        syncAll();
    }
    return is_ok;
}

static void ResetDeviceState(int rank_id, int n_ranks, const DeviceBuffers& buf)
{
    (void)rank_id;
    (void)n_ranks;
    aclrtMemset(buf.gemm_output, buf.packedSize, 0, buf.packedSize);
    aclrtMemset(buf.reduced_output, buf.packedSize, 0, buf.packedSize);
    aclrtMemset(buf.row_output, buf.rowOutputSize, 0, buf.rowOutputSize);
    // Clears window groupDone + groupReady.
    aclrtMemset(buf.signal_matrix, buf.signalBytes, 0, buf.signalBytes);
}

static void LaunchCompute(const DeviceBuffers& buf, int rank_id, int n_ranks, aclrtStream s)
{
    launchCcuGemmArCompute(
        reinterpret_cast<uint8_t*>(buf.gemm_output), reinterpret_cast<uint8_t*>(buf.src0_dev),
        reinterpret_cast<uint8_t*>(buf.src1_dev), reinterpret_cast<int32_t*>(buf.groupDone_dev), rank_id, s,
        COMPUTE_BLOCK_NUM, G_K, static_cast<uint32_t>(n_ranks));
}

// Launch the AIV progress monitor after any prior pass on the same stream has finished.
static void LaunchCcuProgress(CcuState& ccu, const DeviceBuffers& buf, int rankId, int nRanks)
{
    ProgressCkeCtx rsCke{};
    for (uint32_t slot = 0; slot < CCU_PROGRESS_SLOTS; ++slot) {
        rsCke.ckeSlotVA[slot] = ccu.rsProgressVA[slot];
        rsCke.ckeMask[slot] = ccu.rsProgressMask[slot];
    }
    rsCke.itemsDoneAddr = MissionRsItemsDoneAddr(buf.progressCtx_dev, 0);
    rsCke.kernelReadyAddr = MissionRsKernelReadyAddr(buf.progressCtx_dev, 0);

    launchCcuGemmArProgress(
        ccu.aivStream, rsCke, reinterpret_cast<int32_t*>(buf.groupDone_dev),
        reinterpret_cast<uint8_t*>(buf.progressCtx_dev), reinterpret_cast<uint8_t*>(buf.signal_matrix),
        reinterpret_cast<uint8_t*>(buf.windowDeviceCtx), G_NUM_TILES, static_cast<uint32_t>(nRanks),
        static_cast<uint32_t>(rankId));
}

static bool InitHcclCommWithRetry(int rank_id, int n_ranks, const HcclRootInfo* rootInfo, HcclComm& hcclComm)
{
    constexpr int kMaxRetries = 3;
    HcclResult hret = HCCL_SUCCESS;
    for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
        hret =
            HcclCommInitRootInfo(static_cast<uint32_t>(n_ranks), rootInfo, static_cast<uint32_t>(rank_id), &hcclComm);
        if (hret == HCCL_SUCCESS) {
            return true;
        }
        std::cerr << "[WARN] Rank " << rank_id << ": HcclCommInitRootInfo failed: " << hret << " (attempt "
                  << (attempt + 1) << "/" << kMaxRetries << "), retrying in 5s..." << std::endl;
        sleep(5);
    }
    std::cerr << "[ERROR] Rank " << rank_id << ": HcclCommInitRootInfo failed\n";
    return false;
}

struct CcuAddrPack {
    uint64_t inputVA;
    uint64_t outputVA;
    uint64_t token;
};

static void ExchangeCcuAddrs(
    int rank_id, int n_ranks, const DeviceBuffers& buf, uint64_t* allInputVA, uint64_t* allOutputVA,
    uint64_t* allTokens)
{
    CcuAddrPack myPack{
        reinterpret_cast<uint64_t>(buf.gemm_output), reinterpret_cast<uint64_t>(buf.reduced_output), buf.ccuToken};
    std::vector<CcuAddrPack> allPacks(n_ranks);
    CommMpiAllgather(&myPack, sizeof(CcuAddrPack), allPacks.data(), sizeof(CcuAddrPack));
    for (int i = 0; i < n_ranks && i < static_cast<int>(kMaxGemmArRanks); ++i) {
        allInputVA[i] = allPacks[i].inputVA;
        allOutputVA[i] = allPacks[i].outputVA;
        allTokens[i] = allPacks[i].token;
    }
    if (VerboseLog()) {
        std::cerr << "[INFO] Rank " << rank_id << ": CCU VA exchange done, " << n_ranks << " peers" << std::endl;
    }
}

struct RankSession {
    aclrtStream computeStream{nullptr};
    aclrtStream commStream{nullptr};
    HcclComm hcclComm{nullptr};
    CcuState ccu{};
    GemmHcclContext windowCtx{};
    DeviceBuffers buf{};
    uint64_t allInputVA[kMaxGemmArRanks]{};
    uint64_t allOutputVA[kMaxGemmArRanks]{};
    uint64_t allTokens[kMaxGemmArRanks]{};
    int status{0};
};

static void DestroyRankSession(RankSession& s)
{
    FreeDeviceBuffers(s.buf);
    if (s.ccu.ccuStream) {
        aclrtDestroyStream(s.ccu.ccuStream);
        s.ccu.ccuStream = nullptr;
    }
    if (s.ccu.aivStream) {
        aclrtDestroyStream(s.ccu.aivStream);
        s.ccu.aivStream = nullptr;
    }
    s.windowCtx.Finalize();
    if (s.hcclComm != nullptr) {
        HcclCommDestroy(s.hcclComm);
        s.hcclComm = nullptr;
    }
    if (s.computeStream) {
        s.status |= aclrtDestroyStream(s.computeStream);
        s.computeStream = nullptr;
    }
    if (s.commStream) {
        s.status |= aclrtDestroyStream(s.commStream);
        s.commStream = nullptr;
    }
}

static bool InitRankSession(
    RankSession& s, int rank_id, int n_ranks, const uint16_t* a_data, size_t a_bytes, const uint16_t* b_data,
    size_t b_bytes, const HcclRootInfo* rootInfo)
{
    s.status |= aclrtCreateStream(&s.computeStream);
    s.status |= aclrtCreateStream(&s.commStream);
    if (!InitHcclCommWithRetry(rank_id, n_ranks, rootInfo, s.hcclComm)) {
        DestroyRankSession(s);
        return false;
    }
    s.status |= aclrtCreateStream(&s.ccu.ccuStream);
    s.status |= aclrtCreateStream(&s.ccu.aivStream);
    if (!InitCcuThreadAndChannels(s.hcclComm, rank_id, n_ranks, s.ccu)) {
        DestroyRankSession(s);
        return false;
    }
    if (!s.windowCtx.InitWindowOnExistingComm(rank_id, n_ranks, s.hcclComm, s.commStream)) {
        std::cerr << "[ERROR] Rank " << rank_id << ": HCCL window context init failed\n";
        DestroyRankSession(s);
        return false;
    }
    s.buf.windowDeviceCtx = s.windowCtx.deviceCtx;
    if (!AllocDeviceBuffers(s.buf, rank_id, n_ranks, s.windowCtx.hostCtx, a_data, a_bytes, b_data, b_bytes)) {
        DestroyRankSession(s);
        return false;
    }
    HcclHostBarrier(s.hcclComm, s.commStream);
    static_assert(MAX_RANKS <= kMaxGemmArRanks, "MAX_RANKS exceeds CCU limit");
    ExchangeCcuAddrs(rank_id, n_ranks, s.buf, s.allInputVA, s.allOutputVA, s.allTokens);
    CommMpiBarrier();
    if (!RegisterCcuGemmArOnce(s.hcclComm, s.ccu, rank_id, n_ranks, s.buf)) {
        DestroyRankSession(s);
        return false;
    }
    StoreCcuLaunchContext(s.hcclComm, rank_id, n_ranks, s.allInputVA, s.allOutputVA, s.allTokens);
    return true;
}

static bool RunGemmAllReducePerRank(
    int rank_id, int n_ranks, const uint16_t* a_data, size_t a_bytes, const uint16_t* b_data, size_t b_bytes,
    const float* golden, const HcclRootInfo* rootInfo)
{
    RankSession s{};
    if (!InitRankSession(s, rank_id, n_ranks, a_data, a_bytes, b_data, b_bytes, rootInfo)) {
        return false;
    }

    auto resetState = [&]() { ResetDeviceState(rank_id, n_ranks, s.buf); };
    auto launchComp = [&](aclrtStream stream) { LaunchCompute(s.buf, rank_id, n_ranks, stream); };
    auto prepareCcu = [&]() {
        return PrepareCcuKernel(s.buf, s.hcclComm, s.ccu, rank_id, n_ranks, s.allInputVA, s.allOutputVA, s.allTokens);
    };
    auto launchProgress = [&]() { LaunchCcuProgress(s.ccu, s.buf, rank_id, n_ranks); };
    auto syncAll = [&]() {
        aclrtSynchronizeStream(s.computeStream);
        SyncCcuStreams(s.ccu, rank_id, "syncAll ccu");
        aclrtSynchronizeStream(s.ccu.aivStream);
        aclrtSynchronizeStream(s.commStream);
        HcclHostBarrier(s.hcclComm, s.commStream);
    };

    const bool is_ok = RunBenchmarkAndVerify(
        rank_id, n_ranks, resetState, launchComp, prepareCcu, launchProgress, syncAll, s.computeStream, s.commStream,
        s.hcclComm, s.ccu, s.buf, golden);
    DestroyRankSession(s);
    return (s.status == 0) && is_ok;
}

// ============================================================================
// MPI-based multi-process launcher
// ============================================================================
static void PrintLaunchBanner(int n_ranks, int first_device_id)
{
    std::cout << "\n================================================================" << std::endl;
    std::cout << "  CCU GEMM AllReduce A5 FP16 (ReduceScatter + AllGather) — HCCL backend" << std::endl;
    std::cout << "  M=" << G_ORIG_M << " K=" << G_K << " N=" << G_ORIG_N;
    if (G_M != G_ORIG_M || G_N != G_ORIG_N)
        std::cout << "  (padded " << G_M << "x" << G_N << ")";
    std::cout << "  tile=" << G_BASE_M << "x" << G_BASE_K << "x" << G_BASE_N << "  tiles=" << G_NUM_TILES << std::endl;
    std::cout << "  ranks=" << n_ranks << "  devices=[" << first_device_id << "," << (first_device_id + n_ranks) << ")"
              << "  compute_blocks=" << COMPUTE_BLOCK_NUM << "  ccu_group_tiles=" << G_COMM_GROUP_TILES << std::endl;
    std::cout << "  mode: independent A per rank, shared B (CCU tile-level fusion)" << std::endl;
    std::cout << "================================================================" << std::endl;
}

static bool InitHcclRootInfoWithRetry(HcclRootInfo& rootInfo)
{
    constexpr int kMaxRetries = 3;
    HcclResult hret = HCCL_SUCCESS;
    for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
        hret = HcclGetRootInfo(&rootInfo);
        if (hret == HCCL_SUCCESS)
            return true;
        std::cerr << "[WARN] HcclGetRootInfo failed: " << hret << " (attempt " << (attempt + 1) << "/" << kMaxRetries
                  << "), retrying in 5s..." << std::endl;
        sleep(5);
    }
    std::cerr << "[ERROR] HcclGetRootInfo failed after " << kMaxRetries << " attempts: " << hret << std::endl;
    return false;
}

static bool RunGemmAllReduce(
    int n_ranks, int first_device_id, const uint16_t* a_parts, const uint16_t* b_data, const float* golden)
{
    if (n_ranks <= 0 || n_ranks > 8) {
        std::cerr << "[ERROR] Invalid n_ranks: " << n_ranks << " (must be 1-8)\n";
        return false;
    }

    int mpiRank = CommMpiRank();
    if (mpiRank == 0)
        PrintLaunchBanner(n_ranks, first_device_id);

    int device_id = mpiRank % n_ranks + first_device_id;

    constexpr int kAclRepeatInit = 100002;
    aclError aRet = aclInit(nullptr);
    if (aRet != ACL_SUCCESS && static_cast<int>(aRet) != kAclRepeatInit) {
        std::cerr << "[ERROR] Rank " << mpiRank << ": aclInit failed: " << (int)aRet << std::endl;
        return false;
    }

    aRet = aclrtSetDevice(device_id);
    if (aRet != ACL_SUCCESS) {
        std::cerr << "[ERROR] Rank " << mpiRank << ": aclrtSetDevice(" << device_id << ") failed\n";
        return false;
    }

    HcclRootInfo rootInfo{};
    if (mpiRank == 0 && !InitHcclRootInfoWithRetry(rootInfo))
        return false;
    CommMpiBcast(&rootInfo, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0);
    CommMpiBarrier();

    size_t a_rank_elems = (size_t)G_M * G_K;
    bool ok = RunGemmAllReducePerRank(
        mpiRank, n_ranks, a_parts + (size_t)mpiRank * a_rank_elems, (size_t)G_M * G_K * sizeof(uint16_t), b_data,
        (size_t)G_N * G_K * sizeof(uint16_t), golden, &rootInfo);
    CommMpiBarrier();
    aclrtResetDevice(device_id);
    aclFinalize();
    return ok;
}

// ============================================================================
// Data generation helpers
// ============================================================================

static uint16_t floatToHalf(float f)
{
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = f;
    uint32_t x = bits.u;
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x007FFFFF;
    if (exp <= 0)
        return (uint16_t)sign;
    if (exp >= 31)
        return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static void gemmBlockedTile(
    const float* B, float* C_row, int K, int N, const float* A_row, int kk, int kEnd, int jj, int jEnd)
{
    for (int k = kk; k < kEnd; k++) {
        float aik = A_row[k];
        for (int j = jj; j < jEnd; j++)
            C_row[(size_t)j] += aik * B[(size_t)k * N + j];
    }
}

static void gemmBlockedRowRange(const float* A, const float* B, float* C, int K, int N, int r0, int r1)
{
    constexpr int BLK = 64;
    for (int i = r0; i < r1; i++) {
        for (int kk = 0; kk < K; kk += BLK) {
            int kEnd = std::min(kk + BLK, K);
            for (int jj = 0; jj < N; jj += BLK)
                gemmBlockedTile(B, &C[(size_t)i * N], K, N, &A[(size_t)i * K], kk, kEnd, jj, std::min(jj + BLK, N));
        }
    }
}

static void computeGolden(const float* A, const float* B, float* C, int M, int K, int N)
{
    memset_s(C, (size_t)M * N * sizeof(float), 0, (size_t)M * N * sizeof(float));

    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0)
        hw = 1;
    unsigned n_threads = std::min(hw, 64u);

    int rows_per = (M + (int)n_threads - 1) / (int)n_threads;
    std::vector<std::thread> threads;

    for (unsigned t = 0; t < n_threads; t++) {
        int r0 = (int)t * rows_per;
        int r1 = std::min(r0 + rows_per, M);
        if (r0 >= M)
            break;
        threads.emplace_back(gemmBlockedRowRange, A, B, C, K, N, r0, r1);
    }
    for (auto& th : threads)
        th.join();
}

// ============================================================================
// Input file caching: save/load binary matrices to avoid regeneration
// ============================================================================

struct CachedEnvVars {
    std::string gemm_ar_dir;
    std::string first_device;
};

static CachedEnvVars g_cached_env;

static std::string SafeGetEnv(const char* name)
{
    std::string prefix = std::string(name) + "=";
    std::ifstream ifs("/proc/self/environ", std::ios::binary);
    if (!ifs.is_open()) {
        return {};
    }
    std::string entry;
    while (std::getline(ifs, entry, '\0')) {
        if (entry.compare(0, prefix.size(), prefix) == 0) {
            return entry.substr(prefix.size());
        }
    }
    return {};
}

static void InitCachedEnv()
{
    g_cached_env.gemm_ar_dir = SafeGetEnv("GEMM_AR_DIR");
    g_cached_env.first_device = SafeGetEnv("GEMM_ALLREDUCE_FIRST_DEVICE");
}

static std::string getInputDir()
{
    const std::string& envDir = g_cached_env.gemm_ar_dir;
    if (!envDir.empty()) {
        return envDir + "/input";
    }
    return "input";
}

static std::string getInputPrefix(int nranks)
{
    std::string dir = getInputDir();
    return dir + "/M" + std::to_string(G_ORIG_M) + "_K" + std::to_string(G_ORIG_K) + "_N" + std::to_string(G_ORIG_N) +
           "_R" + std::to_string(nranks);
}

static void ensureDirExists(const std::string& dir) { mkdir(dir.c_str(), 0755); }

template <typename T>
static bool saveBinary(const std::string& path, const T* data, size_t count)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
        return false;
    ofs.write(reinterpret_cast<const char*>(data), count * sizeof(T));
    return ofs.good();
}

template <typename T>
static bool loadBinary(const std::string& path, T* data, size_t count)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        return false;
    ifs.read(reinterpret_cast<char*>(data), count * sizeof(T));
    return ifs.good();
}

static bool inputFilesExist(int nranks)
{
    std::string prefix = getInputPrefix(nranks);
    std::string aFile = prefix + "_A.bin";
    std::string bFile = prefix + "_B.bin";
    std::string gFile = prefix + "_golden.bin";
    struct stat st;
    return (stat(aFile.c_str(), &st) == 0 && stat(bFile.c_str(), &st) == 0 && stat(gFile.c_str(), &st) == 0);
}

static bool loadInputFiles(
    int nranks, std::vector<uint16_t>& a_parts, std::vector<uint16_t>& b_data, std::vector<float>& golden)
{
    std::string prefix = getInputPrefix(nranks);
    std::string aFile = prefix + "_A.bin";
    std::string bFile = prefix + "_B.bin";
    std::string gFile = prefix + "_golden.bin";

    size_t a_total = (size_t)nranks * G_M * G_K;
    size_t b_total = (size_t)G_N * G_K;
    size_t g_total = (size_t)G_M * G_N;

    a_parts.resize(a_total);
    b_data.resize(b_total);
    golden.resize(g_total);

    if (VerboseLog())
        printf("Loading cached input from: %s_*.bin\n", prefix.c_str());

    if (!loadBinary(aFile, a_parts.data(), a_total)) {
        fprintf(stderr, "[ERROR] Failed to load %s\n", aFile.c_str());
        return false;
    }
    if (!loadBinary(bFile, b_data.data(), b_total)) {
        fprintf(stderr, "[ERROR] Failed to load %s\n", bFile.c_str());
        return false;
    }
    if (!loadBinary(gFile, golden.data(), g_total)) {
        fprintf(stderr, "[ERROR] Failed to load %s\n", gFile.c_str());
        return false;
    }

    if (VerboseLog())
        printf("  Loaded A[%d ranks × %d × %d], B[%d × %d], golden[%d × %d]\n", nranks, G_M, G_K, G_N, G_K, G_M, G_N);
    return true;
}

static bool saveInputFiles(
    int nranks, const std::vector<uint16_t>& a_parts, const std::vector<uint16_t>& b_data,
    const std::vector<float>& golden)
{
    ensureDirExists(getInputDir());
    std::string prefix = getInputPrefix(nranks);
    std::string aFile = prefix + "_A.bin";
    std::string bFile = prefix + "_B.bin";
    std::string gFile = prefix + "_golden.bin";

    size_t a_total = (size_t)nranks * G_M * G_K;
    size_t b_total = (size_t)G_N * G_K;
    size_t g_total = (size_t)G_M * G_N;

    if (VerboseLog())
        printf("Saving input data to: %s_*.bin\n", prefix.c_str());

    if (!saveBinary(aFile, a_parts.data(), a_total)) {
        fprintf(stderr, "[WARN] Failed to save %s\n", aFile.c_str());
        return false;
    }
    if (!saveBinary(bFile, b_data.data(), b_total)) {
        fprintf(stderr, "[WARN] Failed to save %s\n", bFile.c_str());
        return false;
    }
    if (!saveBinary(gFile, golden.data(), g_total)) {
        fprintf(stderr, "[WARN] Failed to save %s\n", gFile.c_str());
        return false;
    }

    double a_mb = a_total * sizeof(uint16_t) / (1024.0 * 1024.0);
    double b_mb = b_total * sizeof(uint16_t) / (1024.0 * 1024.0);
    double g_mb = g_total * sizeof(float) / (1024.0 * 1024.0);
    if (VerboseLog())
        printf("  Saved: A=%.1f MB, B=%.1f MB, golden=%.1f MB\n", a_mb, b_mb, g_mb);
    return true;
}

static void convertFp32ToFp16Padded(
    const std::vector<std::vector<float>>& A_fp32_all, const std::vector<float>& B_fp32, int nranks,
    std::vector<uint16_t>& a_parts, std::vector<uint16_t>& b_data)
{
    size_t a_rank_elems = (size_t)G_M * G_K;
    size_t b_elems = (size_t)G_N * G_K;
    a_parts.assign((size_t)nranks * a_rank_elems, 0);
    b_data.assign(b_elems, 0);

    for (int r = 0; r < nranks; r++) {
        uint16_t* a_dst = a_parts.data() + (size_t)r * a_rank_elems;
        for (int i = 0; i < (int)G_ORIG_M; i++)
            for (int j = 0; j < (int)G_K; j++)
                a_dst[(size_t)i * G_K + j] = floatToHalf(A_fp32_all[r][(size_t)i * G_K + j]);
    }

    uint16_t* b_dst = b_data.data();
    for (int i = 0; i < (int)G_ORIG_N; i++)
        for (int j = 0; j < (int)G_K; j++)
            b_dst[(size_t)i * G_K + j] = floatToHalf(B_fp32[(size_t)j * G_ORIG_N + i]);
}

static void padGoldenToAligned(const std::vector<float>& golden_orig, std::vector<float>& golden)
{
    golden.assign((size_t)G_M * G_N, 0.0f);
    for (int i = 0; i < (int)G_ORIG_M; i++)
        memcpy_s(
            &golden[(size_t)i * G_N], G_N * sizeof(float), &golden_orig[(size_t)i * G_ORIG_N],
            G_ORIG_N * sizeof(float));
}

static bool generateData(
    int nranks, std::vector<uint16_t>& a_parts, std::vector<uint16_t>& b_data, std::vector<float>& golden)
{
    if (VerboseLog()) {
        printf(
            "Data Parallel: each rank has independent A[%d,%d], shared B[%d,%d], %d ranks\n", G_ORIG_M, G_K, G_K,
            G_ORIG_N, nranks);
    }

    std::mt19937 gen(42);
    float scale = std::sqrt(65000.0f / ((float)G_K * nranks * 4.0f));
    std::uniform_real_distribution<float> dist(-scale, scale);

    std::vector<std::vector<float>> A_fp32_all(nranks);
    for (int r = 0; r < nranks; r++) {
        A_fp32_all[r].resize((size_t)G_ORIG_M * G_K);
        for (auto& v : A_fp32_all[r])
            v = dist(gen);
    }
    std::vector<float> B_fp32((size_t)G_K * G_ORIG_N);
    for (auto& v : B_fp32)
        v = dist(gen);

    if (VerboseLog())
        printf("  Computing golden reference (sum of %d CPU GEMMs %d×%d×%d)...\n", nranks, G_ORIG_M, G_K, G_ORIG_N);
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<float> golden_orig((size_t)G_ORIG_M * G_ORIG_N, 0.0f);
    std::vector<float> tmp((size_t)G_ORIG_M * G_ORIG_N);
    for (int r = 0; r < nranks; r++) {
        memset_s(tmp.data(), tmp.size() * sizeof(float), 0, tmp.size() * sizeof(float));
        computeGolden(A_fp32_all[r].data(), B_fp32.data(), tmp.data(), G_ORIG_M, G_K, G_ORIG_N);
        for (size_t i = 0; i < golden_orig.size(); i++)
            golden_orig[i] += tmp[i];
    }

    double secs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    if (VerboseLog())
        printf("  Golden computed in %.1f s\n", secs);

    padGoldenToAligned(golden_orig, golden);
    convertFp32ToFp16Padded(A_fp32_all, B_fp32, nranks, a_parts, b_data);

    double gsum = 0.0;
    for (auto v : golden_orig)
        gsum += v;
    if (VerboseLog())
        printf("  Golden = sum(A_i × B): shape=(%d, %d), sum=%.2f\n", G_ORIG_M, G_ORIG_N, gsum);
    return true;
}

// ============================================================================
// Entry point
// ============================================================================

static int parseFirstDevice(int argc, char* argv[])
{
    const std::string& envVal = g_cached_env.first_device;
    if (!envVal.empty()) {
        int val = atoi(envVal.c_str());
        if (val >= 0)
            return val;
    }
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "--first-device") == 0) {
            int val = atoi(argv[i + 1]);
            if (val >= 0)
                return val;
        }
    }
    return 0;
}

static bool PrepareInputData(
    int n_ranks, std::vector<uint16_t>& a_parts, std::vector<uint16_t>& b_data, std::vector<float>& golden)
{
    size_t a_total = (size_t)n_ranks * G_M * G_K;
    size_t b_total = (size_t)G_N * G_K;
    size_t g_total = (size_t)G_M * G_N;

    int data_ok = 0;
    if (CommMpiRank() == 0) {
        if (inputFilesExist(n_ranks)) {
            if (VerboseLog())
                printf("[INFO] Found cached input files, loading...\n");
            if (!loadInputFiles(n_ranks, a_parts, b_data, golden))
                data_ok = 1;
        } else {
            if (VerboseLog())
                printf("[INFO] No cached input files, generating...\n");
            if (!generateData(n_ranks, a_parts, b_data, golden)) {
                data_ok = 1;
            } else {
                saveInputFiles(n_ranks, a_parts, b_data, golden);
            }
        }
    } else {
        a_parts.resize(a_total);
        b_data.resize(b_total);
        golden.resize(g_total);
    }

    CommMpiBcast(&data_ok, 1, COMM_MPI_INT, 0);
    if (data_ok != 0)
        return false;

    CommMpiBcast(a_parts.data(), (int)(a_total * sizeof(uint16_t)), COMM_MPI_CHAR, 0);
    CommMpiBcast(b_data.data(), (int)(b_total * sizeof(uint16_t)), COMM_MPI_CHAR, 0);
    CommMpiBcast(golden.data(), (int)(g_total * sizeof(float)), COMM_MPI_CHAR, 0);
    return true;
}

int main(int argc, char* argv[])
{
    InitCachedEnv();

    if (!CommMpiInit(&argc, &argv)) {
        fprintf(stderr, "[ERROR] MPI_Init failed. Launch with: mpirun -n <NRANKS> ./ccu_gemm_allreduce\n");
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: mpirun -n <NRANKS> %s [--first-device ID]\n", argv[0]);
            CommMpiFinalize();
            return 0;
        }
    }

    int n_ranks = CommMpiSize();
    int first_device_id = parseFirstDevice(argc, argv);

    if (CommMpiRank() == 0) {
        printf(
            "CCU GEMM AllReduce A5 FP16 (HCCL): ranks=%d, devices=[%d, %d)\n\n", n_ranks, first_device_id,
            first_device_id + n_ranks);
    }

    std::vector<uint16_t> a_parts;
    std::vector<uint16_t> b_data;
    std::vector<float> golden;

    if (!PrepareInputData(n_ranks, a_parts, b_data, golden)) {
        CommMpiFinalize();
        return 1;
    }

    bool ok = RunGemmAllReduce(n_ranks, first_device_id, a_parts.data(), b_data.data(), golden.data());

    if (CommMpiRank() == 0) {
        printf(ok ? "\nCCU GEMM AllReduce demo completed successfully.\n" : "\nCCU GEMM AllReduce demo FAILED.\n");
    }

    CommMpiFinalize();
    return ok ? 0 : 1;
}
