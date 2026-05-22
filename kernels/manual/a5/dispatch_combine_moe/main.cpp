/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl_types.h"

#include "kernel_launch.hpp"
#include "op_host/comm_mpi.h"
#include "op_host/data_utils.hpp"
#include "op_host/runtime_context.hpp"
#include "op_host/tiling_builder.hpp"

extern "C" rtError_t rtSetDevice(int32_t device);

namespace {

constexpr int kDefaultWarmupIters = 3;
constexpr int kDefaultMeasureIters = 5;
constexpr double kMicrosecondsPerSecond = 1000.0 * 1000.0;
constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;

struct DeviceBuffer {
    void *ptr = nullptr;
    size_t bytes = 0;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    DeviceBuffer(DeviceBuffer &&other) noexcept : ptr(other.ptr), bytes(other.bytes)
    {
        other.ptr = nullptr;
        other.bytes = 0;
    }

    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
    {
        if (this != &other) {
            if (ptr != nullptr) {
                aclrtFree(ptr);
            }
            ptr = other.ptr;
            bytes = other.bytes;
            other.ptr = nullptr;
            other.bytes = 0;
        }
        return *this;
    }

    ~DeviceBuffer()
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
};

struct EventHandle {
    aclrtEvent event = nullptr;

    ~EventHandle()
    {
        if (event != nullptr) {
            aclrtDestroyEvent(event);
        }
    }
};

struct PerfStats {
    double avg = 0.0;
    double min = 0.0;
    double max = 0.0;
    double stddev = 0.0;
};

struct PerfThroughput {
    double input_tokens_per_s = 0.0;
    double routed_tokens_per_s = 0.0;
    double tflops = 0.0;
    double gbs = 0.0;
};

struct TimingSamples {
    std::vector<double> kernel_times_us;
    std::vector<double> e2e_times_us;
};

struct HostInputData {
    std::vector<uint8_t> x;
    std::vector<uint8_t> weight1;
    std::vector<uint8_t> weight2;
    std::vector<uint8_t> expert_idx;
    std::vector<uint8_t> scale1;
    std::vector<uint8_t> scale2;
    std::vector<uint8_t> probs;
    std::vector<uint8_t> x_active_mask;
    std::vector<uint16_t> expected_out;
};

struct DeviceLaunchBuffers {
    const DeviceBuffer &x;
    const DeviceBuffer &weight1;
    const DeviceBuffer &weight2;
    const DeviceBuffer &expert_idx;
    const DeviceBuffer &scale1;
    const DeviceBuffer &scale2;
    const DeviceBuffer &probs;
    const DeviceBuffer &x_active_mask;
    const DeviceBuffer &out;
    const DeviceBuffer &expert_token_nums;
    const DeviceBuffer &workspace;
    const DeviceBuffer &tiling;
};

struct DeviceBufferSet {
    DeviceBuffer x;
    DeviceBuffer weight1;
    DeviceBuffer weight2;
    DeviceBuffer expert_idx;
    DeviceBuffer scale1;
    DeviceBuffer scale2;
    DeviceBuffer probs;
    DeviceBuffer x_active_mask;
    DeviceBuffer out;
    DeviceBuffer expert_token_nums;
    DeviceBuffer workspace;
    DeviceBuffer tiling;
};

DeviceBuffer MakeDeviceBuffer(size_t bytes, const void *host_src = nullptr)
{
    DeviceBuffer buffer;
    buffer.bytes = bytes;
    if (bytes == 0) {
        return buffer;
    }
    if (aclrtMalloc(&buffer.ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        throw std::runtime_error("aclrtMalloc failed");
    }
    if (host_src != nullptr &&
        aclrtMemcpy(buffer.ptr, bytes, host_src, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        throw std::runtime_error("aclrtMemcpy host->device failed");
    }
    return buffer;
}

std::vector<uint16_t> BytesToU16(const std::vector<uint8_t> &bytes)
{
    if (bytes.size() % sizeof(uint16_t) != 0) {
        throw std::runtime_error("fp16 file size is not aligned");
    }
    std::vector<uint16_t> out(bytes.size() / sizeof(uint16_t));
    auto *out_bytes = reinterpret_cast<uint8_t *>(out.data());
    std::copy(bytes.begin(), bytes.end(), out_bytes);
    return out;
}

int ParseEnvInt(const char *name, int default_value)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    try {
        return std::stoi(value);
    } catch (const std::exception &) {
        throw std::runtime_error(std::string("invalid integer in env: ") + name);
    }
}

bool ZeroWindowMemory(const StandaloneRankRuntime &runtime)
{
    const uint64_t window_bytes = runtime.hccl.host_remote_window_ctx.windowBytes;
    for (uint32_t i = 0; i < runtime.hccl.host_remote_window_ctx.rankSize; ++i) {
        void *window_ptr = reinterpret_cast<void *>(runtime.hccl.host_remote_window_ctx.windowIn[i]);
        if (aclrtMemset(window_ptr, window_bytes, 0, window_bytes) != ACL_SUCCESS) {
            return false;
        }
    }
    return true;
}

void ZeroDeviceBuffer(const DeviceBuffer &buffer, const char *name)
{
    if (buffer.bytes == 0) {
        return;
    }
    if (aclrtMemset(buffer.ptr, buffer.bytes, 0, buffer.bytes) != ACL_SUCCESS) {
        throw std::runtime_error(std::string("failed to zero ") + name);
    }
}

void PrepareIterationState(const StandaloneRankRuntime &runtime, const DeviceBuffer &out_dev,
                           const DeviceBuffer &expert_token_nums_dev, const DeviceBuffer &workspace_dev)
{
    if (!ZeroWindowMemory(runtime)) {
        throw std::runtime_error("failed to zero HCCL windows");
    }
    ZeroDeviceBuffer(out_dev, "out buffer");
    ZeroDeviceBuffer(expert_token_nums_dev, "expert_token_nums");
    ZeroDeviceBuffer(workspace_dev, "workspace");
}

PerfStats CalcStats(const std::vector<double> &samples)
{
    PerfStats stats;
    if (samples.empty()) {
        return stats;
    }
    stats.min = *std::min_element(samples.begin(), samples.end());
    stats.max = *std::max_element(samples.begin(), samples.end());
    stats.avg = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
    double variance = 0.0;
    for (double sample : samples) {
        const double delta = sample - stats.avg;
        variance += delta * delta;
    }
    stats.stddev = std::sqrt(variance / static_cast<double>(samples.size()));
    return stats;
}

PerfThroughput CalcThroughput(const CaseConfig &cfg, double avg_us)
{
    PerfThroughput throughput;
    if (avg_us <= 0.0) {
        return throughput;
    }
    throughput.input_tokens_per_s = cfg.input_tokens_all_ranks * kMicrosecondsPerSecond / avg_us;
    throughput.routed_tokens_per_s = cfg.routed_tokens_all_ranks * kMicrosecondsPerSecond / avg_us;
    throughput.tflops = cfg.compute_flops_all_ranks * kMicrosecondsPerSecond / avg_us / 1e12;
    throughput.gbs = cfg.comm_bytes_all_ranks * kMicrosecondsPerSecond / avg_us / kBytesPerGiB;
    return throughput;
}

std::vector<double> GatherMaxSamplesToRoot(const std::vector<double> &local_samples, int rank_id, int world_size)
{
    if (local_samples.empty()) {
        return {};
    }
    const size_t sample_count = local_samples.size();
    const int bytes_per_rank = static_cast<int>(sample_count * sizeof(double));
    std::vector<double> gathered;
    if (rank_id == 0) {
        gathered.resize(sample_count * static_cast<size_t>(world_size));
    }
    CommMpiGather(local_samples.data(), bytes_per_rank, COMM_MPI_CHAR,
                  rank_id == 0 ? static_cast<void *>(gathered.data()) : nullptr, bytes_per_rank, COMM_MPI_CHAR, 0);
    if (rank_id != 0) {
        return {};
    }

    std::vector<double> max_samples(sample_count, 0.0);
    for (size_t sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
        double max_value = gathered[sample_idx];
        for (int rank = 1; rank < world_size; ++rank) {
            max_value = std::max(max_value, gathered[static_cast<size_t>(rank) * sample_count + sample_idx]);
        }
        max_samples[sample_idx] = max_value;
    }
    return max_samples;
}

std::string BuildAccuracyReportText(int rank_id, const AccuracyReport &report, double atol, double rtol)
{
    std::ostringstream os;
    os << std::fixed << std::setprecision(6) << "rank=" << rank_id << " compare(total=" << report.total_count
       << ", mismatch=" << report.mismatch_count << ", nan_or_inf=" << report.nan_or_inf_count
       << ", max_abs_err=" << report.max_abs_err << ", max_rel_err=" << report.max_rel_err
       << ", mean_abs_err=" << report.mean_abs_err << ", rmse=" << report.rmse << ", atol=" << atol << ", rtol=" << rtol
       << ")";
    if (report.has_first_bad) {
        os << '\n'
           << std::fixed << std::setprecision(6) << "rank=" << rank_id << " first_bad(index=" << report.first_bad_index
           << ", expected=" << report.first_expected << ", actual=" << report.first_actual << ")";
    }
    return os.str();
}

void PrintOrderedByRank(int rank_id, int world_size, const std::string &text)
{
    for (int turn = 0; turn < world_size; ++turn) {
        CommMpiBarrier();
        if (turn == rank_id) {
            std::cout << text << std::endl;
        }
    }
    CommMpiBarrier();
}

void PrintStatsLine(const char *prefix, const PerfStats &stats)
{
    std::cout << prefix << stats.avg << " us"
              << " min=" << stats.min << " us"
              << " max=" << stats.max << " us"
              << " std=" << stats.stddev << " us\n";
}

void PrintThroughputLine(const PerfThroughput &throughput)
{
    std::cout << "    input_tokens/s=" << throughput.input_tokens_per_s
              << " routed_tokens/s=" << throughput.routed_tokens_per_s << " eq_compute=" << throughput.tflops
              << " TFLOPS"
              << " eq_comm=" << throughput.gbs << " GB/s\n";
}

void PrintPerfSummary(const CaseConfig &cfg, int warmup_iters, int measure_iters,
                      const std::vector<double> &kernel_samples_us, const std::vector<double> &e2e_samples_us)
{
    if (kernel_samples_us.empty() || e2e_samples_us.empty()) {
        return;
    }
    const PerfStats kernel_stats = CalcStats(kernel_samples_us);
    const PerfStats e2e_stats = CalcStats(e2e_samples_us);
    const PerfThroughput kernel_throughput = CalcThroughput(cfg, kernel_stats.avg);
    const PerfThroughput e2e_throughput = CalcThroughput(cfg, e2e_stats.avg);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n===============================================================\n";
    std::cout << "[PROFILE] dispatch_combine_moe\n";
    std::cout << "  shape: m=" << cfg.m << " k=" << cfg.k << " n=" << cfg.n << " topk=" << cfg.topk
              << " expert_per_rank=" << cfg.expert_per_rank << " world_size=" << cfg.world_size << '\n';
    std::cout << "  iters: warmup=" << warmup_iters << " measure=" << measure_iters << '\n';
    std::cout << "  logical work(all ranks): input_tokens=" << cfg.input_tokens_all_ranks
              << " routed_tokens=" << cfg.routed_tokens_all_ranks
              << " remote_routed_tokens=" << cfg.remote_routed_tokens_all_ranks
              << " compute_flops=" << cfg.compute_flops_all_ranks << " comm_bytes=" << cfg.comm_bytes_all_ranks << '\n';
    PrintStatsLine("  kernel(max rank per iter): avg=", kernel_stats);
    PrintThroughputLine(kernel_throughput);
    PrintStatsLine("  e2e(max rank per iter):    avg=", e2e_stats);
    PrintThroughputLine(e2e_throughput);
    std::cout
        << "  note: equivalent compute/comm are derived from case.json logical workload, not hardware counters.\n";
    std::cout << "===============================================================\n" << std::endl;
}

HostInputData LoadHostInputData(const RankFileSet &files)
{
    HostInputData data;
    data.x = ReadBinaryFile(files.x);
    data.weight1 = ReadBinaryFile(files.weight1);
    data.weight2 = ReadBinaryFile(files.weight2);
    data.expert_idx = ReadBinaryFile(files.expert_idx);
    data.scale1 = ReadBinaryFile(files.scale1);
    data.scale2 = ReadBinaryFile(files.scale2);
    data.probs = ReadBinaryFile(files.probs);
    data.x_active_mask = ReadBinaryFile(files.x_active_mask);
    data.expected_out = BytesToU16(ReadBinaryFile(files.expected_out));
    return data;
}

DeviceBufferSet MakeDeviceBufferSet(const CaseConfig &cfg, const DispatchCombineMoeBuildResult &build,
                                    const HostInputData &host_data)
{
    return DeviceBufferSet{MakeDeviceBuffer(host_data.x.size(), host_data.x.data()),
                           MakeDeviceBuffer(host_data.weight1.size(), host_data.weight1.data()),
                           MakeDeviceBuffer(host_data.weight2.size(), host_data.weight2.data()),
                           MakeDeviceBuffer(host_data.expert_idx.size(), host_data.expert_idx.data()),
                           MakeDeviceBuffer(host_data.scale1.size(), host_data.scale1.data()),
                           MakeDeviceBuffer(host_data.scale2.size(), host_data.scale2.data()),
                           MakeDeviceBuffer(host_data.probs.size(), host_data.probs.data()),
                           MakeDeviceBuffer(host_data.x_active_mask.size(), host_data.x_active_mask.data()),
                           MakeDeviceBuffer(static_cast<size_t>(cfg.m) * cfg.k * sizeof(uint16_t)),
                           MakeDeviceBuffer(static_cast<size_t>(cfg.expert_per_rank) * sizeof(int32_t)),
                           MakeDeviceBuffer(build.workspace_bytes),
                           MakeDeviceBuffer(sizeof(build.tiling), &build.tiling)};
}

DeviceLaunchBuffers MakeLaunchBufferView(const DeviceBufferSet &buffers)
{
    return DeviceLaunchBuffers{
        buffers.x,         buffers.weight1, buffers.weight2,       buffers.expert_idx, buffers.scale1,
        buffers.scale2,    buffers.probs,   buffers.x_active_mask, buffers.out,        buffers.expert_token_nums,
        buffers.workspace, buffers.tiling};
}

DispatchCombineMoeLaunchArgs BuildLaunchArgs(const DispatchCombineMoeBuildResult &build,
                                             const DeviceLaunchBuffers &buffers)
{
    DispatchCombineMoeLaunchArgs args;
    args.block_dim = build.block_dim;
    args.tiling = buffers.tiling.ptr;
    args.workspace = buffers.workspace.ptr;
    args.x = buffers.x.ptr;
    args.weight1 = buffers.weight1.ptr;
    args.weight2 = buffers.weight2.ptr;
    args.expert_idx = buffers.expert_idx.ptr;
    args.scale1 = buffers.scale1.ptr;
    args.scale2 = buffers.scale2.ptr;
    args.probs = buffers.probs.ptr;
    args.x_active_mask = buffers.x_active_mask.ptr;
    args.out = buffers.out.ptr;
    args.expert_token_nums = buffers.expert_token_nums.ptr;
    return args;
}

void LaunchAndSync(const DispatchCombineMoeLaunchArgs &args, aclrtStream stream)
{
    launchDispatchCombineMoe(args, stream);
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        throw std::runtime_error("stream sync failed");
    }
}

void RunWarmupIters(const StandaloneRankRuntime &runtime, const DeviceLaunchBuffers &buffers,
                    const DispatchCombineMoeLaunchArgs &args, int warmup_iters)
{
    CommMpiBarrier();
    for (int iter = 0; iter < warmup_iters; ++iter) {
        PrepareIterationState(runtime, buffers.out, buffers.expert_token_nums, buffers.workspace);
        CommMpiBarrier();
        LaunchAndSync(args, runtime.compute_stream);
        CommMpiBarrier();
    }
}

void RunTimedIteration(const StandaloneRankRuntime &runtime, const DeviceLaunchBuffers &buffers,
                       const DispatchCombineMoeLaunchArgs &args, const EventHandle &kernel_start,
                       const EventHandle &kernel_end, TimingSamples &samples)
{
    PrepareIterationState(runtime, buffers.out, buffers.expert_token_nums, buffers.workspace);
    CommMpiBarrier();
    const auto host_start = std::chrono::high_resolution_clock::now();
    if (aclrtRecordEvent(kernel_start.event, runtime.compute_stream) != ACL_SUCCESS) {
        throw std::runtime_error("failed to record kernel start event");
    }
    launchDispatchCombineMoe(args, runtime.compute_stream);
    if (aclrtRecordEvent(kernel_end.event, runtime.compute_stream) != ACL_SUCCESS) {
        throw std::runtime_error("failed to record kernel end event");
    }
    if (aclrtSynchronizeStream(runtime.compute_stream) != ACL_SUCCESS) {
        throw std::runtime_error("stream sync failed");
    }
    CommMpiBarrier();
    const auto host_end = std::chrono::high_resolution_clock::now();

    float kernel_ms = 0.0f;
    if (aclrtEventElapsedTime(&kernel_ms, kernel_start.event, kernel_end.event) != ACL_SUCCESS) {
        throw std::runtime_error("failed to query kernel elapsed time");
    }
    samples.kernel_times_us.push_back(static_cast<double>(kernel_ms) * 1000.0);
    samples.e2e_times_us.push_back(std::chrono::duration<double, std::micro>(host_end - host_start).count());
}

TimingSamples RunMeasureIters(const StandaloneRankRuntime &runtime, const DeviceLaunchBuffers &buffers,
                              const DispatchCombineMoeLaunchArgs &args, const EventHandle &kernel_start,
                              const EventHandle &kernel_end, int measure_iters)
{
    TimingSamples samples;
    samples.kernel_times_us.reserve(static_cast<size_t>(measure_iters));
    samples.e2e_times_us.reserve(static_cast<size_t>(measure_iters));
    for (int iter = 0; iter < measure_iters; ++iter) {
        RunTimedIteration(runtime, buffers, args, kernel_start, kernel_end, samples);
    }
    return samples;
}

bool RunAccuracyCheck(int rank_id, int world_size, const std::string &case_dir, const CaseConfig &cfg,
                      const StandaloneRankRuntime &runtime, const DeviceLaunchBuffers &buffers,
                      const DispatchCombineMoeLaunchArgs &args, const std::vector<uint16_t> &expected_out)
{
    PrepareIterationState(runtime, buffers.out, buffers.expert_token_nums, buffers.workspace);
    CommMpiBarrier();
    LaunchAndSync(args, runtime.compute_stream);
    CommMpiBarrier();

    std::vector<uint16_t> actual_out(static_cast<size_t>(cfg.m) * cfg.k);
    if (aclrtMemcpy(actual_out.data(), actual_out.size() * sizeof(uint16_t), buffers.out.ptr,
                    actual_out.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host output copy failed");
    }

    WriteBinaryFile(case_dir + "/output_rank" + std::to_string(rank_id) + ".bin", actual_out.data(),
                    actual_out.size() * sizeof(uint16_t));
    const AccuracyReport report = CompareFp16File(expected_out, actual_out, cfg.compare_atol, cfg.compare_rtol);
    PrintOrderedByRank(rank_id, world_size,
                       BuildAccuracyReportText(rank_id, report, cfg.compare_atol, cfg.compare_rtol) + "\n" +
                           (report.pass ? "PASS" : "FAIL") + std::string(" rank=") + std::to_string(rank_id));
    return report.pass;
}

bool RunOneRank(int rank_id, int world_size, const std::string &case_dir, const HcclRootInfo &root_info)
{
    StandaloneRankRuntime runtime;
    if (!InitStandaloneRankRuntime(runtime, rank_id, world_size, root_info)) {
        return false;
    }

    bool ok = false;
    try {
        const int warmup_iters = ParseEnvInt("DISPATCH_COMBINE_MOE_WARMUP_ITERS", kDefaultWarmupIters);
        const int measure_iters = ParseEnvInt("DISPATCH_COMBINE_MOE_MEASURE_ITERS", kDefaultMeasureIters);
        if (warmup_iters < 0 || measure_iters < 0) {
            throw std::runtime_error("warmup/measure iters must be non-negative");
        }

        const CaseConfig cfg = LoadCaseConfig(case_dir + "/case.json");
        const RankFileSet files = BuildRankFileSet(case_dir, rank_id);
        const DispatchCombineMoeBuildResult build = BuildDispatchCombineMoeTiling(cfg, runtime);
        const HostInputData host_data = LoadHostInputData(files);

        DeviceBufferSet owned_buffers = MakeDeviceBufferSet(cfg, build, host_data);
        DeviceLaunchBuffers buffers = MakeLaunchBufferView(owned_buffers);

        EventHandle kernel_start;
        EventHandle kernel_end;
        if (measure_iters > 0) {
            if (aclrtCreateEvent(&kernel_start.event) != ACL_SUCCESS ||
                aclrtCreateEvent(&kernel_end.event) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create ACL events");
            }
        }

        const DispatchCombineMoeLaunchArgs args = BuildLaunchArgs(build, buffers);
        RunWarmupIters(runtime, buffers, args, warmup_iters);
        const TimingSamples samples = RunMeasureIters(runtime, buffers, args, kernel_start, kernel_end, measure_iters);

        const std::vector<double> kernel_max_samples =
            GatherMaxSamplesToRoot(samples.kernel_times_us, rank_id, world_size);
        const std::vector<double> e2e_max_samples = GatherMaxSamplesToRoot(samples.e2e_times_us, rank_id, world_size);
        if (rank_id == 0) {
            PrintPerfSummary(cfg, warmup_iters, measure_iters, kernel_max_samples, e2e_max_samples);
        }

        ok = RunAccuracyCheck(rank_id, world_size, case_dir, cfg, runtime, buffers, args, host_data.expected_out);
    } catch (const std::exception &ex) {
        std::cerr << "rank=" << rank_id << " error: " << ex.what() << std::endl;
        ok = false;
    }

    DestroyStandaloneRankRuntime(runtime);
    return ok;
}

} // namespace

int main(int argc, char **argv)
{
    if (!CommMpiInit(&argc, &argv)) {
        return 1;
    }

    const int rank_id = CommMpiRank();
    const int world_size = CommMpiSize();
    const char *case_dir_env = std::getenv("DISPATCH_COMBINE_MOE_CASE_DIR");
    const std::string case_dir = case_dir_env ? case_dir_env : "../out";

    if (aclInit(nullptr) != ACL_SUCCESS) {
        CommMpiFinalize();
        return 1;
    }
    if (rtSetDevice(rank_id) != 0) {
        aclFinalize();
        CommMpiFinalize();
        return 1;
    }
    if (aclrtSetDevice(rank_id) != ACL_SUCCESS) {
        aclFinalize();
        CommMpiFinalize();
        return 1;
    }

    HcclRootInfo root_info{};
    if (rank_id == 0 && HcclGetRootInfo(&root_info) != HCCL_SUCCESS) {
        aclrtResetDevice(rank_id);
        aclFinalize();
        CommMpiFinalize();
        return 1;
    }
    CommMpiBcast(&root_info, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0);
    CommMpiBarrier();

    const bool ok = RunOneRank(rank_id, world_size, case_dir, root_info);

    CommMpiBarrier();
    aclFinalize();
    CommMpiFinalize();
    return ok ? 0 : 1;
}
