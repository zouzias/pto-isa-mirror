#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl_types.h"

#include "comm_mpi.h"
#include "data_utils.hpp"
#include "kernel_launch.hpp"
#include "op_kernel/utils/const_args.hpp"
#include "runtime_context.hpp"
#include "tiling_builder.hpp"

extern "C" rtError_t rtSetDevice(int32_t device);
extern "C" rtError_t rtGetC2cCtrlAddr(uint64_t *addr, uint32_t *len);

namespace {

constexpr int kDefaultWarmupIters = 3;
constexpr int kDefaultMeasureIters = 5;
constexpr double kMicrosecondsPerSecond = 1000.0 * 1000.0;
constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;
static double g_sys_cnt_multiple = 20.0; // Default A2/A3, in ns per SYS_CNT tick.
constexpr uint32_t kHostCombineImplDirectAuto = 3U;

struct DeviceBuffer {
    void *ptr = nullptr;
    size_t bytes = 0;

    ~DeviceBuffer()
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
};

struct HostBuffer {
    void *ptr = nullptr;
    size_t bytes = 0;

    ~HostBuffer()
    {
        if (ptr != nullptr) {
            aclrtFreeHost(ptr);
        }
    }
};

struct PerfStats {
    double avg = 0.0;
    double min = 0.0;
    double max = 0.0;
    double stddev = 0.0;
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

HostBuffer MakeHostBuffer(size_t bytes)
{
    HostBuffer buffer;
    buffer.bytes = bytes;
    if (bytes == 0) {
        return buffer;
    }
    if (aclrtMallocHost(&buffer.ptr, bytes) != ACL_SUCCESS) {
        throw std::runtime_error("aclrtMallocHost failed");
    }
    return buffer;
}

std::vector<uint16_t> BytesToU16(const std::vector<uint8_t> &bytes)
{
    if (bytes.size() % sizeof(uint16_t) != 0) {
        throw std::runtime_error("fp16 file size is not aligned");
    }
    std::vector<uint16_t> out(bytes.size() / sizeof(uint16_t));
    std::memcpy(out.data(), bytes.data(), bytes.size());
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

bool TraceEnabled()
{
    return ParseEnvInt("DISPATCH_MEGA_COMBINE_TRACE", 0) != 0;
}

uint64_t AlignUpU64(uint64_t value, uint64_t align)
{
    return (value + align - 1U) / align * align;
}

uint64_t SwigluFullRowUbBytes(uint32_t n)
{
    auto align_ub = [](uint64_t value) { return AlignUpU64(value, 32U); };
    uint64_t ub_offset = 0;
    ub_offset += align_ub(static_cast<uint64_t>(n) * sizeof(float));  // activation
    ub_offset += align_ub(static_cast<uint64_t>(n) * sizeof(float));  // gate
    ub_offset += align_ub(static_cast<uint64_t>(n) * sizeof(float));  // temp / sigmoid
    ub_offset += align_ub(static_cast<uint64_t>(n) * sizeof(float));  // quant work
    ub_offset += align_ub(static_cast<uint64_t>(n) * sizeof(int8_t)); // int8 output
    ub_offset += 2U * 32U;                                            // max/scale scratch
    return ub_offset;
}

void Trace(int rank_id, const std::string &message)
{
    if (!TraceEnabled()) {
        return;
    }
    std::cerr << "[trace] rank=" << rank_id << " " << message << std::endl;
}

bool ZeroWindowMemory(const StandaloneRankRuntime &runtime)
{
    const uint64_t window_bytes = runtime.hccl.WindowBytes();
    void *window_ptr = runtime.hccl.WindowIn(static_cast<uint32_t>(runtime.hccl.rank_id));
    if (aclrtMemset(window_ptr, window_bytes, 0, window_bytes) != ACL_SUCCESS) {
        return false;
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
                           const DeviceBuffer &expert_token_nums_dev, const DeviceBuffer &workspace_dev,
                           const DeviceBuffer &profile_dev)
{
    if (!ZeroWindowMemory(runtime)) {
        throw std::runtime_error("failed to zero HCCL windows");
    }
    ZeroDeviceBuffer(out_dev, "out buffer");
    ZeroDeviceBuffer(expert_token_nums_dev, "expert_token_nums");
    ZeroDeviceBuffer(workspace_dev, "workspace");
    ZeroDeviceBuffer(profile_dev, "profile buffer");
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

double ToTokensPerSecond(double tokens, double us)
{
    return us > 0.0 ? tokens * kMicrosecondsPerSecond / us : 0.0;
}

double ToTflops(double flops, double us)
{
    return us > 0.0 ? flops * kMicrosecondsPerSecond / us / 1e12 : 0.0;
}

double ToGbs(double bytes, double us)
{
    return us > 0.0 ? bytes * kMicrosecondsPerSecond / us / kBytesPerGiB : 0.0;
}

double SysCntTicksToUs(uint64_t ticks)
{
    return static_cast<double>(ticks) * g_sys_cnt_multiple / 1000.0;
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

double ReadKernelProfileUs(const DeviceBuffer &profile_dev, HostBuffer &profile_host, uint32_t block_dim)
{
    if (profile_dev.bytes == 0 || profile_host.bytes == 0 || block_dim == 0) {
        return 0.0;
    }
    if (aclrtMemcpy(profile_host.ptr, profile_host.bytes, profile_dev.ptr, profile_dev.bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host profile copy failed");
    }

    uint64_t start_min = std::numeric_limits<uint64_t>::max();
    uint64_t end_max = 0;
    const auto *profile = static_cast<const uint8_t *>(profile_host.ptr);
    for (uint32_t block = 0; block < block_dim; ++block) {
        for (size_t profile_idx = 0; profile_idx < kMegaMoeProfileEntriesPerBlock; ++profile_idx) {
            const uint64_t *entry =
                reinterpret_cast<const uint64_t *>(profile + static_cast<size_t>(block) * kMegaMoeProfileBytesPerBlock +
                                                   profile_idx * kMegaMoeProfileEntryBytes);
            const uint64_t start = entry[kMegaMoeProfileKernelStart];
            const uint64_t end = entry[kMegaMoeProfileKernelEnd];
            if (start == 0 && end == 0) {
                continue;
            }
            start_min = std::min(start_min, start);
            end_max = std::max(end_max, end);
        }
    }
    if (start_min == std::numeric_limits<uint64_t>::max() || end_max < start_min) {
        return 0.0;
    }
    const uint64_t duration_ticks = end_max - start_min;
    return SysCntTicksToUs(duration_ticks);
}

std::string BuildAccuracyReportText(int rank_id, const AccuracyReport &report)
{
    std::ostringstream os;
    os << std::setprecision(6) << "rank=" << rank_id << " max_diff=" << report.max_abs_err
       << " max_ratio=" << report.max_rel_err << " err=" << report.mismatch_count << "/" << report.err_threshold
       << " -> " << (report.pass ? "PASS" : "FAIL");
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

void PrintPerfSummary(const CaseConfig &cfg, uint32_t launch_block_dim, int warmup_iters, int measure_iters,
                      const std::vector<double> &kernel_samples_us)
{
    if (kernel_samples_us.empty()) {
        return;
    }
    const PerfStats kernel_stats = CalcStats(kernel_samples_us);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n===============================================================\n";
    std::cout << "[PROFILE] dispatch_mega_combine\n";
    std::cout << "  shape: m=" << cfg.m << " k=" << cfg.k << " n=" << cfg.n << " topk=" << cfg.topk
              << " expert_per_rank=" << cfg.expert_per_rank << " world_size=" << cfg.world_size << '\n';
    std::cout << "  schedule: aic_num=" << cfg.aic_num << " aiv_num=" << cfg.aiv_num
              << " launch_block_dim=" << launch_block_dim << " front_reorder_aiv_num=" << cfg.aiv_num << '\n';
    std::cout << "  iters: warmup=" << warmup_iters << " measure=" << measure_iters << '\n';
    std::cout << "  logical work(all ranks): input_tokens=" << cfg.input_tokens_all_ranks
              << " routed_tokens=" << cfg.routed_tokens_all_ranks
              << " remote_routed_tokens=" << cfg.remote_routed_tokens_all_ranks
              << " compute_flops=" << cfg.compute_flops_all_ranks << " comm_bytes=" << cfg.comm_bytes_all_ranks << '\n';
    std::cout << "  kernel(syscnt max rank per iter): avg=" << kernel_stats.avg << " us"
              << " min=" << kernel_stats.min << " us"
              << " max=" << kernel_stats.max << " us"
              << " std=" << kernel_stats.stddev << " us\n";
    std::cout << "    input_tokens/s=" << ToTokensPerSecond(cfg.input_tokens_all_ranks, kernel_stats.avg)
              << " routed_tokens/s=" << ToTokensPerSecond(cfg.routed_tokens_all_ranks, kernel_stats.avg)
              << " eq_compute=" << ToTflops(cfg.compute_flops_all_ranks, kernel_stats.avg) << " TFLOPS"
              << " eq_comm=" << ToGbs(cfg.comm_bytes_all_ranks, kernel_stats.avg) << " GB/s\n";
    std::cout
        << "  note: equivalent compute/comm are derived from case.json logical workload, not hardware counters.\n";
    std::cout << "===============================================================\n" << std::endl;
}

int32_t LoadI32(const std::vector<uint8_t> &bytes, size_t index)
{
    int32_t value = 0;
    const size_t byte_offset = index * sizeof(int32_t);
    if (byte_offset + sizeof(int32_t) <= bytes.size()) {
        std::memcpy(&value, bytes.data() + byte_offset, sizeof(int32_t));
    }
    return value;
}

void PrintWorkloadAuditIfEnabled(const CaseConfig &cfg, int rank_id, int world_size,
                                 const std::vector<uint8_t> &expert_idx, const std::vector<uint8_t> &x_active_mask,
                                 size_t actual_output_elems, bool skip_accuracy)
{
    const size_t global_expert_num = static_cast<size_t>(cfg.world_size) * cfg.expert_per_rank;
    const size_t fixed_fields = 6;
    const size_t per_rank_offset = fixed_fields;
    const size_t per_expert_offset = per_rank_offset + cfg.world_size;
    const size_t fields = per_expert_offset + global_expert_num;
    std::vector<uint64_t> local(fields, 0);

    uint64_t active_tokens = 0;
    uint64_t valid_routes = 0;
    uint64_t remote_routes = 0;
    uint64_t invalid_expert_routes = 0;
    const size_t expert_idx_count = expert_idx.size() / sizeof(int32_t);
    for (uint32_t token = 0; token < cfg.m; ++token) {
        const bool active = token < x_active_mask.size() && x_active_mask[token] != 0;
        if (!active) {
            continue;
        }
        ++active_tokens;
        for (uint32_t topk = 0; topk < cfg.topk; ++topk) {
            const size_t slot = static_cast<size_t>(token) * cfg.topk + topk;
            if (slot >= expert_idx_count) {
                ++invalid_expert_routes;
                continue;
            }
            const int32_t expert = LoadI32(expert_idx, slot);
            if (expert < 0 || static_cast<size_t>(expert) >= global_expert_num) {
                ++invalid_expert_routes;
                continue;
            }
            const uint32_t dst_rank = static_cast<uint32_t>(expert) / cfg.expert_per_rank;
            ++valid_routes;
            if (static_cast<int>(dst_rank) != rank_id) {
                ++remote_routes;
            }
            ++local[per_rank_offset + dst_rank];
            ++local[per_expert_offset + static_cast<size_t>(expert)];
        }
    }

    local[0] = active_tokens;
    local[1] = active_tokens * cfg.topk;
    local[2] = valid_routes;
    local[3] = remote_routes;
    local[4] = invalid_expert_routes;
    local[5] = actual_output_elems;

    const int local_bytes = static_cast<int>(local.size() * sizeof(uint64_t));
    std::vector<uint64_t> gathered(rank_id == 0 ? fields * static_cast<size_t>(world_size) : 0, 0);
    CommMpiGather(local.data(), local_bytes, COMM_MPI_CHAR, rank_id == 0 ? gathered.data() : nullptr, local_bytes,
                  COMM_MPI_CHAR, 0);

    if (rank_id != 0) {
        return;
    }

    std::vector<uint64_t> total(fields, 0);
    for (int rank = 0; rank < world_size; ++rank) {
        const uint64_t *rank_fields = gathered.data() + static_cast<size_t>(rank) * fields;
        for (size_t idx = 0; idx < fields; ++idx) {
            total[idx] += rank_fields[idx];
        }
    }

    const auto dest_begin = total.begin() + static_cast<std::ptrdiff_t>(per_rank_offset);
    const auto dest_end = dest_begin + cfg.world_size;
    const auto expert_begin = total.begin() + static_cast<std::ptrdiff_t>(per_expert_offset);
    const auto expert_end = expert_begin + static_cast<std::ptrdiff_t>(global_expert_num);
    const uint64_t min_dest_rows = dest_begin == dest_end ? 0 : *std::min_element(dest_begin, dest_end);
    const uint64_t max_dest_rows = dest_begin == dest_end ? 0 : *std::max_element(dest_begin, dest_end);
    const uint64_t min_expert_rows = expert_begin == expert_end ? 0 : *std::min_element(expert_begin, expert_end);
    const uint64_t max_expert_rows = expert_begin == expert_end ? 0 : *std::max_element(expert_begin, expert_end);
    const size_t nonzero_experts =
        static_cast<size_t>(std::count_if(expert_begin, expert_end, [](uint64_t rows) { return rows != 0; }));
    const uint64_t expected_output_elems =
        static_cast<uint64_t>(cfg.world_size) * static_cast<uint64_t>(cfg.m) * static_cast<uint64_t>(cfg.k);
    const bool routes_match_case = static_cast<double>(total[0]) == cfg.input_tokens_all_ranks &&
                                   static_cast<double>(total[2]) == cfg.routed_tokens_all_ranks &&
                                   static_cast<double>(total[3]) == cfg.remote_routed_tokens_all_ranks;
    const bool output_shape_match = total[5] == expected_output_elems;

    std::cout << "[WORKLOAD_AUDIT] dispatch_mega_combine "
              << (routes_match_case && output_shape_match && total[4] == 0 ? "PASS" : "CHECK")
              << " accuracy=" << (skip_accuracy ? "SKIP" : "FULL") << '\n';
    std::cout << "  source active_tokens=" << total[0] << " topk_slots=" << total[1] << " valid_routes=" << total[2]
              << " remote_routes=" << total[3] << " invalid_expert_routes=" << total[4] << '\n';
    std::cout << "  case_json input_tokens=" << static_cast<uint64_t>(cfg.input_tokens_all_ranks)
              << " routed_tokens=" << static_cast<uint64_t>(cfg.routed_tokens_all_ranks)
              << " remote_routed_tokens=" << static_cast<uint64_t>(cfg.remote_routed_tokens_all_ranks) << '\n';
    std::cout << "  dest_rows total=" << total[2] << " per_rank_min=" << min_dest_rows
              << " per_rank_max=" << max_dest_rows << " max_output_size=" << cfg.max_output_size << '\n';
    std::cout << "  expert_rows nonzero=" << nonzero_experts << "/" << global_expert_num << " min=" << min_expert_rows
              << " max=" << max_expert_rows << '\n';
    std::cout << "  output_elements total=" << total[5] << " expected=" << expected_output_elems << std::endl;
}

void ValidateFullPathConstraints(const CaseConfig &cfg)
{
    if (cfg.expert_per_rank + 1U > MEGA_MOE_D2C_MAX_LOGICAL_GROUP_EVENTS) {
        throw std::runtime_error(
            "D2C hard flag budget exceeded: expert_per_rank=" + std::to_string(cfg.expert_per_rank) +
            " max=" + std::to_string(MEGA_MOE_D2C_MAX_LOGICAL_GROUP_EVENTS));
    }
    if (cfg.expert_per_rank > MEGA_MOE_GMM2_TO_COMBINE_MAX_LOGICAL_GROUP_EVENTS) {
        throw std::runtime_error(
            "GMM2->combine hard flag budget exceeded: expert_per_rank=" + std::to_string(cfg.expert_per_rank) +
            " max=" + std::to_string(MEGA_MOE_GMM2_TO_COMBINE_MAX_LOGICAL_GROUP_EVENTS));
    }
    if (cfg.k % 128U != 0U) {
        throw std::runtime_error("GMM1 requires K % 128 == 0");
    }
    if (cfg.n % 32U != 0U) {
        throw std::runtime_error("GMM1 requires N % 32 == 0");
    }
    if (cfg.n % 64U != 0U) {
        throw std::runtime_error("SwiGLU requires N % 64 == 0");
    }
    if (SwigluFullRowUbBytes(cfg.n) > AtlasA2::UB_SIZE) {
        throw std::runtime_error(
            "SwiGLU full-row UB capacity exceeded: ub_bytes=" + std::to_string(SwigluFullRowUbBytes(cfg.n)) +
            " max=" + std::to_string(AtlasA2::UB_SIZE));
    }
    if ((cfg.k * sizeof(uint16_t)) % 32U != 0U) {
        throw std::runtime_error("combine/unpermute requires K * sizeof(float16) to be 32-byte aligned");
    }
    if (cfg.max_output_size < cfg.m * cfg.topk) {
        throw std::runtime_error("unpermute requires max_output_size >= M * topK for large dropless path");
    }
}

bool RunOneRank(int rank_id, int world_size, const std::string &case_dir, const HcclRootInfo &root_info)
{
    StandaloneRankRuntime runtime;
    if (!InitStandaloneRankRuntime(runtime, rank_id, world_size, root_info)) {
        return false;
    }
    Trace(rank_id, "runtime initialized");

    bool ok = false;
    try {
        const int warmup_iters = ParseEnvInt("DISPATCH_MEGA_COMBINE_WARMUP_ITERS", kDefaultWarmupIters);
        const int measure_iters = ParseEnvInt("DISPATCH_MEGA_COMBINE_MEASURE_ITERS", kDefaultMeasureIters);
        const bool skip_accuracy = ParseEnvInt("DISPATCH_MEGA_COMBINE_SKIP_ACCURACY", 0) != 0;
        const bool start_sync_debug = ParseEnvInt("DISPATCH_MEGA_COMBINE_START_SYNC_DEBUG", 0) != 0;
        const bool workload_audit = ParseEnvInt("DISPATCH_MEGA_COMBINE_WORKLOAD_AUDIT", 0) != 0;
        if (warmup_iters < 0 || measure_iters < 0) {
            throw std::runtime_error("warmup/measure iters must be non-negative");
        }

        const CaseConfig cfg = LoadCaseConfig(case_dir + "/case.json");
        const RankFileSet files = BuildRankFileSet(case_dir, rank_id);
        MegaMoeBuildResult build = BuildMegaMoeTiling(cfg, runtime);
        const auto &front = build.tiling.frontReorderTiling;
        const bool front_capable = FrontCaseIsSupported(front.frontCase);
        if (!front_capable) {
            throw std::runtime_error("front unsupported case has no legacy fallback");
        }
        auto &combine_tiling = build.tiling.combineTiling;
        combine_tiling.combineImplMode = kHostCombineImplDirectAuto;
        ValidateFullPathConstraints(cfg);
        Trace(rank_id, "tiling built frontCase=" + std::to_string(front.frontCase) +
                           " block_dim=" + std::to_string(build.block_dim));

        const std::vector<uint8_t> x = ReadBinaryFile(files.x);
        const std::vector<uint8_t> weight1 = ReadBinaryFile(files.weight1);
        const std::vector<uint8_t> weight2 = ReadBinaryFile(files.weight2);
        const std::vector<uint8_t> expert_idx = ReadBinaryFile(files.expert_idx);
        const std::vector<uint8_t> scale1 = ReadBinaryFile(files.scale1);
        const std::vector<uint8_t> scale2 = ReadBinaryFile(files.scale2);
        const std::vector<uint8_t> probs = ReadBinaryFile(files.probs);
        const std::vector<uint8_t> x_active_mask = ReadBinaryFile(files.x_active_mask);
        std::vector<uint16_t> expected_out;
        if (!skip_accuracy) {
            const std::vector<uint8_t> expected_out_bytes = ReadBinaryFile(files.expected_out);
            expected_out = BytesToU16(expected_out_bytes);
        }

        DeviceBuffer x_dev = MakeDeviceBuffer(x.size(), x.data());
        DeviceBuffer weight1_dev = MakeDeviceBuffer(weight1.size(), weight1.data());
        DeviceBuffer weight2_dev = MakeDeviceBuffer(weight2.size(), weight2.data());
        DeviceBuffer expert_idx_dev = MakeDeviceBuffer(expert_idx.size(), expert_idx.data());
        DeviceBuffer scale1_dev = MakeDeviceBuffer(scale1.size(), scale1.data());
        DeviceBuffer scale2_dev = MakeDeviceBuffer(scale2.size(), scale2.data());
        DeviceBuffer probs_dev = MakeDeviceBuffer(probs.size(), probs.data());
        DeviceBuffer out_dev = MakeDeviceBuffer(static_cast<size_t>(cfg.m) * cfg.k * sizeof(uint16_t));
        DeviceBuffer expert_token_nums_dev =
            MakeDeviceBuffer(static_cast<size_t>(cfg.expert_per_rank) * sizeof(int32_t));
        DeviceBuffer workspace_dev = MakeDeviceBuffer(build.workspace_bytes);
        DeviceBuffer tiling_dev = MakeDeviceBuffer(sizeof(build.tiling), &build.tiling);
        const size_t profile_bytes = static_cast<size_t>(build.block_dim) * kMegaMoeProfileBytesPerBlock;
        DeviceBuffer profile_dev = MakeDeviceBuffer(profile_bytes);
        HostBuffer profile_host = MakeHostBuffer(profile_bytes);
        Trace(rank_id, "buffers allocated");

        MegaMoeLaunchArgs args;
        uint64_t ffts_addr = 0;
        uint32_t ffts_len = 0;
        if (rtGetC2cCtrlAddr(&ffts_addr, &ffts_len) != 0) {
            throw std::runtime_error("rtGetC2cCtrlAddr failed");
        }

        args.ffts = reinterpret_cast<void *>(ffts_addr);
        args.block_dim = build.block_dim;
        args.tiling = tiling_dev.ptr;
        args.workspace = workspace_dev.ptr;
        args.x = x_dev.ptr;
        args.weight1 = weight1_dev.ptr;
        args.weight2 = weight2_dev.ptr;
        args.expert_idx = expert_idx_dev.ptr;
        args.scale1 = scale1_dev.ptr;
        args.scale2 = scale2_dev.ptr;
        args.probs = probs_dev.ptr;
        args.out = out_dev.ptr;
        args.expert_token_nums = expert_token_nums_dev.ptr;
        args.profile_data = profile_dev.ptr;
        args.start_sync_debug = start_sync_debug ? 1U : 0U;

        auto launch_once = [&]() {
            Trace(rank_id, "launch begin");
            launchMegaMoe(args, runtime.compute_stream);
            Trace(rank_id, "launch submitted");
            if (aclrtSynchronizeStream(runtime.compute_stream) != ACL_SUCCESS) {
                throw std::runtime_error("stream sync failed");
            }
            Trace(rank_id, "launch synced");
        };

        std::vector<double> kernel_times_us;
        kernel_times_us.reserve(static_cast<size_t>(measure_iters));

        CommMpiBarrier();
        for (int iter = 0; iter < warmup_iters; ++iter) {
            PrepareIterationState(runtime, out_dev, expert_token_nums_dev, workspace_dev, profile_dev);
            CommMpiBarrier();
            launch_once();
            CommMpiBarrier();
        }

        for (int iter = 0; iter < measure_iters; ++iter) {
            PrepareIterationState(runtime, out_dev, expert_token_nums_dev, workspace_dev, profile_dev);
            CommMpiBarrier();
            Trace(rank_id, "measure launch begin");
            launchMegaMoe(args, runtime.compute_stream);
            Trace(rank_id, "measure launch submitted");
            if (aclrtSynchronizeStream(runtime.compute_stream) != ACL_SUCCESS) {
                throw std::runtime_error("stream sync failed");
            }
            Trace(rank_id, "measure launch synced");
            CommMpiBarrier();

            kernel_times_us.push_back(ReadKernelProfileUs(profile_dev, profile_host, build.block_dim));
        }

        const std::vector<double> kernel_max_samples = GatherMaxSamplesToRoot(kernel_times_us, rank_id, world_size);
        if (rank_id == 0) {
            PrintPerfSummary(cfg, build.block_dim, warmup_iters, measure_iters, kernel_max_samples);
        }

        PrepareIterationState(runtime, out_dev, expert_token_nums_dev, workspace_dev, profile_dev);
        CommMpiBarrier();
        launch_once();
        CommMpiBarrier();

        std::vector<uint16_t> actual_out(static_cast<size_t>(cfg.m) * cfg.k);
        if (aclrtMemcpy(actual_out.data(), actual_out.size() * sizeof(uint16_t), out_dev.ptr,
                        actual_out.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            throw std::runtime_error("device->host output copy failed");
        }
        if (workload_audit) {
            PrintWorkloadAuditIfEnabled(cfg, rank_id, world_size, expert_idx, x_active_mask, actual_out.size(),
                                        skip_accuracy);
        }

        WriteBinaryFile(case_dir + "/output_rank" + std::to_string(rank_id) + ".bin", actual_out.data(),
                        actual_out.size() * sizeof(uint16_t));
        if (skip_accuracy) {
            ok = true;
            PrintOrderedByRank(rank_id, world_size,
                               "rank=" + std::to_string(rank_id) + " accuracy=SKIP\n" + (ok ? "PASS" : "FAIL") +
                                   std::string(" rank=") + std::to_string(rank_id));
        } else {
            const AccuracyReport report = CompareFp16File(expected_out, actual_out, cfg.compare_atol, cfg.compare_rtol);
            ok = report.pass;
            PrintOrderedByRank(rank_id, world_size,
                               BuildAccuracyReportText(rank_id, report) + "\n" + (ok ? "PASS" : "FAIL") +
                                   std::string(" rank=") + std::to_string(rank_id));
        }
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
    const char *case_dir_env = std::getenv("DISPATCH_MEGA_COMBINE_CASE_DIR");
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
