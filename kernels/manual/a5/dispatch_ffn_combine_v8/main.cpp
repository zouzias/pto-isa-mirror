#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
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
#include "debug_checks.hpp"
#include "kernel_launch.hpp"
#include "op_kernel/utils/const_args.hpp"
#include "runtime_context.hpp"
#include "tiling_builder.hpp"

extern "C" rtError_t rtSetDevice(int32_t device);
extern "C" rtError_t rtGetC2cCtrlAddr(uint64_t *addr, uint32_t *len);

namespace {

constexpr int kDefaultWarmupIters = 3;
constexpr int kDefaultMeasureIters = 5;
constexpr int kDefaultAicNum = 28;
constexpr int kDefaultAivNum = 56;
constexpr uint64_t kHostMiB = 1024U * 1024U;
constexpr double kMicrosecondsPerSecond = 1000.0 * 1000.0;
constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;
static double g_sys_cnt_multiple = 1.0; // A5 default, in ns per SYS_CNT tick.
constexpr uint32_t kHostCombineImplDirectAuto = 3U;
constexpr uint32_t kHostGmm2CombineCvModeDirect = 1U;
constexpr uint32_t kHostCombineSmallTokenThreshold = 4096U;

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

std::vector<int32_t> BytesToI32(const std::vector<uint8_t> &bytes)
{
    if (bytes.size() % sizeof(int32_t) != 0) {
        throw std::runtime_error("int32 file size is not aligned");
    }
    std::vector<int32_t> out(bytes.size() / sizeof(int32_t));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

std::vector<float> BytesToF32(const std::vector<uint8_t> &bytes)
{
    if (bytes.size() % sizeof(float) != 0) {
        throw std::runtime_error("float file size is not aligned");
    }
    std::vector<float> out(bytes.size() / sizeof(float));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

float Bf16ToFloat(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

uint16_t FloatToBf16RoundToNearestEven(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16U) & 1U;
    const uint32_t rounding_bias = 0x7FFFU + lsb;
    return static_cast<uint16_t>((bits + rounding_bias) >> 16U);
}

uint16_t FloatToHalfTrunc(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16U) & 0x8000U);
    const uint32_t exponent = (bits >> 23U) & 0xFFU;
    const uint32_t mantissa = bits & 0x7FFFFFU;
    if (exponent == 0U) {
        return sign;
    }
    if (exponent == 0xFFU) {
        return static_cast<uint16_t>(sign | (mantissa == 0U ? 0x7C00U : 0x7E00U));
    }

    const int32_t half_exp = static_cast<int32_t>(exponent) - 127 + 15;
    if (half_exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    if (half_exp <= 0) {
        if (half_exp < -10) {
            return sign;
        }
        const uint32_t mantissa_with_hidden = mantissa | 0x800000U;
        const uint32_t shift = static_cast<uint32_t>(14 - half_exp);
        return static_cast<uint16_t>(sign | (mantissa_with_hidden >> shift));
    }
    return static_cast<uint16_t>(sign | (static_cast<uint16_t>(half_exp) << 10U) |
                                 static_cast<uint16_t>(mantissa >> 13U));
}

uint32_t RoundRightShiftToEven(uint32_t value, uint32_t shift)
{
    if (shift == 0U) {
        return value;
    }
    if (shift >= 32U) {
        return 0;
    }
    const uint32_t truncated = value >> shift;
    const uint32_t remainder_mask = (1U << shift) - 1U;
    const uint32_t remainder = value & remainder_mask;
    const uint32_t halfway = 1U << (shift - 1U);
    if (remainder > halfway || (remainder == halfway && (truncated & 1U) != 0U)) {
        return truncated + 1U;
    }
    return truncated;
}

uint16_t FloatToHalfRoundToNearestEven(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16U) & 0x8000U);
    const uint32_t exponent = (bits >> 23U) & 0xFFU;
    const uint32_t mantissa = bits & 0x7FFFFFU;
    if (exponent == 0U) {
        return sign;
    }
    if (exponent == 0xFFU) {
        return static_cast<uint16_t>(sign | (mantissa == 0U ? 0x7C00U : 0x7E00U));
    }

    int32_t half_exp = static_cast<int32_t>(exponent) - 127 + 15;
    if (half_exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    if (half_exp <= 0) {
        const uint32_t mantissa_with_hidden = mantissa | 0x800000U;
        const uint32_t shift = static_cast<uint32_t>(14 - half_exp);
        const uint32_t rounded = RoundRightShiftToEven(mantissa_with_hidden, shift);
        return static_cast<uint16_t>(sign | static_cast<uint16_t>(rounded));
    }

    uint32_t rounded_mantissa = RoundRightShiftToEven(mantissa, 13U);
    if (rounded_mantissa == 0x400U) {
        rounded_mantissa = 0;
        ++half_exp;
        if (half_exp >= 31) {
            return static_cast<uint16_t>(sign | 0x7C00U);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint16_t>(half_exp) << 10U) |
                                 static_cast<uint16_t>(rounded_mantissa));
}

int32_t HalfOrderedValue(uint16_t value)
{
    if ((value & 0x8000U) != 0U) {
        return static_cast<int32_t>(0x8000U - static_cast<uint32_t>(value & 0x7FFFU));
    }
    return static_cast<int32_t>(0x8000U + static_cast<uint32_t>(value));
}

uint32_t HalfUlpDistance(uint16_t expected, uint16_t actual)
{
    return static_cast<uint32_t>(std::abs(HalfOrderedValue(expected) - HalfOrderedValue(actual)));
}

int8_t RoundHalfAwayToInt8(float value)
{
    float rounded = value >= 0.0f ? std::floor(value + 0.5f) : std::ceil(value - 0.5f);
    rounded = std::max(-128.0f, std::min(127.0f, rounded));
    return static_cast<int8_t>(rounded);
}

int8_t RoundHalfUpToInt8(float value)
{
    const float clamped = std::max(-128.0f, std::min(127.0f, value));
    return static_cast<int8_t>(std::floor(clamped + 0.5f));
}

std::vector<int8_t> QuantizeBf16RowLikeV3(const std::vector<uint16_t> &x, uint32_t token, uint32_t k, float &scale)
{
    std::vector<int8_t> quant(k, 0);
    float max_abs = 0.0f;
    const size_t row_offset = static_cast<size_t>(token) * k;
    for (uint32_t col = 0; col < k; ++col) {
        max_abs = std::max(max_abs, std::fabs(Bf16ToFloat(x[row_offset + col])));
    }

    scale = max_abs / 127.0f;
    if (scale == 0.0f) {
        scale = 1.0e-6f / 127.0f;
    }
    for (uint32_t col = 0; col < k; ++col) {
        const float divided = Bf16ToFloat(x[row_offset + col]) / scale;
        quant[col] = RoundHalfAwayToInt8(Fp16ToFloat(FloatToHalfTrunc(divided)));
    }
    return quant;
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

double ParseEnvDouble(const char *name, double default_value)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    try {
        return std::stod(value);
    } catch (const std::exception &) {
        throw std::runtime_error(std::string("invalid double in env: ") + name);
    }
}

bool TraceEnabled()
{
    return ParseEnvInt("DISPATCH_FFN_COMBINE_V8_TRACE", 0) != 0;
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
    const uint64_t window_bytes = runtime.hccl.WindowClearBytes();
    void *window_ptr = runtime.hccl.WindowClearBase(static_cast<uint32_t>(runtime.hccl.rank_id));
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

std::string FormatSystemTimeUs(std::chrono::system_clock::time_point tp)
{
    const auto epoch_us = std::chrono::duration_cast<std::chrono::microseconds>(tp.time_since_epoch()).count();
    const std::time_t time = std::chrono::system_clock::to_time_t(tp);
    std::tm local_time{};
    localtime_r(&time, &local_time);

    char buffer[32] = {};
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &local_time);
    std::ostringstream os;
    os << buffer << '.' << std::setw(6) << std::setfill('0') << (epoch_us % 1000000LL) << std::setfill(' ');
    return os.str();
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

std::vector<double> GatherAllSamplesToRoot(const std::vector<double> &local_samples, int rank_id, int world_size)
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
    return rank_id == 0 ? gathered : std::vector<double>{};
}

uint64_t ReadKernelProfileTicks(const DeviceBuffer &profile_dev, HostBuffer &profile_host, uint32_t block_dim)
{
    if (profile_dev.bytes == 0 || profile_host.bytes == 0 || block_dim == 0) {
        return 0U;
    }
    if (aclrtMemcpy(profile_host.ptr, profile_host.bytes, profile_dev.ptr, profile_dev.bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host profile copy failed");
    }

    uint64_t start_min = std::numeric_limits<uint64_t>::max();
    uint64_t end_max = 0;
    const auto *profile = static_cast<const uint8_t *>(profile_host.ptr);
    for (uint32_t block = 0; block < block_dim; ++block) {
        for (size_t profile_idx = 0; profile_idx < kDispatchFFNCombineProfileEntriesPerBlock; ++profile_idx) {
            const uint64_t *entry = reinterpret_cast<const uint64_t *>(
                profile + static_cast<size_t>(block) * kDispatchFFNCombineProfileBytesPerBlock +
                profile_idx * kDispatchFFNCombineProfileEntryBytes);
            const uint64_t start = entry[kDispatchFFNCombineProfileKernelStart];
            const uint64_t end = entry[kDispatchFFNCombineProfileKernelEnd];
            if (start == 0 && end == 0) {
                continue;
            }
            start_min = std::min(start_min, start);
            end_max = std::max(end_max, end);
        }
    }
    if (start_min == std::numeric_limits<uint64_t>::max() || end_max < start_min) {
        return 0U;
    }
    return end_max - start_min;
}

std::string BuildAccuracyReportText(int rank_id, const AccuracyReport &report, double atol, double rtol)
{
    (void)atol;
    (void)rtol;
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
                      const std::vector<double> &kernel_ticks_samples, const std::vector<double> &host_wall_samples_us,
                      const std::vector<double> &inferred_ns_per_tick_samples)
{
    if (kernel_ticks_samples.empty()) {
        return;
    }
    std::vector<double> kernel_samples_us;
    kernel_samples_us.reserve(kernel_ticks_samples.size());
    for (double ticks : kernel_ticks_samples) {
        kernel_samples_us.push_back(SysCntTicksToUs(static_cast<uint64_t>(ticks)));
    }
    const PerfStats kernel_stats = CalcStats(kernel_samples_us);
    const PerfStats kernel_ticks_stats = CalcStats(kernel_ticks_samples);
    const PerfStats host_wall_stats = CalcStats(host_wall_samples_us);
    const PerfStats inferred_ns_per_tick_stats = CalcStats(inferred_ns_per_tick_samples);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n===============================================================\n";
    std::cout << "[PROFILE] dispatch_ffn_combine_v8\n";
    std::cout << "  shape: m=" << cfg.m << " k=" << cfg.k << " n=" << cfg.n << " topk=" << cfg.topk
              << " expert_per_rank=" << cfg.expert_per_rank << " world_size=" << cfg.world_size << '\n';
    std::cout << "  schedule: aic_num=" << cfg.aic_num << " aiv_num=" << cfg.aiv_num
              << " launch_block_dim=" << launch_block_dim << " front_reorder_aiv_num=" << cfg.aiv_num << '\n';
    std::cout << "  iters: warmup=" << warmup_iters << " measure=" << measure_iters << '\n';
    std::cout << "  logical work(all ranks): input_tokens=" << cfg.input_tokens_all_ranks
              << " routed_tokens=" << cfg.routed_tokens_all_ranks
              << " remote_routed_tokens=" << cfg.remote_routed_tokens_all_ranks
              << " compute_flops=" << cfg.compute_flops_all_ranks << " comm_bytes=" << cfg.comm_bytes_all_ranks << '\n';
    if (!host_wall_samples_us.empty()) {
        std::cout << "  host wall launch+sync(max rank per iter): avg=" << host_wall_stats.avg << " us"
                  << " min=" << host_wall_stats.min << " us"
                  << " max=" << host_wall_stats.max << " us"
                  << " std=" << host_wall_stats.stddev << " us\n";
    }
    std::cout << "  kernel raw syscnt ticks(max rank per iter): avg=" << kernel_ticks_stats.avg
              << " min=" << kernel_ticks_stats.min << " max=" << kernel_ticks_stats.max
              << " std=" << kernel_ticks_stats.stddev << '\n';
    if (!inferred_ns_per_tick_samples.empty()) {
        std::cout << "  syscnt inferred from host wall: avg=" << inferred_ns_per_tick_stats.avg << " ns/tick"
                  << " min=" << inferred_ns_per_tick_stats.min << " ns/tick"
                  << " max=" << inferred_ns_per_tick_stats.max << " ns/tick"
                  << " current_convert=" << g_sys_cnt_multiple << " ns/tick\n";
    }
    std::cout << "  kernel(syscnt max rank per iter, convert=" << g_sys_cnt_multiple
              << " ns/tick): avg=" << kernel_stats.avg << " us"
              << " min=" << kernel_stats.min << " us"
              << " max=" << kernel_stats.max << " us"
              << " std=" << kernel_stats.stddev << " us\n";
    std::cout << "    input_tokens/s=" << ToTokensPerSecond(cfg.input_tokens_all_ranks, kernel_stats.avg)
              << " routed_tokens/s=" << ToTokensPerSecond(cfg.routed_tokens_all_ranks, kernel_stats.avg)
              << " eq_compute=" << ToTflops(cfg.compute_flops_all_ranks, kernel_stats.avg) << " TFLOPS"
              << " eq_comm=" << ToGbs(cfg.comm_bytes_all_ranks, kernel_stats.avg) << " GB/s\n";
    std::cout
        << "  note: equivalent compute/comm are derived from case.json logical workload, not hardware counters.\n";
    std::cout << "  note: host wall includes launch and stream synchronization overhead; use inferred ns/tick only as a"
                 " SYS_CNT conversion sanity check.\n";
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

    std::cout << "[WORKLOAD_AUDIT] dispatch_ffn_combine_v8 "
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

void ValidateStageHardFlagBudget(const CaseConfig &cfg, uint32_t stageNum)
{
    if (stageNum >= 10U && cfg.expert_per_rank > V5_DISPATCH_V2C_MAX_LOGICAL_GROUP_EVENTS) {
        throw std::runtime_error(
            "stage10 dispatch V2C hard flag budget exceeded: expert_per_rank=" + std::to_string(cfg.expert_per_rank) +
            " max=" + std::to_string(V5_DISPATCH_V2C_MAX_LOGICAL_GROUP_EVENTS));
    }
    if (stageNum >= 10U && cfg.k % 128U != 0U) {
        throw std::runtime_error("stage10 GMM1 requires K % 128 == 0");
    }
    if (stageNum >= 10U && cfg.n % 32U != 0U) {
        throw std::runtime_error("stage10 GMM1 requires N % 32 == 0");
    }
    if (stageNum >= 11U && cfg.n % 64U != 0U) {
        throw std::runtime_error("stage11 SwiGLU requires N % 64 == 0");
    }
    if (stageNum >= 11U && SwigluFullRowUbBytes(cfg.n) > AtlasA5::UB_SIZE) {
        throw std::runtime_error(
            "stage11 SwiGLU full-row UB capacity exceeded: ub_bytes=" + std::to_string(SwigluFullRowUbBytes(cfg.n)) +
            " max=" + std::to_string(AtlasA5::UB_SIZE));
    }
    if (stageNum >= 13U && (cfg.k * sizeof(uint16_t)) % 32U != 0U) {
        throw std::runtime_error("stage13 combine requires K * sizeof(float16) to be 32-byte aligned");
    }
    if (stageNum >= 14U && cfg.max_output_size < cfg.m * cfg.topk) {
        throw std::runtime_error("stage14 unpermute requires max_output_size >= M * topK for large dropless path");
    }
    if (stageNum >= 14U && (cfg.k * sizeof(uint16_t)) % 32U != 0U) {
        throw std::runtime_error("stage14 unpermute requires K * sizeof(float16) to be 32-byte aligned");
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
        const int warmup_iters = ParseEnvInt("DISPATCH_FFN_COMBINE_V8_WARMUP_ITERS", kDefaultWarmupIters);
        const int measure_iters = ParseEnvInt("DISPATCH_FFN_COMBINE_V8_MEASURE_ITERS", kDefaultMeasureIters);
        const bool skip_accuracy = ParseEnvInt("DISPATCH_FFN_COMBINE_V8_SKIP_ACCURACY", 0) != 0;
        const bool stage_profile = ParseEnvInt("DISPATCH_FFN_COMBINE_V8_STAGE_PROFILE", 0) != 0;
        const bool start_sync_debug = ParseEnvInt("DISPATCH_FFN_COMBINE_V8_START_SYNC_DEBUG", 0) != 0;
        const bool workload_audit = ParseEnvInt("DISPATCH_FFN_COMBINE_V8_WORKLOAD_AUDIT", 0) != 0;
        g_sys_cnt_multiple = ParseEnvDouble("DISPATCH_FFN_COMBINE_V8_SYS_CNT_NS", g_sys_cnt_multiple);
        if (g_sys_cnt_multiple <= 0.0) {
            throw std::runtime_error("DISPATCH_FFN_COMBINE_V8_SYS_CNT_NS must be positive");
        }
        if (warmup_iters < 0 || measure_iters < 0) {
            throw std::runtime_error("warmup/measure iters must be non-negative");
        }

        CaseConfig cfg = LoadCaseConfig(case_dir + "/case.json");
        const int aic_num = ParseEnvInt("DISPATCH_FFN_COMBINE_V8_AIC_NUM", kDefaultAicNum);
        const int aiv_num = ParseEnvInt("DISPATCH_FFN_COMBINE_V8_AIV_NUM", kDefaultAivNum);
        if (aic_num <= 0 || aiv_num <= 0) {
            throw std::runtime_error("DISPATCH_FFN_COMBINE_V8_AIC_NUM/AIV_NUM must be positive");
        }
        if (aiv_num != aic_num * 2) {
            throw std::runtime_error("A5 V8 currently expects a 1:2 mixed-core shape: AIV_NUM must equal AIC_NUM*2");
        }
        cfg.aic_num = static_cast<uint32_t>(aic_num);
        cfg.aiv_num = static_cast<uint32_t>(aiv_num);
        const RankFileSet files = BuildRankFileSet(case_dir, rank_id);
        DispatchFFNCombineBuildResult build = BuildDispatchFFNCombineTiling(cfg, runtime);
        auto &front_tiling = build.tiling.frontReorderTiling;
        auto &frontDebug = build.tiling.frontDebugTiling;
        const int front_stages =
            ParseEnvInt("DISPATCH_FFN_COMBINE_V8_FRONT_STAGES", static_cast<int>(front_tiling.stageNum));
        if (front_stages < 0 || front_stages > 14) {
            throw std::runtime_error("DISPATCH_FFN_COMBINE_V8_FRONT_STAGES must be in [0, 14]");
        }
        front_tiling.stageNum = static_cast<uint16_t>(front_stages);
        frontDebug.frontCheckMode = 0U;
        frontDebug.frontCheckDebugMode = 0U;
        frontDebug.frontCheckStopStep = 0U;
        frontDebug.frontSortCheckDebugStep = 0U;
        frontDebug.frontCheckDebugSync = 0U;
        const int front_stop_step = ParseEnvInt("DISPATCH_FFN_COMBINE_V8_FRONT_STOP_STEP", 0);
        if (front_stop_step < 0 || front_stop_step > 8) {
            throw std::runtime_error("DISPATCH_FFN_COMBINE_V8_FRONT_STOP_STEP must be in [0, 8]");
        }
        frontDebug.frontStopStep = static_cast<uint16_t>(front_stop_step);
        const bool front_capable =
            front_tiling.frontCase == 21000U || front_tiling.frontCase == 11000U || front_tiling.frontCase == 11010U;
        if (!front_capable) {
            throw std::runtime_error("front unsupported case has no legacy fallback");
        }
        frontDebug.frontMode = 1U;
        frontDebug.frontDebugMode = ParseEnvInt("DISPATCH_FFN_COMBINE_V8_FRONT_DEBUG", 0) != 0 ? 1U : 0U;
        frontDebug.frontSortDebugStep = 0U;
        if (frontDebug.frontStopStep != 0U) {
            frontDebug.frontMode = 1U;
            frontDebug.frontDebugMode = 1U;
            if (front_tiling.stageNum >= 9U) {
                front_tiling.stageNum = 8U;
            }
        }
        frontDebug.smallFrontMode = 0U;
        frontDebug.smallFrontDebugMode = 0U;
        auto &dispatch_tiling = build.tiling.dispatchTiling;
        dispatch_tiling.dispatchGatherMode = 1U;
        dispatch_tiling.dispatchGatherDebugMode =
            ParseEnvInt("DISPATCH_FFN_COMBINE_V8_DISPATCH_GATHER_DEBUG", 0) != 0 ? 1U : 0U;
        const int dispatch_gather_stop_step = ParseEnvInt("DISPATCH_FFN_COMBINE_V8_DISPATCH_GATHER_STOP_STEP", 0);
        if (dispatch_gather_stop_step < 0 || dispatch_gather_stop_step > 4) {
            throw std::runtime_error("DISPATCH_FFN_COMBINE_V8_DISPATCH_GATHER_STOP_STEP must be in [0, 4]");
        }
        dispatch_tiling.dispatchGatherStopStep = static_cast<uint32_t>(dispatch_gather_stop_step);
        dispatch_tiling.dispatchGatherDebugMaxGroup = std::numeric_limits<uint32_t>::max();
        dispatch_tiling.dispatchGatherDebugMaxSrcRank = std::numeric_limits<uint32_t>::max();
        dispatch_tiling.dispatchGatherDebugMaxBatches = std::numeric_limits<uint32_t>::max();
        dispatch_tiling.dispatchGatherDebugCutPoint = 0U;
        if (dispatch_tiling.dispatchGatherStopStep != 0U) {
            dispatch_tiling.dispatchGatherMode = 1U;
            dispatch_tiling.dispatchGatherDebugMode = 1U;
            if (frontDebug.frontStopStep == 0U) {
                frontDebug.frontStopStep = 8U;
                frontDebug.frontDebugMode = frontDebug.frontMode;
            }
            front_tiling.stageNum = static_cast<uint16_t>(dispatch_tiling.dispatchGatherStopStep >= 3U ? 10U : 9U);
        }
        auto &combine_tiling = build.tiling.combineTiling;
        combine_tiling.combineImplMode = kHostCombineImplDirectAuto;
        const int combine_stop_step = ParseEnvInt("DISPATCH_FFN_COMBINE_V8_COMBINE_STOP_STEP", 0);
        if (combine_stop_step < 0 || combine_stop_step > 7) {
            throw std::runtime_error("DISPATCH_FFN_COMBINE_V8_COMBINE_STOP_STEP must be in [0, 7]");
        }
        combine_tiling.combineStopStep = static_cast<uint32_t>(combine_stop_step);
        const bool gmm2_combine_cv_debug =
            ParseEnvInt("DISPATCH_FFN_COMBINE_V8_GMM2_COMBINE_CV_DEBUG", 0) != 0;
        combine_tiling.gmm2CombineCvMode = kHostGmm2CombineCvModeDirect;
        combine_tiling.gmm2CombineCvDebugMode = gmm2_combine_cv_debug ? 1U : 0U;
        ValidateStageHardFlagBudget(cfg, build.tiling.frontReorderTiling.stageNum);
        dispatch_tiling.dispatchDebugMode = 0U;
        const bool gmm1_debug = ParseEnvInt("DISPATCH_FFN_COMBINE_V8_GMM1_DEBUG", 0) != 0;
        const bool dispatch_gather_gmm1_boundary = dispatch_tiling.dispatchGatherStopStep >= 3U;
        build.tiling.gmm1Tiling.gmm1DebugMode = (gmm1_debug || dispatch_gather_gmm1_boundary) ? 1U : 0U;
        const bool swiglu_debug = ParseEnvInt("DISPATCH_FFN_COMBINE_V8_SWIGLU_DEBUG", 0) != 0;
        build.tiling.swigluTiling.swigluDebugMode = swiglu_debug ? 1U : 0U;
        const bool combine_debug = ParseEnvInt("DISPATCH_FFN_COMBINE_V8_COMBINE_DEBUG", 0) != 0 ||
                                   ParseEnvInt("DISPATCH_FFN_COMBINE_V8_DUMP_COMBINE", 0) != 0 ||
                                   combine_stop_step != 0 || gmm2_combine_cv_debug;
        build.tiling.combineTiling.combineDebugMode = combine_debug ? 1U : 0U;
        const bool gmm2_debug = ParseEnvInt("DISPATCH_FFN_COMBINE_V8_GMM2_DEBUG", 0) != 0 || combine_debug;
        build.tiling.gmm2Tiling.gmm2DebugMode = gmm2_debug ? 1U : 0U;
        const bool unpermute_debug = ParseEnvInt("DISPATCH_FFN_COMBINE_V8_UNPERMUTE_DEBUG", 0) != 0;
        build.tiling.unpermuteTiling.unpermuteDebugMode = unpermute_debug ? 1U : 0U;
        const auto &front = build.tiling.frontReorderTiling;
        const auto &frontDebugTrace = build.tiling.frontDebugTiling;
        Trace(
            rank_id,
            "tiling built stage=" + std::to_string(front.stageNum) + " frontPath=" + std::to_string(front.frontPath) +
                " smallRouteElems=" + std::to_string(front.smallRouteElems) +
                " smallFrontUbBytes=" + std::to_string(front.smallFrontUbBytes) +
                " smallFrontMode=" + std::to_string(frontDebugTrace.smallFrontMode) +
                " smallFrontDebugMode=" + std::to_string(frontDebugTrace.smallFrontDebugMode) +
                " smallFrontRouteElems=" + std::to_string(front.smallFrontRouteElems) +
                " smallFrontAlignedRouteElems=" + std::to_string(front.smallFrontAlignedRouteElems) +
                " smallFrontNeedCoreNum=" + std::to_string(frontDebugTrace.smallFrontNeedCoreNum) +
                " smallFrontPerCoreRoutes=" + std::to_string(front.smallFrontPerCoreRoutes) +
                " smallFrontLastCoreRoutes=" + std::to_string(front.smallFrontLastCoreRoutes) +
                " smallFrontSortUbBytes=" + std::to_string(front.smallFrontSortUbBytes) +
                " smallFrontDebugOffset=" + std::to_string(frontDebugTrace.smallFrontDebugOffset) +
                " smallFrontDebugBytes=" + std::to_string(frontDebugTrace.smallFrontDebugBytes) +
                " smallFrontDebugBytesPerWorker=" + std::to_string(frontDebugTrace.smallFrontDebugBytesPerWorker) +
                " frontCheckMode=" + std::to_string(frontDebugTrace.frontCheckMode) +
                " frontCase=" + std::to_string(front.frontCase) +
                " frontCheckDebugMode=" + std::to_string(frontDebugTrace.frontCheckDebugMode) +
                " frontCheckStopStep=" + std::to_string(frontDebugTrace.frontCheckStopStep) +
                " frontSortCheckDebugStep=" + std::to_string(frontDebugTrace.frontSortCheckDebugStep) +
                " frontCheckDebugSync=" + std::to_string(frontDebugTrace.frontCheckDebugSync) + " routeElems=" +
                std::to_string(front.routeElems) + " alignedRouteElems=" + std::to_string(front.alignedRouteElems) +
                " sortLoopMaxElement=" + std::to_string(front.sortLoopMaxElement) +
                " fullLoadMaxRouteElems=" + std::to_string(front.fullLoadMaxRouteElems) + " sortNeedCoreNum=" +
                std::to_string(front.sortNeedCoreNum) + " sortPerCoreElems=" + std::to_string(front.sortPerCoreElems) +
                " sortLastCoreElems=" + std::to_string(front.sortLastCoreElems) +
                " sortPerCoreLoops=" + std::to_string(front.sortPerCoreLoops) +
                " sortPerCorePerLoopElems=" + std::to_string(front.sortPerCorePerLoopElems) +
                " sortPerCoreLastLoopElems=" + std::to_string(front.sortPerCoreLastLoopElems) +
                " sortLastCoreLoops=" + std::to_string(front.sortLastCoreLoops) +
                " sortLastCorePerLoopElems=" + std::to_string(front.sortLastCorePerLoopElems) +
                " sortLastCoreLastLoopElems=" + std::to_string(front.sortLastCoreLastLoopElems) +
                " sortVmsMiddleNeedCoreNum=" + std::to_string(front.sortVmsMiddleNeedCoreNum) +
                " sortOutLoopMaxElems=" + std::to_string(front.sortOutLoopMaxElems) + " frontMode=" +
                std::to_string(frontDebug.frontMode) + " frontDebugMode=" + std::to_string(frontDebug.frontDebugMode) +
                " frontStopStep=" + std::to_string(frontDebug.frontStopStep) +
                " frontExpandedExpertOffset=" + std::to_string(front.frontExpandedExpertOffset) +
                " frontExpandDstToSrcOffset=" + std::to_string(front.frontExpandDstToSrcOffset) +
                " frontSortWs0Offset=" + std::to_string(front.frontSortWs0Offset) +
                " frontSortWs1Offset=" + std::to_string(front.frontSortWs1Offset) +
                " frontQuantTmpOffset=" + std::to_string(front.frontQuantTmpOffset) +
                " frontQuantTmpBytes=" + std::to_string(front.frontQuantTmpBytes) +
                " frontDebugOffset=" + std::to_string(frontDebug.frontDebugOffset) +
                " frontDebugBytes=" + std::to_string(frontDebug.frontDebugBytes) +
                " frontSortCheckOffset=" + std::to_string(frontDebug.frontSortCheckOffset) +
                " frontSortCheckBytes=" + std::to_string(frontDebug.frontSortCheckBytes) +
                " frontMergeCheckOffset=" + std::to_string(frontDebug.frontMergeCheckOffset) +
                " frontMergeCheckBytes=" + std::to_string(frontDebug.frontMergeCheckBytes) +
                " frontCountScratchOffset=" + std::to_string(front.frontCountScratchOffset) +
                " frontCountScratchBytes=" + std::to_string(front.frontCountScratchBytes) +
                " frontCheckDebugOffset=" + std::to_string(frontDebug.frontCheckDebugOffset) +
                " frontCheckDebugBytes=" + std::to_string(frontDebug.frontCheckDebugBytes) +
                " frontCheckDebugBytesPerWorker=" + std::to_string(frontDebug.frontCheckDebugBytesPerWorker) +
                " dispatchGatherMode=" + std::to_string(build.tiling.dispatchTiling.dispatchGatherMode) +
                " dispatchGatherDebugMode=" + std::to_string(build.tiling.dispatchTiling.dispatchGatherDebugMode) +
                " dispatchGatherStopStep=" + std::to_string(build.tiling.dispatchTiling.dispatchGatherStopStep) +
                " dispatchGatherDebugMaxGroup=" +
                std::to_string(build.tiling.dispatchTiling.dispatchGatherDebugMaxGroup) +
                " dispatchGatherDebugMaxSrcRank=" +
                std::to_string(build.tiling.dispatchTiling.dispatchGatherDebugMaxSrcRank) +
                " dispatchGatherDebugMaxBatches=" +
                std::to_string(build.tiling.dispatchTiling.dispatchGatherDebugMaxBatches) +
                " dispatchGatherScratchOffset=" +
                std::to_string(build.tiling.dispatchTiling.dispatchGatherScratchOffset) +
                " dispatchGatherScratchBytes=" +
                std::to_string(build.tiling.dispatchTiling.dispatchGatherScratchBytes) +
                " dispatchGatherScratchBytesPerAiv=" +
                std::to_string(build.tiling.dispatchTiling.dispatchGatherScratchBytesPerAiv) +
                " dispatchGatherDebugOffset=" + std::to_string(build.tiling.dispatchTiling.dispatchGatherDebugOffset) +
                " dispatchGatherDebugBytes=" + std::to_string(build.tiling.dispatchTiling.dispatchGatherDebugBytes) +
                " dispatchGatherTileBytes=" + std::to_string(build.tiling.dispatchTiling.dispatchGatherTileBytes) +
                " dispatchDebugMode=" + std::to_string(build.tiling.dispatchTiling.dispatchDebugMode) +
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
        DeviceBuffer x_active_mask_dev = MakeDeviceBuffer(x_active_mask.size(), x_active_mask.data());
        DeviceBuffer out_dev = MakeDeviceBuffer(static_cast<size_t>(cfg.m) * cfg.k * sizeof(uint16_t));
        DeviceBuffer expert_token_nums_dev =
            MakeDeviceBuffer(static_cast<size_t>(cfg.expert_per_rank) * sizeof(int32_t));
        DeviceBuffer workspace_dev = MakeDeviceBuffer(build.workspace_bytes);
        DeviceBuffer tiling_dev = MakeDeviceBuffer(sizeof(build.tiling), &build.tiling);
        const size_t profile_bytes = static_cast<size_t>(build.block_dim) * kDispatchFFNCombineProfileBytesPerBlock;
        DeviceBuffer profile_dev = MakeDeviceBuffer(profile_bytes);
        HostBuffer profile_host = MakeHostBuffer(profile_bytes);
        Trace(rank_id, "buffers allocated");

        DispatchFFNCombineLaunchArgs args;
        uint64_t ffts_addr = 0;
        uint32_t ffts_len = 0;
        const rtError_t ffts_ret = rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
        if (ffts_ret != 0) {
            std::cerr << "rank=" << rank_id << " warning: rtGetC2cCtrlAddr failed ret=" << ffts_ret
                      << ", continue with ffts=0" << std::endl;
            ffts_addr = 0;
            ffts_len = 0;
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
        args.x_active_mask = x_active_mask_dev.ptr;
        args.out = out_dev.ptr;
        args.expert_token_nums = expert_token_nums_dev.ptr;
        args.profile_data = profile_dev.ptr;
        args.stage_profile = stage_profile ? 1U : 0U;
        args.start_sync_debug = start_sync_debug ? 1U : 0U;

        auto launch_once = [&]() {
            Trace(rank_id, "launch begin");
            launchDispatchFFNCombine(args, runtime.compute_stream);
            Trace(rank_id, "launch submitted");
            if (aclrtSynchronizeStream(runtime.compute_stream) != ACL_SUCCESS) {
                throw std::runtime_error("stream sync failed");
            }
            Trace(rank_id, "launch synced");
        };

        std::vector<double> kernel_ticks;
        std::vector<double> host_wall_times_us;
        std::vector<double> inferred_ns_per_tick_samples;
        std::vector<std::string> host_timestamp_lines;
        kernel_ticks.reserve(static_cast<size_t>(measure_iters));
        host_wall_times_us.reserve(static_cast<size_t>(measure_iters));
        inferred_ns_per_tick_samples.reserve(static_cast<size_t>(measure_iters));
        host_timestamp_lines.reserve(static_cast<size_t>(measure_iters));
        bool measured_profile_ready = false;

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
            const auto host_start_time = std::chrono::system_clock::now();
            const auto host_start_steady = std::chrono::steady_clock::now();
            Trace(rank_id, "measure launch begin");
            launchDispatchFFNCombine(args, runtime.compute_stream);
            Trace(rank_id, "measure launch submitted");
            if (aclrtSynchronizeStream(runtime.compute_stream) != ACL_SUCCESS) {
                throw std::runtime_error("stream sync failed");
            }
            const auto host_end_steady = std::chrono::steady_clock::now();
            const auto host_end_time = std::chrono::system_clock::now();
            Trace(rank_id, "measure launch synced");
            CommMpiBarrier();

            const uint64_t ticks = ReadKernelProfileTicks(profile_dev, profile_host, build.block_dim);
            const double host_wall_us =
                static_cast<double>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(host_end_steady - host_start_steady).count()) /
                1000.0;
            kernel_ticks.push_back(static_cast<double>(ticks));
            host_wall_times_us.push_back(host_wall_us);
            const double inferred_ns_per_tick = ticks == 0U ? 0.0 : host_wall_us * 1000.0 / static_cast<double>(ticks);
            if (ticks != 0U) {
                inferred_ns_per_tick_samples.push_back(inferred_ns_per_tick);
            }
            std::ostringstream ts;
            ts << std::fixed << std::setprecision(2) << "rank=" << rank_id << " hostMeasureTimestamp iter=" << iter
               << " start=\"" << FormatSystemTimeUs(host_start_time) << "\""
               << " end=\"" << FormatSystemTimeUs(host_end_time) << "\""
               << " wall_us=" << host_wall_us << " syscnt_ticks=" << ticks
               << " syscnt_us_at_current_convert=" << SysCntTicksToUs(ticks)
               << " inferred_ns_per_tick=" << inferred_ns_per_tick
               << " current_convert_ns_per_tick=" << g_sys_cnt_multiple;
            host_timestamp_lines.push_back(ts.str());
            measured_profile_ready = true;
        }

        if (!host_timestamp_lines.empty()) {
            std::ostringstream os;
            os << "rank=" << rank_id << " hostMeasureTimestamps fields=iter,start,end,wall_us,syscnt_ticks,"
               << "syscnt_us_at_current_convert,inferred_ns_per_tick,current_convert_ns_per_tick\n";
            for (size_t i = 0; i < host_timestamp_lines.size(); ++i) {
                os << "  " << host_timestamp_lines[i];
                if (i + 1U < host_timestamp_lines.size()) {
                    os << '\n';
                }
            }
            PrintOrderedByRank(rank_id, world_size, os.str());
        }

        const std::vector<double> kernel_max_samples = GatherMaxSamplesToRoot(kernel_ticks, rank_id, world_size);
        const std::vector<double> host_wall_max_samples =
            GatherMaxSamplesToRoot(host_wall_times_us, rank_id, world_size);
        const std::vector<double> inferred_ns_per_tick_all_samples =
            GatherAllSamplesToRoot(inferred_ns_per_tick_samples, rank_id, world_size);
        if (rank_id == 0) {
            PrintPerfSummary(cfg, build.block_dim, warmup_iters, measure_iters, kernel_max_samples,
                             host_wall_max_samples, inferred_ns_per_tick_all_samples);
        }
        if (stage_profile) {
            std::vector<int32_t> expert_token_nums_host(static_cast<size_t>(cfg.expert_per_rank), 0);
            const size_t expert_token_nums_bytes = expert_token_nums_host.size() * sizeof(int32_t);
            if (expert_token_nums_bytes != 0U &&
                aclrtMemcpy(expert_token_nums_host.data(), expert_token_nums_bytes, expert_token_nums_dev.ptr,
                            expert_token_nums_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
                throw std::runtime_error("device->host expert_token_nums profile copy failed");
            }
            const std::string report =
                measured_profile_ready ?
                    BuildStageProfileReportText(rank_id, {profile_host.ptr, profile_host.bytes}, build.block_dim,
                                                g_sys_cnt_multiple, start_sync_debug, &cfg, &build,
                                                &expert_token_nums_host) :
                    "rank=" + std::to_string(rank_id) + " stageProfile EMPTY measure_iters=0";
            PrintOrderedByRank(rank_id, world_size, report);
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

        const bool stage_check_pass = RunStageDebugChecks(
            rank_id, world_size, cfg, build, runtime, {workspace_dev.ptr, workspace_dev.bytes},
            {expert_token_nums_dev.ptr, expert_token_nums_dev.bytes}, {weight1_dev.ptr, weight1_dev.bytes},
            {scale1_dev.ptr, scale1_dev.bytes}, {weight2_dev.ptr, weight2_dev.bytes},
            {scale2_dev.ptr, scale2_dev.bytes}, {probs_dev.ptr, probs_dev.bytes}, {out_dev.ptr, out_dev.bytes},
            expert_idx, x, weight1, scale1, weight2, scale2, probs, actual_out, case_dir, skip_accuracy);

        WriteBinaryFile(case_dir + "/output_rank" + std::to_string(rank_id) + ".bin", actual_out.data(),
                        actual_out.size() * sizeof(uint16_t));
        if (skip_accuracy) {
            ok = stage_check_pass;
            PrintOrderedByRank(rank_id, world_size,
                               "rank=" + std::to_string(rank_id) + " accuracy=SKIP\n" + (ok ? "PASS" : "FAIL") +
                                   std::string(" rank=") + std::to_string(rank_id));
        } else {
            const AccuracyReport report = CompareFp16File(expected_out, actual_out, cfg.compare_atol, cfg.compare_rtol);
            ok = report.pass && stage_check_pass;
            PrintOrderedByRank(rank_id, world_size,
                               BuildAccuracyReportText(rank_id, report, cfg.compare_atol, cfg.compare_rtol) + "\n" +
                                   (ok ? "PASS" : "FAIL") + std::string(" rank=") + std::to_string(rank_id));
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
    const char *case_dir_env = std::getenv("DISPATCH_FFN_COMBINE_V8_CASE_DIR");
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
