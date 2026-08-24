/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CPU_STUB_RUNTIME_INL_HPP
#define CPU_STUB_RUNTIME_INL_HPP

namespace pto::cpu_sim {
struct RuntimeConfig {
    bool initialized = false;
    uint32_t device_id = 0;
    uint32_t num_cores = 1;
    bool trace_enabled = kInstructionTraceEnabled;
    std::filesystem::path trace_root = "cpu_sim_traces";
    uint64_t next_stream_id = 1;
    uint64_t next_launch_id = 0;
    std::mutex mutex;
};

struct StreamState {
    uint64_t id = 0;
};

inline const auto g_time_origin = std::chrono::steady_clock::now();

using SetExecutionContextHookFn = void (*)(uint32_t block_idx, uint32_t subblock_id, uint32_t subblock_dim);
using GetExecutionContextHookFn = void (*)(uint32_t* block_idx, uint32_t* subblock_id, uint32_t* subblock_dim);
using GetSharedStorageHookFn = void* (*)(std::string key, size_t size);
using GetTaskCookieHookFn = uint64_t (*)();
using GetSubblockIdInjectedHookFn = uint32_t (*)();
using GetPipeSharedStateInjectedHookFn = void* (*)(uint64_t pipe_key, size_t size);

inline void set_execution_context(uint32_t block_idx, uint32_t subblock_id, uint32_t subblock_dim);
inline void reset_execution_context();

inline GetSubblockIdInjectedHookFn injected_subblock_id_hook = nullptr;
inline GetPipeSharedStateInjectedHookFn injected_pipe_shared_state_hook = nullptr;

inline RuntimeConfig& runtime_config()
{
    static RuntimeConfig config;
    return config;
}

inline uint32_t ReadEnvU32(const char* name, uint32_t fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0') {
        return fallback;
    }
    return parsed == 0 ? fallback : static_cast<uint32_t>(parsed);
}

inline bool ReadEnvBool(const char* name, bool fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    if (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 || std::strcmp(value, "FALSE") == 0) {
        return false;
    }
    if (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 || std::strcmp(value, "TRUE") == 0) {
        return true;
    }
    return fallback;
}

inline void InitializeRuntime()
{
    auto& config = runtime_config();
    std::scoped_lock lock(config.mutex);
    config.device_id = 0;
    config.num_cores = ReadEnvU32("PTO_CPU_SIM_NUM_CORES", 4);
    config.trace_enabled = kInstructionTraceEnabled && ReadEnvBool("PTO_CPU_SIM_TRACE_ENABLE", true);
    if (const char* trace_dir = std::getenv("PTO_CPU_SIM_TRACE_DIR"); trace_dir != nullptr && *trace_dir != '\0') {
        config.trace_root = trace_dir;
    } else {
        config.trace_root = "cpu_sim_traces";
    }
    config.next_stream_id = 1;
    if (config.trace_enabled) {
        std::filesystem::create_directories(config.trace_root);
    }
    config.initialized = true;
}

inline void ShutdownRuntime()
{
    auto& config = runtime_config();
    std::scoped_lock lock(config.mutex);
    config.initialized = false;
    config.next_stream_id = 1;
}

inline void EnsureRuntimeInitialized()
{
    if (!runtime_config().initialized) {
        InitializeRuntime();
    }
}

inline uint32_t GetConfiguredCoreCount()
{
    EnsureRuntimeInitialized();
    return runtime_config().num_cores;
}

inline bool IsTraceEnabled()
{
    EnsureRuntimeInitialized();
    return runtime_config().trace_enabled;
}

inline std::filesystem::path GetTraceRoot()
{
    EnsureRuntimeInitialized();
    return runtime_config().trace_root;
}

inline uint64_t NextLaunchId()
{
    auto& config = runtime_config();
    std::scoped_lock lock(config.mutex);
    return config.next_launch_id++;
}

inline std::filesystem::path CreateKernelTraceDir(const std::string& kernel_name)
{
    EnsureRuntimeInitialized();
    auto dir = GetTraceRoot();
    dir.append(kernel_name);
    dir.append("launch_" + std::to_string(NextLaunchId()));
    if (IsTraceEnabled()) {
        std::filesystem::create_directories(dir);
    }
    return dir;
}

struct KernelLaunchOptions {
    std::string kernel_name;
    uint32_t requested_cores = 0;
    uint32_t total_work_items = 0;
    uint32_t work_quantum = 1;
    uint32_t subblocks_per_block = 1;
    uint32_t explicit_block_count = 0;
    bool write_trace_files = true;
};

inline uint32_t ResolveActiveCoreCount(
    uint32_t requested_cores, uint32_t total_work_items = 0, uint32_t work_quantum = 1)
{
    const uint32_t configured = requested_cores == 0 ? GetConfiguredCoreCount() : requested_cores;
    uint32_t active = std::max<uint32_t>(1, configured);
    if (total_work_items == 0) {
        return active;
    }

    active = std::max<uint32_t>(1, std::min(active, total_work_items));
    const uint32_t quantum = std::max<uint32_t>(1, work_quantum);
    while (active > 1) {
        const uint32_t chunk = (total_work_items + active - 1) / active;
        if (chunk % quantum == 0) {
            break;
        }
        --active;
    }
    return active;
}

struct ResolvedKernelLaunch {
    uint32_t subblocks_per_block = 1;
    uint32_t active_cores = 1;
    bool trace_enabled = false;
    std::filesystem::path trace_dir;
};

inline ResolvedKernelLaunch ResolveKernelLaunch(const KernelLaunchOptions& options)
{
    ResolvedKernelLaunch launch;
    launch.subblocks_per_block = std::max<uint32_t>(1, options.subblocks_per_block);
    const uint32_t active_blocks =
        options.explicit_block_count != 0 ?
            options.explicit_block_count :
            ResolveActiveCoreCount(options.requested_cores, options.total_work_items, options.work_quantum);
    launch.active_cores = active_blocks * launch.subblocks_per_block;
    launch.trace_enabled = IsTraceEnabled() && options.write_trace_files;
    launch.trace_dir = launch.trace_enabled ? CreateKernelTraceDir(options.kernel_name) : std::filesystem::path{};
    return launch;
}

template <typename KernelFn>
inline void RunKernelCore(
    const ResolvedKernelLaunch& launch, uint32_t core_id, KernelFn& kernel, std::vector<std::string>& trace_chunks)
{
    const uint32_t block_idx = core_id / launch.subblocks_per_block;
    const uint32_t subblock_id = core_id % launch.subblocks_per_block;
    ResetInstructionTrace();
    reset_execution_context();
    set_execution_context(block_idx, subblock_id, launch.subblocks_per_block);
    kernel();
    if (launch.trace_enabled) {
        std::ostringstream out;
        DumpInstructionTraceJson(out);
        trace_chunks[core_id] = out.str();
    }
    reset_execution_context();
}

inline void CaptureFirstKernelError(std::exception_ptr& first_error, std::mutex& error_mutex)
{
    std::scoped_lock lock(error_mutex);
    if (first_error == nullptr) {
        first_error = std::current_exception();
    }
}

inline void WriteKernelTraceChunks(const std::filesystem::path& trace_dir, const std::vector<std::string>& trace_chunks)
{
    std::ofstream combined(trace_dir / "trace.jsonl", std::ios::trunc);
    for (const auto& chunk : trace_chunks) {
        combined << chunk;
    }
}

template <typename KernelFn>
inline void LaunchKernelMultiCore(const KernelLaunchOptions& options, aclrtStream stream, KernelFn&& kernel)
{
    (void)stream;
    EnsureRuntimeInitialized();

    const ResolvedKernelLaunch launch = ResolveKernelLaunch(options);
    std::vector<std::thread> workers;
    workers.reserve(launch.active_cores);
    std::vector<std::string> trace_chunks(launch.active_cores);
    std::exception_ptr first_error;
    std::mutex error_mutex;

    for (uint32_t core_id = 0; core_id < launch.active_cores; ++core_id) {
        workers.emplace_back([&, core_id]() {
            try {
                RunKernelCore(launch, core_id, kernel, trace_chunks);
            } catch (...) {
                CaptureFirstKernelError(first_error, error_mutex);
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }

    if (first_error != nullptr) {
        std::rethrow_exception(first_error);
    }

    if (launch.trace_enabled) {
        WriteKernelTraceChunks(launch.trace_dir, trace_chunks);
    }
}

inline StreamState* ToStreamState(aclrtStream stream) { return reinterpret_cast<StreamState*>(stream); }

inline SetExecutionContextHookFn ResolveSetExecutionContextHook()
{
    static auto hook =
        reinterpret_cast<SetExecutionContextHookFn>(dlsym(RTLD_DEFAULT, "pto_cpu_sim_set_execution_context"));
    return hook;
}

inline GetExecutionContextHookFn ResolveExecutionContextHook()
{
    static auto hook =
        reinterpret_cast<GetExecutionContextHookFn>(dlsym(RTLD_DEFAULT, "pto_cpu_sim_get_execution_context"));
    return hook;
}

inline GetSharedStorageHookFn ResolveSharedStorageHook()
{
    static auto hook = reinterpret_cast<GetSharedStorageHookFn>(dlsym(RTLD_DEFAULT, "pto_cpu_sim_get_shared_storage"));
    return hook;
}

inline GetTaskCookieHookFn ResolveTaskCookieHook()
{
    static auto hook = reinterpret_cast<GetTaskCookieHookFn>(dlsym(RTLD_DEFAULT, "pto_cpu_sim_get_task_cookie"));
    return hook;
}

inline GetSubblockIdInjectedHookFn ResolveSubblockIdHook()
{
    static auto hook = reinterpret_cast<GetSubblockIdInjectedHookFn>(dlsym(RTLD_DEFAULT, "pto_sim_get_subblock_id"));
    return hook;
}

inline GetPipeSharedStateInjectedHookFn ResolvePipeSharedStateHook()
{
    static auto hook =
        reinterpret_cast<GetPipeSharedStateInjectedHookFn>(dlsym(RTLD_DEFAULT, "pto_sim_get_pipe_shared_state"));
    return hook;
}

struct ExecutionContext {
    uint32_t block_idx = 0;
    uint32_t subblock_id = 0;
    uint32_t subblock_dim = 1;
    uint64_t task_cookie = 0;
};

inline thread_local ExecutionContext execution_context{};

inline void register_hooks(void* get_subblock_id, void* get_pipe_shared_state)
{
    injected_subblock_id_hook = reinterpret_cast<GetSubblockIdInjectedHookFn>(get_subblock_id);
    injected_pipe_shared_state_hook = reinterpret_cast<GetPipeSharedStateInjectedHookFn>(get_pipe_shared_state);
}

inline void set_execution_context(uint32_t block_idx, uint32_t subblock_id, uint32_t subblock_dim = 1)
{
    execution_context.block_idx = block_idx;
    execution_context.subblock_id = subblock_id;
    execution_context.subblock_dim = (subblock_dim == 0) ? 1 : subblock_dim;
    if (auto hook = ResolveSetExecutionContextHook(); hook != nullptr) {
        hook(execution_context.block_idx, execution_context.subblock_id, execution_context.subblock_dim);
    }
}

inline void reset_execution_context() { execution_context = {}; }

inline void set_task_cookie(uint64_t task_cookie) { execution_context.task_cookie = task_cookie; }

class ScopedExecutionContext {
public:
    ScopedExecutionContext(uint32_t block_idx, uint32_t subblock_id, uint32_t subblock_dim = 1)
        : saved_(execution_context)
    {
        set_execution_context(block_idx, subblock_id, subblock_dim);
    }

    ~ScopedExecutionContext() { execution_context = saved_; }

private:
    ExecutionContext saved_{};
};
} // namespace pto::cpu_sim

namespace cce {
template <typename... Args>
inline int printf(const char* fmt, Args... args)
{
    return std::printf(fmt, args...);
}
} // namespace cce

inline int aclInit(const char*)
{
    pto::cpu_sim::InitializeRuntime();
    return 0;
}

inline int aclrtSetDevice(int device_id)
{
    auto& config = pto::cpu_sim::runtime_config();
    std::scoped_lock lock(config.mutex);
    config.device_id = static_cast<uint32_t>(std::max(0, device_id));
    return 0;
}

inline int aclrtCreateStream(aclrtStream* stream)
{
    pto::cpu_sim::EnsureRuntimeInitialized();
    auto& config = pto::cpu_sim::runtime_config();
    auto* state = new pto::cpu_sim::StreamState();
    {
        std::scoped_lock lock(config.mutex);
        state->id = config.next_stream_id++;
    }
    *stream = reinterpret_cast<aclrtStream>(state);
    return 0;
}

inline uint32_t get_block_idx()
{
    if (auto hook = pto::cpu_sim::ResolveExecutionContextHook(); hook != nullptr) {
        uint32_t block_idx = 0;
        uint32_t subblock_id = 0;
        uint32_t subblock_dim = 1;
        hook(&block_idx, &subblock_id, &subblock_dim);
        return block_idx;
    }
    return pto::cpu_sim::execution_context.block_idx;
}

inline uint32_t get_subblockid()
{
    if (pto::cpu_sim::injected_subblock_id_hook != nullptr) {
        return pto::cpu_sim::injected_subblock_id_hook();
    }
    if (auto hook = pto::cpu_sim::ResolveSubblockIdHook(); hook != nullptr) {
        return hook();
    }
    if (auto hook = pto::cpu_sim::ResolveExecutionContextHook(); hook != nullptr) {
        uint32_t block_idx = 0;
        uint32_t subblock_id = 0;
        uint32_t subblock_dim = 1;
        hook(&block_idx, &subblock_id, &subblock_dim);
        return subblock_id;
    }
    return pto::cpu_sim::execution_context.subblock_id;
}

inline uint32_t get_subblockdim()
{
    if (auto hook = pto::cpu_sim::ResolveExecutionContextHook(); hook != nullptr) {
        uint32_t block_idx = 0;
        uint32_t subblock_id = 0;
        uint32_t subblock_dim = 1;
        hook(&block_idx, &subblock_id, &subblock_dim);
        return subblock_dim;
    }
    return pto::cpu_sim::execution_context.subblock_dim;
}

inline uint32_t get_block_num() { return pto::cpu_sim::GetConfiguredCoreCount(); }

inline uint32_t get_coreid() { return get_block_idx(); }

inline uint64_t get_task_cookie()
{
    if (auto hook = pto::cpu_sim::ResolveTaskCookieHook(); hook != nullptr) {
        return hook();
    }
    return pto::cpu_sim::execution_context.task_cookie;
}

template <typename T>
struct is_event : std::false_type {};

template <typename... Ts>
inline constexpr bool all_events_v = (is_event<Ts>::value && ...);

#if defined(__CPU_SIM)
namespace pto {
template <SyncCoreType CoreType = SyncCoreType::AIVOnly>
inline void SYNCALL_IMPL()
{
    (void)CoreType;
}

template <SyncCoreType CoreType = SyncCoreType::AIVOnly>
inline void SYNCALL_SOFT_IMPL(int32_t* gmWorkspace, int32_t usedCores)
{
    (void)CoreType;
    (void)gmWorkspace;
    (void)usedCores;
}

inline void SYNCALL_SOFT_AIC_IMPL(int32_t* gmWorkspace, int32_t usedCores)
{
    (void)gmWorkspace;
    (void)usedCores;
}

template <SyncCoreType CoreType = SyncCoreType::Mix>
inline void SYNCALL_SOFT_MIX_IMPL(int32_t* gmWorkspace, int32_t usedCores)
{
    (void)CoreType;
    (void)gmWorkspace;
    (void)usedCores;
}
} // namespace pto
#endif // __CPU_SIM

#endif // CPU_STUB_RUNTIME_INL_HPP
