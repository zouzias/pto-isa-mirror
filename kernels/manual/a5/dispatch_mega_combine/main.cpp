/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
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
extern "C" rtError_t rtGetC2cCtrlAddr(uint64_t* addr, uint32_t* len);

namespace {

constexpr int kDefaultAicNum = 28;
constexpr int kDefaultAivNum = 56;

struct DeviceBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr), bytes(other.bytes)
    {
        other.ptr = nullptr;
        other.bytes = 0;
    }

    ~DeviceBuffer()
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
};

struct RankBuffers {
    DeviceBuffer x;
    DeviceBuffer weight1;
    DeviceBuffer weight2;
    DeviceBuffer expertIdx;
    DeviceBuffer scale1;
    DeviceBuffer scale2;
    DeviceBuffer probs;
    DeviceBuffer out;
    DeviceBuffer expertTokenNums;
    DeviceBuffer workspace;
    DeviceBuffer tiling;
    std::vector<uint16_t> expectedOut;
};

DeviceBuffer MakeDeviceBuffer(size_t bytes, const void* hostSrc = nullptr)
{
    DeviceBuffer buffer;
    buffer.bytes = bytes;
    if (bytes == 0U) {
        return buffer;
    }
    if (aclrtMalloc(&buffer.ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        throw std::runtime_error("aclrtMalloc failed");
    }
    if (hostSrc != nullptr &&
        aclrtMemcpy(buffer.ptr, bytes, hostSrc, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        throw std::runtime_error("aclrtMemcpy host->device failed");
    }
    return buffer;
}

DeviceBuffer LoadDeviceBuffer(const std::string& path)
{
    const std::vector<uint8_t> data = ReadBinaryFile(path);
    return MakeDeviceBuffer(data.size(), data.data());
}

std::vector<uint16_t> BytesToU16(const std::vector<uint8_t>& bytes)
{
    if (bytes.size() % sizeof(uint16_t) != 0U) {
        throw std::runtime_error("fp16 file size is not aligned");
    }
    std::vector<uint16_t> out(bytes.size() / sizeof(uint16_t));
    for (size_t i = 0; i < out.size(); ++i) {
        const size_t byteOffset = i * sizeof(uint16_t);
        out[i] = static_cast<uint16_t>(bytes[byteOffset]) | (static_cast<uint16_t>(bytes[byteOffset + 1U]) << 8U);
    }
    return out;
}

int ParseEnvInt(const char* name, int defaultValue)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid integer in env: ") + name);
    }
}

bool ParseFirstDevice(int argc, char** argv, int worldSize, int& firstDevice)
{
    firstDevice = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) != "--first-device") {
            continue;
        }
        if (i + 1 >= argc) {
            std::cerr << "--first-device requires a non-negative device ID" << std::endl;
            return false;
        }

        const std::string valueText = argv[++i];
        size_t parsed = 0;
        long long value = -1;
        try {
            value = std::stoll(valueText, &parsed);
        } catch (const std::exception&) {
            parsed = 0;
        }
        if (parsed != valueText.size() || value < 0 || value > std::numeric_limits<int>::max()) {
            std::cerr << "invalid --first-device value: " << valueText << std::endl;
            return false;
        }
        firstDevice = static_cast<int>(value);
    }

    if (worldSize <= 0 || firstDevice > std::numeric_limits<int>::max() - (worldSize - 1)) {
        std::cerr << "device range overflows int: first-device=" << firstDevice << " world-size=" << worldSize
                  << std::endl;
        return false;
    }
    return true;
}

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    if (alignment == 0U) {
        throw std::invalid_argument("alignment must be non-zero");
    }
    return (value + alignment - 1U) / alignment * alignment;
}

uint64_t SwigluFullRowUbBytes(uint32_t n)
{
    auto alignUb = [](uint64_t value) { return AlignUp(value, 32U); };
    uint64_t bytes = 0;
    bytes += alignUb(static_cast<uint64_t>(n) * sizeof(float));
    bytes += alignUb(static_cast<uint64_t>(n) * sizeof(float));
    bytes += alignUb(static_cast<uint64_t>(n) * sizeof(float));
    bytes += alignUb(static_cast<uint64_t>(n) * sizeof(float));
    bytes += alignUb(static_cast<uint64_t>(n) * sizeof(int8_t));
    bytes += 2U * 32U;
    return bytes;
}

void ValidateConfiguration(const CaseConfig& cfg)
{
    if (cfg.world_size == 0U || cfg.world_size > 16U) {
        throw std::runtime_error("front cumsum requires world_size in [1, 16]");
    }
    if (cfg.expert_per_rank > MEGA_MOE_D2C_MAX_LOGICAL_GROUP_EVENTS) {
        throw std::runtime_error(
            "dispatch V2C hard flag budget exceeded: expert_per_rank=" + std::to_string(cfg.expert_per_rank) +
            " max=" + std::to_string(MEGA_MOE_D2C_MAX_LOGICAL_GROUP_EVENTS));
    }
    if (cfg.k % 128U != 0U) {
        throw std::runtime_error("GMM1 requires K % 128 == 0");
    }
    if (cfg.n % 64U != 0U) {
        throw std::runtime_error("SwiGLU requires N % 64 == 0");
    }
    if (SwigluFullRowUbBytes(cfg.n) > AtlasA5::UB_SIZE) {
        throw std::runtime_error(
            "SwiGLU full-row UB capacity exceeded: ub_bytes=" + std::to_string(SwigluFullRowUbBytes(cfg.n)) +
            " max=" + std::to_string(AtlasA5::UB_SIZE));
    }
    if ((cfg.k * sizeof(uint16_t)) % 32U != 0U) {
        throw std::runtime_error("combine/unpermute requires K * sizeof(float16) to be 32-byte aligned");
    }
    if (static_cast<uint64_t>(cfg.max_output_size) < static_cast<uint64_t>(cfg.m) * cfg.topk) {
        throw std::runtime_error("unpermute requires max_output_size >= M * topK");
    }
}

CaseConfig LoadAndValidateCase(const std::string& caseDir, int worldSize)
{
    CaseConfig cfg = LoadCaseConfig(caseDir + "/case.json");
    if (cfg.world_size != static_cast<uint32_t>(worldSize)) {
        throw std::runtime_error("case world_size does not match MPI world size");
    }

    const int aicNum = ParseEnvInt("DISPATCH_MEGA_COMBINE_AIC_NUM", kDefaultAicNum);
    const int aivNum = ParseEnvInt("DISPATCH_MEGA_COMBINE_AIV_NUM", kDefaultAivNum);
    if (aicNum <= 0 || aivNum != aicNum * 2) {
        throw std::runtime_error("A5 requires positive AIC_NUM and AIV_NUM == AIC_NUM * 2");
    }
    cfg.aic_num = static_cast<uint32_t>(aicNum);
    cfg.aiv_num = static_cast<uint32_t>(aivNum);
    ValidateConfiguration(cfg);
    return cfg;
}

void ZeroDeviceBuffer(const DeviceBuffer& buffer, const char* name)
{
    if (buffer.bytes != 0U && aclrtMemset(buffer.ptr, buffer.bytes, 0, buffer.bytes) != ACL_SUCCESS) {
        throw std::runtime_error(std::string("failed to zero ") + name);
    }
}

void PrepareLaunchBuffers(
    const StandaloneRankRuntime& runtime, const DeviceBuffer& out, const DeviceBuffer& expertTokenNums,
    const DeviceBuffer& workspace)
{
    const uint64_t windowBytes = runtime.hccl.WindowClearBytes();
    void* window = runtime.hccl.WindowClearBase(static_cast<uint32_t>(runtime.hccl.rank_id));
    if (aclrtMemset(window, windowBytes, 0, windowBytes) != ACL_SUCCESS) {
        throw std::runtime_error("failed to zero HCCL window");
    }
    ZeroDeviceBuffer(out, "out buffer");
    ZeroDeviceBuffer(expertTokenNums, "expert_token_nums");
    ZeroDeviceBuffer(workspace, "workspace");
}

std::string BuildAccuracyReport(int rankId, const AccuracyReport& report)
{
    std::ostringstream os;
    os << std::setprecision(6) << "rank=" << rankId << " max_diff=" << report.max_abs_err
       << " max_ratio=" << report.max_rel_err << " err=" << report.mismatch_count << "/" << report.err_threshold
       << " -> " << (report.pass ? "PASS" : "FAIL");
    return os.str();
}

void PrintOrderedByRank(int rankId, int worldSize, const std::string& text)
{
    for (int turn = 0; turn < worldSize; ++turn) {
        CommMpiBarrier();
        if (turn == rankId) {
            std::cout << text << std::endl;
        }
    }
    CommMpiBarrier();
}

RankBuffers CreateRankBuffers(const RankFileSet& files, const CaseConfig& cfg, const MegaMoeBuildResult& build)
{
    return {
        LoadDeviceBuffer(files.x),
        LoadDeviceBuffer(files.weight1),
        LoadDeviceBuffer(files.weight2),
        LoadDeviceBuffer(files.expert_idx),
        LoadDeviceBuffer(files.scale1),
        LoadDeviceBuffer(files.scale2),
        LoadDeviceBuffer(files.probs),
        MakeDeviceBuffer(static_cast<size_t>(cfg.m) * cfg.k * sizeof(uint16_t)),
        MakeDeviceBuffer(static_cast<size_t>(cfg.expert_per_rank) * sizeof(int32_t)),
        MakeDeviceBuffer(build.workspace_bytes),
        MakeDeviceBuffer(sizeof(build.tiling), &build.tiling),
        BytesToU16(ReadBinaryFile(files.expected_out)),
    };
}

void* GetFftsAddress()
{
    uint64_t fftsAddr = 0;
    uint32_t fftsLen = 0;
    if (rtGetC2cCtrlAddr(&fftsAddr, &fftsLen) != 0) {
        fftsAddr = 0;
    }
    return reinterpret_cast<void*>(fftsAddr);
}

MegaMoeLaunchArgs CreateLaunchArgs(const RankBuffers& buffers, const MegaMoeBuildResult& build)
{
    MegaMoeLaunchArgs args;
    args.ffts = GetFftsAddress();
    args.x = buffers.x.ptr;
    args.weight1 = buffers.weight1.ptr;
    args.weight2 = buffers.weight2.ptr;
    args.expert_idx = buffers.expertIdx.ptr;
    args.scale1 = buffers.scale1.ptr;
    args.scale2 = buffers.scale2.ptr;
    args.probs = buffers.probs.ptr;
    args.out = buffers.out.ptr;
    args.expert_token_nums = buffers.expertTokenNums.ptr;
    args.workspace = buffers.workspace.ptr;
    args.tiling = buffers.tiling.ptr;
    args.block_dim = build.block_dim;
    args.start_sync = ParseEnvInt("DISPATCH_MEGA_COMBINE_START_SYNC", 0) != 0 ? 1U : 0U;
    return args;
}

void RunKernel(const StandaloneRankRuntime& runtime, const RankBuffers& buffers, const MegaMoeLaunchArgs& args)
{
    PrepareLaunchBuffers(runtime, buffers.out, buffers.expertTokenNums, buffers.workspace);
    CommMpiBarrier();
    launchMegaMoe(args, runtime.compute_stream);
    if (aclrtSynchronizeStream(runtime.compute_stream) != ACL_SUCCESS) {
        throw std::runtime_error("stream sync failed");
    }
    CommMpiBarrier();
}

bool VerifyRankOutput(
    int rankId, int worldSize, const std::string& caseDir, const CaseConfig& cfg, const RankBuffers& buffers)
{
    std::vector<uint16_t> actualOut(static_cast<size_t>(cfg.m) * cfg.k);
    const size_t outputBytes = actualOut.size() * sizeof(uint16_t);
    if (aclrtMemcpy(actualOut.data(), outputBytes, buffers.out.ptr, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST) !=
        ACL_SUCCESS) {
        throw std::runtime_error("device->host output copy failed");
    }
    WriteBinaryFile(caseDir + "/output_rank" + std::to_string(rankId) + ".bin", actualOut.data(), outputBytes);

    const AccuracyReport report = CompareFp16File(buffers.expectedOut, actualOut, cfg.compare_atol, cfg.compare_rtol);
    PrintOrderedByRank(rankId, worldSize, BuildAccuracyReport(rankId, report));
    return report.pass;
}

bool ExecuteRankCase(int rankId, int worldSize, const std::string& caseDir, const StandaloneRankRuntime& runtime)
{
    const CaseConfig cfg = LoadAndValidateCase(caseDir, worldSize);
    const RankFileSet files = BuildRankFileSet(caseDir, rankId);
    const MegaMoeBuildResult build = BuildMegaMoeTiling(cfg, runtime);
    const RankBuffers buffers = CreateRankBuffers(files, cfg, build);
    const MegaMoeLaunchArgs args = CreateLaunchArgs(buffers, build);
    RunKernel(runtime, buffers, args);
    return VerifyRankOutput(rankId, worldSize, caseDir, cfg, buffers);
}

bool RunOneRank(int rankId, int worldSize, int deviceId, const std::string& caseDir, const HcclRootInfo& rootInfo)
{
    StandaloneRankRuntime runtime;
    if (!InitStandaloneRankRuntime(runtime, rankId, worldSize, deviceId, rootInfo)) {
        return false;
    }

    bool ok = false;
    try {
        ok = ExecuteRankCase(rankId, worldSize, caseDir, runtime);
    } catch (const std::exception& ex) {
        std::cerr << "rank=" << rankId << " error: " << ex.what() << std::endl;
        ok = false;
    }

    DestroyStandaloneRankRuntime(runtime);
    return ok;
}

} // namespace

int main(int argc, char** argv)
{
    if (!CommMpiInit(&argc, &argv)) {
        return 1;
    }

    const int rankId = CommMpiRank();
    const int worldSize = CommMpiSize();
    int firstDevice = 0;
    if (!ParseFirstDevice(argc, argv, worldSize, firstDevice)) {
        CommMpiFinalize();
        return 1;
    }
    const int deviceId = firstDevice + rankId;
    const char* caseDirEnv = std::getenv("DISPATCH_MEGA_COMBINE_CASE_DIR");
    const std::string caseDir = caseDirEnv == nullptr ? "../out" : caseDirEnv;

    if (rankId == 0) {
        std::cout << "rank/device mapping: ranks=[0," << worldSize << ") physical_devices=[" << firstDevice << ","
                  << firstDevice + worldSize << ")" << std::endl;
    }

    if (aclInit(nullptr) != ACL_SUCCESS) {
        CommMpiFinalize();
        return 1;
    }
    if (rtSetDevice(deviceId) != 0 || aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        aclFinalize();
        CommMpiFinalize();
        return 1;
    }

    HcclRootInfo rootInfo{};
    if (rankId == 0 && HcclGetRootInfo(&rootInfo) != HCCL_SUCCESS) {
        aclrtResetDevice(deviceId);
        aclFinalize();
        CommMpiFinalize();
        return 1;
    }
    CommMpiBcast(&rootInfo, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0);
    CommMpiBarrier();

    const bool ok = RunOneRank(rankId, worldSize, deviceId, caseDir, rootInfo);

    CommMpiBarrier();
    aclFinalize();
    CommMpiFinalize();
    return ok ? 0 : 1;
}
