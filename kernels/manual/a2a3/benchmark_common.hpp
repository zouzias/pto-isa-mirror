/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef BENCHMARK_COMMON_HPP_
#define BENCHMARK_COMMON_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <pto/pto-inst.hpp>

#include "common.hpp"
#include "pto/comm/async/sdma/sdma_types.hpp"

namespace benchmark {

constexpr size_t kBytesPerKiB = 1024;
constexpr uint32_t kMaxDeviceBaselinePosts = 512;
constexpr double kDeviceCycleSeconds = 20.0e-9;

inline uint64_t ReadEnvUint64(const char* name, uint64_t defaultValue)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    return end != value && *end == '\0' ? static_cast<uint64_t>(parsed) : defaultValue;
}

template <typename T>
T DeviceBaselinePattern(int rankId, int outerIndex, bool measured, uint32_t block, uint32_t post, size_t elemIndex)
{
    constexpr uint64_t kPatternSpan = 7000000;
    constexpr uint64_t kWarmupBase = 1;
    constexpr uint64_t kMeasuredBase = 8000001;
    const uint64_t key = static_cast<uint64_t>(rankId) * 524287 + static_cast<uint64_t>(outerIndex) * 131071 +
                         static_cast<uint64_t>(block) * 8191 + static_cast<uint64_t>(post) * 4099 + elemIndex;
    return static_cast<T>((measured ? kMeasuredBase : kWarmupBase) + key % kPatternSpan);
}

inline AICORE uint64_t GetSyscnt()
{
    uint64_t syscnt;
    asm volatile("MOV %0, SYS_CNT\n" : "+l"(syscnt));
    return syscnt;
}

inline bool CheckAclCall(aclError ret, const char* op)
{
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] " << op << " failed: " << static_cast<int>(ret) << std::endl;
        return false;
    }
    return true;
}

struct DeviceBaselineConfig {
    uint64_t transferBytesValue;
    uint64_t blockDivisorValue;
    uint64_t queueNumValue;
    uint64_t postCountValue;
    uint64_t blockNumValue;
    uint64_t outerWarmupValue;
    uint64_t outerItersValue;
    uint64_t innerWarmupValue;
    uint64_t innerItersValue;
    bool waitEachEvent;
    size_t transferBytes;
    size_t bytesPerBlock;
    size_t totalBytes;
    size_t elemCount;
    size_t profileCount;
    uint32_t blockBytes;
    uint32_t queueNum;
    uint32_t postCount;
    uint32_t blockNum;
    int outerWarmup;
    int outerIters;
    int innerWarmup;
    int innerIters;
};

struct DeviceBaselineEnvNames {
    const char* transferBytes;
    const char* blockDivisor;
    const char* queueNum;
    const char* postCount;
    const char* blockNum;
    const char* outerWarmup;
    const char* outerIters;
    const char* innerWarmup;
    const char* innerIters;
    const char* waitEachEvent;
};

template <typename T>
bool IsTransferConfigValid(const DeviceBaselineConfig& config)
{
    return config.transferBytesValue > 0 && config.transferBytesValue <= UINT32_MAX &&
           config.transferBytesValue % sizeof(T) == 0 && config.blockDivisorValue > 0 &&
           config.transferBytesValue % config.blockDivisorValue == 0 &&
           config.transferBytesValue / config.blockDivisorValue >= pto::comm::sdma::kSdmaMinTransferBytes;
}

inline bool IsParallelConfigValid(const DeviceBaselineConfig& config)
{
    return config.queueNumValue > 0 && config.queueNumValue <= pto::comm::sdma::kSdmaMaxChannel &&
           config.postCountValue > 0 && config.postCountValue <= kMaxDeviceBaselinePosts && config.blockNumValue > 0 &&
           config.blockNumValue <= pto::comm::sdma::kSdmaMaxChannel / config.queueNumValue;
}

inline bool IsIterationConfigValid(const DeviceBaselineConfig& config)
{
    return config.outerItersValue > 0 && config.outerItersValue <= INT32_MAX &&
           config.outerWarmupValue <= INT32_MAX - config.outerItersValue && config.innerItersValue > 0 &&
           config.innerItersValue <= INT32_MAX && config.innerWarmupValue <= INT32_MAX;
}

inline bool IsDeviceAllocationSizeValid(const DeviceBaselineConfig& config)
{
    return config.transferBytesValue <= SIZE_MAX / config.postCountValue &&
           config.transferBytesValue * config.postCountValue <= SIZE_MAX / config.blockNumValue;
}

template <typename T>
bool FinalizeDeviceBaselineConfig(DeviceBaselineConfig& config)
{
    if (!IsTransferConfigValid<T>(config) || !IsParallelConfigValid(config) || !IsIterationConfigValid(config) ||
        !IsDeviceAllocationSizeValid(config)) {
        return false;
    }
    config.transferBytes = static_cast<size_t>(config.transferBytesValue);
    config.bytesPerBlock = config.transferBytes * static_cast<size_t>(config.postCountValue);
    config.totalBytes = config.bytesPerBlock * static_cast<size_t>(config.blockNumValue);
    config.elemCount = config.transferBytes / sizeof(T);
    config.profileCount = static_cast<size_t>(config.blockNumValue) * 2;
    config.blockBytes = static_cast<uint32_t>(config.transferBytesValue / config.blockDivisorValue);
    config.queueNum = static_cast<uint32_t>(config.queueNumValue);
    config.postCount = static_cast<uint32_t>(config.postCountValue);
    config.blockNum = static_cast<uint32_t>(config.blockNumValue);
    config.outerWarmup = static_cast<int>(config.outerWarmupValue);
    config.outerIters = static_cast<int>(config.outerItersValue);
    config.innerWarmup = static_cast<int>(config.innerWarmupValue);
    config.innerIters = static_cast<int>(config.innerItersValue);
    return true;
}

template <typename T>
bool LoadDeviceBaselineConfig(const DeviceBaselineEnvNames& names, DeviceBaselineConfig& config)
{
    config.transferBytesValue = ReadEnvUint64(names.transferBytes, 128 * kBytesPerKiB);
    config.blockDivisorValue = ReadEnvUint64(names.blockDivisor, 1);
    config.queueNumValue = ReadEnvUint64(names.queueNum, 1);
    config.postCountValue = ReadEnvUint64(names.postCount, 1);
    config.blockNumValue = ReadEnvUint64(names.blockNum, 1);
    config.outerWarmupValue = ReadEnvUint64(names.outerWarmup, 1);
    config.outerItersValue = ReadEnvUint64(names.outerIters, 5);
    config.innerWarmupValue = ReadEnvUint64(names.innerWarmup, 1);
    config.innerItersValue = ReadEnvUint64(names.innerIters, 10);
    config.waitEachEvent = ReadEnvUint64(names.waitEachEvent, 0) != 0;
    return FinalizeDeviceBaselineConfig<T>(config);
}

inline void PrintDeviceBaselineConfig(const char* instr, const DeviceBaselineConfig& config)
{
    std::cout << "\n================ " << instr << " Device Baseline ================" << std::endl;
    std::cout << "bytes=" << config.transferBytes << " block_bytes=" << config.blockBytes
              << " queue_num=" << config.queueNum << " block_num=" << config.blockNum
              << " post_count=" << config.postCount << " outer_warmup=" << config.outerWarmup
              << " outer_iters=" << config.outerIters << " inner_warmup=" << config.innerWarmup
              << " inner_iters=" << config.innerIters << " wait_each_event=" << config.waitEachEvent
              << " verify_each_outer=1" << std::endl;
}

inline void PrintDeviceBaselineResult(const char* instr, const DeviceBaselineConfig& config, uint64_t aggregateCycles)
{
    const double kernelTotalCycles =
        static_cast<double>(aggregateCycles) / static_cast<double>(config.outerIters * config.innerIters);
    const double bandwidthGBps =
        static_cast<double>(config.totalBytes) / (kernelTotalCycles * kDeviceCycleSeconds) / 1.0e9;
    std::cout << "[VERIFY] instr=" << instr << " status=PASS checked_blocks=" << config.blockNum
              << " checked_posts_per_block=" << config.postCount << std::endl;
    std::cout << std::fixed << std::setprecision(4) << "[DEVICE_BASELINE] instr=" << instr
              << " bytes=" << config.transferBytes << " block_divisor=" << config.blockDivisorValue
              << " block_bytes=" << config.blockBytes << " queue_num=" << config.queueNum
              << " block_num=" << config.blockNum << " sqe_num_per_post=" << config.blockDivisorValue
              << " sqe_num_per_block=" << config.blockDivisorValue * config.postCountValue
              << " sqe_num_total=" << config.blockDivisorValue * config.postCountValue * config.blockNumValue
              << " post_count=" << config.postCount << " outer_warmup=" << config.outerWarmup
              << " outer_iters=" << config.outerIters << " inner_warmup=" << config.innerWarmup
              << " inner_iters=" << config.innerIters << " wait_each_event=" << config.waitEachEvent
              << " verify_each_outer=1 kernel_total_cycles=" << kernelTotalCycles
              << " device_bandwidth_GBps=" << bandwidthGBps << std::endl;
}

template <typename T>
struct DeviceBaselineResources {
    T* inputHost = nullptr;
    T* verifyHost = nullptr;
    uint64_t* profileBufDev = nullptr;
    uint64_t* profileBufHost = nullptr;
    T* shmem = nullptr;
    T* sendShmem = nullptr;
    T* recvShmem = nullptr;
    SdmaWorkspaceManager sdmaMgr;

    bool Allocate(size_t totalBytes, size_t profileCount)
    {
        return CheckAclCall(
                   aclrtMallocHost(reinterpret_cast<void**>(&inputHost), totalBytes),
                   "aclrtMallocHost(device baseline input)") &&
               CheckAclCall(
                   aclrtMallocHost(reinterpret_cast<void**>(&verifyHost), totalBytes),
                   "aclrtMallocHost(device baseline verify)") &&
               CheckAclCall(
                   aclrtMalloc(
                       reinterpret_cast<void**>(&profileBufDev), profileCount * sizeof(uint64_t),
                       ACL_MEM_MALLOC_HUGE_FIRST),
                   "aclrtMalloc(device baseline profile)") &&
               CheckAclCall(
                   aclrtMallocHost(reinterpret_cast<void**>(&profileBufHost), profileCount * sizeof(uint64_t)),
                   "aclrtMallocHost(device baseline profile)");
    }

    void MapWindow(TestContext& ctx, int rankId, size_t totalBytes, size_t requiredWindowBytes)
    {
        uint64_t localWinBase = ctx.hostCtx.windowsIn[rankId];
        size_t winOffset = 0;
        shmem = reinterpret_cast<T*>(WindowAlloc(localWinBase, winOffset, requiredWindowBytes));
        auto* shmemBytes = reinterpret_cast<uint8_t*>(shmem);
        sendShmem = reinterpret_cast<T*>(shmemBytes + 64 * sizeof(int32_t));
        recvShmem = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(sendShmem) + totalBytes);
    }

    void Release(TestContext& ctx, bool finalizeSdma)
    {
        if (inputHost != nullptr) {
            ctx.aclStatus |= aclrtFreeHost(inputHost);
        }
        if (verifyHost != nullptr) {
            ctx.aclStatus |= aclrtFreeHost(verifyHost);
        }
        if (profileBufDev != nullptr) {
            ctx.aclStatus |= aclrtFree(profileBufDev);
        }
        if (profileBufHost != nullptr) {
            ctx.aclStatus |= aclrtFreeHost(profileBufHost);
        }
        if (finalizeSdma) {
            sdmaMgr.Finalize();
        }
    }
};

template <typename T>
void FillDeviceBaselinePattern(
    T* buffer, int rankId, int patternOuter, bool measured, uint32_t blockNum, uint32_t postCount, size_t bytesPerBlock,
    size_t elemCount);

template <typename T>
bool PrepareDeviceBaselineSource(
    int rankId, int sourceRank, int patternOuter, bool measured, const DeviceBaselineConfig& config,
    DeviceBaselineResources<T>& resources)
{
    bool sourceOk = true;
    if (rankId == sourceRank) {
        FillDeviceBaselinePattern(
            resources.inputHost, rankId, patternOuter, measured, config.blockNum, config.postCount,
            config.bytesPerBlock, config.elemCount);
        sourceOk = CheckAclCall(
            aclrtMemcpy(
                resources.sendShmem, config.totalBytes, resources.inputHost, config.totalBytes,
                ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy(device baseline source)");
    }
    char sourceStatus = sourceOk ? 1 : 0;
    CommMpiBcast(&sourceStatus, 1, COMM_MPI_CHAR, sourceRank);
    return sourceStatus != 0;
}

template <typename T>
void FillDeviceBaselinePattern(
    T* buffer, int rankId, int patternOuter, bool measured, uint32_t blockNum, uint32_t postCount, size_t bytesPerBlock,
    size_t elemCount)
{
    for (uint32_t block = 0; block < blockNum; ++block) {
        for (uint32_t post = 0; post < postCount; ++post) {
            for (size_t i = 0; i < elemCount; ++i) {
                const size_t index =
                    static_cast<size_t>(block) * bytesPerBlock / sizeof(T) + static_cast<size_t>(post) * elemCount + i;
                buffer[index] = DeviceBaselinePattern<T>(rankId, patternOuter, measured, block, post, i);
            }
        }
    }
}

template <typename T>
bool VerifyDeviceBaselinePattern(
    const T* buffer, int expectedRank, int patternOuter, bool measured, int outer, uint32_t blockNum,
    uint32_t postCount, size_t bytesPerBlock, size_t elemCount)
{
    for (uint32_t block = 0; block < blockNum; ++block) {
        for (uint32_t post = 0; post < postCount; ++post) {
            for (size_t i = 0; i < elemCount; ++i) {
                const size_t index =
                    static_cast<size_t>(block) * bytesPerBlock / sizeof(T) + static_cast<size_t>(post) * elemCount + i;
                const T expected = DeviceBaselinePattern<T>(expectedRank, patternOuter, measured, block, post, i);
                if (buffer[index] != expected) {
                    std::cerr << "[VERIFY] status=FAIL outer=" << outer << " block=" << block << " post=" << post
                              << " index=" << i << " expected=" << static_cast<float>(expected)
                              << " actual=" << static_cast<float>(buffer[index]) << std::endl;
                    return false;
                }
            }
        }
    }
    return true;
}

AICORE inline bool WaitAsyncEvents(
    pto::comm::AsyncEvent* events, uint32_t postedCount, pto::comm::AsyncSession& session, bool waitEachEvent)
{
    if (!waitEachEvent) {
        return postedCount == 0 || events[postedCount - 1].Wait(session);
    }
    bool waitsOk = true;
    for (uint32_t post = 0; post < postedCount; ++post) {
        const bool waitOk = events[post].Wait(session);
        waitsOk = waitOk && waitsOk;
    }
    return waitsOk;
}

template <bool IsTGet, typename T, typename Global, typename Shape, typename Stride>
AICORE inline uint32_t PostAsyncTransfers(
    __gm__ T* localBuffer, __gm__ T* remoteBuffer, const Shape& shape, const Stride& stride,
    pto::comm::AsyncSession& session, pto::comm::AsyncEvent* events, uint32_t postCount, int elemCount, bool& success)
{
    uint32_t postedCount = 0;
    for (uint32_t post = 0; post < postCount; ++post) {
        const size_t offset = static_cast<size_t>(post) * static_cast<size_t>(elemCount);
        Global localGlobal(localBuffer + offset, shape, stride);
        Global remoteGlobal(remoteBuffer + offset, shape, stride);
        if constexpr (IsTGet) {
            events[post] = pto::comm::TGET_ASYNC(localGlobal, remoteGlobal, session);
        } else {
            events[post] = pto::comm::TPUT_ASYNC(remoteGlobal, localGlobal, session);
        }
        if (!events[post].valid()) {
            success = false;
            break;
        }
        postedCount = post + 1;
    }
    return postedCount;
}

template <bool IsTGet, typename T, typename Global, typename Shape, typename Stride>
AICORE inline bool RunAsyncIterations(
    __gm__ T* localBuffer, __gm__ T* remoteBuffer, const Shape& shape, const Stride& stride,
    pto::comm::AsyncSession& session, pto::comm::AsyncEvent* events, int iterations, uint32_t postCount, int elemCount,
    bool waitEachEvent)
{
    bool success = true;
    for (int iter = 0; iter < iterations && success; ++iter) {
        const uint32_t postedCount = PostAsyncTransfers<IsTGet, T, Global>(
            localBuffer, remoteBuffer, shape, stride, session, events, postCount, elemCount, success);
        const bool waitsOk = WaitAsyncEvents(events, postedCount, session, waitEachEvent);
        success = success && waitsOk;
        pipe_barrier(PIPE_ALL);
    }
    return success;
}

template <bool IsTGet, typename T>
__global__ AICORE void ProfileAsyncDeviceBaselineKernel(
    __gm__ uint64_t* profile, __gm__ T* recvShmem, __gm__ T* shmem, int nranks, int rootRank, int peerRank,
    int elemCount, int warmupIters, int timedIters, __gm__ CommDeviceContext* hcclCtx, __gm__ uint8_t* sdmaWorkspace,
    uint32_t sdmaSyncId, uint32_t queueNum, uint32_t blockBytes, uint32_t postCount, bool waitEachEvent)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;
    if (nranks <= 0 || elemCount <= 0 || postCount == 0 || postCount > kMaxDeviceBaselinePosts) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    __gm__ uint8_t* shmemBytes = reinterpret_cast<__gm__ uint8_t*>(shmem);
    __gm__ T* sendShmem = reinterpret_cast<__gm__ T*>(shmemBytes + 64 * sizeof(int32_t));
    if (static_cast<int>(hcclCtx->rankId) == rootRank) {
        const uint32_t blockId = static_cast<uint32_t>(get_block_idx());
        const size_t blockOffset =
            static_cast<size_t>(blockId) * static_cast<size_t>(postCount) * static_cast<size_t>(elemCount);
        __gm__ uint64_t* blockProfile = profile == nullptr ? nullptr : profile + blockId * 2;
        const ShapeDyn shape(1, 1, 1, 1, elemCount);
        const StrideDyn stride(elemCount, elemCount, elemCount, elemCount, 1);
        ScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        pto::comm::sdma::SdmaBaseConfig baseConfig{blockBytes, 0, queueNum};
        if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId, baseConfig)) {
            pipe_barrier(PIPE_ALL);
            return;
        }
        __gm__ T* localBuffer;
        __gm__ T* remoteBuffer;
        if constexpr (IsTGet) {
            localBuffer = recvShmem + blockOffset;
            remoteBuffer = CommRemotePtr(hcclCtx, sendShmem, peerRank) + blockOffset;
        } else {
            localBuffer = sendShmem + blockOffset;
            remoteBuffer = CommRemotePtr(hcclCtx, recvShmem, peerRank) + blockOffset;
        }
        pto::comm::AsyncEvent events[kMaxDeviceBaselinePosts];
        bool success = RunAsyncIterations<IsTGet, T, Global>(
            localBuffer, remoteBuffer, shape, stride, session, events, warmupIters, postCount, elemCount,
            waitEachEvent);
        const uint64_t begin = GetSyscnt();
        success = success && RunAsyncIterations<IsTGet, T, Global>(
                                 localBuffer, remoteBuffer, shape, stride, session, events, timedIters, postCount,
                                 elemCount, waitEachEvent);
        const uint64_t end = GetSyscnt();
        if (blockProfile != nullptr) {
            blockProfile[0] = end - begin;
            blockProfile[1] = success ? 1 : 0;
        }
    }
    pipe_barrier(PIPE_ALL);
}

template <bool IsTGet, typename T>
bool LaunchDeviceBaselineKernel(
    TestContext& ctx, uint64_t* profileBuf, T* recvShmem, T* shmem, int nranks, int rootRank, int peerRank,
    int elemCount, int warmupIters, int timedIters, uint8_t* sdmaWorkspace, uint32_t queueNum, uint32_t blockBytes,
    uint32_t postCount, uint32_t blockNum, bool waitEachEvent)
{
    ProfileAsyncDeviceBaselineKernel<IsTGet, T><<<blockNum, nullptr, ctx.stream>>>(
        profileBuf, recvShmem, shmem, nranks, rootRank, peerRank, elemCount, warmupIters, timedIters, ctx.deviceCtx,
        sdmaWorkspace, 0, queueNum, blockBytes, postCount, waitEachEvent);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);
    if (ctx.aclStatus != 0) {
        std::cerr << "[ERROR] aclrtSynchronizeStream failed in device baseline kernel: " << ctx.aclStatus << std::endl;
        return false;
    }
    return true;
}

template <typename T, typename DirectionPolicy>
bool RunDeviceBaselineOuterIterations(
    int rankId, int nRanks, int rootRank, int peerRank, TestContext& ctx, const DeviceBaselineConfig& config,
    DeviceBaselineResources<T>& resources, uint64_t& aggregateCycles)
{
    const int totalOuter = config.outerWarmup + config.outerIters;
    for (int outer = 0; outer < totalOuter; ++outer) {
        const bool measured = outer >= config.outerWarmup;
        const int patternOuter = measured ? outer - config.outerWarmup : outer;
        const int sourceRank = DirectionPolicy::SourceRank(rootRank, peerRank);
        if (!PrepareDeviceBaselineSource(rankId, sourceRank, patternOuter, measured, config, resources) ||
            !DirectionPolicy::template Reset<T>(rankId, rootRank, peerRank, config, resources)) {
            return false;
        }
        HcclHostBarrier(ctx.comm, ctx.stream);
        const bool launchOk = LaunchDeviceBaselineKernel<DirectionPolicy::kIsTGet>(
            ctx, resources.profileBufDev, resources.recvShmem, resources.shmem, nRanks, rootRank, peerRank,
            static_cast<int>(config.elemCount), config.innerWarmup, config.innerIters,
            rankId == rootRank ? reinterpret_cast<uint8_t*>(resources.sdmaMgr.GetWorkspaceAddr()) : nullptr,
            config.queueNum, config.blockBytes, config.postCount, config.blockNum, config.waitEachEvent);
        if (!launchOk) {
            return false;
        }
        HcclHostBarrier(ctx.comm, ctx.stream);
        uint64_t outerCriticalCycles = 0;
        if (!DirectionPolicy::template Complete<T>(
                rankId, rootRank, peerRank, patternOuter, measured, outer, config, resources, outerCriticalCycles)) {
            return false;
        }
        if (measured && rankId == rootRank) {
            aggregateCycles += outerCriticalCycles;
        }
    }
    return true;
}

inline bool CheckDeviceBaselineWindow(const TestContext& ctx, int rankId, int rootRank, size_t requiredWindowBytes)
{
    if (ctx.hostCtx.winSize == 0 || requiredWindowBytes <= ctx.hostCtx.winSize) {
        return true;
    }
    if (rankId == rootRank) {
        std::cerr << "[ERROR] device baseline requires " << requiredWindowBytes
                  << " symmetric window bytes, available=" << ctx.hostCtx.winSize << std::endl;
    }
    return false;
}

template <typename T, typename DirectionPolicy>
bool RunDeviceBaselineKernel(
    int rankId, int nRanks, int nDevices, int firstRankId, int firstDeviceId, const HcclRootInfo* rootInfo,
    const DeviceBaselineEnvNames& envNames, const char* envLabel, const char* instr)
{
    const int rootRank = firstRankId;
    if (nRanks < 2) {
        return false;
    }
    TestContext ctx;
    if (!ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo)) {
        return false;
    }
    const int peerRank = (rootRank + 1) % nRanks;
    DeviceBaselineConfig config{};
    if (!LoadDeviceBaselineConfig<T>(envNames, config)) {
        if (rankId == rootRank) {
            std::cerr << "[ERROR] invalid " << envLabel << " device baseline environment" << std::endl;
        }
        return ctx.Finalize() && false;
    }
    DeviceBaselineResources<T> resources;
    const size_t requiredWindowBytes = 64 * sizeof(int32_t) + 2 * config.totalBytes;
    bool ok = CheckDeviceBaselineWindow(ctx, rankId, rootRank, requiredWindowBytes);
    ok = ok && resources.Allocate(config.totalBytes, config.profileCount);
    if (ok) {
        resources.MapWindow(ctx, rankId, config.totalBytes, requiredWindowBytes);
    }
    if (ok && rankId == rootRank && !resources.sdmaMgr.Init()) {
        std::cerr << "[ERROR] SdmaWorkspaceManager Init failed" << std::endl;
        ok = false;
    }
    if (ok && rankId == rootRank) {
        PrintDeviceBaselineConfig(instr, config);
    }
    uint64_t aggregateCycles = 0;
    ok = ok && RunDeviceBaselineOuterIterations<T, DirectionPolicy>(
                   rankId, nRanks, rootRank, peerRank, ctx, config, resources, aggregateCycles);
    if (ok && rankId == rootRank) {
        PrintDeviceBaselineResult(instr, config, aggregateCycles);
    }
    resources.Release(ctx, rankId == rootRank);
    return ctx.Finalize() && ok;
}

} // namespace benchmark

#endif // BENCHMARK_COMMON_HPP_
