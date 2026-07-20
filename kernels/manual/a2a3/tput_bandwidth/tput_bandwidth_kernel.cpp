/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "tput_bandwidth_kernel.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

#include <pto/pto-inst.hpp>
#include "pto/common/pto_tile.hpp"
#include "pto/comm/async/sdma/sdma_types.hpp"
#include "common.hpp"

constexpr size_t kBytesPerKiB = 1024;
constexpr uint32_t kMaxDeviceBaselinePosts = 512;
constexpr double kDeviceCycleSeconds = 20.0e-9;

uint64_t ReadEnvUint64(const char* name, uint64_t defaultValue)
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

inline AICORE uint64_t get_syscnt()
{
    uint64_t syscnt;
    asm volatile("MOV %0, SYS_CNT\n" : "+l"(syscnt));
    return syscnt;
}

bool CheckAclCall(aclError ret, const char* op)
{
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] " << op << " failed: " << static_cast<int>(ret) << std::endl;
        return false;
    }
    return true;
}

template <typename T>
__global__ AICORE void ProfileTPutAsyncDeviceBaselineKernel(
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
        const size_t elemsPerBlock = static_cast<size_t>(postCount) * static_cast<size_t>(elemCount);
        const size_t blockOffset = static_cast<size_t>(blockId) * elemsPerBlock;
        __gm__ uint64_t* blockProfile = profile == nullptr ? nullptr : profile + blockId * 2;
        ShapeDyn shape(1, 1, 1, 1, elemCount);
        StrideDyn stride(elemCount, elemCount, elemCount, elemCount, 1);
        ScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        pto::comm::sdma::SdmaBaseConfig baseConfig{blockBytes, 0, queueNum};
        if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId, baseConfig)) {
            pipe_barrier(PIPE_ALL);
            return;
        }

        __gm__ T* blockSendShmem = sendShmem + blockOffset;
        __gm__ T* remoteRecvShmem = CommRemotePtr(hcclCtx, recvShmem, peerRank) + blockOffset;
        pto::comm::AsyncEvent events[kMaxDeviceBaselinePosts];
        bool success = true;

        for (int iter = 0; iter < warmupIters && success; ++iter) {
            uint32_t postedCount = 0;
            for (uint32_t post = 0; post < postCount; ++post) {
                const size_t offset = static_cast<size_t>(post) * static_cast<size_t>(elemCount);
                Global sendG(blockSendShmem + offset, shape, stride);
                Global remoteRecvG(remoteRecvShmem + offset, shape, stride);
                events[post] = pto::comm::TPUT_ASYNC(remoteRecvG, sendG, session);
                if (!events[post].valid()) {
                    success = false;
                    break;
                }
                postedCount = post + 1;
            }
            if (waitEachEvent) {
                bool waitsOk = true;
                for (uint32_t post = 0; post < postedCount; ++post) {
                    const bool waitOk = events[post].Wait(session);
                    waitsOk = waitOk && waitsOk;
                }
                success = success && waitsOk;
            } else if (postedCount > 0) {
                const bool waitOk = events[postedCount - 1].Wait(session);
                success = success && waitOk;
            }
            pipe_barrier(PIPE_ALL);
        }

        const uint64_t begin = get_syscnt();
        for (int iter = 0; iter < timedIters && success; ++iter) {
            uint32_t postedCount = 0;
            for (uint32_t post = 0; post < postCount; ++post) {
                const size_t offset = static_cast<size_t>(post) * static_cast<size_t>(elemCount);
                Global sendG(blockSendShmem + offset, shape, stride);
                Global remoteRecvG(remoteRecvShmem + offset, shape, stride);
                events[post] = pto::comm::TPUT_ASYNC(remoteRecvG, sendG, session);
                if (!events[post].valid()) {
                    success = false;
                    break;
                }
                postedCount = post + 1;
            }
            if (waitEachEvent) {
                bool waitsOk = true;
                for (uint32_t post = 0; post < postedCount; ++post) {
                    const bool waitOk = events[post].Wait(session);
                    waitsOk = waitOk && waitsOk;
                }
                success = success && waitsOk;
            } else if (postedCount > 0) {
                const bool waitOk = events[postedCount - 1].Wait(session);
                success = success && waitOk;
            }
            pipe_barrier(PIPE_ALL);
        }
        const uint64_t end = get_syscnt();
        if (blockProfile != nullptr) {
            blockProfile[0] = end - begin;
            blockProfile[1] = success ? 1 : 0;
        }
    }

    pipe_barrier(PIPE_ALL);
}

template <typename T>
bool LaunchDeviceBaselineKernel(
    TestContext& ctx, uint64_t* profileBuf, T* recvShmem, T* shmem, int nranks, int rootRank, int peerRank,
    int elemCount, int warmupIters, int timedIters, uint8_t* sdmaWorkspace, uint32_t queueNum, uint32_t blockBytes,
    uint32_t postCount, uint32_t blockNum, bool waitEachEvent)
{
    ProfileTPutAsyncDeviceBaselineKernel<T><<<blockNum, nullptr, ctx.stream>>>(
        profileBuf, recvShmem, shmem, nranks, rootRank, peerRank, elemCount, warmupIters, timedIters, ctx.deviceCtx,
        sdmaWorkspace, 0, queueNum, blockBytes, postCount, waitEachEvent);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);
    if (ctx.aclStatus != 0) {
        std::cerr << "[ERROR] aclrtSynchronizeStream failed in device baseline kernel: " << ctx.aclStatus << std::endl;
        return false;
    }
    return true;
}

template <typename T>
bool RunTPutDeviceBaselineKernel(
    int rankId, int nRanks, int nDevices, int firstRankId, int firstDeviceId, const HcclRootInfo* rootInfo)
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
    const uint64_t transferBytesValue = ReadEnvUint64("TPUT_DEVICE_BASELINE_BYTES", 128 * kBytesPerKiB);
    const uint64_t blockDivisorValue = ReadEnvUint64("TPUT_DEVICE_BASELINE_BLOCK_DIVISOR", 1);
    const uint64_t queueNumValue = ReadEnvUint64("TPUT_DEVICE_BASELINE_QUEUE_NUM", 1);
    const uint64_t postCountValue = ReadEnvUint64("TPUT_DEVICE_BASELINE_POST_COUNT", 1);
    const uint64_t blockNumValue = ReadEnvUint64("TPUT_DEVICE_BASELINE_BLOCK_NUM", 1);
    const uint64_t outerWarmupValue = ReadEnvUint64("TPUT_DEVICE_BASELINE_OUTER_WARMUP", 1);
    const uint64_t outerItersValue = ReadEnvUint64("TPUT_DEVICE_BASELINE_OUTER_ITERS", 5);
    const uint64_t innerWarmupValue = ReadEnvUint64("TPUT_DEVICE_BASELINE_INNER_WARMUP", 1);
    const uint64_t innerItersValue = ReadEnvUint64("TPUT_DEVICE_BASELINE_INNER_ITERS", 10);
    const bool waitEachEvent = ReadEnvUint64("TPUT_DEVICE_BASELINE_WAIT_EACH_EVENT", 0) != 0;

    const bool configValid =
        transferBytesValue > 0 && transferBytesValue <= UINT32_MAX && transferBytesValue % sizeof(T) == 0 &&
        blockDivisorValue > 0 && transferBytesValue % blockDivisorValue == 0 &&
        transferBytesValue / blockDivisorValue >= pto::comm::sdma::kSdmaMinTransferBytes && queueNumValue > 0 &&
        queueNumValue <= pto::comm::sdma::kSdmaMaxChannel && postCountValue > 0 &&
        postCountValue <= kMaxDeviceBaselinePosts && outerItersValue > 0 && blockNumValue > 0 &&
        blockNumValue <= pto::comm::sdma::kSdmaMaxChannel / queueNumValue && outerItersValue <= INT32_MAX &&
        outerWarmupValue <= INT32_MAX - outerItersValue && innerItersValue > 0 && innerItersValue <= INT32_MAX &&
        innerWarmupValue <= INT32_MAX && transferBytesValue <= SIZE_MAX / postCountValue &&
        transferBytesValue * postCountValue <= SIZE_MAX / blockNumValue;
    if (!configValid) {
        if (rankId == rootRank) {
            std::cerr << "[ERROR] invalid TPUT device baseline environment" << std::endl;
        }
        return ctx.Finalize() && false;
    }

    const size_t transferBytes = static_cast<size_t>(transferBytesValue);
    const size_t bytesPerBlock = transferBytes * static_cast<size_t>(postCountValue);
    const size_t totalBytes = bytesPerBlock * static_cast<size_t>(blockNumValue);
    const size_t elemCount = transferBytes / sizeof(T);
    const uint32_t blockBytes = static_cast<uint32_t>(transferBytesValue / blockDivisorValue);
    const uint32_t queueNum = static_cast<uint32_t>(queueNumValue);
    const uint32_t postCount = static_cast<uint32_t>(postCountValue);
    const uint32_t blockNum = static_cast<uint32_t>(blockNumValue);
    const int outerWarmup = static_cast<int>(outerWarmupValue);
    const int outerIters = static_cast<int>(outerItersValue);
    const int innerWarmup = static_cast<int>(innerWarmupValue);
    const int innerIters = static_cast<int>(innerItersValue);
    const size_t profileCount = static_cast<size_t>(blockNum) * 2;

    T* inputHost = nullptr;
    T* verifyHost = nullptr;
    uint64_t* profileBufDev = nullptr;
    uint64_t* profileBufHost = nullptr;
    T* shmem = nullptr;
    SdmaWorkspaceManager sdmaMgr;
    bool ok = false;

    do {
        const size_t requiredWindowBytes = 64 * sizeof(int32_t) + 2 * totalBytes;
        if (ctx.hostCtx.winSize != 0 && requiredWindowBytes > ctx.hostCtx.winSize) {
            if (rankId == rootRank) {
                std::cerr << "[ERROR] device baseline requires " << requiredWindowBytes
                          << " symmetric window bytes, available=" << ctx.hostCtx.winSize << std::endl;
            }
            break;
        }

        if (!CheckAclCall(
                aclrtMallocHost(reinterpret_cast<void**>(&inputHost), totalBytes),
                "aclrtMallocHost(device baseline input)") ||
            !CheckAclCall(
                aclrtMallocHost(reinterpret_cast<void**>(&verifyHost), totalBytes),
                "aclrtMallocHost(device baseline verify)") ||
            !CheckAclCall(
                aclrtMalloc(
                    reinterpret_cast<void**>(&profileBufDev), profileCount * sizeof(uint64_t),
                    ACL_MEM_MALLOC_HUGE_FIRST),
                "aclrtMalloc(device baseline profile)") ||
            !CheckAclCall(
                aclrtMallocHost(reinterpret_cast<void**>(&profileBufHost), profileCount * sizeof(uint64_t)),
                "aclrtMallocHost(device baseline profile)")) {
            break;
        }

        uint64_t localWinBase = ctx.hostCtx.windowsIn[rankId];
        size_t winOffset = 0;
        shmem = reinterpret_cast<T*>(WindowAlloc(localWinBase, winOffset, requiredWindowBytes));
        uint8_t* shmemBytes = reinterpret_cast<uint8_t*>(shmem);
        T* sendShmem = reinterpret_cast<T*>(shmemBytes + 64 * sizeof(int32_t));
        T* recvShmem = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(sendShmem) + totalBytes);

        if (rankId == rootRank && !sdmaMgr.Init()) {
            std::cerr << "[ERROR] SdmaWorkspaceManager Init failed" << std::endl;
            break;
        }

        if (rankId == rootRank) {
            std::cout << "\n================ TPUT_ASYNC Device Baseline ================" << std::endl;
            std::cout << "bytes=" << transferBytes << " block_bytes=" << blockBytes << " queue_num=" << queueNum
                      << " block_num=" << blockNum << " post_count=" << postCount << " outer_warmup=" << outerWarmup
                      << " outer_iters=" << outerIters << " inner_warmup=" << innerWarmup
                      << " inner_iters=" << innerIters << " wait_each_event=" << waitEachEvent << " verify_each_outer=1"
                      << std::endl;
        }

        uint64_t aggregateCycles = 0;
        bool runOk = true;
        const int totalOuter = outerWarmup + outerIters;
        for (int outer = 0; outer < totalOuter; ++outer) {
            const bool measured = outer >= outerWarmup;
            const int patternOuter = measured ? outer - outerWarmup : outer;
            bool sourceOk = true;
            if (rankId == rootRank) {
                for (uint32_t block = 0; block < blockNum; ++block) {
                    for (uint32_t post = 0; post < postCount; ++post) {
                        for (size_t i = 0; i < elemCount; ++i) {
                            const size_t index = static_cast<size_t>(block) * bytesPerBlock / sizeof(T) +
                                                 static_cast<size_t>(post) * elemCount + i;
                            inputHost[index] = DeviceBaselinePattern<T>(rankId, patternOuter, measured, block, post, i);
                        }
                    }
                }
                sourceOk = CheckAclCall(
                    aclrtMemcpy(sendShmem, totalBytes, inputHost, totalBytes, ACL_MEMCPY_HOST_TO_DEVICE),
                    "aclrtMemcpy(device baseline source)");
            }
            char sourceStatus = sourceOk ? 1 : 0;
            CommMpiBcast(&sourceStatus, 1, COMM_MPI_CHAR, rootRank);
            if (sourceStatus == 0) {
                runOk = false;
                break;
            }

            if ((rankId == peerRank &&
                 !CheckAclCall(
                     aclrtMemset(recvShmem, totalBytes, 0xFF, totalBytes), "aclrtMemset(device baseline dst)")) ||
                (rankId == rootRank &&
                 !CheckAclCall(
                     aclrtMemset(profileBufDev, profileCount * sizeof(uint64_t), 0, profileCount * sizeof(uint64_t)),
                     "aclrtMemset(device baseline profile)"))) {
                runOk = false;
            }
            char launchStatus = runOk ? 1 : 0;
            CommMpiBcast(&launchStatus, 1, COMM_MPI_CHAR, rootRank);
            if (launchStatus == 0) {
                runOk = false;
                break;
            }

            HcclHostBarrier(ctx.comm, ctx.stream);
            if (!LaunchDeviceBaselineKernel(
                    ctx, profileBufDev, recvShmem, shmem, nRanks, rootRank, peerRank, static_cast<int>(elemCount),
                    innerWarmup, innerIters,
                    rankId == rootRank ? reinterpret_cast<uint8_t*>(sdmaMgr.GetWorkspaceAddr()) : nullptr, queueNum,
                    blockBytes, postCount, blockNum, waitEachEvent)) {
                runOk = false;
                break;
            }
            HcclHostBarrier(ctx.comm, ctx.stream);

            bool profileOk = true;
            uint64_t outerCriticalCycles = 0;
            if (rankId == rootRank) {
                profileOk = CheckAclCall(
                    aclrtMemcpy(
                        profileBufHost, profileCount * sizeof(uint64_t), profileBufDev, profileCount * sizeof(uint64_t),
                        ACL_MEMCPY_DEVICE_TO_HOST),
                    "aclrtMemcpy(device baseline profile)");
                for (uint32_t block = 0; profileOk && block < blockNum; ++block) {
                    profileOk = profileBufHost[static_cast<size_t>(block) * 2 + 1] == 1;
                    outerCriticalCycles = std::max(outerCriticalCycles, profileBufHost[static_cast<size_t>(block) * 2]);
                }
            }
            char profileStatus = profileOk ? 1 : 0;
            CommMpiBcast(&profileStatus, 1, COMM_MPI_CHAR, rootRank);

            bool verifyOk = true;
            if (rankId == peerRank) {
                verifyOk = CheckAclCall(
                    aclrtMemcpy(verifyHost, totalBytes, recvShmem, totalBytes, ACL_MEMCPY_DEVICE_TO_HOST),
                    "aclrtMemcpy(device baseline verify)");
                for (uint32_t block = 0; verifyOk && block < blockNum; ++block) {
                    for (uint32_t post = 0; verifyOk && post < postCount; ++post) {
                        for (size_t i = 0; verifyOk && i < elemCount; ++i) {
                            const size_t index = static_cast<size_t>(block) * bytesPerBlock / sizeof(T) +
                                                 static_cast<size_t>(post) * elemCount + i;
                            const T expected =
                                DeviceBaselinePattern<T>(rootRank, patternOuter, measured, block, post, i);
                            if (verifyHost[index] != expected) {
                                std::cerr << "[VERIFY] status=FAIL outer=" << outer << " block=" << block
                                          << " post=" << post << " index=" << i
                                          << " expected=" << static_cast<float>(expected)
                                          << " actual=" << static_cast<float>(verifyHost[index]) << std::endl;
                                verifyOk = false;
                            }
                        }
                    }
                }
            }
            char verifyStatus = verifyOk ? 1 : 0;
            CommMpiBcast(&verifyStatus, 1, COMM_MPI_CHAR, peerRank);
            if (profileStatus == 0 || verifyStatus == 0) {
                runOk = false;
                break;
            }
            if (measured && rankId == rootRank) {
                aggregateCycles += outerCriticalCycles;
            }
        }

        if (runOk && rankId == rootRank) {
            const double kernelTotalCycles =
                static_cast<double>(aggregateCycles) / static_cast<double>(outerIters * innerIters);
            const double batchBytes = static_cast<double>(totalBytes);
            const double bandwidthGBps = batchBytes / (kernelTotalCycles * kDeviceCycleSeconds) / 1.0e9;
            std::cout << "[VERIFY] instr=TPUT_ASYNC status=PASS checked_blocks=" << blockNum
                      << " checked_posts_per_block=" << postCount << std::endl;
            std::cout << std::fixed << std::setprecision(4)
                      << "[DEVICE_BASELINE] instr=TPUT_ASYNC bytes=" << transferBytes
                      << " block_divisor=" << blockDivisorValue << " block_bytes=" << blockBytes
                      << " queue_num=" << queueNum << " block_num=" << blockNum
                      << " sqe_num_per_post=" << blockDivisorValue
                      << " sqe_num_per_block=" << blockDivisorValue * postCountValue
                      << " sqe_num_total=" << blockDivisorValue * postCountValue * blockNumValue
                      << " post_count=" << postCount << " outer_warmup=" << outerWarmup << " outer_iters=" << outerIters
                      << " inner_warmup=" << innerWarmup << " inner_iters=" << innerIters
                      << " wait_each_event=" << waitEachEvent << " verify_each_outer=1"
                      << " kernel_total_cycles=" << kernelTotalCycles << " device_bandwidth_GBps=" << bandwidthGBps
                      << std::endl;
        }
        ok = runOk;
    } while (false);

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
    if (rankId == rootRank) {
        sdmaMgr.Finalize();
    }
    return ctx.Finalize() && ok;
}

bool RunTPutDeviceBaseline(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunTPutDeviceBaselineKernel<float>(
                rankId, n_ranks, n_devices, first_rank_id, first_device_id, rootInfo);
        });
}
