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

#include "../benchmark_common.hpp"

using benchmark::CheckAclCall;
using benchmark::DeviceBaselineConfig;
using benchmark::DeviceBaselineResources;

constexpr benchmark::DeviceBaselineEnvNames kTPutEnvNames{
    "TPUT_DEVICE_BASELINE_BYTES",           "TPUT_DEVICE_BASELINE_BLOCK_DIVISOR", "TPUT_DEVICE_BASELINE_QUEUE_NUM",
    "TPUT_DEVICE_BASELINE_POST_COUNT",      "TPUT_DEVICE_BASELINE_BLOCK_NUM",     "TPUT_DEVICE_BASELINE_OUTER_WARMUP",
    "TPUT_DEVICE_BASELINE_OUTER_ITERS",     "TPUT_DEVICE_BASELINE_INNER_WARMUP",  "TPUT_DEVICE_BASELINE_INNER_ITERS",
    "TPUT_DEVICE_BASELINE_WAIT_EACH_EVENT",
};

struct TPutDeviceBaselinePolicy {
    static constexpr bool kIsTGet = false;

    static int SourceRank(int rootRank, int) { return rootRank; }

    template <typename T>
    static bool Reset(
        int rankId, int rootRank, int peerRank, const DeviceBaselineConfig& config,
        DeviceBaselineResources<T>& resources)
    {
        bool resetOk = true;
        if (rankId == peerRank) {
            resetOk = CheckAclCall(
                aclrtMemset(resources.recvShmem, config.totalBytes, 0xFF, config.totalBytes),
                "aclrtMemset(device baseline dst)");
        }
        if (rankId == rootRank) {
            resetOk = CheckAclCall(
                          aclrtMemset(
                              resources.profileBufDev, config.profileCount * sizeof(uint64_t), 0,
                              config.profileCount * sizeof(uint64_t)),
                          "aclrtMemset(device baseline profile)") &&
                      resetOk;
        }
        char launchStatus = resetOk ? 1 : 0;
        CommMpiBcast(&launchStatus, 1, COMM_MPI_CHAR, rootRank);
        return launchStatus != 0;
    }

    template <typename T>
    static bool ReadProfile(
        int rankId, int rootRank, const DeviceBaselineConfig& config, DeviceBaselineResources<T>& resources,
        uint64_t& outerCriticalCycles)
    {
        bool profileOk = true;
        if (rankId == rootRank) {
            profileOk = CheckAclCall(
                aclrtMemcpy(
                    resources.profileBufHost, config.profileCount * sizeof(uint64_t), resources.profileBufDev,
                    config.profileCount * sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST),
                "aclrtMemcpy(device baseline profile)");
            for (uint32_t block = 0; profileOk && block < config.blockNum; ++block) {
                profileOk = resources.profileBufHost[static_cast<size_t>(block) * 2 + 1] == 1;
                outerCriticalCycles =
                    std::max(outerCriticalCycles, resources.profileBufHost[static_cast<size_t>(block) * 2]);
            }
        }
        char profileStatus = profileOk ? 1 : 0;
        CommMpiBcast(&profileStatus, 1, COMM_MPI_CHAR, rootRank);
        return profileStatus != 0;
    }

    template <typename T>
    static bool Verify(
        int rankId, int rootRank, int peerRank, int patternOuter, bool measured, int outer,
        const DeviceBaselineConfig& config, DeviceBaselineResources<T>& resources)
    {
        bool verifyOk = true;
        if (rankId == peerRank) {
            verifyOk = CheckAclCall(
                aclrtMemcpy(
                    resources.verifyHost, config.totalBytes, resources.recvShmem, config.totalBytes,
                    ACL_MEMCPY_DEVICE_TO_HOST),
                "aclrtMemcpy(device baseline verify)");
            verifyOk = verifyOk && benchmark::VerifyDeviceBaselinePattern(
                                       resources.verifyHost, rootRank, patternOuter, measured, outer, config.blockNum,
                                       config.postCount, config.bytesPerBlock, config.elemCount);
        }
        char verifyStatus = verifyOk ? 1 : 0;
        CommMpiBcast(&verifyStatus, 1, COMM_MPI_CHAR, peerRank);
        return verifyStatus != 0;
    }

    template <typename T>
    static bool Complete(
        int rankId, int rootRank, int peerRank, int patternOuter, bool measured, int outer,
        const DeviceBaselineConfig& config, DeviceBaselineResources<T>& resources, uint64_t& outerCriticalCycles)
    {
        const bool profileOk = ReadProfile(rankId, rootRank, config, resources, outerCriticalCycles);
        const bool verifyOk = Verify(rankId, rootRank, peerRank, patternOuter, measured, outer, config, resources);
        return profileOk && verifyOk;
    }
};

bool RunTPutDeviceBaseline(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo* rootInfo) {
            return benchmark::RunDeviceBaselineKernel<float, TPutDeviceBaselinePolicy>(
                rankId, n_ranks, n_devices, first_rank_id, first_device_id, rootInfo, kTPutEnvNames, "TPUT",
                "TPUT_ASYNC");
        });
}
