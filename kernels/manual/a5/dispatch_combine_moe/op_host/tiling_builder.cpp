/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "tiling_builder.hpp"

#include <algorithm>
#include <stdexcept>

#include "op_kernel/utils/const_args.hpp"
#include "token_reorder/routing/moe_init_routing_quant_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024U * 1024U;

void FillCoCTiling(CoCTiling &coc, const CaseConfig &cfg)
{
    coc.m0 = 128;
    coc.k0 = 256;
    coc.n0 = 256;
    coc.swizzleDirect = 1;
    coc.swizzleOffset = 7;
    coc.ubMoveNum = 16 * 1024;
    coc.pValue = 1;
    coc.commNpuSplit = cfg.world_size;
    coc.commDataSplit = 1;
    coc.lenPerLoop = coc.m0 * coc.n0 / 2;
}

auto *GetPlatform(const CaseConfig &cfg)
{
    if (cfg.soc_version.empty()) {
        return platform_ascendc::PlatformAscendCManager::GetInstance();
    }
    return platform_ascendc::PlatformAscendCManager::GetInstance(cfg.soc_version.c_str());
}

uint32_t GetAivNum(const CaseConfig &cfg)
{
    auto *platform = GetPlatform(cfg);
    if (platform == nullptr) {
        throw std::runtime_error("PlatformAscendCManager::GetInstance failed for " + cfg.soc_version);
    }
    return platform->GetCoreNumAiv();
}

uint32_t GetBlockDim(const CaseConfig &cfg, uint32_t aivNum)
{
    auto *platform = GetPlatform(cfg);
    if (platform == nullptr) {
        throw std::runtime_error("PlatformAscendCManager::GetInstance failed for " + cfg.soc_version);
    }
    return platform->CalcTschBlockDim(aivNum, platform->GetCoreNumAic(), aivNum);
}

void ValidateRemoteWindowCapacity(const CaseConfig &cfg, const StandaloneRankRuntime &runtime)
{
    const uint64_t segmentBytes = runtime.hccl.host_remote_window_ctx.windowBytes;
    if (segmentBytes <= 3 * MB_SIZE) {
        throw std::runtime_error("HCCL remote window is too small for dispatch_combine_moe layout");
    }

    const uint64_t offsetPeerPerTokenScale = ((segmentBytes / 3) + 512 - 1) / 512 * 512;
    const uint64_t offsetD = offsetPeerPerTokenScale + MB_SIZE;
    const uint64_t offsetPeerTokenPerExpert = segmentBytes - 2 * MB_SIZE;
    const uint64_t peerPerTokenScaleBytes = static_cast<uint64_t>(cfg.max_output_size) * sizeof(float);
    const uint64_t dBytes = static_cast<uint64_t>(cfg.max_output_size) * cfg.k * sizeof(int16_t);
    const uint64_t peerTokenPerExpertBytes =
        static_cast<uint64_t>(cfg.world_size) * cfg.expert_per_rank * sizeof(int32_t);
    const uint64_t signalBaseBytes = segmentBytes - MB_SIZE;

    if (offsetPeerPerTokenScale + peerPerTokenScaleBytes > offsetD) {
        throw std::runtime_error("HCCL remote window per-token-scale region is too small");
    }
    if (offsetD + dBytes > offsetPeerTokenPerExpert) {
        throw std::runtime_error("HCCL remote window dispatch output region is too small");
    }
    if (offsetPeerTokenPerExpert + peerTokenPerExpertBytes > signalBaseBytes) {
        throw std::runtime_error("HCCL remote window token-count region overlaps signal region");
    }
}

void FillInitRoutingTiling(CoCTiling &coc, const CaseConfig &cfg, uint32_t aivNum)
{
    optiling::MoeInitRoutingQuantTilingBase tilingBase;
    const int64_t inputXDtypeSize = sizeof(int16_t);
    const int64_t scaleDim0 = 0;
    const int64_t ubSize = 196352;
    const int64_t expertCapacity = 0;
    const int64_t expertNum = static_cast<int64_t>(cfg.expert_per_rank) * cfg.world_size + 1;
    const int64_t activeNum = 0;
    const int64_t dropPadMode = 0;
    const int64_t expertTokensCountOrCumsumFlag = 2;
    const bool expertTokensBeforeCapacityFlag = false;
    const int64_t quantMode = 1;
    const int64_t aivNumInitRouting = static_cast<int64_t>(aivNum);
    if (!tilingBase.DoTiling(cfg.m, cfg.k, cfg.topk, expertCapacity, expertNum, activeNum, dropPadMode,
                             expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag, inputXDtypeSize, quantMode,
                             scaleDim0, aivNumInitRouting, ubSize)) {
        throw std::runtime_error("MoeInitRoutingQuantTilingBase::DoTiling failed");
    }
    coc.initRoutingQuantTilingKey = tilingBase.tilingKey_;
    coc.moeInitRoutingQuantTilingData = tilingBase.quantTilingData;
}
} // namespace

DispatchFFNCombineBuildResult BuildDispatchFFNCombineTiling(const CaseConfig &cfg, const StandaloneRankRuntime &runtime)
{
    ValidateRemoteWindowCapacity(cfg, runtime);
    DispatchFFNCombineBuildResult result;
    auto &info = result.tiling.dispatchFFNCombineInfo;
    info.M = cfg.m;
    info.K = cfg.k;
    info.N = cfg.n;
    info.topK = cfg.topk;
    info.expertPerRank = cfg.expert_per_rank;
    info.worldSize = cfg.world_size;
    info.maxOutputSize = cfg.max_output_size;
    info.isTransposeB = 0;
    info.isWeightNz = 1;
    info.listLen = cfg.list_len;
    info.aivNum = GetAivNum(cfg);
    info.totalUbSize = 196352;

    FillCoCTiling(result.tiling.cocTiling, cfg);
    FillInitRoutingTiling(result.tiling.cocTiling, cfg, info.aivNum);

    result.tiling.runtimeInfo.remoteWindowContext = reinterpret_cast<uint64_t>(runtime.hccl.remote_window_ctx);
    result.tiling.runtimeInfo.rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    result.tiling.runtimeInfo.rankSize = static_cast<uint32_t>(runtime.hccl.world_size);

    const uint32_t n2 = cfg.k;
    const uint32_t k2 = cfg.n / 2;
    const uint64_t cocWorkspace =
        ((cfg.m + 255) / 256) * 256 * cfg.topk * sizeof(int32_t) +
        static_cast<uint64_t>(cfg.world_size) * cfg.world_size * cfg.expert_per_rank * sizeof(int32_t) * 3 +
        static_cast<uint64_t>(cfg.max_output_size) * sizeof(float) * 2 +
        static_cast<uint64_t>(cfg.max_output_size) * cfg.n * sizeof(int16_t) +
        static_cast<uint64_t>(cfg.max_output_size) * n2 * sizeof(int16_t) +
        static_cast<uint64_t>(cfg.max_output_size) * cfg.k * sizeof(int8_t) +
        static_cast<uint64_t>(cfg.max_output_size) * k2 * sizeof(int8_t) +
        static_cast<uint64_t>(cfg.world_size) * sizeof(int32_t) * 16 +
        static_cast<uint64_t>(cfg.expert_per_rank + cfg.world_size) * sizeof(int32_t) * 16;

    result.block_dim = GetBlockDim(cfg, info.aivNum);
    result.workspace_bytes = SYSTEM_NEED_WORKSPACE + cocWorkspace;
    result.tiling.launchConfig.blockDim = result.block_dim;
    result.tiling.launchConfig.tilingKey = 1000010;
    result.tiling.launchConfig.workspaceBytes = result.workspace_bytes;
    return result;
}
