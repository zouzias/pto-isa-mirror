/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>

#include "tiling/platform/platform_ascendc.h"

#include "utils.h"
#include "aclrtlaunch_moe_grouped_dual_proj_custom.h"
#include "aclrtlaunch_moe_grouped_dual_proj_packed_custom.h"
#include "aclrtlaunch_moe_grouped_dual_proj_packed_nd_custom.h"
#include "aclrtlaunch_moe_grouped_gate_proj_custom.h"
#include "aclrtlaunch_moe_grouped_gemm_custom.h"
#include "aclrtlaunch_moe_grouped_gemm_with_intermediates_custom.h"
#include "aclrtlaunch_moe_grouped_up_proj_custom.h"
#include "../kernel/moe_grouped_ffn_custom.h"

namespace ascendc_path {

namespace {

struct WorkList {
    at::Tensor expertIds;
    at::Tensor rowOffsets;
    at::Tensor validRows;
    uint32_t blockDim;
};

// Work items for the projection kernels (gate/up).
// Each block computes a single [kMoeBaseM, kMoeBaseN] tile, identified by (expert, rowOffset, colTile).
// Keeping colTile explicit improves core utilization versus looping all N tiles inside one block.
struct ProjectionWorkList {
    at::Tensor groupOffsets;
    at::Tensor tileOffsets;
    uint32_t numExperts;
    uint32_t blockDim;
};

struct PaddedLayout {
    at::Tensor x;
    at::Tensor groupOffsetsCpu;
    std::vector<int32_t> originalOffsets;
    std::vector<int32_t> paddedOffsets;
    bool enabled;
};

struct Stage1Outputs {
    at::Tensor out;
    at::Tensor gateProj;
    at::Tensor upProj;
};

struct Projections {
    at::Tensor gateProj;
    at::Tensor upProj;
};

struct CachedGroupedWeight {
    at::Tensor weightKn;
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    const void *dataPtr;
    uint32_t version;
    bool hasVersion;
};

struct RoutingPlanKey {
    std::vector<int32_t> offsets;
    int64_t deviceIndex;

    bool operator==(const RoutingPlanKey &other) const
    {
        return deviceIndex == other.deviceIndex && offsets == other.offsets;
    }
};

struct RoutingPlanKeyHash {
    size_t operator()(const RoutingPlanKey &key) const
    {
        size_t hashValue = std::hash<int64_t>{}(key.deviceIndex);
        for (int32_t value : key.offsets) {
            hashValue ^= std::hash<int32_t>{}(value) + 0x9e3779b9 + (hashValue << 6) + (hashValue >> 2);
        }
        return hashValue;
    }
};

struct CachedRoutingPlan {
    at::Tensor groupOffsetsCpu;
    std::vector<int32_t> originalOffsets;
    std::vector<int32_t> paddedOffsets;
    WorkList workList;
    ProjectionWorkList projectionWorkList;
    bool enabled;
};

enum class ForwardImpl {
    GroupedMatmul,
    Fused,
    CustomSplit,
};

constexpr size_t kMaxGroupedWeightCacheEntries = 8;
constexpr size_t kMaxRoutingPlanCacheEntries = 16;

int32_t RoundUpToMultiple(int32_t value, int32_t align)
{
    if (value == 0) {
        return 0;
    }
    return ((value + align - 1) / align) * align;
}

at::Tensor MakeCpuIntTensor(const std::vector<int32_t> &values)
{
    auto options = at::TensorOptions().dtype(at::kInt).device(at::kCPU);
    auto tensor = at::empty({static_cast<int64_t>(values.size())}, options);
    if (!values.empty()) {
        std::memcpy(tensor.data_ptr<int32_t>(), values.data(), values.size() * sizeof(int32_t));
    }
    return tensor;
}

uint32_t GetTensorVersion(const at::Tensor &tensor, bool &hasVersion)
{
    const auto &versionCounter = tensor.unsafeGetTensorImpl()->version_counter();
    hasVersion = versionCounter.enabled();
    return hasVersion ? versionCounter.current_version() : 0;
}

bool MatchesCachedGroupedWeight(const CachedGroupedWeight &cachedWeight, const at::Tensor &weightDn)
{
    if (!cachedWeight.weightKn.defined() || cachedWeight.dataPtr != weightDn.data_ptr()) {
        return false;
    }

    const bool sameShape = cachedWeight.sizes == weightDn.sizes().vec();
    const bool sameStride = cachedWeight.strides == weightDn.strides().vec();
    if (!sameShape || !sameStride) {
        return false;
    }

    bool hasVersion = false;
    const uint32_t version = GetTensorVersion(weightDn, hasVersion);
    return cachedWeight.hasVersion == hasVersion && (!hasVersion || cachedWeight.version == version);
}

at::Tensor GetGroupedWeightKn(const at::Tensor &weightDn)
{
    static std::mutex cacheMutex;
    static std::unordered_map<const c10::TensorImpl *, CachedGroupedWeight> cache;

    const auto *tensorImpl = weightDn.unsafeGetTensorImpl();
    {
        std::lock_guard<std::mutex> guard(cacheMutex);
        const auto cachedIt = cache.find(tensorImpl);
        if (cachedIt != cache.end() && MatchesCachedGroupedWeight(cachedIt->second, weightDn)) {
            return cachedIt->second.weightKn;
        }
    }

    // GroupedMatmul rejects transpose views in separated-weight mode, so cache a
    // full [experts, hidden, inter] contiguous copy and slice that instead.
    auto weightKn = weightDn.transpose(1, 2).contiguous();

    CachedGroupedWeight cachedWeight;
    cachedWeight.weightKn = weightKn;
    cachedWeight.sizes = weightDn.sizes().vec();
    cachedWeight.strides = weightDn.strides().vec();
    cachedWeight.dataPtr = weightDn.data_ptr();
    cachedWeight.version = GetTensorVersion(weightDn, cachedWeight.hasVersion);

    {
        std::lock_guard<std::mutex> guard(cacheMutex);
        if (cache.size() >= kMaxGroupedWeightCacheEntries) {
            cache.clear();
        }
        cache[tensorImpl] = std::move(cachedWeight);
    }
    return weightKn;
}

std::vector<int32_t> ExtractCpuOffsets(const at::Tensor &groupOffsetsCpu)
{
    const auto *offsetPtr = groupOffsetsCpu.data_ptr<int32_t>();
    return std::vector<int32_t>(offsetPtr, offsetPtr + groupOffsetsCpu.numel());
}

WorkList BuildWorkList(const at::Tensor &groupOffsetsCpu)
{
    const int64_t numExperts = groupOffsetsCpu.size(0) - 1;
    const auto *offsetPtr = groupOffsetsCpu.data_ptr<int32_t>();

    std::vector<int32_t> expertIds;
    std::vector<int32_t> rowOffsets;
    std::vector<int32_t> validRows;

    const int64_t mTiles =
        (offsetPtr[numExperts] + static_cast<int32_t>(kMoeBaseM) - 1) / static_cast<int32_t>(kMoeBaseM);
    expertIds.reserve(numExperts * mTiles);
    rowOffsets.reserve(numExperts * mTiles);
    validRows.reserve(numExperts * mTiles);

    for (int64_t expert = 0; expert < numExperts; ++expert) {
        const int32_t start = offsetPtr[expert];
        const int32_t end = offsetPtr[expert + 1];
        for (int32_t row = start; row < end; row += kMoeBaseM) {
            const int32_t rows = std::min<int32_t>(kMoeBaseM, end - row);
            expertIds.push_back(static_cast<int32_t>(expert));
            rowOffsets.push_back(row);
            validRows.push_back(rows);
        }
    }

    WorkList workList;
    workList.blockDim = static_cast<uint32_t>(expertIds.size());
    workList.expertIds = CopyTensorHostToDevice(MakeCpuIntTensor(expertIds));
    workList.rowOffsets = CopyTensorHostToDevice(MakeCpuIntTensor(rowOffsets));
    workList.validRows = CopyTensorHostToDevice(MakeCpuIntTensor(validRows));
    return workList;
}

ProjectionWorkList BuildProjectionWorkList(const at::Tensor &groupOffsetsCpu)
{
    const int64_t numExperts = groupOffsetsCpu.size(0) - 1;
    const auto *offsetPtr = groupOffsetsCpu.data_ptr<int32_t>();

    std::vector<int32_t> tileOffsets;
    tileOffsets.reserve(static_cast<size_t>(numExperts) + 1);
    tileOffsets.push_back(0);

    for (int64_t expert = 0; expert < numExperts; ++expert) {
        const int32_t start = offsetPtr[expert];
        const int32_t end = offsetPtr[expert + 1];
        const int32_t rows = std::max<int32_t>(end - start, 0);
        const int32_t expertTiles = rows == 0 ? 0 : RoundUpToMultiple(rows, static_cast<int32_t>(kMoeBaseM)) /
                                                        static_cast<int32_t>(kMoeBaseM);
        tileOffsets.push_back(tileOffsets.back() + expertTiles);
    }

    ProjectionWorkList workList;
    workList.groupOffsets = CopyTensorHostToDevice(groupOffsetsCpu);
    workList.tileOffsets = CopyTensorHostToDevice(MakeCpuIntTensor(tileOffsets));
    workList.numExperts = static_cast<uint32_t>(numExperts);
    workList.blockDim =
        static_cast<uint32_t>(static_cast<int64_t>(tileOffsets.back()) * static_cast<int64_t>(kMoeNumNTiles));
    return workList;
}

CachedRoutingPlan BuildRoutingPlan(const std::vector<int32_t> &originalOffsets)
{
    const int64_t numExperts = static_cast<int64_t>(originalOffsets.size()) - 1;

    CachedRoutingPlan plan;
    plan.enabled = false;
    plan.originalOffsets = originalOffsets;
    plan.paddedOffsets.reserve(static_cast<size_t>(numExperts) + 1);
    plan.paddedOffsets.push_back(0);

    bool needsPadding = false;
    for (int64_t expert = 0; expert < numExperts; ++expert) {
        const int32_t start = originalOffsets[expert];
        const int32_t end = originalOffsets[expert + 1];
        const int32_t rows = end - start;
        const int32_t paddedRows = RoundUpToMultiple(rows, static_cast<int32_t>(kMoeBaseM));
        needsPadding |= (paddedRows != rows);
        plan.paddedOffsets.push_back(plan.paddedOffsets.back() + paddedRows);
    }

    if (!needsPadding) {
        plan.groupOffsetsCpu = MakeCpuIntTensor(plan.originalOffsets);
        plan.workList = BuildWorkList(plan.groupOffsetsCpu);
        plan.projectionWorkList = BuildProjectionWorkList(plan.groupOffsetsCpu);
        return plan;
    }

    plan.enabled = true;
    plan.groupOffsetsCpu = MakeCpuIntTensor(plan.paddedOffsets);
    plan.workList = BuildWorkList(plan.groupOffsetsCpu);
    // Projection kernels support ragged M via tileMeta, so keep the unpadded worklist
    // to avoid extra GM copies in MaybePadInputs/RestoreOutputLayout for the split path.
    plan.projectionWorkList = BuildProjectionWorkList(MakeCpuIntTensor(plan.originalOffsets));
    return plan;
}

CachedRoutingPlan GetCachedRoutingPlan(const at::Tensor &groupOffsetsCpu, const at::Tensor &x)
{
    static std::mutex cacheMutex;
    static std::unordered_map<RoutingPlanKey, CachedRoutingPlan, RoutingPlanKeyHash> cache;

    RoutingPlanKey key{ExtractCpuOffsets(groupOffsetsCpu), x.device().index()};
    {
        std::lock_guard<std::mutex> guard(cacheMutex);
        const auto cachedIt = cache.find(key);
        if (cachedIt != cache.end()) {
            return cachedIt->second;
        }
    }

    auto routingPlan = BuildRoutingPlan(key.offsets);
    {
        std::lock_guard<std::mutex> guard(cacheMutex);
        if (cache.size() >= kMaxRoutingPlanCacheEntries) {
            cache.clear();
        }
        cache[key] = routingPlan;
    }
    return routingPlan;
}

PaddedLayout MaybePadInputs(const at::Tensor &x, const CachedRoutingPlan &routingPlan)
{
    PaddedLayout layout;
    layout.enabled = routingPlan.enabled;
    layout.groupOffsetsCpu = routingPlan.groupOffsetsCpu;
    layout.originalOffsets = routingPlan.originalOffsets;
    layout.paddedOffsets = routingPlan.paddedOffsets;

    if (!layout.enabled) {
        layout.x = x;
        return layout;
    }

    layout.x = at::zeros({layout.paddedOffsets.back(), x.size(1)}, x.options());

    const int64_t numExperts = static_cast<int64_t>(layout.originalOffsets.size()) - 1;
    for (int64_t expert = 0; expert < numExperts; ++expert) {
        const int32_t start = layout.originalOffsets[expert];
        const int32_t end = layout.originalOffsets[expert + 1];
        const int32_t rows = end - start;
        if (rows == 0) {
            continue;
        }
        layout.x.narrow(0, layout.paddedOffsets[expert], rows).copy_(x.narrow(0, start, rows));
    }

    return layout;
}

at::Tensor RestoreOutputLayout(const at::Tensor &paddedOut, const PaddedLayout &layout, const at::Tensor &x)
{
    if (!layout.enabled) {
        return paddedOut;
    }

    auto out = at::empty({x.size(0), kMoeInterSize}, x.options().dtype(at::kFloat));
    const int64_t numExperts = static_cast<int64_t>(layout.originalOffsets.size()) - 1;
    for (int64_t expert = 0; expert < numExperts; ++expert) {
        const int32_t start = layout.originalOffsets[expert];
        const int32_t end = layout.originalOffsets[expert + 1];
        const int32_t rows = end - start;
        if (rows == 0) {
            continue;
        }
        out.narrow(0, start, rows).copy_(paddedOut.narrow(0, layout.paddedOffsets[expert], rows));
    }
    return out;
}

Stage1Outputs RestoreStage1Outputs(const Stage1Outputs &paddedOutputs, const PaddedLayout &layout, const at::Tensor &x)
{
    if (!layout.enabled) {
        return paddedOutputs;
    }

    Stage1Outputs outputs;
    outputs.out = at::empty({x.size(0), kMoeInterSize}, x.options().dtype(at::kFloat));
    outputs.gateProj = at::empty_like(outputs.out);
    outputs.upProj = at::empty_like(outputs.out);

    const int64_t numExperts = static_cast<int64_t>(layout.originalOffsets.size()) - 1;
    for (int64_t expert = 0; expert < numExperts; ++expert) {
        const int32_t start = layout.originalOffsets[expert];
        const int32_t end = layout.originalOffsets[expert + 1];
        const int32_t rows = end - start;
        if (rows == 0) {
            continue;
        }

        const int32_t paddedStart = layout.paddedOffsets[expert];
        outputs.out.narrow(0, start, rows).copy_(paddedOutputs.out.narrow(0, paddedStart, rows));
        outputs.gateProj.narrow(0, start, rows).copy_(paddedOutputs.gateProj.narrow(0, paddedStart, rows));
        outputs.upProj.narrow(0, start, rows).copy_(paddedOutputs.upProj.narrow(0, paddedStart, rows));
    }
    return outputs;
}

void ValidateInputs(const at::Tensor &x, const at::Tensor &gateWeightDn, const at::Tensor &upWeightDn,
                    const at::Tensor &groupOffsets)
{
    TORCH_CHECK(x.device().type() == DEVICE_TYPE, "x must be a NPU tensor (PrivateUse1)");
    TORCH_CHECK(gateWeightDn.device().type() == DEVICE_TYPE, "gate_weight_dn must be a NPU tensor (PrivateUse1)");
    TORCH_CHECK(upWeightDn.device().type() == DEVICE_TYPE, "up_weight_dn must be a NPU tensor (PrivateUse1)");
    TORCH_CHECK(groupOffsets.device().is_cpu() || groupOffsets.device().type() == DEVICE_TYPE,
                "group_offsets must be a CPU or NPU tensor");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(gateWeightDn.scalar_type() == at::kBFloat16, "gate_weight_dn must be bfloat16");
    TORCH_CHECK(upWeightDn.scalar_type() == at::kBFloat16, "up_weight_dn must be bfloat16");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(gateWeightDn.dim() == 3, "gate_weight_dn must be 3D");
    TORCH_CHECK(upWeightDn.dim() == 3, "up_weight_dn must be 3D");
    TORCH_CHECK(groupOffsets.dim() == 1, "group_offsets must be 1D");
    TORCH_CHECK(x.size(1) == kMoeHiddenSize, "x shape must be [tokens, ", kMoeHiddenSize, "]");
    TORCH_CHECK(gateWeightDn.size(1) == kMoeInterSize && gateWeightDn.size(2) == kMoeHiddenSize,
                "gate_weight_dn shape must be [experts, ", kMoeInterSize, ", ", kMoeHiddenSize, "]");
    TORCH_CHECK(upWeightDn.sizes() == gateWeightDn.sizes(), "up_weight_dn shape must match gate_weight_dn");
    TORCH_CHECK(gateWeightDn.size(0) >= 0, "number of experts must be non-negative");
    TORCH_CHECK(gateWeightDn.size(0) == groupOffsets.size(0) - 1,
                "group_offsets length must be num_experts + 1");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(gateWeightDn.is_contiguous(), "gate_weight_dn must be contiguous");
    TORCH_CHECK(upWeightDn.is_contiguous(), "up_weight_dn must be contiguous");
}

void ValidateNdWeightInput(const at::Tensor &x, const at::Tensor &weightNd, const at::Tensor &groupOffsets)
{
    TORCH_CHECK(x.device().type() == DEVICE_TYPE, "x must be a NPU tensor (PrivateUse1)");
    TORCH_CHECK(weightNd.device().type() == DEVICE_TYPE, "weight_nd must be a NPU tensor (PrivateUse1)");
    TORCH_CHECK(groupOffsets.device().is_cpu() || groupOffsets.device().type() == DEVICE_TYPE,
                "group_offsets must be a CPU or NPU tensor");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(weightNd.scalar_type() == at::kBFloat16, "weight_nd must be bfloat16");
    TORCH_CHECK(x.dim() == 2, "x must be rank-2 [M, H]");
    TORCH_CHECK(weightNd.dim() == 3, "weight_nd must be rank-3 [E, H, 2I]");
    TORCH_CHECK(groupOffsets.dim() == 1, "group_offsets must be rank-1 [E+1]");
    TORCH_CHECK(x.size(1) == kMoeHiddenSize, "x hidden size must match kernel specialization");
    TORCH_CHECK(weightNd.size(1) == kMoeHiddenSize, "weight_nd hidden size must match kernel specialization");
    TORCH_CHECK(weightNd.size(2) == kMoeInterSize * 2, "weight_nd last dim must be 2 * intermediate size");
    TORCH_CHECK(weightNd.is_contiguous(), "weight_nd must be contiguous");
}

at::Tensor NormalizeGroupOffsetsToCpuInt(const at::Tensor &groupOffsets)
{
    at::Tensor groupOffsetsCpu = groupOffsets.device().is_cpu() ? groupOffsets.contiguous() : groupOffsets.cpu().contiguous();
    if (groupOffsetsCpu.scalar_type() != at::kInt) {
        groupOffsetsCpu = groupOffsetsCpu.to(at::kInt);
    }
    return groupOffsetsCpu;
}

Projections LaunchGroupedFFNGroupedMatmulProjections(const at::Tensor &x, const at::Tensor &gateWeightDn,
                                                     const at::Tensor &upWeightDn, const at::Tensor &groupOffsetsCpu);

at::Tensor LaunchGroupedFFN(const at::Tensor &x, const at::Tensor &gateWeightDn, const at::Tensor &upWeightDn,
                            const at::Tensor &groupOffsetsCpu, const WorkList &workList)
{
    (void)groupOffsetsCpu;
    auto out = at::empty({x.size(0), kMoeInterSize}, x.options().dtype(at::kFloat));
    if (workList.blockDim == 0) {
        return out.zero_();
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const std::size_t systemWorkspaceSize = static_cast<std::size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    const uint32_t blockDim = workList.blockDim;

    const std::size_t userWorkspaceSize = GetMoeUserWorkspaceBytes(blockDim);
    const std::size_t workspaceSize = std::max<std::size_t>(userWorkspaceSize + systemWorkspaceSize, 1);
    auto workspaceTensor = at::empty({static_cast<int64_t>(workspaceSize)},
                                     at::TensorOptions().dtype(at::kByte).device(x.options().device()));

    EXEC_KERNEL_CMD(moe_grouped_gemm_custom, blockDim, x, gateWeightDn, upWeightDn, out, workList.expertIds,
                    workList.rowOffsets, workList.validRows, workspaceTensor);
    return out;
}

Stage1Outputs LaunchGroupedFFNFusedWithIntermediates(const at::Tensor &x,
                                                     const at::Tensor &gateWeightDn,
                                                     const at::Tensor &upWeightDn,
                                                     const at::Tensor &groupOffsetsCpu,
                                                     const WorkList &workList)
{
    (void)groupOffsetsCpu;
    Stage1Outputs outputs;
    outputs.out = at::empty({x.size(0), kMoeInterSize}, x.options().dtype(at::kFloat));
    outputs.gateProj = at::empty_like(outputs.out);
    outputs.upProj = at::empty_like(outputs.out);

    if (workList.blockDim == 0) {
        outputs.out.zero_();
        outputs.gateProj.zero_();
        outputs.upProj.zero_();
        return outputs;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const std::size_t systemWorkspaceSize = static_cast<std::size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    const uint32_t blockDim = workList.blockDim;

    const std::size_t userWorkspaceSize = GetMoeUserWorkspaceBytes(blockDim);
    const std::size_t workspaceSize = std::max<std::size_t>(userWorkspaceSize + systemWorkspaceSize, 1);
    auto workspaceTensor = at::empty({static_cast<int64_t>(workspaceSize)},
                                     at::TensorOptions().dtype(at::kByte).device(x.options().device()));

    EXEC_KERNEL_CMD(moe_grouped_gemm_with_intermediates_custom, blockDim, x, gateWeightDn, upWeightDn, outputs.out,
                    outputs.gateProj, outputs.upProj, workList.expertIds, workList.rowOffsets, workList.validRows,
                    workspaceTensor);
    return outputs;
}

Projections LaunchGroupedFFNSplitProjections(const at::Tensor &x, const at::Tensor &gateWeightDn,
                                             const at::Tensor &upWeightDn, const ProjectionWorkList &workList)
{
    Projections projections;
    projections.gateProj = at::empty({x.size(0), kMoeInterSize}, x.options().dtype(at::kFloat));
    if (workList.blockDim == 0) {
        projections.gateProj.zero_();
        projections.upProj = at::zeros_like(projections.gateProj);
        return projections;
    }

    projections.upProj = at::empty_like(projections.gateProj);

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const std::size_t systemWorkspaceSize = static_cast<std::size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    const std::size_t workspaceSize = std::max<std::size_t>(systemWorkspaceSize, 1);
    auto workspaceTensor = at::empty({static_cast<int64_t>(workspaceSize)},
                                     at::TensorOptions().dtype(at::kByte).device(x.options().device()));

    const uint32_t blockDim = workList.blockDim;
    EXEC_KERNEL_CMD(moe_grouped_dual_proj_custom, blockDim, x, gateWeightDn, upWeightDn, projections.gateProj,
                    projections.upProj, workList.groupOffsets, workList.tileOffsets, workList.numExperts,
                    workspaceTensor);
    return projections;
}

Stage1Outputs LaunchGroupedFFNSplit(const at::Tensor &x, const at::Tensor &gateWeightDn, const at::Tensor &upWeightDn,
                                    const ProjectionWorkList &workList)
{
    auto projections = LaunchGroupedFFNSplitProjections(x, gateWeightDn, upWeightDn, workList);
    Stage1Outputs outputs;
    outputs.gateProj = projections.gateProj;
    outputs.upProj = projections.upProj;

    outputs.out = at::silu(outputs.gateProj);
    outputs.out.mul_(outputs.upProj);
    return outputs;
}

at::Tensor LaunchGroupedFFNSplitStage1Input(const at::Tensor &x, const at::Tensor &gateWeightDn,
                                            const at::Tensor &upWeightDn, const ProjectionWorkList &workList)
{
    if (workList.blockDim == 0) {
        return at::zeros({x.size(0), kMoeInterSize * 2}, x.options().dtype(at::kBFloat16));
    }

    auto stage1Input = at::empty({x.size(0), kMoeInterSize * 2}, x.options().dtype(at::kBFloat16));

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const std::size_t systemWorkspaceSize = static_cast<std::size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    const std::size_t workspaceSize = std::max<std::size_t>(systemWorkspaceSize, 1);
    auto workspaceTensor = at::empty({static_cast<int64_t>(workspaceSize)},
                                     at::TensorOptions().dtype(at::kByte).device(x.options().device()));

    const uint32_t blockDim = workList.blockDim;
    EXEC_KERNEL_CMD(moe_grouped_dual_proj_packed_custom, blockDim, x, gateWeightDn, upWeightDn, stage1Input,
                    workList.groupOffsets, workList.tileOffsets, workList.numExperts, workspaceTensor);
    return stage1Input;
}

at::Tensor LaunchGroupedFFNSplitStage1InputNd(const at::Tensor &x, const at::Tensor &weightNd,
                                              const ProjectionWorkList &workList)
{
    if (workList.blockDim == 0) {
        return at::zeros({x.size(0), kMoeInterSize * 2}, x.options().dtype(at::kBFloat16));
    }

    auto stage1Input = at::empty({x.size(0), kMoeInterSize * 2}, x.options().dtype(at::kBFloat16));

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const std::size_t systemWorkspaceSize = static_cast<std::size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    const std::size_t workspaceSize = std::max<std::size_t>(systemWorkspaceSize, 1);
    auto workspaceTensor = at::empty({static_cast<int64_t>(workspaceSize)},
                                     at::TensorOptions().dtype(at::kByte).device(x.options().device()));

    const uint32_t blockDim = workList.blockDim;
    EXEC_KERNEL_CMD(moe_grouped_dual_proj_packed_nd_custom, blockDim, x, weightNd, stage1Input, workList.groupOffsets,
                    workList.tileOffsets, workList.numExperts, workspaceTensor);
    return stage1Input;
}

std::vector<at::Tensor> BuildGroupedInputs(const at::Tensor &x, const at::Tensor &groupOffsetsCpu)
{
    std::vector<at::Tensor> groupedInputs;
    const auto *offsetPtr = groupOffsetsCpu.data_ptr<int32_t>();
    const int64_t numExperts = groupOffsetsCpu.size(0) - 1;
    groupedInputs.reserve(static_cast<size_t>(numExperts));
    for (int64_t expert = 0; expert < numExperts; ++expert) {
        const int32_t start = offsetPtr[expert];
        const int32_t end = offsetPtr[expert + 1];
        if (start == end) {
            continue;
        }
        groupedInputs.push_back(x.narrow(0, start, end - start));
    }
    return groupedInputs;
}

std::vector<at::Tensor> BuildGroupedWeights(const at::Tensor &weightKn, const at::Tensor &groupOffsetsCpu)
{
    std::vector<at::Tensor> groupedWeights;
    const auto *offsetPtr = groupOffsetsCpu.data_ptr<int32_t>();
    const int64_t numExperts = groupOffsetsCpu.size(0) - 1;
    groupedWeights.reserve(static_cast<size_t>(numExperts));
    for (int64_t expert = 0; expert < numExperts; ++expert) {
        if (offsetPtr[expert] == offsetPtr[expert + 1]) {
            continue;
        }
        groupedWeights.push_back(weightKn[expert]);
    }
    return groupedWeights;
}

Projections LaunchGroupedFFNGroupedMatmulProjections(const at::Tensor &x, const at::Tensor &gateWeightDn,
                                                     const at::Tensor &upWeightDn, const at::Tensor &groupOffsetsCpu)
{
    static auto groupedMatmulOp =
        c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_grouped_matmul", "List");

    auto callGroupedMatmul = [&](const std::vector<at::Tensor> &inputs,
                                 const std::vector<at::Tensor> &weights) -> std::vector<at::Tensor> {
        c10::Stack stack;
        stack.emplace_back(inputs);
        stack.emplace_back(weights);
        stack.emplace_back();
        stack.emplace_back();
        stack.emplace_back();
        stack.emplace_back();
        stack.emplace_back();
        stack.emplace_back();
        stack.emplace_back();
        stack.emplace_back();
        stack.emplace_back();
        stack.emplace_back();
        stack.emplace_back(int64_t{0});
        stack.emplace_back(int64_t{-1});
        stack.emplace_back(int64_t{0});
        stack.emplace_back(int64_t{0});
        stack.emplace_back();
        groupedMatmulOp.callBoxed(&stack);
        return std::move(stack.front()).toTensorVector();
    };

    auto groupedInputs = BuildGroupedInputs(x, groupOffsetsCpu);
    if (groupedInputs.empty()) {
        Projections projections;
        projections.gateProj = at::zeros({x.size(0), kMoeInterSize}, x.options().dtype(at::kFloat));
        projections.upProj = at::zeros_like(projections.gateProj);
        return projections;
    }

    auto groupedGateWeights = BuildGroupedWeights(GetGroupedWeightKn(gateWeightDn), groupOffsetsCpu);
    auto groupedUpWeights = BuildGroupedWeights(GetGroupedWeightKn(upWeightDn), groupOffsetsCpu);

    Projections projections;
    projections.gateProj = at::cat(callGroupedMatmul(groupedInputs, groupedGateWeights), 0).to(at::kFloat);
    projections.upProj = at::cat(callGroupedMatmul(groupedInputs, groupedUpWeights), 0).to(at::kFloat);
    return projections;
}

Stage1Outputs LaunchGroupedFFNGroupedMatmul(const at::Tensor &x, const at::Tensor &gateWeightDn, const at::Tensor &upWeightDn,
                                            const at::Tensor &groupOffsetsCpu)
{
    auto projections = LaunchGroupedFFNGroupedMatmulProjections(x, gateWeightDn, upWeightDn, groupOffsetsCpu);

    Stage1Outputs outputs;
    outputs.gateProj = projections.gateProj;
    outputs.upProj = projections.upProj;
    outputs.out = at::silu(outputs.gateProj);
    outputs.out.mul_(outputs.upProj);
    return outputs;
}

ForwardImpl GetForwardImpl()
{
    const char *fusedEnv = std::getenv("PTO_MOE_GROUPED_FFN_USE_FUSED");
    if (fusedEnv != nullptr && std::strcmp(fusedEnv, "0") != 0) {
        return ForwardImpl::Fused;
    }

    const char *splitEnv = std::getenv("PTO_MOE_GROUPED_FFN_USE_CUSTOM_SPLIT");
    if (splitEnv != nullptr && std::strcmp(splitEnv, "0") != 0) {
        return ForwardImpl::CustomSplit;
    }

    return ForwardImpl::GroupedMatmul;
}

} // namespace

at::Tensor run_moe_grouped_ffn(const at::Tensor &x, const at::Tensor &gateWeightDn, const at::Tensor &upWeightDn,
                               const at::Tensor &groupOffsets)
{
    ValidateInputs(x, gateWeightDn, upWeightDn, groupOffsets);

    at::Tensor groupOffsetsCpu = NormalizeGroupOffsetsToCpuInt(groupOffsets);

    const auto *offsetPtr = groupOffsetsCpu.data_ptr<int32_t>();
    TORCH_CHECK(offsetPtr[0] == 0, "group_offsets[0] must be 0");
    for (int64_t idx = 0; idx < groupOffsetsCpu.size(0) - 1; ++idx) {
        TORCH_CHECK(offsetPtr[idx] <= offsetPtr[idx + 1], "group_offsets must be non-decreasing");
    }
    TORCH_CHECK(offsetPtr[groupOffsetsCpu.size(0) - 1] == x.size(0),
                "group_offsets[-1] must equal x.size(0)");

    const auto forwardImpl = GetForwardImpl();
    if (forwardImpl == ForwardImpl::GroupedMatmul) {
        return LaunchGroupedFFNGroupedMatmul(x, gateWeightDn, upWeightDn, groupOffsetsCpu).out;
    }

    const auto routingPlan = GetCachedRoutingPlan(groupOffsetsCpu, x);
    if (forwardImpl == ForwardImpl::CustomSplit) {
        // The projection worklist encodes ragged validM, so we can avoid padding copies.
        return LaunchGroupedFFNSplit(x, gateWeightDn, upWeightDn, routingPlan.projectionWorkList).out;
    }

    if (forwardImpl == ForwardImpl::Fused) {
        auto paddedLayout = MaybePadInputs(x, routingPlan);
        auto paddedOut = LaunchGroupedFFN(paddedLayout.x, gateWeightDn, upWeightDn, groupOffsetsCpu, routingPlan.workList);
        return RestoreOutputLayout(paddedOut, paddedLayout, x);
    }

    TORCH_CHECK(false, "Unsupported forward implementation");
}

at::Tensor run_moe_grouped_ffn_stage1_input(const at::Tensor &x, const at::Tensor &gateWeightDn,
                                            const at::Tensor &upWeightDn, const at::Tensor &groupOffsets)
{
    ValidateInputs(x, gateWeightDn, upWeightDn, groupOffsets);

    at::Tensor groupOffsetsCpu = NormalizeGroupOffsetsToCpuInt(groupOffsets);

    const auto *offsetPtr = groupOffsetsCpu.data_ptr<int32_t>();
    TORCH_CHECK(offsetPtr[0] == 0, "group_offsets[0] must be 0");
    for (int64_t idx = 0; idx < groupOffsetsCpu.size(0) - 1; ++idx) {
        TORCH_CHECK(offsetPtr[idx] <= offsetPtr[idx + 1], "group_offsets must be non-decreasing");
    }
    TORCH_CHECK(offsetPtr[groupOffsetsCpu.size(0) - 1] == x.size(0),
                "group_offsets[-1] must equal x.size(0)");

    const auto forwardImpl = GetForwardImpl();
    TORCH_CHECK(forwardImpl == ForwardImpl::CustomSplit,
                "pto_moe_grouped_ffn_stage1_input requires PTO_MOE_GROUPED_FFN_USE_CUSTOM_SPLIT=1");

    const auto routingPlan = GetCachedRoutingPlan(groupOffsetsCpu, x);
    return LaunchGroupedFFNSplitStage1Input(x, gateWeightDn, upWeightDn, routingPlan.projectionWorkList);
}

at::Tensor run_moe_grouped_ffn_stage1_input_nd(const at::Tensor &x, const at::Tensor &weightNd,
                                               const at::Tensor &groupOffsets)
{
    ValidateNdWeightInput(x, weightNd, groupOffsets);

    at::Tensor groupOffsetsCpu = NormalizeGroupOffsetsToCpuInt(groupOffsets);

    const auto *offsetPtr = groupOffsetsCpu.data_ptr<int32_t>();
    TORCH_CHECK(offsetPtr[0] == 0, "group_offsets[0] must be 0");
    for (int64_t idx = 0; idx < groupOffsetsCpu.size(0) - 1; ++idx) {
        TORCH_CHECK(offsetPtr[idx] <= offsetPtr[idx + 1], "group_offsets must be non-decreasing");
    }
    TORCH_CHECK(offsetPtr[groupOffsetsCpu.size(0) - 1] == x.size(0),
                "group_offsets[-1] must equal x.size(0)");

    const auto forwardImpl = GetForwardImpl();
    TORCH_CHECK(forwardImpl == ForwardImpl::CustomSplit,
                "pto_moe_grouped_ffn_stage1_input_nd requires PTO_MOE_GROUPED_FFN_USE_CUSTOM_SPLIT=1");

    const auto routingPlan = GetCachedRoutingPlan(groupOffsetsCpu, x);
    return LaunchGroupedFFNSplitStage1InputNd(x, weightNd, routingPlan.projectionWorkList);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> run_moe_grouped_ffn_with_intermediates(
    const at::Tensor &x,
    const at::Tensor &gateWeightDn,
    const at::Tensor &upWeightDn,
    const at::Tensor &groupOffsets)
{
    ValidateInputs(x, gateWeightDn, upWeightDn, groupOffsets);

    at::Tensor groupOffsetsCpu = NormalizeGroupOffsetsToCpuInt(groupOffsets);

    const auto *offsetPtr = groupOffsetsCpu.data_ptr<int32_t>();
    TORCH_CHECK(offsetPtr[0] == 0, "group_offsets[0] must be 0");
    for (int64_t idx = 0; idx < groupOffsetsCpu.size(0) - 1; ++idx) {
        TORCH_CHECK(offsetPtr[idx] <= offsetPtr[idx + 1], "group_offsets must be non-decreasing");
    }
    TORCH_CHECK(offsetPtr[groupOffsetsCpu.size(0) - 1] == x.size(0),
                "group_offsets[-1] must equal x.size(0)");

    const auto forwardImpl = GetForwardImpl();

    Stage1Outputs outputs;
    if (forwardImpl == ForwardImpl::GroupedMatmul) {
        outputs = LaunchGroupedFFNGroupedMatmul(x, gateWeightDn, upWeightDn, groupOffsetsCpu);
    } else if (forwardImpl == ForwardImpl::CustomSplit) {
        const auto routingPlan = GetCachedRoutingPlan(groupOffsetsCpu, x);
        outputs = LaunchGroupedFFNSplit(x, gateWeightDn, upWeightDn, routingPlan.projectionWorkList);
    } else if (forwardImpl == ForwardImpl::Fused) {
        const auto routingPlan = GetCachedRoutingPlan(groupOffsetsCpu, x);
        auto paddedLayout = MaybePadInputs(x, routingPlan);
        auto paddedOutputs =
            LaunchGroupedFFNFusedWithIntermediates(paddedLayout.x, gateWeightDn, upWeightDn, groupOffsetsCpu,
                                                   routingPlan.workList);
        outputs = RestoreStage1Outputs(paddedOutputs, paddedLayout, x);
    } else {
        TORCH_CHECK(false, "Unsupported forward implementation");
    }
    return std::make_tuple(outputs.out, outputs.gateProj, outputs.upProj);
}

} // namespace ascendc_path

namespace {
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("pto_moe_grouped_ffn(Tensor x, Tensor gate_weight_dn, Tensor up_weight_dn, Tensor group_offsets) -> Tensor");
    m.def("pto_moe_grouped_ffn_stage1_input(Tensor x, Tensor gate_weight_dn, Tensor up_weight_dn, Tensor group_offsets) -> Tensor");
    m.def("pto_moe_grouped_ffn_stage1_input_nd(Tensor x, Tensor weight_nd, Tensor group_offsets) -> Tensor");
    m.def(
        "pto_moe_grouped_ffn_with_intermediates(Tensor x, Tensor gate_weight_dn, Tensor up_weight_dn, Tensor group_offsets) -> (Tensor, Tensor, Tensor)"
    );
}
} // namespace

namespace {
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("pto_moe_grouped_ffn", TORCH_FN(ascendc_path::run_moe_grouped_ffn));
    m.impl("pto_moe_grouped_ffn_stage1_input", TORCH_FN(ascendc_path::run_moe_grouped_ffn_stage1_input));
    m.impl("pto_moe_grouped_ffn_stage1_input_nd", TORCH_FN(ascendc_path::run_moe_grouped_ffn_stage1_input_nd));
    m.impl(
        "pto_moe_grouped_ffn_with_intermediates",
        TORCH_FN(ascendc_path::run_moe_grouped_ffn_with_intermediates)
    );
}
} // namespace
