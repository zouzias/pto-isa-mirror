/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_PERF_SIM_COSTMODEL_PROVIDER_HPP
#define PTO_PERF_SIM_COSTMODEL_PROVIDER_HPP

#include "config.hpp"
#include "latency.hpp"
#include "recorder.hpp"

#include <pto/costmodel/common/input_mapping.hpp>

#include <string>

namespace pto::perf_sim {

// ── Magic number constants ──
static constexpr uint64_t kPercentMultiplier = 100;
static constexpr uint64_t kMatrixLatencyBase = 4;
static constexpr uint64_t kElemsPerMatrixUnit = 16;
static constexpr uint64_t kGMLatencyBase = 3;
static constexpr uint64_t kGMCycleMultiplier = 2;
static constexpr uint64_t kGMElemsPerUnit = 64;
static constexpr uint64_t kMTE1LatencyBase = 1;
static constexpr uint64_t kMTE1ElemsPerUnit = 64;
static constexpr uint64_t kDefaultLatencyBase = 2;
static constexpr uint64_t kDefaultElemsPerUnit = 32;

// ── Runtime context passed to costmodel ──

struct CostModelRuntimeCtx {
    uint32_t block_dim = 1;
    uint32_t npu_arch = 2201;
};

inline CostModelRuntimeCtx& GetCostModelCtx()
{
    static thread_local CostModelRuntimeCtx ctx;
    return ctx;
}

// ── String → lightweight costmodel mappings ──

using ::pto::mocker::lightweight::TryMapOpcode;
using ::pto::mocker::lightweight::TryMapDType;

#if !defined(__NPU_ARCH__) || (__NPU_ARCH__ == 2201)
// ── LightweightFormula backend ──

inline uint64_t EstimateLightweightCycles(const std::string& opcode, int rows, int cols, const std::string& dtype)
{
    using namespace ::pto::mocker::lightweight;

    PtoOpcode pto_op;
    if (!TryMapOpcode(opcode, pto_op))
        return 0;

    CostModelInput input{};
    input.op = pto_op;
    if (!TryMapDType(dtype, input.dtype))
        return 0;
    input.rows = rows;
    input.cols = cols;
    input.data_size = static_cast<int64_t>(rows) * cols;

    if (pto_op == PtoOpcode::TMATMUL || pto_op == PtoOpcode::TGEMV) {
        input.k = cols;
    }

    CostModelResult result = EstimateCycles(input);
    if (result.cycles > 0.0)
        return static_cast<uint64_t>(result.cycles);
    return 0;
}

#endif

// ── Fallback for unsupported instructions ──

inline uint64_t FallbackCycles(const std::string& opcode, int rows, int cols)
{
    uint64_t elems = static_cast<uint64_t>(rows) * cols;
    PipeStage stage = StaticPipeStageLookup(opcode);
    if (stage == PipeStage::Scalar)
        return 1;
    if (stage == PipeStage::Matrix)
        return kMatrixLatencyBase + elems / kElemsPerMatrixUnit;
    if (IsGMAccessPipe(stage))
        return kGMLatencyBase + elems * kGMCycleMultiplier / kGMElemsPerUnit;
    if (stage == PipeStage::MTE1)
        return kMTE1LatencyBase + elems / kMTE1ElemsPerUnit;
    return kDefaultLatencyBase + elems / kDefaultElemsPerUnit;
}

// ── Unified estimation entry (used when CCE mock trace is unavailable) ──

inline uint64_t EstimateInstrCycles(const std::string& opcode, int rows, int cols, const std::string& dtype)
{
#if !defined(__NPU_ARCH__) || (__NPU_ARCH__ == 2201)
    if (opcode == "TDIVS" && (dtype == "int16" || dtype == "int32")) {
        return 4;
    }
    uint64_t cycles = EstimateLightweightCycles(opcode, rows, cols, dtype);
    return cycles > 0 ? cycles : FallbackCycles(opcode, rows, cols);
#else
    (void)dtype;
    return FallbackCycles(opcode, rows, cols);
#endif
}

} // namespace pto::perf_sim

#endif
