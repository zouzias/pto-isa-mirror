/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_COSTMODEL_VFSIM_PTO_ADAPTER_VFSIM_COST_MODEL_H_
#define PTO_COSTMODEL_VFSIM_PTO_ADAPTER_VFSIM_COST_MODEL_H_

#include <cstdint>
#include <string>
#include <vector>

#include "pto/costmodel/a5/cce_costmodel/vf_info.hpp"

namespace pto::mocker::vf {

enum class VfPredictionStatus : uint8_t {
    VF_SIM_HIT,
    PARTIAL_FALLBACK,
    UNSUPPORTED_FORM,
    INVALID_TRACE,
    EMPTY_PROGRAM,
    SIMULATOR_ERROR,
};

enum class VfSimLogLevel : uint8_t {
    OFF,
    ERRORS,
    SUMMARY,
    DETAILED,
};

struct VfPredictionOptions {
    VfSimLogLevel logLevel = VfSimLogLevel::OFF;
    std::string configDir;
};

struct VfPredictionResult {
    uint64_t cycles = 0;
    VfPredictionStatus status = VfPredictionStatus::EMPTY_PROGRAM;
    uint32_t vfSimHitCount = 0;
    uint32_t fallbackCount = 0;
    uint32_t ignoredInstructionCount = 0;
    std::vector<std::string> diagnostics;
};

const char* toString(VfPredictionStatus status);

// PTO's internal VfInfo remains the integration boundary. The adapter lowers it to the native VfSim API.
VfPredictionResult predictVfCyclesWithVfSim(const std::vector<VfInfo>& vfs, const VfPredictionOptions& options = {});

} // namespace pto::mocker::vf

#endif // PTO_COSTMODEL_VFSIM_PTO_ADAPTER_VFSIM_COST_MODEL_H_
