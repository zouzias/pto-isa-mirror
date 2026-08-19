// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: CANN-1.0

#ifndef VFSIM_NATIVE_CANONICAL_PROGRAM_LOWERING_H_
#define VFSIM_NATIVE_CANONICAL_PROGRAM_LOWERING_H_

#include <unordered_map>
#include <set>
#include <vector>

#include "api/native/CanonicalVfInfo.h"
#include "api/native/VfInfo.h"
#include "native/IFU.h"

namespace vfsim {

struct CanonicalRuntimeProgram {
    std::vector<DynamicInst> instructions;
    std::unordered_map<std::string, ValueInfo> values;
    ProgramAnalysis::ParamMap params;
    std::unordered_map<int, std::vector<int64_t>> topBlockLoopBounds;
    int64_t totalTopBlocks = 1;
    std::string dtype = "fp32";
};

CanonicalRuntimeProgram lowerCanonicalProgram(const CanonicalVfInfo& vfInfo, const ParamDb* db = nullptr);
UarchConfig resolveCanonicalUarch(const CanonicalVfInfo& vfInfo, const UarchConfig& defaults);
std::set<std::string> cppResolvedUarchOverrideFields();

} // namespace vfsim

#endif // VFSIM_NATIVE_CANONICAL_PROGRAM_LOWERING_H_
