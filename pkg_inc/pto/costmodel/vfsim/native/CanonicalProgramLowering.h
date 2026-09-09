/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef VFSIM_NATIVE_CANONICAL_PROGRAM_LOWERING_H
#define VFSIM_NATIVE_CANONICAL_PROGRAM_LOWERING_H

#include "api/native/CanonicalVfInfo.h"
#include "api/native/RuntimeValue.h"
#include "native/IFU.h"
#include "native/ParamDB.h"

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vfsim {

struct CanonicalRuntimeProgram {
    std::vector<DynamicInst> instructions;
    std::unordered_map<std::string, ValueInfo> values;
    std::unordered_map<int, std::vector<int64_t>> topBlockLoopBounds;
    std::unordered_set<int64_t> emptyTopBlocks;
    int64_t totalTopBlocks = 1;
    std::string dtype = "fp32";
};

CanonicalRuntimeProgram lowerCanonicalProgram(const CanonicalVfInfo& vfInfo, const ParamDB* db = nullptr);
UarchConfig resolveCanonicalUarch(const CanonicalVfInfo& vfInfo, const UarchConfig& defaults);

} // namespace vfsim

#endif // VFSIM_NATIVE_CANONICAL_PROGRAM_LOWERING_H
