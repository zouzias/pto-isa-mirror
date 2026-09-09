/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_COSTMODEL_VFSIM_PTO_ADAPTER_PTO_CANONICAL_LOWERING_H_
#define PTO_COSTMODEL_VFSIM_PTO_ADAPTER_PTO_CANONICAL_LOWERING_H_

#include <cstdint>

#include "pto/costmodel/a5/cce_costmodel/vf_info.hpp"

#include "api/native/CanonicalVfInfo.h"

namespace pto::mocker::vf {

struct PtoCanonicalLoweringResult {
    vfsim::CanonicalVfInfo program;
    uint32_t ignoredInstructionCount = 0;
};

PtoCanonicalLoweringResult lowerPtoVfToCanonical(const VfInfo& vf);

} // namespace pto::mocker::vf

#endif // PTO_COSTMODEL_VFSIM_PTO_ADAPTER_PTO_CANONICAL_LOWERING_H_
