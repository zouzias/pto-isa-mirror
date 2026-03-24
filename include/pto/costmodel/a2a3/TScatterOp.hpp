/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TSCATTER_OP_HPP
#define TSCATTER_OP_HPP

#include <vector>
#include "pto/costmodel/costmodel_types.hpp"

namespace pto {

// TSCATTER: pure scalar element-wise loop — no CCE pipeline instructions issued.
PTO_INTERNAL std::vector<CostModelStats> runScatterOp()
{
    std::vector<CostModelStats> stats;
    stats.emplace_back("scatter", 1);
    return stats; // no CCE pipeline instructions
}

} // namespace pto
#endif // TSCATTER_OP_HPP
