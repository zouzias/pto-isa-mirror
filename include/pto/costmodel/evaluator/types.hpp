/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MOCKER_EVALUATOR_TYPES_HPP
#define PTO_MOCKER_EVALUATOR_TYPES_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace pto::mocker::evaluator {

struct CycleEstimate {
    uint64_t cycles = 0;
    bool supported = true;
    std::string note;
};

struct CceLatencyRecord {
    std::string name;
    uint64_t cycles = 0;
    bool supported = true;
    std::string note;
};

struct PtoLatencyRecord {
    std::string name;
    std::vector<CceLatencyRecord> cce_calls;
    uint64_t total_cycles = 0;
    bool supported = true;
};

struct TraceLatencyReport {
    std::string arch_name;
    std::vector<PtoLatencyRecord> executed_pto;
    std::vector<CceLatencyRecord> raw_cce_calls;
    uint64_t total_cycles = 0;
    bool supported = true;
};

} // namespace pto::mocker::evaluator

#endif
