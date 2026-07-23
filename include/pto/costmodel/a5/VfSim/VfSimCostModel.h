// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
#ifndef PTO_COSTMODEL_A5_VFSIM_COST_MODEL_H
#define PTO_COSTMODEL_A5_VFSIM_COST_MODEL_H

#include "pto/costmodel/a5/cce_costmodel/vf_info.hpp"

#include <cstdint>
#include <vector>

namespace pto::mocker::vf {

// The public integration boundary is VfInfo. Lowering to VfSimulator's native
// program representation remains private to the implementation.
uint64_t PredictVfCyclesWithVfSim(const std::vector<VfInfo> &vfs);

}  // namespace pto::mocker::vf

#endif
