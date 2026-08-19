// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef VFSIM_NATIVE_ISA_TRAITS_H_
#define VFSIM_NATIVE_ISA_TRAITS_H_

#include <string>

#include "native/ParamDB.h"

namespace vfsim {

enum class OpClass {
    LOAD,
    STORE,
    COMPUTE,
    UNKNOWN,
};

OpClass getOpClass(const ParamDb& db, const std::string& op, const std::string& dtype = "fp32");
bool isLoadOp(const ParamDb& db, const std::string& op, const std::string& dtype = "fp32");
bool isStoreOp(const ParamDb& db, const std::string& op, const std::string& dtype = "fp32");
bool isComputeOp(const ParamDb& db, const std::string& op, const std::string& dtype = "fp32");
bool usesLsq(const ParamDb& db, const std::string& op, const std::string& dtype = "fp32");
bool usesShqQueue(const ParamDb& db, const std::string& op, const std::string& dtype = "fp32");
bool usesSharedShqCredit(const ParamDb& db, const std::string& op, const std::string& dtype = "fp32");

} // namespace vfsim

#endif // VFSIM_NATIVE_ISA_TRAITS_H_
