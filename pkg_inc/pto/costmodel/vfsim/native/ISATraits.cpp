// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <cctype>
#include <algorithm>

#include "native/ISATraits.h"

namespace vfsim {
namespace {

std::string canonicalOp(std::string op)
{
    std::transform(
        op.begin(), op.end(), op.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return op;
}

OpClass opClassFromString(const std::string& text)
{
    const std::string canon = canonicalOp(text);
    if (canon == "LOAD")
        return OpClass::LOAD;
    if (canon == "STORE")
        return OpClass::STORE;
    if (canon == "COMPUTE")
        return OpClass::COMPUTE;
    return OpClass::UNKNOWN;
}

OpClass opClassFromNameFallback(const std::string& op)
{
    const std::string canon = canonicalOp(op);
    if (canon.rfind("VLD", 0) == 0)
        return OpClass::LOAD;
    if (canon.rfind("VST", 0) == 0)
        return OpClass::STORE;
    return OpClass::COMPUTE;
}

} // namespace

OpClass getOpClass(const ParamDb& db, const std::string& op, const std::string& dtype)
{
    const std::string canonOp = canonicalOp(op);
    if (db.hasInst(canonOp, dtype)) {
        const InstConfig& cfg = db.inst(canonOp, dtype);
        const OpClass opClass = opClassFromString(cfg.opClass);
        if (opClass != OpClass::UNKNOWN)
            return opClass;
        if (!cfg.exu.empty() || !cfg.dispatchExu.empty())
            return OpClass::COMPUTE;
    }
    return opClassFromNameFallback(canonOp);
}

bool isLoadOp(const ParamDb& db, const std::string& op, const std::string& dtype)
{
    return getOpClass(db, op, dtype) == OpClass::LOAD;
}

bool isStoreOp(const ParamDb& db, const std::string& op, const std::string& dtype)
{
    return getOpClass(db, op, dtype) == OpClass::STORE;
}

bool isComputeOp(const ParamDb& db, const std::string& op, const std::string& dtype)
{
    return getOpClass(db, op, dtype) == OpClass::COMPUTE;
}

bool usesLsq(const ParamDb& db, const std::string& op, const std::string& dtype)
{
    const OpClass cls = getOpClass(db, op, dtype);
    return cls == OpClass::LOAD || cls == OpClass::STORE;
}

bool usesShqQueue(const ParamDb& db, const std::string& op, const std::string& dtype)
{
    return isComputeOp(db, op, dtype);
}

bool usesSharedShqCredit(const ParamDb& db, const std::string& op, const std::string& dtype)
{
    const OpClass cls = getOpClass(db, op, dtype);
    return cls == OpClass::COMPUTE || cls == OpClass::STORE;
}

} // namespace vfsim
