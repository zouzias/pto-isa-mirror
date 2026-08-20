// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <algorithm>
#include <utility>

#include "api/native/CanonicalVfInfo.h"

namespace vfsim {

CanonicalNode CanonicalNode::makeInstruction(CanonicalInstruction value) { return CanonicalNode{std::move(value)}; }

CanonicalNode CanonicalNode::makeLoop(CanonicalLoop value)
{
    return CanonicalNode{std::shared_ptr<const CanonicalLoop>(std::make_shared<CanonicalLoop>(std::move(value)))};
}

CanonicalNode CanonicalNode::makeMembar(CanonicalMembar value) { return CanonicalNode{std::move(value)}; }

bool CanonicalValidationResult::ok() const
{
    return std::none_of(diagnostics.begin(), diagnostics.end(), [](const CanonicalValidationDiagnostic& diagnostic) {
        return diagnostic.severity == "error";
    });
}

} // namespace vfsim
