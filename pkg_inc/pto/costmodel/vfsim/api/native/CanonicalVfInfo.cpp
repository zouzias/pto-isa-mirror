// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: CANN-1.0

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
