// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef VFSIM_API_NATIVE_UARCH_OVERRIDE_SCHEMA_H_
#define VFSIM_API_NATIVE_UARCH_OVERRIDE_SCHEMA_H_

#include <optional>
#include <set>
#include <string>

namespace vfsim {

enum class UarchOverrideFieldType { INTEGER, BOOLEAN, STRING };
enum class UarchOverrideTarget { PYTHON, CPP };

std::string normalizeUarchOverrideFieldName(const std::string& name);
std::optional<UarchOverrideFieldType> uarchOverrideFieldType(const std::string& name);
bool isDeprecatedUarchOverrideField(const std::string& name);
bool uarchOverrideFieldSupportsTarget(const std::string& name, UarchOverrideTarget target);
std::set<std::string> uarchOverrideFieldsForTarget(UarchOverrideTarget target);
const char* uarchOverrideFieldTypeName(UarchOverrideFieldType type);

} // namespace vfsim

#endif // VFSIM_API_NATIVE_UARCH_OVERRIDE_SCHEMA_H_
