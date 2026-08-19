// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: CANN-1.0

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
