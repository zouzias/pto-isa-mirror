// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: CANN-1.0

#include <iterator>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "api/native/UarchOverrideSchema.h"

namespace vfsim {
namespace {

struct GeneratedUarchField {
    const char* name;
    const char* type;
    bool supportsPython;
    bool supportsCpp;
};

#include "api/native/generated/UarchOverrideSchemaData.inc"

UarchOverrideFieldType parseType(const std::string& type)
{
    if (type == "integer")
        return UarchOverrideFieldType::INTEGER;
    if (type == "boolean")
        return UarchOverrideFieldType::BOOLEAN;
    if (type == "string")
        return UarchOverrideFieldType::STRING;
    throw std::runtime_error("Unsupported generated uarch field type: " + type);
}

const std::unordered_map<std::string, UarchOverrideFieldType>& fieldTypes()
{
    static const auto fields = [] {
        std::unordered_map<std::string, UarchOverrideFieldType> result;
        for (const auto& field : GENERATED_UARCH_FIELDS)
            result.emplace(field.name, parseType(field.type));
        return result;
    }();
    return fields;
}

const std::unordered_set<std::string>& deprecatedFields()
{
    static const std::unordered_set<std::string> fields(
        std::begin(GENERATED_DEPRECATED_UARCH_FIELDS), std::end(GENERATED_DEPRECATED_UARCH_FIELDS));
    return fields;
}

} // namespace

std::string normalizeUarchOverrideFieldName(const std::string& name)
{
    static const std::unordered_map<std::string, std::string> aliases{
        {"IDU_window_width", "idu_window_width"},
        {"IDU_issue_width", "idu_issue_width"},
        {"LDQ_width", "ldq_width"},
    };
    const auto found = aliases.find(name);
    return found == aliases.end() ? name : found->second;
}

std::optional<UarchOverrideFieldType> uarchOverrideFieldType(const std::string& name)
{
    const auto found = fieldTypes().find(normalizeUarchOverrideFieldName(name));
    if (found == fieldTypes().end())
        return std::nullopt;
    return found->second;
}

bool isDeprecatedUarchOverrideField(const std::string& name) { return deprecatedFields().count(name) != 0; }

bool uarchOverrideFieldSupportsTarget(const std::string& name, UarchOverrideTarget target)
{
    const std::string normalizedName = normalizeUarchOverrideFieldName(name);
    for (const auto& field : GENERATED_UARCH_FIELDS) {
        if (normalizedName != field.name)
            continue;
        return target == UarchOverrideTarget::PYTHON ? field.supportsPython : field.supportsCpp;
    }
    return false;
}

std::set<std::string> uarchOverrideFieldsForTarget(UarchOverrideTarget target)
{
    std::set<std::string> result;
    for (const auto& field : GENERATED_UARCH_FIELDS) {
        const bool supported = target == UarchOverrideTarget::PYTHON ? field.supportsPython : field.supportsCpp;
        if (supported)
            result.emplace(field.name);
    }
    return result;
}

const char* uarchOverrideFieldTypeName(UarchOverrideFieldType type)
{
    switch (type) {
        case UarchOverrideFieldType::INTEGER:
            return "integer";
        case UarchOverrideFieldType::BOOLEAN:
            return "boolean";
        case UarchOverrideFieldType::STRING:
            return "string";
    }
    return "unknown";
}

} // namespace vfsim
