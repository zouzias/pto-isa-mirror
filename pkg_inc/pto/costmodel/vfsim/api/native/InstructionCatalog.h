// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef VFSIM_API_NATIVE_INSTRUCTION_CATALOG_H_
#define VFSIM_API_NATIVE_INSTRUCTION_CATALOG_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vfsim {

enum class CatalogInstructionClass { LOAD, STORE, COMPUTE, CONTROL };
enum class CatalogOperandDirection { INPUT, OUTPUT, IGNORE };
enum class CatalogArgumentKind { REGISTER, UB, SCALAR, PREDICATE, CONFIG, REGISTER_OR_SCALAR };

struct NativeOperandSpec {
    std::string name;
    int argumentIndex = 0;
    CatalogOperandDirection direction = CatalogOperandDirection::INPUT;
    std::string role;
    CatalogArgumentKind kind = CatalogArgumentKind::REGISTER;
    bool optional = false;
    bool allowIntegerExpression = false;
    std::unordered_set<std::string> allowedValues;
};

struct NativeCallVariant {
    int argumentCount = 0;
    std::unordered_map<int, std::unordered_set<std::string>> argumentValues;
};

struct NativeInstructionSpec {
    std::string opcode;
    CatalogInstructionClass instructionClass = CatalogInstructionClass::COMPUTE;
    std::string signature;
    std::string formRule;
    std::string fixedForm;
    bool virtualOpcode = false;
    std::unordered_set<std::string> forms;
    std::unordered_map<std::string, std::string> specializations;
    std::vector<NativeOperandSpec> operands;
    std::vector<NativeCallVariant> callVariants;
};

class InstructionCatalog {
public:
    std::string canonicalOpcode(const std::string& opcode) const;
    std::string specializeOpcode(const std::string& opcode, const std::string& form) const;
    const NativeInstructionSpec* lookup(const std::string& opcode) const;

private:
    friend const InstructionCatalog& defaultInstructionCatalog();
    InstructionCatalog();

    std::unordered_map<std::string, NativeInstructionSpec> specs_;
    std::unordered_map<std::string, std::string> aliases_;
};

const InstructionCatalog& defaultInstructionCatalog();

} // namespace vfsim

#endif // VFSIM_API_NATIVE_INSTRUCTION_CATALOG_H_
