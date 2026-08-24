// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <algorithm>
#include <cctype>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pto/costmodel/vfsim/pto_adapter/pto_canonical_lowering.hpp"

#include "api/native/InstructionCatalog.h"

namespace pto::mocker::vf {
namespace {

using Environment = std::unordered_map<std::string, std::string>;

struct LogicalValue {
    vfsim::CanonicalStorageKind storage = vfsim::CanonicalStorageKind::UNKNOWN;
    std::string dtype;
};

std::string toLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

std::string toUpper(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::toupper(character));
    });
    return value;
}

std::string safeId(const std::string& value)
{
    std::string result;
    result.reserve(value.size());
    for (unsigned char character : value) {
        result.push_back(
            std::isalnum(character) || character == '_' || character == '.' || character == '-' ?
                static_cast<char>(character) :
                '_');
    }
    return result.empty() ? "value" : result;
}

std::string normalizeDtype(const std::string& dtype)
{
    const std::string normalized = toLower(dtype);
    static const std::unordered_map<std::string, std::string> aliases = {
        {"f32", "fp32"},      {"float32", "fp32"}, {"fp32", "fp32"},     {"f16", "fp16"},
        {"float16", "fp16"},  {"fp16", "fp16"},    {"bfloat16", "bf16"}, {"bf16", "bf16"},
        {"s32", "int32"},     {"i32", "int32"},    {"int32", "int32"},   {"u32", "uint32"},
        {"uint32", "uint32"}, {"bool", "bool"},    {"boolean", "bool"},  {"predicate", "bool"},
    };
    const auto alias = aliases.find(normalized);
    return alias == aliases.end() ? normalized : alias->second;
}

std::string compactDtype(const std::string& dtype)
{
    const std::string normalized = normalizeDtype(dtype);
    if (normalized == "fp32")
        return "f32";
    if (normalized == "fp16")
        return "f16";
    if (normalized == "int32")
        return "s32";
    if (normalized == "uint32")
        return "u32";
    return normalized;
}

std::string publicDtype(const MemInfo& memory)
{
    if (memory.dtype.empty() || memory.dtype == "unknown" || memory.dtype == "bool")
        return {};
    return normalizeDtype(memory.dtype);
}

std::string integerBitForm(const std::string& dtype)
{
    const std::string normalized = toUpper(dtype);
    if (normalized == "INT16" || normalized == "UINT16" || normalized == "I16" || normalized == "S16" ||
        normalized == "U16") {
        return "b16";
    }
    if (normalized == "INT32" || normalized == "UINT32" || normalized == "I32" || normalized == "S32" ||
        normalized == "U32") {
        return "b32";
    }
    return {};
}

std::vector<const MemInfo*> dataOperands(const std::vector<MemInfo>& operands)
{
    std::vector<const MemInfo*> result;
    result.reserve(operands.size());
    for (const MemInfo& operand : operands) {
        if (operand.location != MemLocation::PredicateRegister)
            result.push_back(&operand);
    }
    return result;
}

std::string inferIntegerBitForm(const VfInst& instruction)
{
    std::string form;
    bool hasTypedOperand = false;
    const auto visit = [&](const std::vector<MemInfo>& operands) {
        for (const MemInfo* operand : dataOperands(operands)) {
            const std::string dtype = publicDtype(*operand);
            if (dtype.empty())
                continue;
            hasTypedOperand = true;
            const std::string operandForm = integerBitForm(dtype);
            if (operandForm.empty() || (!form.empty() && form != operandForm))
                return false;
            form = operandForm;
        }
        return true;
    };
    if (!visit(instruction.dst) || !visit(instruction.src) || !hasTypedOperand)
        return {};
    return form;
}

std::string firstDtype(const std::vector<MemInfo>& operands)
{
    for (const MemInfo* operand : dataOperands(operands)) {
        const std::string dtype = publicDtype(*operand);
        if (!dtype.empty())
            return dtype;
    }
    return {};
}

std::string inferForm(const VfInst& instruction, const vfsim::NativeInstructionSpec* specification)
{
    if (specification != nullptr && !specification->fixedForm.empty())
        return specification->fixedForm;
    if (const std::string bitForm = inferIntegerBitForm(instruction); !bitForm.empty())
        return bitForm;

    const std::string sourceDtype = firstDtype(instruction.src);
    const std::string destinationDtype = firstDtype(instruction.dst);
    if (!sourceDtype.empty() && !destinationDtype.empty() && sourceDtype != destinationDtype)
        return compactDtype(sourceDtype) + "_to_" + compactDtype(destinationDtype);
    if (!destinationDtype.empty())
        return normalizeDtype(destinationDtype);
    if (!sourceDtype.empty())
        return normalizeDtype(sourceDtype);
    return "fp32";
}

vfsim::CanonicalStorageKind canonicalStorage(MemLocation location)
{
    return location == MemLocation::UB ? vfsim::CanonicalStorageKind::UB : vfsim::CanonicalStorageKind::REGISTER;
}

vfsim::CanonicalInstructionClass canonicalClass(vfsim::CatalogInstructionClass instructionClass)
{
    switch (instructionClass) {
        case vfsim::CatalogInstructionClass::LOAD:
            return vfsim::CanonicalInstructionClass::LOAD;
        case vfsim::CatalogInstructionClass::STORE:
            return vfsim::CanonicalInstructionClass::STORE;
        case vfsim::CatalogInstructionClass::COMPUTE:
            return vfsim::CanonicalInstructionClass::COMPUTE;
        case vfsim::CatalogInstructionClass::CONTROL:
            return vfsim::CanonicalInstructionClass::CONTROL;
    }
    return vfsim::CanonicalInstructionClass::UNKNOWN;
}

vfsim::CanonicalOperandRole canonicalRole(const std::string& role)
{
    if (role == "source")
        return vfsim::CanonicalOperandRole::SOURCE;
    if (role == "destination")
        return vfsim::CanonicalOperandRole::DESTINATION;
    if (role == "memory")
        return vfsim::CanonicalOperandRole::MEMORY;
    if (role == "scalar")
        return vfsim::CanonicalOperandRole::SCALAR;
    if (role == "predicate")
        return vfsim::CanonicalOperandRole::PREDICATE;
    if (role == "config")
        return vfsim::CanonicalOperandRole::CONFIG;
    return vfsim::CanonicalOperandRole::UNKNOWN;
}

bool isIgnoredPredicateSetup(const std::string& operation)
{
    const std::string normalized = toUpper(operation);
    return normalized == "PLT_B8" || normalized == "PLT_B16" || normalized == "PLT_B32";
}

class PtoCanonicalLowerer {
public:
    explicit PtoCanonicalLowerer(const VfInfo& input) : input_(input)
    {
        output_.source.emplace("adapter", std::string("pto_vf_info"));
        output_.source.emplace("pto_op", input.op);
        output_.source.emplace("pto_shape", input.shape);
    }

    PtoCanonicalLoweringResult run()
    {
        Environment environment;
        output_.context = lowerNodes(input_.tree, environment);
        const vfsim::CanonicalValidationResult validation = vfsim::validateCanonicalVfInfo(output_);
        if (!validation.ok()) {
            std::string message = "PTO VfInfo could not be lowered to CanonicalVfInfo";
            for (const vfsim::CanonicalValidationDiagnostic& diagnostic : validation.diagnostics)
                message += "; " + diagnostic.code + " at " + diagnostic.path;
            throw std::runtime_error(message);
        }
        return PtoCanonicalLoweringResult{std::move(output_), ignoredInstructionCount_};
    }

private:
    const VfInfo& input_;
    vfsim::CanonicalVfInfo output_;
    std::unordered_map<std::string, LogicalValue> logicalValues_;
    std::unordered_set<std::string> nodeIds_;
    uint64_t instructionCounter_ = 0;
    uint64_t loopCounter_ = 0;
    uint64_t membarCounter_ = 0;
    uint64_t definitionCounter_ = 0;
    uint64_t scalarCounter_ = 0;
    uint32_t ignoredInstructionCount_ = 0;

    std::string nodeId(const std::string& kind)
    {
        uint64_t* counter = kind == "instruction" ? &instructionCounter_ :
                            kind == "loop"        ? &loopCounter_ :
                                                    &membarCounter_;
        while (true) {
            const std::string candidate = kind + "." + std::to_string((*counter)++);
            if (nodeIds_.insert(candidate).second)
                return candidate;
        }
    }

    void registerLogicalValue(const MemInfo& memory)
    {
        if (memory.name.empty())
            throw std::runtime_error("PTO operand has no trace value ID");
        LogicalValue value{canonicalStorage(memory.location), publicDtype(memory)};
        if (value.dtype.empty())
            value.dtype = "fp32";
        const auto [iterator, inserted] = logicalValues_.try_emplace(memory.name, value);
        if (!inserted && (iterator->second.storage != value.storage || iterator->second.dtype != value.dtype)) {
            throw std::runtime_error("PTO trace value metadata is inconsistent: " + memory.name);
        }
    }

    std::string registerScalar(const VfInst& instruction, const std::string& prefix)
    {
        const std::string logicalId = prefix + std::to_string(scalarCounter_++);
        std::string dtype = firstDtype(instruction.dst);
        if (dtype.empty())
            dtype = firstDtype(instruction.src);
        if (dtype.empty())
            dtype = "fp32";
        logicalValues_.emplace(logicalId, LogicalValue{vfsim::CanonicalStorageKind::SCALAR, normalizeDtype(dtype)});
        return logicalId;
    }

    const LogicalValue& logicalValue(const std::string& logicalId) const
    {
        const auto value = logicalValues_.find(logicalId);
        if (value == logicalValues_.end())
            throw std::runtime_error("PTO operand has no canonical value metadata: " + logicalId);
        return value->second;
    }

    std::string newDefinition(const std::string& logicalId, const std::optional<std::string>& producer)
    {
        const LogicalValue& logical = logicalValue(logicalId);
        const std::string definitionId = safeId(logicalId) + ".def" + std::to_string(definitionCounter_++);
        vfsim::CanonicalValue value;
        value.definitionId = definitionId;
        value.logicalId = logicalId;
        value.storage = logical.storage;
        value.dtype = logical.dtype;
        value.producerNodeId = producer;
        if (logical.storage == vfsim::CanonicalStorageKind::UB) {
            const std::string objectId = "ub." + safeId(logicalId);
            value.storageObjectId = objectId;
            output_.storageObjects.try_emplace(
                objectId, vfsim::CanonicalStorageObject{objectId, vfsim::CanonicalStorageKind::UB, {}, std::nullopt});
        }
        output_.values.emplace(definitionId, std::move(value));
        return definitionId;
    }

    std::string ensureEntry(const std::string& logicalId, Environment& environment)
    {
        const auto definition = environment.find(logicalId);
        if (definition != environment.end())
            return definition->second;
        const std::string definitionId = newDefinition(logicalId, std::nullopt);
        environment.emplace(logicalId, definitionId);
        return definitionId;
    }

    vfsim::CanonicalOperand operand(
        const std::string& definitionId, const std::string& logicalId, vfsim::CanonicalOperandRole role,
        bool output) const
    {
        const LogicalValue& logical = logicalValue(logicalId);
        vfsim::CanonicalOperand result;
        result.valueId = definitionId;
        result.role = role;
        result.dtype = logical.dtype;
        if (logical.storage == vfsim::CanonicalStorageKind::UB) {
            vfsim::CanonicalMemoryAccess memory;
            memory.baseObjectId = "ub." + safeId(logicalId);
            memory.accessKind = output ? vfsim::CanonicalAccessKind::WRITE : vfsim::CanonicalAccessKind::READ;
            result.memoryAccess = std::move(memory);
        }
        return result;
    }

    std::vector<const vfsim::NativeOperandSpec*> trackedOperands(
        const vfsim::NativeInstructionSpec* specification, vfsim::CatalogOperandDirection direction) const
    {
        std::vector<const vfsim::NativeOperandSpec*> result;
        if (specification == nullptr)
            return result;
        for (const vfsim::NativeOperandSpec& operand : specification->operands) {
            if (operand.direction == direction)
                result.push_back(&operand);
        }
        std::sort(result.begin(), result.end(), [](const auto* left, const auto* right) {
            return left->argumentIndex < right->argumentIndex;
        });
        return result;
    }

    std::optional<std::string> capturedScalar(const VfInst& instruction, const vfsim::NativeOperandSpec& expected)
    {
        const auto argument =
            std::find_if(instruction.arguments.begin(), instruction.arguments.end(), [&](const VfArgInfo& candidate) {
                return candidate.kind == VfArgKind::Immediate &&
                       candidate.argumentIndex == static_cast<uint32_t>(expected.argumentIndex);
            });
        if (argument == instruction.arguments.end())
            return std::nullopt;
        if (expected.kind != vfsim::CatalogArgumentKind::SCALAR &&
            expected.kind != vfsim::CatalogArgumentKind::REGISTER_OR_SCALAR) {
            return std::nullopt;
        }
        return registerScalar(instruction, "__pto_immediate_");
    }

    std::vector<std::pair<std::string, vfsim::CanonicalOperandRole>> instructionInputs(
        const VfInst& instruction, const vfsim::NativeInstructionSpec* specification)
    {
        const std::vector<const MemInfo*> sources = dataOperands(instruction.src);
        for (const MemInfo* source : sources)
            registerLogicalValue(*source);

        std::vector<std::pair<std::string, vfsim::CanonicalOperandRole>> result;
        const auto expectedInputs = trackedOperands(specification, vfsim::CatalogOperandDirection::INPUT);
        std::size_t sourceIndex = 0;
        for (const vfsim::NativeOperandSpec* expected : expectedInputs) {
            if (auto scalar = capturedScalar(instruction, *expected)) {
                const vfsim::CanonicalOperandRole catalogRole = canonicalRole(expected->role);
                result.emplace_back(
                    std::move(*scalar), catalogRole == vfsim::CanonicalOperandRole::UNKNOWN ?
                                            vfsim::CanonicalOperandRole::SCALAR :
                                            catalogRole);
                continue;
            }
            if (sourceIndex < sources.size()) {
                const MemInfo& source = *sources[sourceIndex++];
                const vfsim::CanonicalOperandRole fallback = source.location == MemLocation::UB ?
                                                                 vfsim::CanonicalOperandRole::MEMORY :
                                                                 vfsim::CanonicalOperandRole::SOURCE;
                const vfsim::CanonicalOperandRole catalogRole = canonicalRole(expected->role);
                result.emplace_back(
                    source.name, catalogRole == vfsim::CanonicalOperandRole::UNKNOWN ? fallback : catalogRole);
                continue;
            }
            if (expected->kind == vfsim::CatalogArgumentKind::SCALAR) {
                result.emplace_back(
                    registerScalar(instruction, "__pto_omitted_scalar_"), vfsim::CanonicalOperandRole::SCALAR);
            }
        }
        while (sourceIndex < sources.size()) {
            const MemInfo& source = *sources[sourceIndex++];
            result.emplace_back(
                source.name, source.location == MemLocation::UB ? vfsim::CanonicalOperandRole::MEMORY :
                                                                  vfsim::CanonicalOperandRole::SOURCE);
        }
        return result;
    }

    vfsim::CanonicalInstruction lowerInstruction(const VfInst& instruction, Environment& environment)
    {
        if (instruction.opName.empty())
            throw std::runtime_error("PTO instruction has no opcode");

        const vfsim::InstructionCatalog& catalog = vfsim::defaultInstructionCatalog();
        const std::string baseOpcode = catalog.canonicalOpcode(instruction.opName);
        const vfsim::NativeInstructionSpec* baseSpecification = catalog.lookup(baseOpcode);
        const std::string form = inferForm(instruction, baseSpecification);
        const std::string opcode = catalog.specializeOpcode(baseOpcode, form);
        const vfsim::NativeInstructionSpec* specification = catalog.lookup(opcode);

        vfsim::CanonicalInstruction result;
        result.instructionId = nodeId("instruction");
        result.opcode = opcode;
        result.form = form;
        result.instructionClass = specification == nullptr ? vfsim::CanonicalInstructionClass::COMPUTE :
                                                             canonicalClass(specification->instructionClass);

        for (const auto& [logicalId, role] : instructionInputs(instruction, specification)) {
            result.inputs.push_back(operand(ensureEntry(logicalId, environment), logicalId, role, false));
        }

        const std::vector<const MemInfo*> destinations = dataOperands(instruction.dst);
        if (destinations.empty())
            throw std::runtime_error("PTO instruction has no non-predicate destination: " + opcode);
        const auto expectedOutputs = trackedOperands(specification, vfsim::CatalogOperandDirection::OUTPUT);
        for (std::size_t index = 0; index < destinations.size(); ++index) {
            const MemInfo& destination = *destinations[index];
            registerLogicalValue(destination);
            vfsim::CanonicalOperandRole role = destination.location == MemLocation::UB ?
                                                   vfsim::CanonicalOperandRole::MEMORY :
                                                   vfsim::CanonicalOperandRole::DESTINATION;
            if (index < expectedOutputs.size()) {
                const vfsim::CanonicalOperandRole catalogRole = canonicalRole(expectedOutputs[index]->role);
                if (catalogRole != vfsim::CanonicalOperandRole::UNKNOWN)
                    role = catalogRole;
            }
            const std::string definitionId = newDefinition(destination.name, result.instructionId);
            result.outputs.push_back(operand(definitionId, destination.name, role, true));
            if (destination.location != MemLocation::UB)
                environment[destination.name] = definitionId;
        }
        return result;
    }

    void collectWrittenRegisters(const std::vector<VfNode>& nodes, std::unordered_set<std::string>& written)
    {
        for (const VfNode& node : nodes) {
            if (IsLoop(node)) {
                collectWrittenRegisters(AsLoop(node).body, written);
                continue;
            }
            if (!IsInst(node) || isIgnoredPredicateSetup(AsInst(node).opName))
                continue;
            for (const MemInfo& destination : AsInst(node).dst) {
                if (destination.location != MemLocation::PhyRegister)
                    continue;
                registerLogicalValue(destination);
                written.insert(destination.name);
            }
        }
    }

    vfsim::CanonicalLoop lowerLoop(const VfLoop& source, Environment& environment)
    {
        if (source.count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
            throw std::runtime_error("PTO loop count exceeds the CanonicalVfInfo range");

        vfsim::CanonicalLoop result;
        result.loopId = nodeId("loop");
        result.induction.variableId = "iter_" + safeId(result.loopId);
        result.count = static_cast<int64_t>(source.count);
        result.unroll = int64_t{1};

        std::unordered_set<std::string> writtenSet;
        collectWrittenRegisters(source.body, writtenSet);
        std::vector<std::string> written(writtenSet.begin(), writtenSet.end());
        std::sort(written.begin(), written.end());

        Environment entryEnvironment = environment;
        std::unordered_map<std::string, std::string> entries;
        for (const std::string& logicalId : written)
            entries.emplace(logicalId, ensureEntry(logicalId, entryEnvironment));

        Environment bodyEnvironment = entryEnvironment;
        result.body = lowerNodes(source.body, bodyEnvironment);
        Environment next = environment;
        std::unordered_map<std::string, std::string> exits;
        for (const std::string& logicalId : written) {
            const std::string backEdge = bodyEnvironment.at(logicalId);
            const std::string exit = newDefinition(logicalId, result.loopId);
            result.carriedValues.push_back(
                vfsim::CanonicalLoopCarriedValue{logicalId, entries.at(logicalId), backEdge, exit});
            exits.emplace(backEdge, exit);
            next[logicalId] = exit;
        }
        if (source.count != 0) {
            for (const auto& [logicalId, definitionId] : bodyEnvironment) {
                if (writtenSet.count(logicalId) != 0)
                    continue;
                const auto exit = exits.find(definitionId);
                next[logicalId] = exit == exits.end() ? definitionId : exit->second;
            }
        }
        environment = std::move(next);
        return result;
    }

    std::vector<vfsim::CanonicalNode> lowerNodes(const std::vector<VfNode>& nodes, Environment& environment)
    {
        std::vector<vfsim::CanonicalNode> result;
        result.reserve(nodes.size());
        for (const VfNode& node : nodes) {
            if (IsLoop(node)) {
                result.push_back(vfsim::CanonicalNode::makeLoop(lowerLoop(AsLoop(node), environment)));
                continue;
            }
            if (IsMemBar(node)) {
                vfsim::CanonicalMembar membar;
                membar.instructionId = nodeId("membar");
                membar.barrier = AsMemBar(node).name.empty() ? "VST_VLD" : toUpper(AsMemBar(node).name);
                result.push_back(vfsim::CanonicalNode::makeMembar(std::move(membar)));
                continue;
            }
            const VfInst& instruction = AsInst(node);
            if (isIgnoredPredicateSetup(instruction.opName)) {
                ++ignoredInstructionCount_;
                continue;
            }
            result.push_back(vfsim::CanonicalNode::makeInstruction(lowerInstruction(instruction, environment)));
        }
        return result;
    }
};

} // namespace

PtoCanonicalLoweringResult lowerPtoVfToCanonical(const VfInfo& vf) { return PtoCanonicalLowerer(vf).run(); }

} // namespace pto::mocker::vf
