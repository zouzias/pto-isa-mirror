// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: CANN-1.0

#include <algorithm>
#include <charconv>
#include <cmath>
#include <unordered_set>
#include <utility>

#include "api/native/CanonicalVfInfo.h"
#include "api/native/InstructionCatalog.h"
#include "api/native/UarchOverrideSchema.h"

namespace vfsim {
namespace {

struct NodeInfo {
    std::vector<std::string> scope;
    int64_t order = 0;
    std::string kind;
};

struct PendingDependency {
    std::string producer;
    std::string consumer;
    std::string path;
};

std::optional<int64_t> parseInt64(const std::string& text)
{
    if (text.empty())
        return std::nullopt;
    int64_t value = 0;
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc{} || result.ptr != end)
        return std::nullopt;
    return value;
}

std::optional<int64_t> resolveIntegerExpression(
    const CanonicalIntegerExpression& expression, const std::unordered_map<std::string, int64_t>& params)
{
    if (const auto* integer = std::get_if<int64_t>(&expression))
        return *integer;
    const std::string& symbol = std::get<std::string>(expression);
    auto parameter = params.find(symbol);
    if (parameter != params.end())
        return parameter->second;
    return parseInt64(symbol);
}

bool validInputRole(CanonicalOperandRole role)
{
    return role == CanonicalOperandRole::SOURCE || role == CanonicalOperandRole::MEMORY ||
           role == CanonicalOperandRole::SCALAR || role == CanonicalOperandRole::PREDICATE ||
           role == CanonicalOperandRole::CONFIG;
}

bool validOutputRole(CanonicalOperandRole role)
{
    return role == CanonicalOperandRole::DESTINATION || role == CanonicalOperandRole::MEMORY;
}

bool validInstructionClass(CanonicalInstructionClass value)
{
    return value == CanonicalInstructionClass::LOAD || value == CanonicalInstructionClass::STORE ||
           value == CanonicalInstructionClass::COMPUTE || value == CanonicalInstructionClass::CONTROL;
}

bool validStorageKind(CanonicalStorageKind value)
{
    return value == CanonicalStorageKind::REGISTER || value == CanonicalStorageKind::UB ||
           value == CanonicalStorageKind::SCALAR;
}

CatalogInstructionClass catalogClass(CanonicalInstructionClass value)
{
    switch (value) {
        case CanonicalInstructionClass::LOAD:
            return CatalogInstructionClass::LOAD;
        case CanonicalInstructionClass::STORE:
            return CatalogInstructionClass::STORE;
        case CanonicalInstructionClass::COMPUTE:
            return CatalogInstructionClass::COMPUTE;
        case CanonicalInstructionClass::CONTROL:
            return CatalogInstructionClass::CONTROL;
        case CanonicalInstructionClass::UNKNOWN:
            return CatalogInstructionClass::COMPUTE;
    }
    return CatalogInstructionClass::COMPUTE;
}

std::string operandRoleName(CanonicalOperandRole role)
{
    switch (role) {
        case CanonicalOperandRole::SOURCE:
            return "source";
        case CanonicalOperandRole::DESTINATION:
            return "destination";
        case CanonicalOperandRole::MEMORY:
            return "memory";
        case CanonicalOperandRole::SCALAR:
            return "scalar";
        case CanonicalOperandRole::PREDICATE:
            return "predicate";
        case CanonicalOperandRole::CONFIG:
            return "config";
        case CanonicalOperandRole::UNKNOWN:
            return "unknown";
    }
    return "unknown";
}

bool scopePrefix(const std::vector<std::string>& prefix, const std::vector<std::string>& scope)
{
    return prefix.size() <= scope.size() && std::equal(prefix.begin(), prefix.end(), scope.begin());
}

bool finiteScalar(const CanonicalScalar& value)
{
    if (const auto* number = std::get_if<double>(&value))
        return std::isfinite(*number);
    return true;
}

const char* scalarTypeName(const CanonicalScalar& value)
{
    if (std::holds_alternative<std::monostate>(value))
        return "NoneType";
    if (std::holds_alternative<bool>(value))
        return "bool";
    if (std::holds_alternative<int64_t>(value))
        return "int";
    if (std::holds_alternative<double>(value))
        return "float";
    return "str";
}

class CanonicalValidator {
public:
    explicit CanonicalValidator(const CanonicalVfInfo& vfInfo) : vfInfo_(vfInfo) {}

    CanonicalValidationResult run()
    {
        validateTopLevelMetadata();
        indexNodes(vfInfo_.context, {});
        validateStorageObjects();
        validateValues();
        validateNodes(vfInfo_.context, "context", {});
        validateProducedDefinitions();
        validatePendingDependencies();
        return std::move(result_);
    }

private:
    const CanonicalVfInfo& vfInfo_;
    CanonicalValidationResult result_;
    std::unordered_map<std::string, NodeInfo> nodeInfo_;
    int64_t nextOrder_ = 0;
    std::unordered_set<std::string> registeredNodeIds_;
    std::vector<PendingDependency> pendingDependencies_;
    std::unordered_map<std::string, std::vector<std::string>> producedDefinitions_;

    void error(
        std::string code, std::string message, std::string path,
        std::optional<CanonicalSourceLocation> location = std::nullopt,
        std::map<std::string, CanonicalScalar> context = {})
    {
        result_.diagnostics.push_back(CanonicalValidationDiagnostic{
            std::move(code), "error", std::move(message), std::move(path), std::move(location), std::move(context)});
    }

    void validateLocation(const std::optional<CanonicalSourceLocation>& location, const std::string& path)
    {
        if (!location)
            return;
        if (location->line && *location->line <= 0)
            error("invalid_int64", "Source line must be positive", path + ".line");
        if (location->column && *location->column <= 0)
            error("invalid_int64", "Source column must be positive", path + ".column");
    }

    void validateScalarMap(const std::map<std::string, CanonicalScalar>& values, const std::string& path)
    {
        for (const auto& [key, value] : values) {
            if (!finiteScalar(value))
                error("invalid_scalar_attribute", "Scalar must be finite", path + "." + key);
        }
    }

    void validateUarch()
    {
        validateScalarMap(vfInfo_.uarch, "uarch");
        for (const auto& [name, value] : vfInfo_.uarch) {
            if (isDeprecatedUarchOverrideField(name)) {
                error("deprecated_uarch_field", "Deprecated uarch field is no longer accepted", "uarch." + name);
                continue;
            }
            const auto expected = uarchOverrideFieldType(name);
            if (!expected) {
                error(
                    "unsupported_uarch_field", "C++ canonical frontend does not support unknown uarch field",
                    "uarch." + name, std::nullopt, {{"field", name}});
                continue;
            }
            if (!uarchOverrideFieldSupportsTarget(name, UarchOverrideTarget::CPP)) {
                error(
                    "unsupported_uarch_target", "uarch field is not supported by the C++ target", "uarch." + name,
                    std::nullopt, {{"field", name}, {"target", "cpp"}});
                continue;
            }
            const bool matches =
                (*expected == UarchOverrideFieldType::INTEGER && std::holds_alternative<int64_t>(value)) ||
                (*expected == UarchOverrideFieldType::BOOLEAN && std::holds_alternative<bool>(value)) ||
                (*expected == UarchOverrideFieldType::STRING && std::holds_alternative<std::string>(value));
            if (!matches) {
                error(
                    "uarch_field_type_mismatch",
                    "uarch." + name + " must use " + uarchOverrideFieldTypeName(*expected) + " type", "uarch." + name,
                    std::nullopt,
                    {{"field", name},
                     {"expected_type", std::string(uarchOverrideFieldTypeName(*expected))},
                     {"actual_type", std::string(scalarTypeName(value))}});
            }
        }
    }

    void validateTopLevelMetadata()
    {
        if (vfInfo_.schemaVersion != CANONICAL_VF_INFO_SCHEMA_VERSION)
            error("unsupported_schema_version", "Unsupported schema version", "schema_version");
        validateUarch();
        validateScalarMap(vfInfo_.source, "source");
    }

    void indexNodes(const std::vector<CanonicalNode>& nodes, const std::vector<std::string>& scope)
    {
        for (const auto& node : nodes) {
            if (const auto* instruction = std::get_if<CanonicalInstruction>(&node.payload)) {
                nodeInfo_.emplace(instruction->instructionId, NodeInfo{scope, nextOrder_++, "instruction"});
                continue;
            }
            if (const auto* loopPtr = std::get_if<std::shared_ptr<const CanonicalLoop>>(&node.payload)) {
                if (!*loopPtr) {
                    ++nextOrder_;
                    continue;
                }
                const CanonicalLoop& loop = **loopPtr;
                nodeInfo_.emplace(loop.loopId, NodeInfo{scope, nextOrder_++, "loop"});
                auto childScope = scope;
                childScope.push_back(loop.loopId);
                indexNodes(loop.body, childScope);
                continue;
            }
            const auto& membar = std::get<CanonicalMembar>(node.payload);
            nodeInfo_.emplace(membar.instructionId, NodeInfo{scope, nextOrder_++, "membar"});
        }
    }

    void validateStorageObjects()
    {
        for (const auto& [objectId, storageObject] : vfInfo_.storageObjects) {
            const std::string path = "storage_objects." + objectId;
            if (objectId.empty() || storageObject.objectId != objectId) {
                error(
                    "invalid_storage_object_identity", "Storage object key must match object_id", path,
                    storageObject.sourceLocation);
            }
            if (storageObject.storage != CanonicalStorageKind::UB) {
                error(
                    "unsupported_storage_object_kind", "Canonical v1 storage objects must use UB", path,
                    storageObject.sourceLocation);
            }
            if (std::any_of(
                    storageObject.shape.begin(), storageObject.shape.end(), [](int64_t dim) { return dim < 0; })) {
                error("invalid_int64", "Storage object shape must be non-negative", path, storageObject.sourceLocation);
            }
            validateLocation(storageObject.sourceLocation, path + ".source_location");
        }
    }

    void validateValues()
    {
        for (const auto& [definitionId, value] : vfInfo_.values) {
            const std::string path = "values." + definitionId;
            if (definitionId.empty() || value.definitionId != definitionId)
                error("invalid_value_identity", "Value key must match definition_id", path, value.sourceLocation);
            if (value.logicalId.empty())
                error("missing_logical_id", "Value must declare logical_id", path, value.sourceLocation);
            if (!validStorageKind(value.storage))
                error("unsupported_storage", "Value has unsupported storage", path, value.sourceLocation);
            if (value.dtype.empty())
                error("missing_value_dtype", "Value must declare dtype", path, value.sourceLocation);
            if (std::any_of(value.shape.begin(), value.shape.end(), [](int64_t dim) { return dim < 0; }))
                error("invalid_int64", "Value shape must be non-negative", path, value.sourceLocation);
            validateLocation(value.sourceLocation, path + ".source_location");
            if (value.storage == CanonicalStorageKind::UB) {
                if (!value.storageObjectId || !vfInfo_.storageObjects.count(*value.storageObjectId)) {
                    error(
                        "unknown_storage_object", "UB value must reference a declared storage object", path,
                        value.sourceLocation);
                }
            } else if (value.storageObjectId) {
                error(
                    "storage_object_on_non_ub_value", "Only UB values may reference a storage object", path,
                    value.sourceLocation);
            }
        }
    }

    void registerNodeId(
        const std::string& nodeId, const std::string& path, const std::optional<CanonicalSourceLocation>& location)
    {
        if (nodeId.empty() || !registeredNodeIds_.insert(nodeId).second)
            error("duplicate_node_id", "Node ID must be globally unique", path, location);
    }

    bool producerVisibleTo(const std::string& producerId, const std::string& consumerId) const
    {
        auto producer = nodeInfo_.find(producerId);
        auto consumer = nodeInfo_.find(consumerId);
        if (producer == nodeInfo_.end() || consumer == nodeInfo_.end() ||
            producer->second.order >= consumer->second.order ||
            !scopePrefix(producer->second.scope, consumer->second.scope)) {
            return false;
        }
        const auto& producerInfo = producer->second;
        const auto& consumerScope = consumer->second.scope;
        return producerInfo.kind != "loop" || consumerScope.size() <= producerInfo.scope.size() ||
               consumerScope[producerInfo.scope.size()] != producerId;
    }

    void validateDependencies(
        const std::vector<CanonicalDependencyRef>& dependencies, const std::string& consumer, const std::string& path)
    {
        for (size_t index = 0; index < dependencies.size(); ++index) {
            const auto& dependency = dependencies[index];
            const std::string itemPath = path + "[" + std::to_string(index) + "]";
            if (dependency.producerNodeId == consumer)
                error("self_dependency", "Node cannot depend on itself", itemPath);
            if (dependency.kind != CanonicalDependencyKind::MEMORY &&
                dependency.kind != CanonicalDependencyKind::CONTROL) {
                error("unsupported_dependency_kind", "Explicit dependency must be memory or control", itemPath);
            }
            if (dependency.operandIndex && *dependency.operandIndex < 0) {
                error("invalid_dependency_operand_index", "Dependency operand_index must be non-negative", itemPath);
            }
            auto producer = nodeInfo_.find(dependency.producerNodeId);
            if (producer != nodeInfo_.end() && !producerVisibleTo(dependency.producerNodeId, consumer)) {
                error(
                    "dependency_producer_not_visible", "Dependency producer is not visible before consumer", itemPath);
            }
            pendingDependencies_.push_back({dependency.producerNodeId, consumer, itemPath});
        }
    }

    const NativeInstructionSpec* validateInstructionHeader(
        const CanonicalInstruction& instruction, const std::string& nodePath)
    {
        if (instruction.opcode.empty()) {
            error("missing_opcode", "Instruction must declare opcode", nodePath, instruction.sourceLocation);
        }
        if (!validInstructionClass(instruction.instructionClass)) {
            error("missing_instruction_class", "Instruction class is invalid", nodePath, instruction.sourceLocation);
        }
        if (instruction.form.empty()) {
            error("missing_instruction_form", "Instruction form is required", nodePath, instruction.sourceLocation);
        }
        const NativeInstructionSpec* catalogSpec = defaultInstructionCatalog().lookup(instruction.opcode);
        if (catalogSpec == nullptr)
            return nullptr;
        if (instruction.opcode != catalogSpec->opcode) {
            error(
                "noncanonical_opcode", "Known opcode must use its canonical Catalog name", nodePath,
                instruction.sourceLocation);
        }
        if (validInstructionClass(instruction.instructionClass) &&
            catalogClass(instruction.instructionClass) != catalogSpec->instructionClass) {
            error(
                "catalog_instruction_class_mismatch", "Instruction class conflicts with Catalog semantics", nodePath,
                instruction.sourceLocation);
        }
        const bool validForm = catalogSpec->virtualOpcode || catalogSpec->forms.count(instruction.form) ||
                               catalogSpec->specializations.count(instruction.form);
        if (!validForm) {
            error(
                "catalog_instruction_form_mismatch", "Instruction form conflicts with Catalog semantics", nodePath,
                instruction.sourceLocation);
        }
        auto specialized = catalogSpec->specializations.find(instruction.form);
        if (specialized != catalogSpec->specializations.end() && specialized->second != instruction.opcode) {
            error(
                "catalog_specialization_required", "Virtual opcode/form must use specialized opcode", nodePath,
                instruction.sourceLocation);
        }
        return catalogSpec;
    }

    bool validateOperands(
        const CanonicalInstruction& instruction, const std::vector<CanonicalOperand>& operands, bool input,
        const std::string& nodePath, const std::unordered_set<std::string>& inductionVariables)
    {
        bool hasMemory = false;
        for (size_t operandIndex = 0; operandIndex < operands.size(); ++operandIndex) {
            const auto& operand = operands[operandIndex];
            const std::string operandPath =
                nodePath + (input ? ".inputs[" : ".outputs[") + std::to_string(operandIndex) + "]";
            auto valueIt = vfInfo_.values.find(operand.valueId);
            if (valueIt == vfInfo_.values.end()) {
                error(
                    "unknown_value_reference", "Operand references unknown value", operandPath,
                    instruction.sourceLocation);
                continue;
            }
            const CanonicalValue& value = valueIt->second;
            if (!(input ? validInputRole(operand.role) : validOutputRole(operand.role))) {
                error(
                    "operand_role_direction_mismatch", "Operand role is invalid for direction", operandPath,
                    instruction.sourceLocation);
            }
            if (operand.dtype && *operand.dtype != value.dtype) {
                error(
                    "operand_dtype_mismatch", "Operand dtype differs from value", operandPath,
                    instruction.sourceLocation);
            }
            const bool isUb = value.storage == CanonicalStorageKind::UB;
            if (isUb && !operand.memoryAccess) {
                error(
                    "missing_memory_access", "UB operand requires memory metadata", operandPath,
                    instruction.sourceLocation);
            }
            if (!isUb && operand.memoryAccess) {
                error(
                    "memory_access_on_non_ub_value", "Only UB operands may carry memory metadata", operandPath,
                    instruction.sourceLocation);
            }
            if (!operand.memoryAccess)
                continue;
            hasMemory =
                validateMemoryAccess(instruction, operand, value, input, operandPath, inductionVariables) || hasMemory;
        }
        return hasMemory;
    }

    bool validateMemoryAccess(
        const CanonicalInstruction& instruction, const CanonicalOperand& operand, const CanonicalValue& value,
        bool input, const std::string& operandPath, const std::unordered_set<std::string>& inductionVariables)
    {
        const auto& memory = *operand.memoryAccess;
        if (operand.role != CanonicalOperandRole::MEMORY) {
            error(
                "memory_operand_role_mismatch", "Memory operand must use memory role", operandPath,
                instruction.sourceLocation);
        }
        if (!value.storageObjectId || memory.baseObjectId != *value.storageObjectId) {
            error(
                "memory_base_object_mismatch", "Memory base must match stable storage object", operandPath,
                instruction.sourceLocation);
        }
        if (!vfInfo_.storageObjects.count(memory.baseObjectId)) {
            error(
                "unknown_memory_base_object", "Memory base object is not declared", operandPath,
                instruction.sourceLocation);
        }
        const CanonicalAccessKind expected = input ? CanonicalAccessKind::READ : CanonicalAccessKind::WRITE;
        const bool directionMatches = memory.accessKind == expected;
        if (!directionMatches) {
            error(
                "memory_access_direction_mismatch", "Memory direction mismatch", operandPath,
                instruction.sourceLocation);
        }
        if (memory.span && *memory.span <= 0) {
            error("invalid_memory_span", "Memory span must be positive", operandPath, instruction.sourceLocation);
        }
        std::unordered_set<std::string> affineVariables;
        for (const auto& term : memory.offset.terms) {
            if (term.variableId.empty() || !affineVariables.insert(term.variableId).second) {
                error(
                    "invalid_affine_term", "Affine variables must be unique", operandPath, instruction.sourceLocation);
            }
            if (!inductionVariables.count(term.variableId) && !vfInfo_.params.count(term.variableId)) {
                error(
                    "undeclared_affine_variable", "Affine variable is not in scope", operandPath,
                    instruction.sourceLocation);
            }
        }
        return directionMatches;
    }

    void validateInputVisibility(const CanonicalInstruction& instruction, const std::string& nodePath)
    {
        for (size_t inputIndex = 0; inputIndex < instruction.inputs.size(); ++inputIndex) {
            auto valueIt = vfInfo_.values.find(instruction.inputs[inputIndex].valueId);
            if (valueIt == vfInfo_.values.end() || !valueIt->second.producerNodeId)
                continue;
            const std::string& producer = *valueIt->second.producerNodeId;
            const std::string inputPath = nodePath + ".inputs[" + std::to_string(inputIndex) + "]";
            if (producer == instruction.instructionId)
                error("self_produced_input", "Input references own output", inputPath);
            else if (!producerVisibleTo(producer, instruction.instructionId))
                error("input_definition_not_visible", "Input definition producer is not visible", inputPath);
        }
    }

    void validateAndRecordOutputs(const CanonicalInstruction& instruction, const std::string& nodePath)
    {
        for (size_t outputIndex = 0; outputIndex < instruction.outputs.size(); ++outputIndex) {
            const auto& output = instruction.outputs[outputIndex];
            auto valueIt = vfInfo_.values.find(output.valueId);
            if (valueIt != vfInfo_.values.end() && valueIt->second.producerNodeId != instruction.instructionId) {
                error(
                    "output_producer_mismatch", "Output definition must name producing instruction",
                    nodePath + ".outputs[" + std::to_string(outputIndex) + "]");
            }
            producedDefinitions_[instruction.instructionId].push_back(output.valueId);
        }
    }

    void validateInstructionMemoryClass(
        const CanonicalInstruction& instruction, const std::string& nodePath, bool hasReadMemory, bool hasWriteMemory,
        bool hasInputMemory, bool hasOutputMemory)
    {
        if (instruction.instructionClass == CanonicalInstructionClass::LOAD && !hasReadMemory)
            error("load_without_memory_read", "Load requires memory read", nodePath);
        if (instruction.instructionClass == CanonicalInstructionClass::LOAD && hasOutputMemory) {
            error("instruction_class_memory_access_mismatch", "Load instructions cannot write memory", nodePath);
        }
        if (instruction.instructionClass == CanonicalInstructionClass::STORE && !hasWriteMemory)
            error("store_without_memory_write", "Store requires memory write", nodePath);
        if (instruction.instructionClass == CanonicalInstructionClass::STORE && hasInputMemory) {
            error("instruction_class_memory_access_mismatch", "Store instructions cannot read memory", nodePath);
        }
        if ((instruction.instructionClass == CanonicalInstructionClass::COMPUTE ||
             instruction.instructionClass == CanonicalInstructionClass::CONTROL) &&
            (hasInputMemory || hasOutputMemory)) {
            error(
                "instruction_class_memory_access_mismatch", "Compute/control instructions cannot access memory",
                nodePath);
        }
    }

    static std::vector<const CanonicalOperand*> actualCatalogOperands(const std::vector<CanonicalOperand>& operands)
    {
        std::vector<const CanonicalOperand*> result;
        for (const auto& operand : operands) {
            if (operand.role != CanonicalOperandRole::PREDICATE && operand.role != CanonicalOperandRole::CONFIG)
                result.push_back(&operand);
        }
        return result;
    }

    static bool catalogStorageMatches(CatalogArgumentKind kind, CanonicalStorageKind storage)
    {
        switch (kind) {
            case CatalogArgumentKind::REGISTER:
                return storage == CanonicalStorageKind::REGISTER;
            case CatalogArgumentKind::UB:
                return storage == CanonicalStorageKind::UB;
            case CatalogArgumentKind::SCALAR:
                return storage == CanonicalStorageKind::SCALAR;
            case CatalogArgumentKind::REGISTER_OR_SCALAR:
                return storage == CanonicalStorageKind::REGISTER || storage == CanonicalStorageKind::SCALAR;
            case CatalogArgumentKind::PREDICATE:
            case CatalogArgumentKind::CONFIG:
                return true;
        }
        return true;
    }

    void validateCatalogOperands(
        const CanonicalInstruction& instruction, const std::vector<const CanonicalOperand*>& actual,
        const std::vector<const NativeOperandSpec*>& expected, const std::string& direction,
        const std::string& nodePath)
    {
        if (actual.size() != expected.size()) {
            error(
                "catalog_operand_count_mismatch", "Operand count conflicts with Catalog signature", nodePath,
                instruction.sourceLocation);
            return;
        }
        for (size_t operandIndex = 0; operandIndex < actual.size(); ++operandIndex) {
            const auto& operand = *actual[operandIndex];
            const auto& operandSpec = *expected[operandIndex];
            const std::string operandPath = nodePath + "." + direction + "[" + std::to_string(operandIndex) + "]";
            if (operandRoleName(operand.role) != operandSpec.role) {
                error(
                    "catalog_operand_role_mismatch", "Operand role conflicts with Catalog signature", operandPath,
                    instruction.sourceLocation);
            }
            auto value = vfInfo_.values.find(operand.valueId);
            if (value == vfInfo_.values.end())
                continue;
            if (!catalogStorageMatches(operandSpec.kind, value->second.storage)) {
                error(
                    "catalog_operand_storage_mismatch", "Operand storage conflicts with Catalog signature", operandPath,
                    instruction.sourceLocation);
            }
        }
    }

    void validateCatalogSignature(
        const CanonicalInstruction& instruction, const NativeInstructionSpec& catalogSpec, const std::string& nodePath)
    {
        std::vector<const NativeOperandSpec*> expectedInputs;
        std::vector<const NativeOperandSpec*> expectedOutputs;
        for (const auto& operand : catalogSpec.operands) {
            if (operand.direction == CatalogOperandDirection::INPUT)
                expectedInputs.push_back(&operand);
            else if (operand.direction == CatalogOperandDirection::OUTPUT)
                expectedOutputs.push_back(&operand);
        }
        validateCatalogOperands(
            instruction, actualCatalogOperands(instruction.inputs), expectedInputs, "inputs", nodePath);
        validateCatalogOperands(
            instruction, actualCatalogOperands(instruction.outputs), expectedOutputs, "outputs", nodePath);
    }

    void validateInstruction(
        const CanonicalInstruction& instruction, const std::string& nodePath,
        const std::unordered_set<std::string>& inductionVariables)
    {
        registerNodeId(instruction.instructionId, nodePath, instruction.sourceLocation);
        validateLocation(instruction.sourceLocation, nodePath + ".source_location");
        const NativeInstructionSpec* catalogSpec = validateInstructionHeader(instruction, nodePath);
        validateScalarMap(instruction.attributes, nodePath + ".attributes");
        const bool hasReadMemory =
            validateOperands(instruction, instruction.inputs, true, nodePath, inductionVariables);
        const bool hasWriteMemory =
            validateOperands(instruction, instruction.outputs, false, nodePath, inductionVariables);
        const bool hasInputMemory = std::any_of(
            instruction.inputs.begin(), instruction.inputs.end(),
            [](const CanonicalOperand& operand) { return operand.memoryAccess.has_value(); });
        const bool hasOutputMemory = std::any_of(
            instruction.outputs.begin(), instruction.outputs.end(),
            [](const CanonicalOperand& operand) { return operand.memoryAccess.has_value(); });
        validateInputVisibility(instruction, nodePath);
        validateAndRecordOutputs(instruction, nodePath);
        validateInstructionMemoryClass(
            instruction, nodePath, hasReadMemory, hasWriteMemory, hasInputMemory, hasOutputMemory);
        if (catalogSpec != nullptr)
            validateCatalogSignature(instruction, *catalogSpec, nodePath);
        validateDependencies(instruction.dependencies, instruction.instructionId, nodePath + ".dependencies");
    }

    void validateLoopBounds(
        const CanonicalLoop& loop, const std::string& nodePath,
        const std::unordered_set<std::string>& inductionVariables)
    {
        const auto count = resolveIntegerExpression(loop.count, vfInfo_.params);
        const auto unroll = resolveIntegerExpression(loop.unroll, vfInfo_.params);
        const auto start = resolveIntegerExpression(loop.induction.start, vfInfo_.params);
        const auto step = resolveIntegerExpression(loop.induction.step, vfInfo_.params);
        if (!count)
            error("unresolved_parameter", "Loop count cannot be resolved", nodePath + ".count");
        else if (*count < 0)
            error("invalid_loop_count", "Loop count must be non-negative", nodePath);
        if (!unroll)
            error("unresolved_parameter", "Loop unroll cannot be resolved", nodePath + ".unroll");
        else if (*unroll <= 0)
            error("invalid_loop_unroll", "Loop unroll must be positive", nodePath);
        if (!start)
            error("unresolved_parameter", "Induction start cannot be resolved", nodePath);
        if (!step)
            error("unresolved_parameter", "Induction step cannot be resolved", nodePath);
        else if (*step == 0)
            error("invalid_induction_step", "Induction step cannot be zero", nodePath);
        const std::string& variableId = loop.induction.variableId;
        if (variableId.empty() || inductionVariables.count(variableId) || vfInfo_.params.count(variableId))
            error("invalid_induction_variable", "Induction variable is invalid", nodePath);
    }

    bool visibleBeforeLoop(
        const std::optional<std::string>& producerId,
        const std::unordered_map<std::string, NodeInfo>::const_iterator& loopInfo) const
    {
        if (!producerId)
            return true;
        auto producer = nodeInfo_.find(*producerId);
        if (producer == nodeInfo_.end() || loopInfo == nodeInfo_.end() ||
            producer->second.order >= loopInfo->second.order ||
            !scopePrefix(producer->second.scope, loopInfo->second.scope)) {
            return false;
        }
        return producer->second.kind != "loop" || loopInfo->second.scope.size() <= producer->second.scope.size() ||
               loopInfo->second.scope[producer->second.scope.size()] != *producerId;
    }

    void validateCarriedValue(
        const CanonicalLoop& loop, const CanonicalLoopCarriedValue& carried, const std::string& carriedPath,
        const std::vector<std::string>& loopScope,
        const std::unordered_map<std::string, NodeInfo>::const_iterator& loopInfo)
    {
        auto entryIt = vfInfo_.values.find(carried.entryValueId);
        auto backIt = vfInfo_.values.find(carried.backEdgeValueId);
        auto exitIt = vfInfo_.values.find(carried.exitValueId);
        if (entryIt == vfInfo_.values.end() || backIt == vfInfo_.values.end() || exitIt == vfInfo_.values.end()) {
            error("unknown_loop_carried_value", "Unknown loop-carried definition", carriedPath);
            return;
        }
        const CanonicalValue& entry = entryIt->second;
        const CanonicalValue& back = backIt->second;
        const CanonicalValue& exit = exitIt->second;
        if (entry.logicalId != carried.logicalId || exit.logicalId != carried.logicalId) {
            error(
                "loop_carried_logical_id_mismatch", "Loop entry and exit logical IDs must match the carried state",
                carriedPath);
        }
        if (entry.storage != back.storage || entry.storage != exit.storage || entry.dtype != back.dtype ||
            entry.dtype != exit.dtype || entry.shape != back.shape || entry.shape != exit.shape ||
            entry.storageObjectId != back.storageObjectId || entry.storageObjectId != exit.storageObjectId) {
            error("loop_carried_type_mismatch", "Loop-carried value metadata must match", carriedPath);
        }
        if (!visibleBeforeLoop(entry.producerNodeId, loopInfo))
            error("loop_entry_not_visible", "Loop entry is not visible before loop", carriedPath);
        if (carried.backEdgeValueId != carried.entryValueId) {
            auto producer = back.producerNodeId ? nodeInfo_.find(*back.producerNodeId) : nodeInfo_.end();
            const bool producedInBody = producer != nodeInfo_.end() && producer->second.scope == loopScope;
            if (!producedInBody && !visibleBeforeLoop(back.producerNodeId, loopInfo)) {
                error("loop_back_edge_out_of_scope", "Back-edge definition must be visible at loop tail", carriedPath);
            }
        }
        if (exit.producerNodeId != loop.loopId)
            error("loop_exit_producer_mismatch", "Loop exit must be produced by loop node", carriedPath);
        producedDefinitions_[loop.loopId].push_back(carried.exitValueId);
    }

    void validateLoopCarriedValues(const CanonicalLoop& loop, const std::string& nodePath)
    {
        auto loopInfo = nodeInfo_.find(loop.loopId);
        std::vector<std::string> loopScope;
        if (loopInfo != nodeInfo_.end()) {
            loopScope = loopInfo->second.scope;
            loopScope.push_back(loop.loopId);
        }
        std::unordered_set<std::string> carriedLogicalIds;
        for (size_t index = 0; index < loop.carriedValues.size(); ++index) {
            const auto& carried = loop.carriedValues[index];
            const std::string carriedPath = nodePath + ".carried_values[" + std::to_string(index) + "]";
            if (carried.logicalId.empty() || !carriedLogicalIds.insert(carried.logicalId).second) {
                error("duplicate_loop_carried_value", "Loop-carried logical_id must be unique", carriedPath);
            }
            validateCarriedValue(loop, carried, carriedPath, loopScope, loopInfo);
        }
    }

    void validateLoop(
        const CanonicalLoop& loop, const std::string& nodePath, std::unordered_set<std::string> inductionVariables)
    {
        registerNodeId(loop.loopId, nodePath, loop.sourceLocation);
        validateLocation(loop.sourceLocation, nodePath + ".source_location");
        validateLoopBounds(loop, nodePath, inductionVariables);
        validateLoopCarriedValues(loop, nodePath);
        inductionVariables.insert(loop.induction.variableId);
        validateNodes(loop.body, nodePath + ".body", std::move(inductionVariables));
    }

    void validateMembar(const CanonicalMembar& membar, const std::string& nodePath)
    {
        registerNodeId(membar.instructionId, nodePath, membar.sourceLocation);
        validateLocation(membar.sourceLocation, nodePath + ".source_location");
        if (membar.barrier.empty())
            error("missing_membar_type", "Membar type is required", nodePath, membar.sourceLocation);
        validateDependencies(membar.dependencies, membar.instructionId, nodePath + ".dependencies");
    }

    void validateNodes(
        const std::vector<CanonicalNode>& nodes, const std::string& path,
        std::unordered_set<std::string> inductionVariables)
    {
        for (size_t index = 0; index < nodes.size(); ++index) {
            const CanonicalNode& node = nodes[index];
            const std::string nodePath = path + "[" + std::to_string(index) + "]";
            if (const auto* instruction = std::get_if<CanonicalInstruction>(&node.payload)) {
                validateInstruction(*instruction, nodePath, inductionVariables);
                continue;
            }
            if (const auto* loopPtr = std::get_if<std::shared_ptr<const CanonicalLoop>>(&node.payload)) {
                if (!*loopPtr) {
                    error("invalid_loop_payload", "Loop payload cannot be null", nodePath);
                    continue;
                }
                validateLoop(**loopPtr, nodePath, inductionVariables);
                continue;
            }
            validateMembar(std::get<CanonicalMembar>(node.payload), nodePath);
        }
    }

    void validateProducedDefinitions()
    {
        for (const auto& [definitionId, value] : vfInfo_.values) {
            if (!value.producerNodeId)
                continue;
            auto producer = nodeInfo_.find(*value.producerNodeId);
            if (producer == nodeInfo_.end()) {
                error(
                    "unknown_value_producer", "Value references unknown producer node",
                    "values." + definitionId + ".producer_node_id", value.sourceLocation);
                continue;
            }
            if (producer->second.kind == "membar") {
                error(
                    "invalid_value_producer_kind", "Membar cannot produce a value definition",
                    "values." + definitionId + ".producer_node_id", value.sourceLocation);
                continue;
            }
            const auto emitted = producedDefinitions_.find(*value.producerNodeId);
            const int64_t count =
                emitted == producedDefinitions_.end() ?
                    0 :
                    static_cast<int64_t>(std::count(emitted->second.begin(), emitted->second.end(), definitionId));
            if (count == 0) {
                error(
                    "producer_definition_not_emitted", "Producer node does not emit the claimed value definition",
                    "values." + definitionId + ".producer_node_id", value.sourceLocation);
            } else if (count > 1) {
                error(
                    "definition_emitted_multiple_times", "Producer node emits the same definition more than once",
                    "values." + definitionId + ".producer_node_id", value.sourceLocation);
            }
        }
    }

    void validatePendingDependencies()
    {
        for (const auto& dependency : pendingDependencies_) {
            if (!nodeInfo_.count(dependency.producer)) {
                error("unknown_dependency_producer", "Dependency references unknown producer node", dependency.path);
            }
        }
    }
};

} // namespace

CanonicalValidationResult validateCanonicalVfInfo(const CanonicalVfInfo& vfInfo)
{
    return CanonicalValidator(vfInfo).run();
}

} // namespace vfsim
