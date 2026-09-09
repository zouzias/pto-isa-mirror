/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pto/costmodel/a5/cce_costmodel/vec_cycle_generated.hpp"
#include "pto/costmodel/perf_sim/config.hpp"
#include "pto/costmodel/vfsim/pto_adapter/pto_canonical_lowering.hpp"
#include "pto/costmodel/vfsim/pto_adapter/vfsim_cost_model.hpp"

#include "api/native/CanonicalVfInfo.h"
#include "native/ParamDB.h"
#include "native/SimulatorRunner.h"

namespace pto::mocker::vf {
namespace {

uint64_t fallbackNodes(const std::vector<VfNode>& nodes, uint64_t multiplier)
{
    uint64_t total = 0;
    for (const VfNode& node : nodes) {
        if (IsLoop(node)) {
            const VfLoop& loop = AsLoop(node);
            total += fallbackNodes(loop.body, multiplier * loop.count);
        } else if (IsInst(node)) {
            total += FallbackVecCycle(AsInst(node).opName) * multiplier;
        } else {
            total += kMemBarPenaltyPlaceholder * multiplier;
        }
    }
    return total;
}

uint64_t fallback(const std::vector<VfInfo>& vfs)
{
    uint64_t total = 0;
    for (const VfInfo& vf : vfs)
        total += fallbackNodes(vf.tree, 1);
    return total;
}

bool formsSupported(const std::vector<vfsim::CanonicalNode>& nodes, const vfsim::ParamDB& db, std::string& reason)
{
    for (const vfsim::CanonicalNode& node : nodes) {
        if (const auto* loop = std::get_if<std::shared_ptr<const vfsim::CanonicalLoop>>(&node.payload)) {
            if (*loop == nullptr) {
                reason = "canonical loop node has no loop body";
                return false;
            }
            if (!formsSupported((*loop)->body, db, reason))
                return false;
            continue;
        }
        if (std::holds_alternative<vfsim::CanonicalMembar>(node.payload))
            continue;

        const vfsim::CanonicalInstruction& instruction = std::get<vfsim::CanonicalInstruction>(node.payload);
        if (instruction.opcode.empty()) {
            reason = "canonical instruction opcode is empty";
            return false;
        }
        if (instruction.form.empty()) {
            reason = "instruction " + instruction.opcode + " has no canonical form";
            return false;
        }
        if (!db.hasInst(instruction.opcode, instruction.form)) {
            reason = "ParamDB has no entry for " + instruction.opcode + "/" + instruction.form;
            return false;
        }
    }
    return true;
}

int statusPriority(VfPredictionStatus status)
{
    switch (status) {
        case VfPredictionStatus::SIMULATOR_ERROR:
            return 4;
        case VfPredictionStatus::INVALID_TRACE:
            return 3;
        case VfPredictionStatus::UNSUPPORTED_FORM:
            return 2;
        case VfPredictionStatus::EMPTY_PROGRAM:
            return 1;
        case VfPredictionStatus::VF_SIM_HIT:
        case VfPredictionStatus::PARTIAL_FALLBACK:
            return 0;
    }
    return 0;
}

void updateFailureStatus(VfPredictionResult& result, VfPredictionStatus status)
{
    if (statusPriority(status) > statusPriority(result.status))
        result.status = status;
}

bool isConfigDirectory(const std::filesystem::path& path)
{
    return std::filesystem::is_regular_file(path / "isa.json") && std::filesystem::is_regular_file(path / "uarch.json");
}

std::optional<std::filesystem::path> asVfSimConfigRoot(const std::filesystem::path& candidate)
{
    if (isConfigDirectory(candidate)) {
        return std::filesystem::absolute(candidate);
    }
    if (isConfigDirectory(candidate / "configs")) {
        return std::filesystem::absolute(candidate);
    }
    return std::nullopt;
}

std::optional<std::filesystem::path> searchFromAncestor(std::filesystem::path current)
{
    while (!current.empty()) {
        const std::filesystem::path internalRoot = current / "pkg_inc/pto/costmodel/vfsim";
        if (auto found = asVfSimConfigRoot(internalRoot)) {
            return found;
        }
        if (auto found = asVfSimConfigRoot(current)) {
            return found;
        }

        std::error_code error;
        for (const auto& entry : std::filesystem::directory_iterator(current, error)) {
            if (!entry.is_directory(error)) {
                continue;
            }
            const std::string name = entry.path().filename().string();
            if (name.size() < 6 || name.substr(name.size() - 6) != "-linux") {
                continue;
            }
            if (auto found = asVfSimConfigRoot(entry.path() / "pkg_inc/pto/costmodel/vfsim")) {
                return found;
            }
        }

        const std::filesystem::path parent = current.parent_path();
        if (parent == current) {
            break;
        }
        current = parent;
    }
    return std::nullopt;
}

std::filesystem::path requireConfigRoot(const std::filesystem::path& candidate, const char* source)
{
    if (auto root = asVfSimConfigRoot(candidate)) {
        return *root;
    }
    throw std::runtime_error(std::string(source) + " does not contain isa.json and uarch.json: " + candidate.string());
}

std::filesystem::path resolveConfigRoot(const VfPredictionOptions& options)
{
    if (!options.configDir.empty()) {
        return requireConfigRoot(options.configDir, "VfPredictionOptions::configDir");
    }

    const std::string& perfSimConfigDir = ::pto::perf_sim::GetConfig().vfsim_config_dir;
    if (!perfSimConfigDir.empty()) {
        return requireConfigRoot(perfSimConfigDir, "PerfSimConfig::vfsim_config_dir");
    }

    if (const char* environment = std::getenv("PTO_VFSIM_CONFIG_DIR"); environment != nullptr && *environment) {
        return requireConfigRoot(environment, "PTO_VFSIM_CONFIG_DIR");
    }

    if (auto root = searchFromAncestor(std::filesystem::current_path())) {
        return *root;
    }
#if defined(__linux__)
    std::error_code error;
    const std::filesystem::path executable = std::filesystem::read_symlink("/proc/self/exe", error);
    if (!error) {
        if (auto root = searchFromAncestor(executable.parent_path())) {
            return *root;
        }
    }
#endif

    throw std::runtime_error("VfSim configs were not found; set VfPredictionOptions::configDir, "
                             "PerfSimConfig::vfsim_config_dir, or PTO_VFSIM_CONFIG_DIR");
}

const vfsim::ParamDB& getParamDb(const std::filesystem::path& configRoot)
{
    static thread_local std::filesystem::path cachedRoot;
    static thread_local std::unique_ptr<vfsim::ParamDB> cachedDb;
    if (!cachedDb || cachedRoot != configRoot) {
        cachedDb = std::make_unique<vfsim::ParamDB>(configRoot);
        cachedRoot = configRoot;
    }
    return *cachedDb;
}

void addDiagnostic(VfPredictionResult& result, std::size_t vfIndex, VfPredictionStatus status, const std::string& reason)
{
    std::ostringstream message;
    message << "vf[" << vfIndex << "]: " << toString(status) << ": " << reason;
    result.diagnostics.push_back(message.str());
}

} // namespace

const char* toString(VfPredictionStatus status)
{
    switch (status) {
        case VfPredictionStatus::VF_SIM_HIT:
            return "VfSimHit";
        case VfPredictionStatus::PARTIAL_FALLBACK:
            return "PartialFallback";
        case VfPredictionStatus::UNSUPPORTED_FORM:
            return "UnsupportedForm";
        case VfPredictionStatus::INVALID_TRACE:
            return "InvalidTrace";
        case VfPredictionStatus::EMPTY_PROGRAM:
            return "EmptyProgram";
        case VfPredictionStatus::SIMULATOR_ERROR:
            return "SimulatorError";
    }
    return "Unknown";
}

VfPredictionResult predictVfCyclesWithVfSim(const std::vector<VfInfo>& vfs, const VfPredictionOptions& options)
{
    VfPredictionResult prediction;
    if (vfs.empty()) {
        prediction.diagnostics.emplace_back("no VfInfo was provided");
        return prediction;
    }

    try {
        const vfsim::ParamDB& db = getParamDb(resolveConfigRoot(options));
        for (std::size_t vfIndex = 0; vfIndex < vfs.size(); ++vfIndex) {
            const VfInfo& vf = vfs[vfIndex];
            PtoCanonicalLoweringResult lowering;
            try {
                lowering = lowerPtoVfToCanonical(vf);
            } catch (const std::exception& exception) {
                prediction.cycles += fallbackNodes(vf.tree, 1);
                ++prediction.fallbackCount;
                updateFailureStatus(prediction, VfPredictionStatus::INVALID_TRACE);
                addDiagnostic(
                    prediction, vfIndex, VfPredictionStatus::INVALID_TRACE,
                    std::string("direct canonical lowering failed: ") + exception.what());
                continue;
            }
            if (lowering.ignoredInstructionCount > 0) {
                prediction.ignoredInstructionCount += lowering.ignoredInstructionCount;
                std::ostringstream diagnostic;
                diagnostic << "vf[" << vfIndex << "]: approximation: ignored " << lowering.ignoredInstructionCount
                           << " predicate setup instruction(s) because predicate registers are not modeled";
                prediction.diagnostics.push_back(diagnostic.str());
            }
            if (lowering.program.context.empty()) {
                ++prediction.fallbackCount;
                updateFailureStatus(prediction, VfPredictionStatus::EMPTY_PROGRAM);
                addDiagnostic(
                    prediction, vfIndex, VfPredictionStatus::EMPTY_PROGRAM, "lowering produced an empty program");
                continue;
            }

            try {
                std::string unsupportedReason;
                if (!formsSupported(lowering.program.context, db, unsupportedReason)) {
                    prediction.cycles += fallbackNodes(vf.tree, 1);
                    ++prediction.fallbackCount;
                    updateFailureStatus(prediction, VfPredictionStatus::UNSUPPORTED_FORM);
                    addDiagnostic(prediction, vfIndex, VfPredictionStatus::UNSUPPORTED_FORM, unsupportedReason);
                    continue;
                }

                const auto result = vfsim::runCanonicalVfInfo(lowering.program, db);
                const uint64_t cycles = static_cast<uint64_t>(std::max<int64_t>(0, result.vfEndCycle));
                prediction.cycles += cycles;
                ++prediction.vfSimHitCount;
            } catch (const std::exception& exception) {
                prediction.cycles += fallbackNodes(vf.tree, 1);
                ++prediction.fallbackCount;
                updateFailureStatus(prediction, VfPredictionStatus::SIMULATOR_ERROR);
                addDiagnostic(prediction, vfIndex, VfPredictionStatus::SIMULATOR_ERROR, exception.what());
                continue;
            }
        }

        if (prediction.fallbackCount == 0 && prediction.vfSimHitCount > 0)
            prediction.status = VfPredictionStatus::VF_SIM_HIT;
        else if (prediction.vfSimHitCount > 0)
            prediction.status = VfPredictionStatus::PARTIAL_FALLBACK;
    } catch (const std::exception& exception) {
        prediction.cycles = fallback(vfs);
        prediction.fallbackCount = static_cast<uint32_t>(vfs.size());
        prediction.status = VfPredictionStatus::SIMULATOR_ERROR;
        addDiagnostic(
            prediction, 0, VfPredictionStatus::SIMULATOR_ERROR,
            std::string("failed to initialize VfSim ParamDB: ") + exception.what());
    }

    return prediction;
}

} // namespace pto::mocker::vf
