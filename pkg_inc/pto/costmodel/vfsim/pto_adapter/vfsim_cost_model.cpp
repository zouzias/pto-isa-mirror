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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pto/costmodel/a5/cce_costmodel/vec_cycle_generated.hpp"
#include "pto/costmodel/vfsim/pto_adapter/pto_canonical_lowering.hpp"
#include "pto/costmodel/a5/cce_costmodel/vf_cost.hpp"

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

bool formsSupported(const std::vector<vfsim::CanonicalNode>& nodes, const vfsim::ParamDB& db)
{
    for (const vfsim::CanonicalNode& node : nodes) {
        if (const auto* loop = std::get_if<std::shared_ptr<const vfsim::CanonicalLoop>>(&node.payload)) {
            if (*loop == nullptr) {
                return false;
            }
            if (!formsSupported((*loop)->body, db))
                return false;
            continue;
        }
        if (std::holds_alternative<vfsim::CanonicalMembar>(node.payload))
            continue;

        const vfsim::CanonicalInstruction& instruction = std::get<vfsim::CanonicalInstruction>(node.payload);
        if (instruction.opcode.empty()) {
            return false;
        }
        if (instruction.form.empty()) {
            return false;
        }
        if (!db.hasInst(instruction.opcode, instruction.form)) {
            return false;
        }
    }
    return true;
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

std::filesystem::path resolveConfigRoot()
{
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

    throw std::runtime_error("VfSim configs were not found; set PTO_VFSIM_CONFIG_DIR");
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

} // namespace

uint64_t PredictVfCyclesWithVfSim(const std::vector<VfInfo>& vfs)
{
    if (vfs.empty())
        return 0;

    try {
        const vfsim::ParamDB& db = getParamDb(resolveConfigRoot());
        uint64_t total = 0;
        for (const VfInfo& vf : vfs) {
            try {
                const auto lowering = lowerPtoVfToCanonical(vf);
                if (lowering.program.context.empty())
                    continue;
                if (!formsSupported(lowering.program.context, db)) {
                    total += fallbackNodes(vf.tree, 1);
                    continue;
                }
                const auto result = vfsim::runCanonicalVfInfo(lowering.program, db);
                total += static_cast<uint64_t>(std::max<int64_t>(0, result.vfEndCycle));
            } catch (const std::exception&) {
                total += fallbackNodes(vf.tree, 1);
            }
        }
        return total;
    } catch (const std::exception&) {
        return fallback(vfs);
    }
}

} // namespace pto::mocker::vf
