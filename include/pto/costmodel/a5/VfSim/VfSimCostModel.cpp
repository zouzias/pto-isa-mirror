// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").

#include "pto/costmodel/a5/VfSim/VfSimCostModel.h"

#include "pto/costmodel/a5/VfSim/ParamDB.h"
#include "pto/costmodel/a5/VfSim/SimulatorRunner.h"
#include "pto/costmodel/a5/VfSim/VfInfo.h"
#include "pto/costmodel/a5/cce_costmodel/vec_cycle_generated.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef PTO_VFSIM_SOURCE_ROOT
#define PTO_VFSIM_SOURCE_ROOT "."
#endif

namespace pto::mocker::vf {
namespace {

std::string Upper(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return value;
}

uint64_t FallbackNodes(const std::vector<VfNode> &nodes, uint64_t multiplier)
{
    uint64_t total = 0;
    for (const VfNode &node : nodes) {
        if (IsLoop(node)) {
            const VfLoop &loop = AsLoop(node);
            total += FallbackNodes(loop.body, multiplier * loop.count);
        } else if (IsInst(node)) {
            total += FallbackVecCycle(AsInst(node).opName) * multiplier;
        } else {
            total += kMemBarPenaltyPlaceholder * multiplier;
        }
    }
    return total;
}

uint64_t Fallback(const std::vector<VfInfo> &vfs)
{
    uint64_t total = 0;
    for (const VfInfo &vf : vfs) total += FallbackNodes(vf.tree, 1);
    return total;
}

bool ValidateName(const MemInfo &mem)
{
    if (mem.location == MemLocation::PhyRegister) return !mem.name.empty() && mem.name.front() == 'V';
    return mem.name.rfind("mem", 0) == 0;
}

std::string ScopedName(const MemInfo &mem, std::size_t vfIndex)
{
    if (mem.location == MemLocation::PhyRegister) {
        return "V" + std::to_string(vfIndex) + "_" + mem.name.substr(1);
    }
    return "mem" + std::to_string(vfIndex) + "_" + mem.name.substr(3);
}

vfsim::ValueStorageKind ToVfSimStorage(MemLocation location)
{
    return location == MemLocation::PhyRegister ? vfsim::ValueStorageKind::Register
                                                : vfsim::ValueStorageKind::UB;
}

std::string PublicDType(const MemInfo &mem)
{
    if (mem.dtype == "unknown" || mem.dtype == "bool") return {};
    return mem.dtype;
}

bool RegisterValue(vfsim::VfInfo &target, const MemInfo &mem, std::size_t vfIndex,
                   std::string &loweredName)
{
    if (!ValidateName(mem)) return false;
    loweredName = ScopedName(mem, vfIndex);

    vfsim::ValueInfo value;
    value.valueId = loweredName;
    value.storage = ToVfSimStorage(mem.location);
    value.dtype = PublicDType(mem);

    auto [it, inserted] = target.values.emplace(loweredName, value);
    if (!inserted) {
        if (it->second.storage != value.storage) return false;
        if (!value.dtype.empty()) {
            if (!it->second.dtype.empty() && it->second.dtype != value.dtype) return false;
            it->second.dtype = value.dtype;
        }
    }
    return true;
}

std::vector<vfsim::ProgramNode> LowerNodes(const std::vector<VfNode> &nodes, std::size_t vfIndex,
                                           uint64_t &loopIndex, vfsim::VfInfo &target,
                                           bool &ok)
{
    std::vector<vfsim::ProgramNode> result;
    result.reserve(nodes.size());
    for (const VfNode &node : nodes) {
        if (IsLoop(node)) {
            const VfLoop &source = AsLoop(node);
            vfsim::ProgramLoopNode loop;
            loop.iters = std::to_string(source.count);
            loop.unroll = "1";
            loop.name = "vf" + std::to_string(vfIndex) + "_loop" + std::to_string(loopIndex++);
            loop.body = LowerNodes(source.body, vfIndex, loopIndex, target, ok);
            result.push_back(vfsim::ProgramNode::makeLoop(std::move(loop)));
            continue;
        }
        if (IsMemBar(node)) {
            ok = false;
            continue;
        }

        const VfInst &inst = AsInst(node);
        if (inst.opName.empty() || inst.dst.empty() || (inst.src.empty() && Upper(inst.opName) != "VDUP")) {
            ok = false;
            continue;
        }
        vfsim::ProgramInstNode lowered;
        lowered.op = Upper(inst.opName);
        for (const MemInfo &mem : inst.dst) {
            std::string name;
            if (!RegisterValue(target, mem, vfIndex, name)) {
                ok = false;
                continue;
            }
            lowered.dst.push_back(std::move(name));
        }
        for (const MemInfo &mem : inst.src) {
            std::string name;
            if (!RegisterValue(target, mem, vfIndex, name)) {
                ok = false;
                continue;
            }
            lowered.src.push_back(std::move(name));
        }
        result.push_back(vfsim::ProgramNode::makeInst(std::move(lowered)));
    }
    return result;
}

bool FormsSupported(const std::vector<vfsim::ProgramNode> &nodes, const vfsim::ParamDB &db)
{
    for (const auto &node : nodes) {
        if (node.kind == vfsim::ProgramNode::Kind::Loop) {
            if (!node.loop || !FormsSupported(node.loop->body, db)) return false;
            continue;
        }
        if (node.inst.op.empty() || node.inst.form.empty() || !db.hasInst(node.inst.op, node.inst.form)) {
            return false;
        }
    }
    return true;
}

}  // namespace

uint64_t PredictVfCyclesWithVfSim(const std::vector<VfInfo> &vfs)
{
    if (vfs.empty()) return 0;

    try {
        static const vfsim::ParamDB db(std::filesystem::path(PTO_VFSIM_SOURCE_ROOT));
        vfsim::VfInfo program;
        program.defaultDtype = "fp32";
        bool ok = true;
        for (std::size_t i = 0; i < vfs.size(); ++i) {
            uint64_t loopIndex = 0;
            auto lowered = LowerNodes(vfs[i].tree, i, loopIndex, program, ok);
            program.body.insert(program.body.end(), std::make_move_iterator(lowered.begin()),
                                std::make_move_iterator(lowered.end()));
        }
        if (!ok) return Fallback(vfs);
        if (program.body.empty()) return 0;

        vfsim::VfInfo canonical = program;
        vfsim::canonicalizeVfInfo(canonical);
        if (!FormsSupported(canonical.body, db)) return Fallback(vfs);

        const auto result = vfsim::runVfInfo(program, db);
        return static_cast<uint64_t>(std::max<int64_t>(0, result.vfEndCycle));
    } catch (const std::exception &) {
        return Fallback(vfs);
    }
}

}  // namespace pto::mocker::vf
