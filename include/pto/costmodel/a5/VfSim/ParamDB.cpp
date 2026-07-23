// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "pto/costmodel/a5/VfSim/ParamDB.h"

#include "pto/costmodel/a5/VfSim/VfSimParamsGenerated.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace vfsim {
namespace {

std::string toString(std::string_view value) {
  return std::string(value.data(), value.size());
}

int64_t parseInt(std::string_view value) {
  return std::stoll(toString(value));
}

bool parseBool(std::string_view value) {
  return value == "true" || value == "1";
}

std::pair<std::string, std::string> splitQualifiedOp(const std::string &name) {
  const std::size_t dot = name.rfind('.');
  if (dot == std::string::npos)
    return {name, ""};
  return {name.substr(0, dot), name.substr(dot + 1)};
}

std::string qualifyOp(const std::string &op, const std::string &form) {
  return form.empty() ? op : op + "." + form;
}

void applyUarchParam(UarchConfig &uarch, std::string_view key,
                     std::string_view type, std::string_view value) {
  (void)type;
  const auto intValue = [&]() { return parseInt(value); };
  const auto boolValue = [&]() { return parseBool(value); };
  const auto stringValue = [&]() { return toString(value); };
  const std::string keyText = toString(key);

  if (keyText == "issue_ports") uarch.issuePorts = intValue();
  else if (keyText == "load_ports") uarch.loadPorts = intValue();
  else if (keyText == "store_ports") uarch.storePorts = intValue();
  else if (keyText == "IDU_window_width") uarch.iduWindowWidth = intValue();
  else if (keyText == "IDU_issue_width") uarch.iduIssueWidth = intValue();
  else if (keyText == "LDQ_width") uarch.ldqWidth = intValue();
  else if (keyText == "vreg_num") uarch.vregNum = intValue();
  else if (keyText == "enable_isu_queue_model") uarch.enableIsuQueueModel = boolValue();
  else if (keyText == "shq_depth") uarch.shqDepth = intValue();
  else if (keyText == "exq_depth") uarch.exqDepth = intValue();
  else if (keyText == "admit_blocked_to_exq") uarch.admitBlockedToExq = boolValue();
  else if (keyText == "enable_shq_credit_model") uarch.enableShqCreditModel = boolValue();
  else if (keyText == "shq_release_delay") uarch.shqReleaseDelay = intValue();
  else if (keyText == "enable_credit_visibility_delay") uarch.enableCreditVisibilityDelay = boolValue();
  else if (keyText == "idu_visible_preg_delay") uarch.iduVisiblePregDelay = intValue();
  else if (keyText == "idu_visible_shq_delay") uarch.iduVisibleShqDelay = intValue();
  else if (keyText == "global_shq_preg_gate") uarch.globalShqPregGate = boolValue();
  else if (keyText == "use_explicit_idu_credit_bank") uarch.useExplicitIduCreditBank = boolValue();
  else if (keyText == "idu_to_ooo_delay") uarch.iduToOooDelay = intValue();
  else if (keyText == "vloop_to_dispatch_delay") uarch.vloopToDispatchDelay = intValue();
  else if (keyText == "idu_dispatch_start_advance") uarch.iduDispatchStartAdvance = intValue();
  else if (keyText == "initial_top_block_vloop_start_cycle") uarch.initialTopBlockVloopStartCycle = intValue();
  else if (keyText == "nested_vloop_initial_start_gap") uarch.nestedVloopInitialStartGap = intValue();
  else if (keyText == "loop1_min_feedback_gap") uarch.loop1MinFeedbackGap = intValue();
  else if (keyText == "innermost_iter_dispatch_stride") uarch.innermostIterDispatchStride = intValue();
  else if (keyText == "consumer_release_start_offset") uarch.consumerReleaseStartOffset = intValue();
  else if (keyText == "load_done_latency") uarch.loadDoneLatency = intValue();
  else if (keyText == "ooo_to_shq_delay") uarch.oooToShqDelay = intValue();
  else if (keyText == "ooo_to_lsq_delay") uarch.oooToLsqDelay = intValue();
  else if (keyText == "exq_recv_delay") uarch.exqRecvDelay = intValue();
  else if (keyText == "shq_to_exq_port_per_cycle") uarch.shqToExqPortPerCycle = intValue();
  else if (keyText == "compute_inflight_cap") uarch.computeInflightCap = intValue();
  else if (keyText == "exq_issue_inflight_cap_per_port") uarch.exqIssueInflightCapPerPort = intValue();
  else if (keyText == "exq_capacity_counts_inflight") uarch.exqCapacityCountsInflight = boolValue();
  else if (keyText == "mem_bar_mode") uarch.memBarMode = stringValue();
  else if (keyText == "enforce_same_cycle_src_hazard") uarch.enforceSameCycleSrcHazard = boolValue();
  else if (keyText == "enable_cross_fu_ii") uarch.enableCrossFuIi = boolValue();
}

void addRelation(std::unordered_map<DTypeName, std::unordered_map<OpName, std::unordered_map<OpName, int64_t>>> &byDtype,
                 std::unordered_map<std::string, std::unordered_map<std::string, int64_t>> &byForm,
                 const std::string &lhs, const std::string &rhs, int64_t cycles) {
  const auto [lhsOp, lhsForm] = splitQualifiedOp(lhs);
  const auto [rhsOp, rhsForm] = splitQualifiedOp(rhs);
  if (!lhsForm.empty()) {
    byForm[lhs][rhs] = cycles;
    if (rhsForm.empty() || rhsForm == lhsForm)
      byDtype[lhsForm][lhsOp][rhsOp] = cycles;
    return;
  }
  byDtype[lhs][rhsOp][rhsForm] = cycles;
}

} // namespace

ParamDB::ParamDB(std::filesystem::path baseDir)
    : baseDir_(resolveBaseDir(std::move(baseDir))) {
  for (const auto &param : generated::kIsaDefaultParams) {
    const std::string key = toString(param.key);
    if (key == "vf_startup_cost")
      bundle_.isaDefaults.vfStartupCost = param.value;
    else if (key == "vf_drain_cost")
      bundle_.isaDefaults.vfDrainCost = param.value;
  }

  for (const auto &param : generated::kIsaInstParams) {
    InstConfig config;
    config.pipelineStartupCost = param.pipelineStartupCost;
    config.latency = param.latency;
    config.throughput = param.throughput;
    config.pipelineDrainCost = param.pipelineDrainCost;
    config.dataLoadCost = param.dataLoadCost;
    config.dataStoreCost = param.dataStoreCost;
    config.exu = toString(param.exu);
    config.dispatchExu = toString(param.dispatchExu);
    config.opClass = toString(param.opClass);
    bundle_.isa[toString(param.op)][toString(param.form)] = std::move(config);
  }

  for (const auto &param : generated::kUarchParams)
    applyUarchParam(bundle_.uarch, param.key, param.type, param.value);

  for (const auto &param : generated::kForwardingParams)
    addRelation(bundle_.forwarding, bundle_.forwardingByForm, toString(param.lhs),
                toString(param.rhs), param.cycles);

  for (const auto &param : generated::kInitiationIntervalParams)
    addRelation(bundle_.initiationInterval, bundle_.initiationIntervalByForm,
                toString(param.lhs), toString(param.rhs), param.cycles);
}

std::filesystem::path ParamDB::resolveBaseDir(std::filesystem::path baseDir) {
  if (baseDir.empty())
    return std::filesystem::absolute(std::filesystem::current_path());
  return std::filesystem::absolute(std::move(baseDir));
}

bool ParamDB::hasInst(const std::string &op, const std::string &dtype) const {
  const auto opIt = bundle_.isa.find(op);
  if (opIt == bundle_.isa.end())
    return false;
  return opIt->second.find(dtype) != opIt->second.end();
}

const InstConfig &ParamDB::inst(const std::string &op, const std::string &dtype) const {
  const auto opIt = bundle_.isa.find(op);
  if (opIt == bundle_.isa.end())
    throw std::runtime_error("Instruction not found: op=" + op + ", dtype=" + dtype);
  const auto dtypeIt = opIt->second.find(dtype);
  if (dtypeIt == opIt->second.end())
    throw std::runtime_error("Instruction not found: op=" + op + ", dtype=" + dtype);
  return dtypeIt->second;
}

int64_t ParamDB::forwardingCycles(const std::string &dtype, const std::string &prod,
                                  const std::string &cons) const {
  const auto dtypeIt = bundle_.forwarding.find(dtype);
  if (dtypeIt != bundle_.forwarding.end()) {
    const auto prodIt = dtypeIt->second.find(prod);
    if (prodIt != dtypeIt->second.end()) {
      const auto consIt = prodIt->second.find(cons);
      if (consIt != prodIt->second.end())
        return std::max<int64_t>(0, consIt->second);
    }
  }
  const int64_t latency = hasInst(prod, dtype) ? inst(prod, dtype).latency : 0;
  return std::max<int64_t>(0, latency - 3);
}

int64_t ParamDB::forwardingCycles(const std::string &prod,
                                  const std::string &prodForm,
                                  const std::string &cons,
                                  const std::string &consForm) const {
  const auto prodIt = bundle_.forwardingByForm.find(qualifyOp(prod, prodForm));
  if (prodIt != bundle_.forwardingByForm.end()) {
    const auto consIt = prodIt->second.find(qualifyOp(cons, consForm));
    if (consIt != prodIt->second.end())
      return std::max<int64_t>(0, consIt->second);
  }
  if (prodForm == consForm)
    return forwardingCycles(prodForm, prod, cons);
  const int64_t latency =
      hasInst(prod, prodForm) ? inst(prod, prodForm).latency : 0;
  return std::max<int64_t>(0, latency - 3);
}

int64_t ParamDB::initiationInterval(const std::string &dtype, const std::string &prev,
                                    const std::string &cur) const {
  const auto dtypeIt = bundle_.initiationInterval.find(dtype);
  if (dtypeIt != bundle_.initiationInterval.end()) {
    const auto prevIt = dtypeIt->second.find(prev);
    if (prevIt != dtypeIt->second.end()) {
      const auto curIt = prevIt->second.find(cur);
      if (curIt != prevIt->second.end())
        return std::max<int64_t>(1, curIt->second);
    }
  }
  return 1;
}

int64_t ParamDB::initiationInterval(const std::string &prev,
                                    const std::string &prevForm,
                                    const std::string &cur,
                                    const std::string &curForm) const {
  const auto prevIt =
      bundle_.initiationIntervalByForm.find(qualifyOp(prev, prevForm));
  if (prevIt != bundle_.initiationIntervalByForm.end()) {
    const auto curIt = prevIt->second.find(qualifyOp(cur, curForm));
    if (curIt != prevIt->second.end())
      return std::max<int64_t>(1, curIt->second);
  }
  if (prevForm == curForm)
    return initiationInterval(prevForm, prev, cur);
  return 1;
}

} // namespace vfsim
