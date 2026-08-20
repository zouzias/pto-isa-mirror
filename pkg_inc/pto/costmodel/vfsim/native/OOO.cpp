// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "native/OOO.h"
#include "native/ControlUnit.h"
#include "native/ISATraits.h"

namespace vfsim {
namespace {

std::string jsonEscape(const std::string& text)
{
    std::string out;
    out.reserve(text.size() + 8);
    for (char c : text) {
        switch (c) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out.push_back(c);
                break;
        }
    }
    return out;
}

template <typename T>
std::string joinJsonArray(const std::vector<T>& values)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i)
            oss << ", ";
        oss << values[i];
    }
    oss << "]";
    return oss.str();
}

template <>
std::string joinJsonArray<std::string>(const std::vector<std::string>& values)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i)
            oss << ", ";
        oss << '"' << jsonEscape(values[i]) << '"';
    }
    oss << "]";
    return oss.str();
}

template <>
std::string joinJsonArray<std::optional<std::string>>(const std::vector<std::optional<std::string>>& values)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i)
            oss << ", ";
        if (values[i].has_value())
            oss << '"' << jsonEscape(*values[i]) << '"';
        else
            oss << "null";
    }
    oss << "]";
    return oss.str();
}

std::string joinIterationPath(const std::vector<std::pair<std::string, int64_t>>& path)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t index = 0; index < path.size(); ++index) {
        if (index)
            oss << ", ";
        oss << "{\"loop_id\":\"" << jsonEscape(path[index].first) << "\",\"iteration\":" << path[index].second << "}";
    }
    oss << "]";
    return oss.str();
}

} // namespace

OooCore::OooCore(
    const UarchConfig& uarch, const ParamDb& db, std::string dtype,
    const std::unordered_map<std::string, ValueInfo>& values)
    : db_(db), dtype_(std::move(dtype)), valueStorage_(values)
{
    theoreticalLimitMode_ = false;
    enableIsuQueueModel_ = uarch.enableIsuQueueModel;
    loadPorts_ = static_cast<int>(uarch.loadPorts);
    issuePorts_ = static_cast<int>(uarch.issuePorts);
    threePortsMode_ = uarch.threePortsMode;
    storePorts_ = static_cast<int>(uarch.storePorts);
    ubSlots_ = static_cast<int>(uarch.ubSlots);
    lsuStorePriorityPregThreshold_ = static_cast<int>(uarch.lsuStorePriorityPregThreshold);
    if (loadPorts_ <= 0 || storePorts_ <= 0 || ubSlots_ <= 0)
        throw std::invalid_argument("load_ports, store_ports, and ub_slots must be positive");
    if (lsuStorePriorityPregThreshold_ < 0)
        throw std::invalid_argument("lsu_store_priority_preg_threshold must be non-negative");
    shqDepth_ = static_cast<int>(uarch.shqDepth);
    lsqDepth_ = static_cast<int>(uarch.ldqWidth ? uarch.ldqWidth : 24);
    pregNum_ = static_cast<int>(uarch.vregNum ? uarch.vregNum : 68);
    vfStartupCost_ = static_cast<int>(db_.isaDefaults().vfStartupCost);
    vfDrainCost_ = static_cast<int>(db_.isaDefaults().vfDrainCost);
    freelist_.clear();
    for (int i = 0; i < pregNum_; ++i)
        freelist_.push_back("p" + std::to_string(i));
    visiblePregFree_ = pregNum_;
    lastIssueCycleAlu_.assign(issuePorts_, -1000000000);
    lastIssueCycleSfu_.assign(issuePorts_, -1000000000);
    lastOpAlu_.assign(issuePorts_, "");
    lastFormAlu_.assign(issuePorts_, "");
    lastOpSfu_.assign(issuePorts_, "");
    lastFormSfu_.assign(issuePorts_, "");
    lastIssueCycleExu_.assign(issuePorts_, -1000000000);
    lastOpExu_.assign(issuePorts_, "");
    lastFormExu_.assign(issuePorts_, "");
    exqInflight_.assign(issuePorts_, 0);
    oooToShqDelay_ = static_cast<int>(uarch.oooToShqDelay ? uarch.oooToShqDelay : 1);
    oooToLsqDelay_ = static_cast<int>(uarch.oooToLsqDelay ? uarch.oooToLsqDelay : 1);
    exqRecvDelay_ = static_cast<int>(uarch.exqRecvDelay ? uarch.exqRecvDelay : 1);
    enforceSameCycleSrcHazard_ = uarch.enforceSameCycleSrcHazard;
    enableExqGreedyBalance_ = false;
    enableShqCreditModel_ = uarch.enableShqCreditModel;
    enableCreditVisibilityDelay_ = uarch.enableCreditVisibilityDelay;
    enableCrossFuIi_ = uarch.enableCrossFuIi;
    exqCapacityCountsInflight_ = uarch.exqCapacityCountsInflight;
    exqDepth_ = static_cast<int>(uarch.exqDepth ? uarch.exqDepth : 26);
    shqToExqPortPerCycle_ = static_cast<int>(uarch.shqToExqPortPerCycle ? uarch.shqToExqPortPerCycle : 1);
    shqExqDispatchPolicy_ = uarch.shqExqDispatchPolicy;
    std::transform(
        shqExqDispatchPolicy_.begin(), shqExqDispatchPolicy_.end(), shqExqDispatchPolicy_.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    exu0ReserveLookahead_ = static_cast<int>(uarch.exu0ReserveLookahead);
    exu0ReserveMinCount_ = std::max<int>(1, static_cast<int>(uarch.exu0ReserveMinCount));
    exqIssueInflightCapPerPort_ = static_cast<int>(uarch.exqIssueInflightCapPerPort);
    computeInflightCap_ = static_cast<int>(uarch.computeInflightCap);
    shqReleaseDelay_ = static_cast<int>(uarch.shqReleaseDelay ? uarch.shqReleaseDelay : 1);
    iduVisiblePregDelay_ = static_cast<int>(uarch.iduVisiblePregDelay);
    iduVisibleShqDelay_ = static_cast<int>(uarch.iduVisibleShqDelay);
    visibleShqUsed_ = 0;
}

int OooCore::getFreePreg() const
{
    if (theoreticalLimitMode_)
        return 1000000000;
    if (enableCreditVisibilityDelay_)
        return std::max(0, visiblePregFree_);
    return static_cast<int>(freelist_.size());
}

int OooCore::getFreeShqQueue() const
{
    return theoreticalLimitMode_ ? 1000000000 : std::max(0, shqDepth_ - static_cast<int>(shq_.size()));
}

int OooCore::getFreeLsq() const
{
    return theoreticalLimitMode_ ? 1000000000 : std::max(0, lsqDepth_ - static_cast<int>(lsq_.size()));
}

int OooCore::getFreeShq() const
{
    if (theoreticalLimitMode_ || !enableShqCreditModel_)
        return 1000000000;
    if (enableCreditVisibilityDelay_)
        return std::max(0, shqDepth_ - visibleShqUsed_);
    return std::max(0, shqDepth_ - shqUsed_);
}

std::optional<int> OooCore::getExuPortForInst(int64_t instId) const
{
    const Uop* uop = findRobUop(instId);
    if (uop == nullptr || uop->exuPort < 0)
        return std::nullopt;
    return uop->exuPort;
}

std::optional<std::string> OooCore::getPregSrcForInst(int64_t instId, size_t index) const
{
    const Uop* uop = findRobUop(instId);
    if (uop == nullptr || index >= uop->pregSrc.size())
        return std::nullopt;
    return uop->pregSrc[index];
}

std::optional<std::string> OooCore::getPregDstForInst(int64_t instId, size_t index) const
{
    const Uop* uop = findRobUop(instId);
    if (uop == nullptr || index >= uop->pregDst.size() || uop->pregDst[index].empty())
        return std::nullopt;
    return uop->pregDst[index];
}

bool OooCore::hasPendingLsuBefore(int64_t streamSeq, const std::string& opClass) const
{
    const std::string target = [&]() {
        std::string value = opClass;
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return static_cast<char>(std::toupper(c));
        });
        return value;
    }();
    for (const auto& u : rob_) {
        if (u.streamSeq < 0 || u.streamSeq >= streamSeq)
            continue;
        if (u.state == "done")
            continue;
        std::string cls;
        if (u.opClass == "LOAD")
            cls = "LOAD";
        else if (u.opClass == "STORE")
            cls = "STORE";
        if (cls == target)
            return true;
    }
    return false;
}

std::unordered_map<std::string, int> OooCore::updateIduVisibility(int64_t cycle)
{
    if (!enableCreditVisibilityDelay_)
        return std::unordered_map<std::string, int>{{"preg_free", 0}, {"shq_release", 0}};
    int pregDelta = iduMailboxPregReleaseDelta_;
    int shqDelta = iduMailboxShqReleaseDelta_;
    iduMailboxPregReleaseDelta_ = 0;
    iduMailboxShqReleaseDelta_ = 0;

    auto pit = visiblePregFreeEvents_.find(cycle);
    if (pit != visiblePregFreeEvents_.end()) {
        visiblePregFree_ += pit->second;
        pregDelta += pit->second;
        visiblePregFreeEvents_.erase(pit);
    }
    auto sit = visibleShqReleaseEvents_.find(cycle);
    if (sit != visibleShqReleaseEvents_.end()) {
        visibleShqUsed_ = std::max(0, visibleShqUsed_ - sit->second);
        shqDelta += sit->second;
        visibleShqReleaseEvents_.erase(sit);
    }
    return std::unordered_map<std::string, int>{{"preg_free", pregDelta}, {"shq_release", shqDelta}};
}

int64_t OooCore::vfEndCycle() const { return lastDoneCycle_ + vfDrainCost_; }

std::string OooCore::classifyOpClass(const std::string& op, const std::string& form) const
{
    return isLoadOp(db_, op, form) ? "LOAD" : (isStoreOp(db_, op, form) ? "STORE" : "COMPUTE");
}

bool OooCore::isRegisterValue(const std::string& name) const { return valueStorage_.isRegister(name); }

int64_t OooCore::computeReadyTimeForSrc(
    const ProducerInfo& producerInfo, const std::string& consumerOp, const std::string& consumerForm) const
{
    const std::string cacheKey = producerInfo.op + "." + producerInfo.form + "\x1f" + consumerOp + "." + consumerForm;
    auto cached = forwardingPairCache_.find(cacheKey);
    int64_t fwd = 0;
    if (cached != forwardingPairCache_.end()) {
        fwd = cached->second;
    } else {
        fwd = db_.forwardingCycles(producerInfo.op, producerInfo.form, consumerOp, consumerForm);
        forwardingPairCache_[cacheKey] = fwd;
    }
    if (isComputeOp(db_, consumerOp, consumerForm) && enableIsuQueueModel_)
        return producerInfo.startCycle + std::max<int64_t>(0, fwd - 1);
    return producerInfo.startCycle + fwd;
}

int64_t OooCore::computeLoadReadyCycle(const Uop& u) const
{
    return std::max<int64_t>(vfStartupCost_, u.lsqReadyCycle);
}

bool OooCore::blockedByControlUnit(const Uop& u) const
{
    if (controlUnit_ == nullptr)
        return false;
    DynamicInst inst;
    inst.type = "inst";
    inst.op = u.op;
    inst.form = u.form;
    inst.streamSeq = u.streamSeq;
    return controlUnit_->blocks(inst, db_, dtype_);
}

std::tuple<int64_t, std::optional<std::string>, std::optional<std::string>, std::optional<int64_t>>
OooCore::computeStoreReadyCycle(const Uop& u) const
{
    for (const auto& ps : u.pregSrc) {
        if (!ps.has_value())
            continue;
        if (pregPending_.count(*ps) && pregProducer_.find(*ps) == pregProducer_.end())
            return {1000000000, std::nullopt, std::nullopt, std::nullopt};
    }

    int64_t bestT = -1;
    std::optional<std::string> pop;
    std::optional<std::string> pform;
    std::optional<int64_t> pst;
    for (const auto& ps : u.pregSrc) {
        if (!ps.has_value())
            continue;
        auto it = pregProducer_.find(*ps);
        if (it == pregProducer_.end())
            continue;
        const auto& kind = it->second.kind;
        if (kind != "COMPUTE" && kind != "LOAD")
            continue;
        const int64_t cand = computeReadyTimeForSrc(it->second, u.op, u.form);
        if (cand > bestT) {
            bestT = cand;
            pop = it->second.op;
            pform = it->second.form;
            pst = it->second.startCycle;
        }
    }
    if (bestT < 0)
        return {1000000000, std::nullopt, std::nullopt, std::nullopt};
    bestT = std::max<int64_t>(bestT, u.lsqReadyCycle);
    return {bestT, pop, pform, pst};
}

std::string OooCore::getFuType(const std::string& op, const std::string& form) const
{
    const auto& cfg = db_.inst(op, form);
    std::string fu = cfg.exu.empty() ? "ALU" : cfg.exu;
    std::transform(
        fu.begin(), fu.end(), fu.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (fu != "ALU" && fu != "SFU")
        fu = "ALU";
    return fu;
}

std::vector<int> OooCore::eligibleExuPorts(const std::string& op, const std::string& form) const
{
    std::string tag = db_.inst(op, form).dispatchExu;
    std::transform(
        tag.begin(), tag.end(), tag.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (tag == "EXU0_ONLY")
        return issuePorts_ > 0 ? std::vector<int>{0} : std::vector<int>{};
    if (tag == "EXU01") {
        if (threePortsMode_) {
            std::vector<int> out;
            for (int port = 0; port < std::min(issuePorts_, 3); ++port)
                out.push_back(port);
            return out;
        } else {
            return issuePorts_ >= 2 ? std::vector<int>{0, 1} : std::vector<int>{0};
        }
    }
    if (tag == "EXU012")
        return issuePorts_ >= 3 ? std::vector<int>{0, 1, 2} : std::vector<int>{0, 1};
    std::vector<int> out;
    for (int i = 0; i < issuePorts_; ++i)
        out.push_back(i);
    return out;
}

std::vector<int> OooCore::eligibleExuPorts(const Uop& u) const
{
    if (u.dispatchExu == "EXU0_ONLY")
        return issuePorts_ > 0 ? std::vector<int>{0} : std::vector<int>{};
    if (u.dispatchExu == "EXU01") {
        std::vector<int> out;
        const int limit = threePortsMode_ ? std::min(issuePorts_, 3) : std::min(issuePorts_, 2);
        for (int port = 0; port < limit; ++port)
            out.push_back(port);
        return out;
    }
    if (u.dispatchExu == "EXU012") {
        std::vector<int> out;
        for (int port = 0; port < std::min(issuePorts_, 3); ++port)
            out.push_back(port);
        return out;
    }
    std::vector<int> out;
    for (int port = 0; port < issuePorts_; ++port)
        out.push_back(port);
    return out;
}

std::vector<int> OooCore::getEligibleExuPorts(const std::string& op, const std::string& form) const
{
    return eligibleExuPorts(op, form);
}

int64_t OooCore::getIi(
    const std::string* prevOp, const std::string* prevForm, const std::string& curOp, const std::string& curForm) const
{
    if (!prevOp || !prevForm || prevOp->empty())
        return 1;
    const std::string cacheKey = *prevOp + "." + *prevForm + "\x1f" + curOp + "." + curForm;
    const auto cached = initiationIntervalPairCache_.find(cacheKey);
    if (cached != initiationIntervalPairCache_.end())
        return cached->second;
    const int64_t value = db_.initiationInterval(*prevOp, *prevForm, curOp, curForm);
    initiationIntervalPairCache_[cacheKey] = value;
    return value;
}

void OooCore::log(const std::string& event, const Uop& u)
{
    history_.push_back(HistoryRecord{
        cycle_, event, u.instId, u.op, u.state, u.blockedReason, u.readyCycle, u.startCycle, u.doneCycle, u.src, u.dst,
        u.pregSrc, u.pregDst, u.pregOld, u.producerOpForStore, u.producerStartForStore, u.staticInstructionId,
        u.iterationPath, u.streamSeq});
}

void OooCore::logMembarBlocked(Uop& u)
{
    const auto oldReason = u.blockedReason;
    u.blockedReason = "membar";
    log("blocked", u);
    u.blockedReason = oldReason;
}

void OooCore::logStartSimple(const Uop& u)
{
    startLogs_.push_back(
        SimpleLogRecord{cycle_, u.instId, u.op, u.dst, u.src, u.staticInstructionId, u.iterationPath, u.streamSeq});
}

void OooCore::logDoneSimple(const Uop& u)
{
    doneLogs_.push_back(SimpleLogRecord{
        u.doneCycle.value_or(cycle_), u.instId, u.op, u.dst, u.src, u.staticInstructionId, u.iterationPath,
        u.streamSeq});
}

void OooCore::dumpHistory(const std::string& path) const
{
    std::ofstream os(path);
    os << "[\n";
    for (size_t i = 0; i < history_.size(); ++i) {
        const auto& h = history_[i];
        os << "  {"
           << "\"cy\":" << h.cy << ","
           << "\"event\":\"" << jsonEscape(h.event) << "\","
           << "\"id\":" << h.id << ","
           << "\"op\":\"" << jsonEscape(h.op) << "\","
           << "\"state\":\"" << jsonEscape(h.state) << "\","
           << "\"blocked_reason\":" << (h.blockedReason ? "\"" + jsonEscape(*h.blockedReason) + "\"" : "null") << ","
           << "\"ready\":" << h.ready << ","
           << "\"start\":" << (h.start ? std::to_string(*h.start) : "null") << ","
           << "\"done\":" << (h.done ? std::to_string(*h.done) : "null") << ","
           << "\"src\":" << joinJsonArray(h.src) << ","
           << "\"dst\":" << joinJsonArray(h.dst) << ","
           << "\"preg_src\":" << joinJsonArray(h.pregSrc) << ","
           << "\"preg_dst\":" << joinJsonArray(h.pregDst) << ","
           << "\"preg_old\":" << joinJsonArray(h.pregOld) << ","
           << "\"producer_op_for_store\":"
           << (h.producerOpForStore ? "\"" + jsonEscape(*h.producerOpForStore) + "\"" : "null") << ","
           << "\"producer_start_for_store\":"
           << (h.producerStartForStore ? std::to_string(*h.producerStartForStore) : "null") << ","
           << "\"static_instruction_id\":\"" << jsonEscape(h.staticInstructionId) << "\","
           << "\"iteration_path\":" << joinIterationPath(h.iterationPath) << ","
           << "\"stream_seq\":" << h.streamSeq << "}";
        if (i + 1 < history_.size())
            os << ",";
        os << "\n";
    }
    os << "]\n";
}

void OooCore::dumpSimpleLogs(const std::string& startPath, const std::string& donePath) const
{
    std::ofstream s(startPath);
    for (const auto& r : startLogs_) {
        s << "{\"cy\":" << r.cy << ",\"inst_id\":" << r.instId << ",\"op\":\"" << jsonEscape(r.op)
          << "\",\"dst\":" << joinJsonArray(r.dst) << ",\"src\":" << joinJsonArray(r.src)
          << ",\"static_instruction_id\":\"" << jsonEscape(r.staticInstructionId) << "\""
          << ",\"iteration_path\":" << joinIterationPath(r.iterationPath) << ",\"stream_seq\":" << r.streamSeq << "}\n";
    }
    std::ofstream d(donePath);
    for (const auto& r : doneLogs_) {
        d << "{\"cy\":" << r.cy << ",\"inst_id\":" << r.instId << ",\"op\":\"" << jsonEscape(r.op)
          << "\",\"dst\":" << joinJsonArray(r.dst) << ",\"src\":" << joinJsonArray(r.src)
          << ",\"static_instruction_id\":\"" << jsonEscape(r.staticInstructionId) << "\""
          << ",\"iteration_path\":" << joinIterationPath(r.iterationPath) << ",\"stream_seq\":" << r.streamSeq << "}\n";
    }
}

Uop* OooCore::findRobUop(int64_t instId)
{
    for (auto& u : rob_) {
        if (u.instId == instId)
            return &u;
    }
    return nullptr;
}

const Uop* OooCore::findRobUop(int64_t instId) const
{
    for (const auto& u : rob_) {
        if (u.instId == instId)
            return &u;
    }
    return nullptr;
}

bool OooCore::isCurrentMapping(const std::string& preg) const
{
    for (const auto& [_, cur] : rat_) {
        if (cur == preg)
            return true;
    }
    return false;
}

void OooCore::scheduleSrcReleaseFromStart(const Uop& u)
{
    if (!u.startCycle.has_value())
        return;
    if (srcReleaseScheduledInstIds_.count(u.instId))
        return;
    const int64_t releaseCycle = *u.startCycle + consumerReleaseStartOffset_;
    auto& bucket = srcReleaseEvents_[releaseCycle];
    for (size_t i = 0; i < u.pregSrc.size(); ++i) {
        const auto& s = u.pregSrc[i];
        if (!s.has_value())
            continue;
        const int64_t gen =
            (i < u.pregSrcGen.size() && u.pregSrcGen[i].has_value()) ? *u.pregSrcGen[i] : pregGeneration_[*s];
        bucket.push_back(SrcReleaseEvent{u.instId, *s, gen});
        srcReleaseExpected_[u.instId] += 1;
    }
    srcReleaseSeen_[u.instId] = 0;
    srcReleaseScheduledInstIds_.insert(u.instId);
}

void OooCore::runSrcReleaseEvents(int64_t cycle)
{
    auto it = srcReleaseEvents_.find(cycle);
    if (it == srcReleaseEvents_.end())
        return;
    for (const auto& ev : it->second) {
        ++srcReleaseSeen_[ev.instId];
        const auto genIt = pregGeneration_.find(ev.preg);
        if (genIt == pregGeneration_.end() || genIt->second != ev.gen)
            continue;
        auto cntIt = pregConsumerCount_.find(ev.preg);
        if (cntIt != pregConsumerCount_.end() && cntIt->second > 0) {
            --cntIt->second;
            if (cntIt->second == 0)
                pregReleaseEligibleCycle_[ev.preg] = cycle;
        }
    }
    srcReleaseEvents_.erase(it);
}

bool OooCore::tryFreePreg(const std::string& preg, int64_t cycle)
{
    if (preg.empty() || isCurrentMapping(preg))
        return false;
    auto cntIt = pregConsumerCount_.find(preg);
    if (cntIt != pregConsumerCount_.end() && cntIt->second > 0)
        return false;
    auto eligIt = pregReleaseEligibleCycle_.find(preg);
    if (eligIt != pregReleaseEligibleCycle_.end() && cycle < eligIt->second)
        return false;
    if (std::find(freelist_.begin(), freelist_.end(), preg) != freelist_.end())
        return false;
    if (pregPending_.count(preg))
        return false;
    pregProducer_.erase(preg);
    pregPending_.erase(preg);
    pregConsumerCount_.erase(preg);
    pregReleaseEligibleCycle_.erase(preg);
    freelist_.push_back(preg);
    if (enableCreditVisibilityDelay_) {
        if (iduVisiblePregDelay_ <= 0) {
            ++visiblePregFree_;
            ++iduMailboxPregReleaseDelta_;
        } else {
            visiblePregFreeEvents_[cycle + iduVisiblePregDelay_] += 1;
        }
    }
    return true;
}

void OooCore::tryFreeEligiblePregs(int64_t cycle)
{
    std::vector<std::string> elig;
    elig.reserve(pregReleaseEligibleCycle_.size());
    for (const auto& [preg, _] : pregReleaseEligibleCycle_)
        elig.push_back(preg);
    for (const auto& preg : elig)
        (void)tryFreePreg(preg, cycle);
}

int OooCore::exqOccupancy(int port) const
{
    if (port < 0 || port >= static_cast<int>(exqInflight_.size()))
        return 0;
    return exqInflight_[static_cast<size_t>(port)] * (exqCapacityCountsInflight_ ? 1 : 0);
}

int OooCore::totalComputeInflight() const
{
    int total = 0;
    for (int x : exqInflight_)
        total += x;
    return total;
}

int64_t OooCore::predictExqIssueCycle(
    int port, const std::string& fuType, const std::string& op, const std::string& form, int64_t recvCycle) const
{
    int64_t pred = recvCycle;
    const std::string* prevOp = nullptr;
    const std::string* prevForm = nullptr;
    int64_t prevIssue = -1000000000;
    if (enableCrossFuIi_) {
        prevOp = &lastOpExu_[static_cast<size_t>(port)];
        prevForm = &lastFormExu_[static_cast<size_t>(port)];
        prevIssue = lastIssueCycleExu_[static_cast<size_t>(port)];
    } else if (fuType == "SFU") {
        prevOp = &lastOpSfu_[static_cast<size_t>(port)];
        prevForm = &lastFormSfu_[static_cast<size_t>(port)];
        prevIssue = lastIssueCycleSfu_[static_cast<size_t>(port)];
    } else {
        prevOp = &lastOpAlu_[static_cast<size_t>(port)];
        prevForm = &lastFormAlu_[static_cast<size_t>(port)];
        prevIssue = lastIssueCycleAlu_[static_cast<size_t>(port)];
    }
    pred = std::max<int64_t>(pred, prevIssue + getIi(prevOp, prevForm, op, form));
    return pred;
}

bool OooCore::useFuRoundRobinFifo() const
{
    return shqExqDispatchPolicy_ == "fu_round_robin_fifo" || shqExqDispatchPolicy_ == "fu_rr_fifo" ||
           shqExqDispatchPolicy_ == "fu_round_robin" || shqExqDispatchPolicy_ == "fu_round_robin_exu0_reserve";
}

bool OooCore::useExu0Reserve() const { return shqExqDispatchPolicy_ == "fu_round_robin_exu0_reserve"; }

int OooCore::exu0OnlyPressureCount(size_t startIndex) const
{
    if (exu0ReserveLookahead_ <= 0)
        return 0;
    int seenCompute = 0;
    int seenExu0Only = 0;
    for (size_t index = startIndex + 1; index < shq_.size(); ++index) {
        const Uop& candidate = shq_[index];
        ++seenCompute;
        if (candidate.dispatchExu == "EXU0_ONLY")
            ++seenExu0Only;
        if (seenCompute >= exu0ReserveLookahead_)
            break;
    }
    return seenExu0Only >= exu0ReserveMinCount_ ? seenExu0Only : 0;
}

int OooCore::selectFuRoundRobinPort(const std::string& fuType, const std::vector<int>& candidates)
{
    if (candidates.empty())
        throw std::invalid_argument("selectFuRoundRobinPort requires candidates");
    const int ptr = shqExqRrPtrByFu_[fuType];
    for (int offset = 0; offset < std::max(1, issuePorts_); ++offset) {
        const int port = (ptr + offset) % std::max(1, issuePorts_);
        if (std::find(candidates.begin(), candidates.end(), port) == candidates.end())
            continue;
        shqExqRrPtrByFu_[fuType] = (port + 1) % std::max(1, issuePorts_);
        return port;
    }
    const int chosen = *std::min_element(candidates.begin(), candidates.end());
    shqExqRrPtrByFu_[fuType] = (chosen + 1) % std::max(1, issuePorts_);
    return chosen;
}

void OooCore::scheduleShqRelease(int64_t cycle, int count)
{
    if (!enableShqCreditModel_ || count <= 0)
        return;
    shqReleaseEvents_[cycle + shqReleaseDelay_] += count;
}

void OooCore::runShqReleaseEvents(int64_t cycle)
{
    if (!enableShqCreditModel_)
        return;
    auto it = shqReleaseEvents_.find(cycle);
    if (it == shqReleaseEvents_.end())
        return;
    const int released = it->second;
    shqReleaseEvents_.erase(it);
    shqUsed_ = std::max(0, shqUsed_ - released);
    if (!enableCreditVisibilityDelay_)
        return;
    if (iduVisibleShqDelay_ <= 0) {
        visibleShqUsed_ = std::max(0, visibleShqUsed_ - released);
        iduMailboxShqReleaseDelta_ += released;
    } else {
        visibleShqReleaseEvents_[cycle + iduVisibleShqDelay_] += released;
    }
}

OooCoreMainline::OooCoreMainline(
    const UarchConfig& uarch, const ParamDb& db, std::string dtype,
    const std::unordered_map<std::string, ValueInfo>& values)
    : OooCore(uarch, db, std::move(dtype), values)
{
    exqInflightPerPort_.assign(issuePorts_, 0);
    exqWait_.resize(static_cast<size_t>(issuePorts_));
    for (auto& port : exqWait_) {
        port["ALU"] = std::deque<Uop>{};
        port["SFU"] = std::deque<Uop>{};
    }
    consumerReleaseStartOffset_ = static_cast<int>(uarch.consumerReleaseStartOffset);
}

void OooCoreMainline::accept(const DynamicInst& inst)
{
    Uop u;
    u.instId = inst.instId;
    u.op = inst.op;
    u.form = inst.form.empty() ? dtype_ : inst.form;
    const InstConfig& profile = db_.inst(u.op, u.form);
    u.opClass = profile.opClass;
    std::transform(u.opClass.begin(), u.opClass.end(), u.opClass.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    if (u.opClass.empty())
        u.opClass = classifyOpClass(u.op, u.form);
    u.fuType = profile.exu.empty() ? "ALU" : profile.exu;
    std::transform(u.fuType.begin(), u.fuType.end(), u.fuType.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    if (u.fuType != "ALU" && u.fuType != "SFU")
        u.fuType = "ALU";
    u.dispatchExu = profile.dispatchExu;
    std::transform(u.dispatchExu.begin(), u.dispatchExu.end(), u.dispatchExu.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    u.latency = std::max<int64_t>(1, profile.latency);
    u.src = inst.src;
    u.dst = inst.dst;
    std::vector<std::string> releasedRatPregs;
    for (size_t sourceIndex = 0; sourceIndex < inst.src.size(); ++sourceIndex) {
        const auto& s = inst.src[sourceIndex];
        if (!isRegisterValue(s)) {
            u.pregSrc.push_back(std::nullopt);
            u.pregSrcGen.push_back(std::nullopt);
            continue;
        }
        auto it = rat_.find(s);
        if (it == rat_.end()) {
            u.pregSrc.push_back(std::nullopt);
            u.pregSrcGen.push_back(std::nullopt);
        } else {
            u.pregSrc.push_back(it->second);
            auto genIt = pregGeneration_.find(it->second);
            u.pregSrcGen.push_back(
                genIt == pregGeneration_.end() ? std::optional<int64_t>{0} : std::optional<int64_t>{genIt->second});
            if (sourceIndex < inst.srcValueRelease.size() && inst.srcValueRelease[sourceIndex]) {
                releasedRatPregs.push_back(it->second);
                rat_.erase(it);
            }
        }
    }
    u.topBlockId = inst.topBlockId;
    u.iterStack = inst.iterStack;
    u.isLastInTopBlock = inst.isLastInTopBlock;
    u.streamSeq = inst.streamSeq;
    u.staticInstructionId = inst.staticInstructionId;
    u.iterationPath = inst.iterationPath;

    for (const auto& preg : u.pregSrc) {
        if (preg) {
            pregConsumerCount_[*preg] += 1;
            pregReleaseEligibleCycle_.erase(*preg);
        }
    }

    int allocCount = 0;
    for (size_t destinationIndex = 0; destinationIndex < u.dst.size(); ++destinationIndex) {
        const auto& d = u.dst[destinationIndex];
        if (!isRegisterValue(d)) {
            u.pregDst.push_back(std::string{});
            continue;
        }
        std::string newPreg;
        if (theoreticalLimitMode_ || freelist_.empty()) {
            newPreg = "p" + std::to_string(nextDynamicPregId_++);
        } else {
            newPreg = freelist_.front();
            freelist_.pop_front();
        }
        std::string oldPreg;
        auto rit = rat_.find(d);
        if (rit != rat_.end())
            oldPreg = rit->second;
        const bool keepMapping = destinationIndex >= inst.dstValueKeep.size() || inst.dstValueKeep[destinationIndex];
        if (keepMapping)
            rat_[d] = newPreg;
        u.pregDst.push_back(newPreg);
        u.pregOld.push_back(oldPreg.empty() ? std::optional<std::string>{} : std::optional<std::string>{oldPreg});
        pregGeneration_[newPreg] += 1;
        pregConsumerCount_[newPreg] = 0;
        pregPending_.insert(newPreg);
        pregReleaseEligibleCycle_.erase(newPreg);
        ++allocCount;
    }
    if (enableCreditVisibilityDelay_ && allocCount > 0)
        visiblePregFree_ = std::max(0, visiblePregFree_ - allocCount);

    if (enableShqCreditModel_ && (u.opClass == "COMPUTE" || u.opClass == "STORE")) {
        ++shqUsed_;
        if (enableCreditVisibilityDelay_)
            ++visibleShqUsed_;
        u.isShqTracked = true;
    }

    if (u.opClass == "LOAD" || u.opClass == "STORE") {
        u.lsqReadyCycle = static_cast<int64_t>(cycle_) + oooToLsqDelay_;
        lsq_.push_back(u);
    } else {
        u.shqReadyCycle = static_cast<int64_t>(cycle_) + oooToShqDelay_;
        shq_.push_back(u);
    }
    rob_.push_back(u);

    for (const auto& releasedPreg : releasedRatPregs)
        (void)tryFreePreg(releasedPreg, cycle_);

    for (const auto& oldPreg : u.pregOld) {
        if (oldPreg)
            (void)tryFreePreg(*oldPreg, cycle_);
    }
}

void OooCoreMainline::freeOldPregs(const Uop& u)
{
    for (const auto& oldPreg : u.pregOld) {
        if (!oldPreg.has_value())
            continue;
        (void)tryFreePreg(*oldPreg, cycle_);
    }
}

void OooCoreMainline::completeRunningUops(int64_t cycle)
{
    for (auto& u : rob_) {
        if (u.state == "running" && u.doneCycle.has_value() && cycle >= *u.doneCycle) {
            u.state = "done";
            if (u.exuPort >= 0 && u.exuPort < static_cast<int>(exqInflight_.size()))
                exqInflight_[static_cast<size_t>(u.exuPort)] =
                    std::max(0, exqInflight_[static_cast<size_t>(u.exuPort)] - 1);
            log("done", u);
            logDoneSimple(u);
            lastDoneCycle_ = std::max(lastDoneCycle_, *u.doneCycle);
            for (const auto& pd : u.pregDst) {
                if (!pd.empty())
                    (void)tryFreePreg(pd, cycle);
            }
        }
    }
}

void OooCoreMainline::retireCompletedUops()
{
    while (!rob_.empty() && rob_.front().state == "done") {
        Uop u = rob_.front();
        rob_.pop_front();
        freeOldPregs(u);
        log("retire", u);
    }
}

void OooCoreMainline::updateLsqReadiness(int64_t cycle, bool storesOnly)
{
    for (auto& u : lsq_) {
        if (u.state == "running" || u.state == "done")
            continue;
        if (storesOnly && u.opClass != "STORE")
            continue;
        if (u.opClass == "LOAD") {
            u.readyCycle = computeLoadReadyCycle(u);
        } else {
            auto ready = computeStoreReadyCycle(u);
            u.readyCycle = std::get<0>(ready);
            u.producerOpForStore = std::get<1>(ready);
            u.producerFormForStore = std::get<2>(ready);
            u.producerStartForStore = std::get<3>(ready);
        }
        u.state = (cycle >= u.readyCycle) ? "ready" : "blocked";
    }
}

void OooCoreMainline::updateShqReadiness(int64_t cycle)
{
    for (auto& u : shq_) {
        if (u.state == "running" || u.state == "done")
            continue;
        int64_t t = std::max<int64_t>(vfStartupCost_, u.shqReadyCycle);
        for (const auto& preg : u.pregSrc) {
            if (!preg.has_value())
                continue;
            auto it = pregProducer_.find(*preg);
            if (it == pregProducer_.end()) {
                if (pregPending_.count(*preg))
                    t = std::max<int64_t>(t, 1000000000);
                continue;
            }
            t = std::max<int64_t>(t, computeReadyTimeForSrc(it->second, u.op, u.form));
        }
        u.readyCycle = t;
        u.state = (cycle >= u.readyCycle) ? "ready" : "blocked";
    }
}

void OooCoreMainline::issueReadyLsu(
    int64_t cycle, int& issuedLoads, int& issuedStores, int& issuedTotal,
    std::unordered_set<int64_t>& membarBlockedLoggedIds)
{
    if (cycle < vfStartupCost_ || issuedTotal >= ubSlots_)
        return;

    struct Candidate {
        int classPriority = 0;
        int64_t streamSeq = 0;
        int64_t instId = 0;
    };

    const bool pregPressure = static_cast<int>(freelist_.size()) < lsuStorePriorityPregThreshold_;
    const std::string preferredClass = pregPressure ? "STORE" : "LOAD";
    std::vector<Candidate> candidates;
    for (const auto& u : lsq_) {
        if (u.state != "ready" || (u.opClass != "LOAD" && u.opClass != "STORE"))
            continue;
        const int64_t age = u.streamSeq >= 0 ? u.streamSeq : u.instId;
        candidates.push_back(Candidate{u.opClass == preferredClass ? 0 : 1, age, u.instId});
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
        return std::tie(lhs.classPriority, lhs.streamSeq, lhs.instId) <
               std::tie(rhs.classPriority, rhs.streamSeq, rhs.instId);
    });

    for (const auto& candidate : candidates) {
        if (issuedTotal >= ubSlots_)
            break;
        auto it = std::find_if(lsq_.begin(), lsq_.end(), [&](const Uop& u) { return u.instId == candidate.instId; });
        if (it == lsq_.end() || it->state != "ready")
            continue;
        Uop& u = *it;
        if (u.opClass == "LOAD" && issuedLoads >= loadPorts_)
            continue;
        if (u.opClass == "STORE" && issuedStores >= storePorts_)
            continue;
        if (blockedByControlUnit(u)) {
            if (membarBlockedLoggedIds.insert(u.instId).second)
                logMembarBlocked(u);
            continue;
        }
        if (u.opClass == "STORE" && !u.producerOpForStore.has_value())
            continue;

        u.startCycle = cycle;
        u.blockedReason.reset();
        u.doneCycle = cycle + u.latency;
        u.state = "running";
        scheduleSrcReleaseFromStart(u);
        if (u.opClass == "LOAD") {
            ++issuedLoads;
            for (const auto& pd : u.pregDst) {
                if (pd.empty())
                    continue;
                pregProducer_[pd] = ProducerInfo{u.op, u.form, *u.startCycle, "LOAD"};
                pregPending_.erase(pd);
            }
        } else {
            ++issuedStores;
            if (u.isShqTracked) {
                scheduleShqRelease(cycle, 1);
                u.isShqTracked = false;
            }
        }
        ++issuedTotal;

        if (auto* robU = findRobUop(u.instId)) {
            robU->producerOpForStore = u.producerOpForStore;
            robU->producerFormForStore = u.producerFormForStore;
            robU->producerStartForStore = u.producerStartForStore;
            robU->startCycle = u.startCycle;
            robU->doneCycle = u.doneCycle;
            robU->state = u.state;
            robU->isShqTracked = u.isShqTracked;
        }
        log("start", u);
        logStartSimple(u);
        lsq_.erase(it);
    }
}

void OooCoreMainline::step()
{
    const int64_t c = cycle_;

    runShqReleaseEvents(c);
    runSrcReleaseEvents(c);
    completeRunningUops(c);
    retireCompletedUops();
    tryFreeEligiblePregs(c);
    updateLsqReadiness(c);
    updateShqReadiness(c);
    int issuedLoads = 0;
    int issuedStores = 0;
    int issuedLsuTotal = 0;
    std::unordered_set<int64_t> membarBlockedLoggedIds;
    issueReadyLsu(c, issuedLoads, issuedStores, issuedLsuTotal, membarBlockedLoggedIds);
    updateShqReadiness(c);

    PortUsage exuUsedThisCycle(static_cast<size_t>(issuePorts_), false);
    IssuedSources issuedSrcsThisCycle;
    issueCompute(c, exuUsedThisCycle, issuedSrcsThisCycle);
    updateLsqReadiness(c, true);
    issueReadyLsu(c, issuedLoads, issuedStores, issuedLsuTotal, membarBlockedLoggedIds);
    ++cycle_;
}

void OooCoreMainline::issueCompute(int64_t c, PortUsage& exuUsedThisCycle, IssuedSources& issuedSrcsThisCycle)
{
    if (!enableIsuQueueModel_) {
        issueDirectCompute(c, exuUsedThisCycle, issuedSrcsThisCycle);
        return;
    }
    dispatchComputeToExq(c, issuedSrcsThisCycle);
    issueComputeFromExq(c, exuUsedThisCycle);
}

bool OooCoreMainline::hasSameCycleSourceHazard(const Uop& uop, const IssuedSources& issuedSources) const
{
    if (!enforceSameCycleSrcHazard_ || theoreticalLimitMode_)
        return false;
    return std::any_of(uop.pregSrc.begin(), uop.pregSrc.end(), [&](const auto& source) {
        return source.has_value() && issuedSources.count(*source) != 0;
    });
}

void OooCoreMainline::recordIssuedSources(const Uop& uop, IssuedSources& issuedSources) const
{
    for (const auto& source : uop.pregSrc) {
        if (source)
            issuedSources.insert(*source);
    }
}

int OooCoreMainline::selectDirectIssuePort(const Uop& uop, int64_t cycle, const PortUsage& portUsage) const
{
    for (int port : eligibleExuPorts(uop)) {
        if (port < 0 || port >= issuePorts_ || portUsage[static_cast<size_t>(port)])
            continue;
        const bool isSfu = uop.fuType == "SFU";
        const auto& previousOp = enableCrossFuIi_ ? lastOpExu_ : (isSfu ? lastOpSfu_ : lastOpAlu_);
        const auto& previousForm = enableCrossFuIi_ ? lastFormExu_ : (isSfu ? lastFormSfu_ : lastFormAlu_);
        const auto& previousIssue =
            enableCrossFuIi_ ? lastIssueCycleExu_ : (isSfu ? lastIssueCycleSfu_ : lastIssueCycleAlu_);
        if (cycle >=
            previousIssue[static_cast<size_t>(port)] +
                getIi(
                    &previousOp[static_cast<size_t>(port)], &previousForm[static_cast<size_t>(port)], uop.op, uop.form))
            return port;
    }
    return -1;
}

void OooCoreMainline::startComputeUop(Uop& uop, int64_t cycle, int port)
{
    uop.startCycle = cycle;
    uop.doneCycle = cycle + uop.latency;
    uop.state = "running";
    uop.exuPort = port;
    scheduleSrcReleaseFromStart(uop);
    if (auto* robUop = findRobUop(uop.instId)) {
        robUop->startCycle = uop.startCycle;
        robUop->doneCycle = uop.doneCycle;
        robUop->state = uop.state;
        robUop->exuPort = uop.exuPort;
    }
    log("start", uop);
    logStartSimple(uop);
}

void OooCoreMainline::updateIssueHistory(const Uop& uop, int64_t cycle, int port)
{
    const size_t index = static_cast<size_t>(port);
    if (enableIsuQueueModel_ || enableCrossFuIi_) {
        lastIssueCycleExu_[index] = cycle;
        lastOpExu_[index] = uop.op;
        lastFormExu_[index] = uop.form;
    }
    if (!enableIsuQueueModel_ && enableCrossFuIi_)
        return;
    if (uop.fuType == "SFU") {
        lastIssueCycleSfu_[index] = cycle;
        lastOpSfu_[index] = uop.op;
        lastFormSfu_[index] = uop.form;
        return;
    }
    lastIssueCycleAlu_[index] = cycle;
    lastOpAlu_[index] = uop.op;
    lastFormAlu_[index] = uop.form;
}

void OooCoreMainline::publishComputeResults(const Uop& uop)
{
    for (const auto& destination : uop.pregDst) {
        if (destination.empty())
            continue;
        pregProducer_[destination] = ProducerInfo{uop.op, uop.form, *uop.startCycle, "COMPUTE"};
        pregPending_.erase(destination);
    }
}

void OooCoreMainline::issueDirectCompute(int64_t c, PortUsage& exuUsedThisCycle, IssuedSources& issuedSrcsThisCycle)
{
    int ex = 0;
    for (auto it = shq_.begin(); it != shq_.end();) {
        auto& u = *it;
        if (u.state != "ready" || u.opClass != "COMPUTE") {
            ++it;
            continue;
        }
        if (ex >= issuePorts_)
            break;
        if (hasSameCycleSourceHazard(u, issuedSrcsThisCycle)) {
            ++it;
            continue;
        }
        const int chosenPort = selectDirectIssuePort(u, c, exuUsedThisCycle);
        if (chosenPort < 0) {
            ++it;
            continue;
        }
        startComputeUop(u, c, chosenPort);
        ++ex;
        exuUsedThisCycle[static_cast<size_t>(chosenPort)] = true;
        recordIssuedSources(u, issuedSrcsThisCycle);
        updateIssueHistory(u, c, chosenPort);
        exqInflight_[static_cast<size_t>(chosenPort)] += 1;
        publishComputeResults(u);
        it = shq_.erase(it);
    }
}

void OooCoreMainline::dispatchComputeToExq(int64_t c, IssuedSources& issuedSrcsThisCycle)
{
    std::vector<int> shqToExqCnt(static_cast<size_t>(issuePorts_), 0);
    int exCount = 0;
    const bool useFuRrFifo = useFuRoundRobinFifo();
    std::unordered_set<std::string> blockedFuTypes;
    for (auto it = shq_.begin(); it != shq_.end();) {
        auto& u = *it;
        const std::string& fuType = u.fuType;
        if (useFuRrFifo && blockedFuTypes.count(fuType)) {
            ++it;
            continue;
        }
        if (u.state != "ready") {
            ++it;
            continue;
        }
        if (exCount >= issuePorts_)
            break;
        if (hasSameCycleSourceHazard(u, issuedSrcsThisCycle)) {
            if (useFuRrFifo)
                blockedFuTypes.insert(fuType);
            ++it;
            continue;
        }
        const std::vector<int> legalPorts = eligibleExuPorts(u);
        std::vector<int> candidates;
        for (int port : legalPorts) {
            if (port < 0 || port >= issuePorts_)
                continue;
            if (shqToExqCnt[static_cast<size_t>(port)] >= shqToExqPortPerCycle_)
                continue;
            const auto& q = exqWait_[static_cast<size_t>(port)];
            int occ = static_cast<int>(q.at("ALU").size() + q.at("SFU").size());
            if (exqCapacityCountsInflight_)
                occ += exqInflight_[static_cast<size_t>(port)];
            if (occ >= exqDepth_)
                continue;
            candidates.push_back(port);
        }

        if (candidates.empty()) {
            bool allPortsLegal = true;
            for (int port = 0; port < issuePorts_; ++port) {
                if (std::find(legalPorts.begin(), legalPorts.end(), port) == legalPorts.end()) {
                    allPortsLegal = false;
                    break;
                }
            }
            if (useFuRrFifo && allPortsLegal)
                blockedFuTypes.insert(fuType);
            ++it;
            continue;
        }

        const size_t shqIndex = static_cast<size_t>(std::distance(shq_.begin(), it));
        if (useExu0Reserve() && std::find(candidates.begin(), candidates.end(), 0) != candidates.end() &&
            legalPorts.size() > 1 && u.dispatchExu != "EXU0_ONLY") {
            const int pressure = exu0OnlyPressureCount(shqIndex);
            if (pressure > 0 && candidates.size() > 1) {
                std::vector<int> occupancy(static_cast<size_t>(issuePorts_), 0);
                for (int port = 0; port < issuePorts_; ++port) {
                    const auto& q = exqWait_[static_cast<size_t>(port)];
                    occupancy[static_cast<size_t>(port)] = static_cast<int>(q.at("ALU").size() + q.at("SFU").size());
                    if (exqCapacityCountsInflight_)
                        occupancy[static_cast<size_t>(port)] += exqInflight_[static_cast<size_t>(port)];
                }

                auto balanceError = [&](int port) {
                    std::vector<int> projected = occupancy;
                    ++projected[static_cast<size_t>(port)];
                    double nonExu0Average = 0.0;
                    for (int other = 1; other < issuePorts_; ++other)
                        nonExu0Average += projected[static_cast<size_t>(other)];
                    nonExu0Average /= std::max(1, issuePorts_ - 1);
                    return std::abs((nonExu0Average - projected[0]) - pressure);
                };

                double bestError = std::numeric_limits<double>::infinity();
                for (int port : candidates)
                    bestError = std::min(bestError, balanceError(port));
                std::vector<int> balancedCandidates;
                std::copy_if(
                    candidates.begin(), candidates.end(), std::back_inserter(balancedCandidates),
                    [&](int port) { return balanceError(port) == bestError; });
                candidates = std::move(balancedCandidates);
            }
        }

        const int64_t recv = c + exqRecvDelay_;
        auto predictCandidate = [&](int port) {
            const auto& fq = exqWait_[static_cast<size_t>(port)].at(fuType);
            if (!fq.empty()) {
                const Uop& prev = fq.back();
                return std::max<int64_t>(recv, prev.exqPredIssue + getIi(&prev.op, &prev.form, u.op, u.form));
            }
            return predictExqIssueCycle(port, fuType, u.op, u.form, recv);
        };

        int chosenPort = -1;
        int64_t chosenPred = 0;
        if (useFuRrFifo) {
            chosenPort = selectFuRoundRobinPort(fuType, candidates);
            chosenPred = predictCandidate(chosenPort);
        } else {
            int chosenOcc = 0;
            for (int port : candidates) {
                const auto& q = exqWait_[static_cast<size_t>(port)];
                int occ = static_cast<int>(q.at("ALU").size() + q.at("SFU").size());
                if (exqCapacityCountsInflight_)
                    occ += exqInflight_[static_cast<size_t>(port)];
                const int64_t pred = predictCandidate(port);
                const auto key = std::make_tuple(pred, occ, port);
                const auto best = std::make_tuple(chosenPred, chosenOcc, chosenPort);
                if (chosenPort < 0 || key < best) {
                    chosenPort = port;
                    chosenPred = pred;
                    chosenOcc = occ;
                }
            }
        }
        u.exuPort = chosenPort;
        u.exqRecvCycle = c + exqRecvDelay_;
        u.exqPredIssue = chosenPred;
        u.state = "exq_wait";
        if (u.opClass == "COMPUTE" || u.opClass == "STORE") {
            scheduleShqRelease(c, 1);
            u.isShqTracked = false;
            if (auto* robU = findRobUop(u.instId))
                robU->isShqTracked = false;
        }
        exqWait_[static_cast<size_t>(chosenPort)][fuType].push_back(u);
        shqToExqCnt[static_cast<size_t>(chosenPort)] += 1;
        ++exCount;
        recordIssuedSources(u, issuedSrcsThisCycle);
        it = shq_.erase(it);
    }
}

void OooCoreMainline::issueComputeFromExq(int64_t c, PortUsage& exuUsedThisCycle)
{
    for (int port = 0; port < issuePorts_; ++port) {
        if (exuUsedThisCycle[static_cast<size_t>(port)])
            continue;
        auto& q = exqWait_[static_cast<size_t>(port)];
        std::string bestFu;
        Uop* bestU = nullptr;
        std::tuple<int64_t, int64_t, int64_t> bestKey{0, 0, 0};
        for (const std::string& fuType : {std::string("ALU"), std::string("SFU")}) {
            auto& fq = q[fuType];
            if (fq.empty())
                continue;
            Uop& cand = fq.front();
            if (exqIssueInflightCapPerPort_ > 0 &&
                exqInflight_[static_cast<size_t>(port)] >= exqIssueInflightCapPerPort_)
                continue;
            if (cand.exqRecvCycle > c)
                continue;
            int64_t ready = std::max<int64_t>(vfStartupCost_, cand.shqReadyCycle);
            bool pending = false;
            for (const auto& preg : cand.pregSrc) {
                if (!preg)
                    continue;
                auto pit = pregProducer_.find(*preg);
                if (pit == pregProducer_.end()) {
                    if (pregPending_.count(*preg))
                        pending = true;
                    continue;
                }
                ready = std::max<int64_t>(ready, computeReadyTimeForSrc(pit->second, cand.op, cand.form));
            }
            if (pending || ready > c)
                continue;
            const int64_t ii = getIi(
                &lastOpExu_[static_cast<size_t>(port)], &lastFormExu_[static_cast<size_t>(port)], cand.op, cand.form);
            if (c < lastIssueCycleExu_[static_cast<size_t>(port)] + ii)
                continue;
            auto key = std::make_tuple(ready, cand.exqRecvCycle, cand.instId);
            if (!bestU || key < bestKey) {
                bestFu = fuType;
                bestU = &cand;
                bestKey = key;
            }
        }
        if (!bestU)
            continue;
        Uop u = *bestU;
        q[bestFu].pop_front();
        startComputeUop(u, c, port);
        updateIssueHistory(u, c, port);
        exuUsedThisCycle[static_cast<size_t>(port)] = true;
        exqInflight_[static_cast<size_t>(port)] += 1;
        publishComputeResults(u);
    }
}

} // namespace vfsim
