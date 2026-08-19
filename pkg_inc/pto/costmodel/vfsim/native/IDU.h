// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. See
// LICENSE in the root of the software repository for the full text of the License.

#ifndef VFSIM_NATIVE_IDU_H_
#define VFSIM_NATIVE_IDU_H_

#include <deque>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "native/IFU.h"

namespace vfsim {

struct IduDispatchBudget {
    int64_t freePreg = 0;
    int64_t freeShqQueue = 0;
    int64_t freeLsq = 0;
    int64_t freeShq = 0;
    int64_t issueBudget = 0;
    bool theoreticalLimitMode = false;
    bool theoreticalLimitVloopOnly = false;
};

struct IduDispatchRecord {
    int64_t cycle = 0;
    int64_t instId = 0;
    std::string op;
    std::vector<std::string> dst;
    std::vector<std::string> src;
    int64_t topBlockId = 0;
    int64_t vreg = 0;
    int64_t shqQueue = 0;
    int64_t lsq = 0;
    int64_t shq = 0;
    std::string staticInstructionId;
    std::vector<std::pair<std::string, int64_t>> iterationPath;
    int64_t streamSeq = -1;
};

struct VLoopTraceRecord {
    int64_t topBlockId = 0;
    std::string loopId;
    std::vector<int64_t> iter;
    int64_t startCycle = 0;
};

class Idu {
public:
    Idu(const UarchConfig& uarch, const ParamDb& db, ProgramAnalysis::ParamMap params = {},
        std::vector<int64_t> loopBounds = {}, int64_t totalTopBlocks = 1,
        std::unordered_map<int, std::vector<int64_t>> topBlockLoopBounds = {}, std::string dtype = "fp32",
        std::unordered_map<std::string, ValueInfo> values = {});

    bool empty() const noexcept { return window_.empty(); }
    bool canAccept() const;
    void accept(const DynamicInst& inst);

    std::vector<DynamicInst> dispatch(int64_t cycle, const IduDispatchBudget& budget);

    const ParamDb& db() const noexcept { return db_; }
    const std::deque<DynamicInst>& window() const noexcept { return window_; }
    const std::vector<IduDispatchRecord>& dispatchLog() const noexcept { return dispatchLog_; }
    const std::vector<VLoopTraceRecord>& vloopTrace() const noexcept { return vloopTrace_; }

private:
    struct LoopFrame {
        int64_t beginIdx = 0;
        int64_t endIdx = 0;
        int64_t loopId = 0;
        int64_t itersTotal = 0;
        int64_t iterNow = 0;
        bool isInnermost = false;
        int64_t unroll = 1;
        int64_t topBlockId = 0;
    };

    const ParamDb& db_;
    std::string dtype_;
    int64_t windowWidth_ = 0;
    int64_t issueWidth_ = 0;
    bool theoreticalLimitMode_ = false;
    bool theoreticalLimitVloopOnly_ = false;
    int64_t vfStartupCost_ = 0;
    int64_t iduDispatchStartAdvance_ = 0;
    int64_t vloopToDispatchDelay_ = 4;
    int64_t initialTopBlockVloopStartCycle_ = 19;
    int64_t nestedVloopInitialStartGap_ = 1;
    int64_t loop1MinFeedbackGap_ = 7;
    int64_t innermostIterDispatchStride_ = 1;
    bool globalShqPregGate_ = false;

    std::deque<DynamicInst> window_;
    ProgramAnalysis analysis_;
    std::vector<int64_t> loopBounds_;
    int64_t totalTopBlocks_ = 1;
    std::unordered_map<int, std::vector<int64_t>> topBlockLoopBounds_;

    std::unordered_map<int64_t, int64_t> topBlockVloopStart_;
    std::unordered_map<int64_t, int64_t> topBlockBodyOpenTime_;
    std::unordered_map<std::string, int64_t> vloopStart_;
    std::unordered_map<std::string, int64_t> bodyOpenTime_;
    std::unordered_map<std::string, int64_t> lastDispatchTime_;
    std::unordered_map<std::string, int64_t> blockBaseCycle_;
    std::vector<VLoopTraceRecord> vloopTrace_;
    std::vector<IduDispatchRecord> dispatchLog_;

    void initVloopStarts();
    void setTopBlockVloop(int64_t topBlockId, int64_t startCycle);
    void initTopBlockNestedStarts(int64_t topBlockId, int64_t topVloopStart);

    std::string makeKey(int64_t topBlockId, const std::string& loopId, const std::vector<int64_t>& iters) const;
    std::optional<std::string> normalizeBlockKey(
        const std::pair<std::string, std::vector<int64_t>>& raw, int64_t topBlockId) const;
    std::optional<std::string> currentInnerBlockKey(const DynamicInst& inst) const;

    void updateLastDispatch(const DynamicInst& inst, int64_t cycle);
    void triggerNextVloops(const DynamicInst& inst, int64_t cycle);
    bool isLastInstOfTopBlock(const DynamicInst& inst) const;
};

} // namespace vfsim

#endif // VFSIM_NATIVE_IDU_H_
