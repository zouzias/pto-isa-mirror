// vf_cost.hpp — VfInfo 嵌套树遍历计费,对齐 tilesim predict_vf_cycles。
// LOOP 节点把 count 乘进 mul 后递归 body,INST 查 vec_cycle(name)×mul,MEMBAR 加固定惩罚×mul。
// 占位常量阶段(vec_cycle_generated.hpp)数值不对齐 formula;Phase 3b NNLS 标定后换真值。
#pragma once

#include "pto/costmodel/a5/VfSim/VfSimCostModel.h"
#include "vec_cycle_generated.hpp"
#include "vf_info.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace pto::mocker::vf {

namespace detail {

// 递归遍历:mul 是「当前乘到叶子的循环次数之积」。
inline void WalkNodes(const std::vector<VfNode> &nodes, uint64_t mul, uint64_t &total)
{
    for (const VfNode &n : nodes) {
        if (IsLoop(n)) {
            const VfLoop &lp = AsLoop(n);
            WalkNodes(lp.body, mul * lp.count, total);  // 进循环体,乘上本层 count
        } else if (IsInst(n)) {
            total += VecCycle(AsInst(n).opName) * mul;
        } else {  // MEMBAR
            total += kMemBarPenaltyPlaceholder * mul;
        }
    }
}

}  // namespace detail

// 树遍历总 cycle(占位常量;3b 标定后变真值)。
inline uint64_t PredictVfCycles(const VfInfo &vf)
{
    return PredictVfCyclesWithVfSim(std::vector<VfInfo>{vf});
}

// 一个 PTO-ISA 指令内多个 VF 的结算。占位实现 = 各 VF cycle 之和(真实建模应考虑跨 VF overlap)。
inline uint64_t PredictVfCycles(const std::vector<VfInfo> &vfs)
{
    return PredictVfCyclesWithVfSim(vfs);
}

// ── 供单测/报告用的辅助 ──

// 顶层 loop 次数乘积(迭代倍数;交叉校验用)。
inline uint64_t LoopProduct(const std::vector<VfNode> &nodes)
{
    uint64_t prod = 1;
    for (const VfNode &n : nodes)
        if (IsLoop(n)) {
            prod *= AsLoop(n).count;
            prod *= LoopProduct(AsLoop(n).body);
        }
    return prod;
}

// 展平叶子微指令名(DFS 序;不含 loop 倍数。校验 body 是否挂最深 loop)。
inline std::vector<std::string> FlattenLeafInstrs(const std::vector<VfNode> &nodes)
{
    std::vector<std::string> out;
    for (const VfNode &n : nodes) {
        if (IsLoop(n)) {
            auto sub = FlattenLeafInstrs(AsLoop(n).body);
            out.insert(out.end(), sub.begin(), sub.end());
        } else if (IsInst(n)) {
            out.push_back(AsInst(n).opName);
        }
    }
    return out;
}

// 嵌套深度(loop 层数)。
inline uint64_t LoopDepth(const std::vector<VfNode> &nodes)
{
    uint64_t best = 0;
    for (const VfNode &n : nodes)
        if (IsLoop(n)) best = std::max(best, 1 + LoopDepth(AsLoop(n).body));
    return best;
}

}  // namespace pto::mocker::vf
