// vec_cycle_generated.hpp — A5 每条 CCE 微指令的 cycle cost 表(查表用)。
// ⚠️ 当前为占位常量(Phase 3a/3c),不对齐 formula 真值;Phase 3b 用 NNLS 从 formula 表反推
//    每指令 cost 并重新生成本文件,接口 VecCycle(name) 不变。树遍历再乘 loop 次数。
#pragma once

#include <cstdint>
#include <initializer_list>
#include <string_view>

namespace pto::mocker::vf {

inline constexpr uint64_t kMemBarPenaltyPlaceholder = 20;  // membar 固定惩罚(占位;3b 重新标定)
inline constexpr uint64_t kUnknownInstructionFallbackCycles = 5;

inline bool IsOneOf(std::string_view name, std::initializer_list<std::string_view> candidates)
{
    for (std::string_view candidate : candidates) {
        if (name == candidate) {
            return true;
        }
    }
    return false;
}

inline uint64_t VecCycle(std::string_view name)
{
    if (IsOneOf(name, {"plt_b8", "plt_b16", "plt_b32", "pset_b8", "pset_b16", "pset_b32"})) {
        return 2;
    }
    if (IsOneOf(name, {"vlds", "vsts", "vdup", "vmov", "vsel"})) {
        return 4;
    }
    if (IsOneOf(name, {"vadd", "vsub", "vmul", "vand", "vor", "vxor", "vshl", "vshr", "vshls", "vshrs",
                       "vabs", "vrelu", "vlrelu", "vnot", "vneg", "vmin", "vmax", "vmins", "vadds",
                       "pand", "por", "pnot"})) {
        return 6;
    }
    if (IsOneOf(name, {"vdiv", "vexp", "vsqrt", "vln", "vcvt", "vmadd", "vmuls", "vmula", "vtrc"}) ||
        name.rfind("vcmp", 0) == 0) {
        return 10;
    }
    if (name == "pipe_barrier") {
        return kMemBarPenaltyPlaceholder;
    }
    return kUnknownInstructionFallbackCycles;
}

// Unsupported VfSim programs use the A2/A3-style repeat-times * per-op-cycle
// calculation. Keep the unknown-op value explicit so it is visible to callers.
inline uint64_t FallbackVecCycle(std::string_view name)
{
    return VecCycle(name);
}

}  // namespace pto::mocker::vf
