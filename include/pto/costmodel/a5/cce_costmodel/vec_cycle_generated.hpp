// vec_cycle_generated.hpp — A5 每条 CCE 微指令的 cycle cost 表(查表用)。
// ⚠️ 当前为占位常量(Phase 3a/3c),不对齐 formula 真值;Phase 3b 用 NNLS 从 formula 表反推
//    每指令 cost 并重新生成本文件,接口 VecCycle(name) 不变。树遍历再乘 loop 次数。
#pragma once

#include <cstdint>
#include <string_view>

namespace pto::mocker::vf {

inline constexpr uint64_t kMemBarPenaltyPlaceholder = 20;  // membar 固定惩罚(占位;3b 重新标定)

inline uint64_t VecCycle(std::string_view name)
{
    // 谓词生成:plt_b*/pset_b*
    if (name == "plt_b8" || name == "plt_b16" || name == "plt_b32" ||
        name == "pset_b8" || name == "pset_b16" || name == "pset_b32") {
        return 2;
    }
    // 搬运:vlds/vsts/vdup/vmov/vsel
    if (name == "vlds" || name == "vsts" || name == "vdup" || name == "vmov" || name == "vsel") {
        return 4;
    }
    // 基础计算:vadd/vsub/vmul/vand/vor/vxor/vshl/vshr/vabs/vrelu/vnot/…
    if (name == "vadd" || name == "vsub" || name == "vmul" || name == "vand" || name == "vor" ||
        name == "vxor" || name == "vshl" || name == "vshr" || name == "vshls" || name == "vshrs" ||
        name == "vabs" || name == "vrelu" || name == "vlrelu" || name == "vnot" || name == "vneg" ||
        name == "vmin" || name == "vmax" || name == "vmins" || name == "vadds" || name == "pand" ||
        name == "por" || name == "pnot") {
        return 6;
    }
    // 较重计算:vdiv/vexp/vsqrt/vln/vcvt/vmadd/vmuls/vmula/vcmp*/vtrc
    if (name == "vdiv" || name == "vexp" || name == "vsqrt" || name == "vln" || name == "vcvt" ||
        name == "vmadd" || name == "vmuls" || name == "vmula" || name == "vtrc" ||
        name.rfind("vcmp", 0) == 0) {
        return 10;
    }
    // 同步屏障
    if (name == "pipe_barrier") {
        return kMemBarPenaltyPlaceholder;
    }
    return 5;  // 兜底(未列名):中等
}

}  // namespace pto::mocker::vf
