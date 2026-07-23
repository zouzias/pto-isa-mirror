// cce_costmodel.hpp — A5 CCE mock 伞头。被 common/arch_select.hpp 在 A5(__NPU_ARCH__==3101/3510)
// 下 include,拉起 host 桩 + VF 模板可编译环境 + VF 结构构建(vf_trace)+ 后端计费(vf_cost)。
// vadd 桩只 rec,ScopeSentinel 析构 BuildVfInfo,PtoInstrScope 析构用 PredictVfCycles 结算。
// 不 include a2a3/cce_costmodel_core.hpp(A5 cycle 走 vf_cost,不用 A2A3 逐指令桩算 cycle)。
#pragma once

#include "a5_vf_stub.hpp"  // host 桩 + VF 模板 + vf_trace(ScopeSentinel/rec/BuildVfInfo)
#include "vf_cost.hpp"     // PredictVfCycles(后端结算,含 vector<VfInfo> 重载)
