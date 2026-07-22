// vf_info.hpp — A5 VF CCE mock 的嵌套树数据结构。
// 由运行期 LLVM-pass 拦截构建(vf_trace.hpp::BuildVfInfo):pass 给 VF 模板 for 插 hook,
// stub rec() 发 op 事件,事件流折叠成本树。树遍历计费(vf_cost.hpp::PredictVfCycles):
// 对 loop 乘 count、对 inst 进入 VfSim 或 fallback、对 membar 加固定惩罚。
// 递归:VfLoop 持 vector<VfNode>,VfNode 持 variant<VfLoop,...>。前置声明 VfNode 后定义 VfLoop,
// 再定义 VfNode——vector<不完整类型>作成员声明合法,使用点两类型均已完整。
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace pto::mocker::vf {

enum class VfNodeKind : uint8_t {
    LOOP,
    INST,
    MEMBAR,
};

enum class MemLocation : uint8_t {
    PhyRegister,
    UB,
};

// dtype 只属于 operand。VfInst/VfInfo 不保存全局 dtype，适配层会从 operand 校验并推导
// VfSim 当前要求的统一 form。
struct MemInfo {
    std::string name;  // VfSim 命名：物理寄存器以 V 开头，UB 地址以 mem 开头
    MemLocation location = MemLocation::PhyRegister;
    std::string dtype;
};

inline bool operator==(const MemInfo &lhs, const MemInfo &rhs)
{
    return lhs.name == rhs.name && lhs.location == rhs.location && lhs.dtype == rhs.dtype;
}

// 字符串持 owning std::string(非 string_view):BuildVfInfo 折叠树后,trace::Events() 会在
// ~ScopeSentinel 内被 Reset() 清空,而 VfInfo 会被 push 进 vf_infos 活到 EndPtoInstr→PredictVfCycles
// 之后才消费。若用 string_view 指向 Event::opName,Reset 后即悬空。owning string 解耦两边生命周期。
struct VfInst {
    std::string opName;  // CCE 微指令名(vlds/vadd/vsts/…，predicate 设置不进入)
    std::vector<MemInfo> dst;
    std::vector<MemInfo> src;
};

inline bool operator==(const VfInst &lhs, const VfInst &rhs)
{
    return lhs.opName == rhs.opName && lhs.dst == rhs.dst && lhs.src == rhs.src;
}

struct VfMemBar {
    std::string name;  // pipe_barrier 等
};

struct VfNode;  // 前置:VfLoop 的 body 需要它(不完整类型可作 vector 元素声明)

struct VfLoop {
    uint64_t count = 0;            // bound 求值后的循环次数(cost 期填)
    std::vector<VfNode> body;      // 循环体:更深 loop 或 inst/membar 叶子
};

struct VfNode {
    VfNodeKind kind;
    std::variant<VfLoop, VfInst, VfMemBar> v;
};

struct VfInfo {
    std::string op;      // "TADD"/"TEXP"/…
    std::string shape;   // "1D_NoPost"/"2D_Post"/…
    std::vector<VfNode> tree; // 顶层节点(通常一个根 loop)
};

// ── 构造助手 ──
inline VfNode MakeLoop(uint64_t count, std::vector<VfNode> body)
{
    return VfNode{VfNodeKind::LOOP, VfLoop{count, std::move(body)}};
}
// 注:name 显式构 std::string —— std::string 的 string_view 构造是 explicit,
// 聚合初始化 VfInst{name}(copy-init 上下文)不能用 explicit ctor,故须先构 string 再 move 进成员。
inline VfNode MakeInst(VfInst inst)
{
    return VfNode{VfNodeKind::INST, std::move(inst)};
}
inline VfNode MakeInst(std::string_view name)
{
    return MakeInst(VfInst{std::string{name}, {}, {}});
}
inline VfNode MakeMemBar(std::string_view name)
{
    return VfNode{VfNodeKind::MEMBAR, VfMemBar{std::string{name}}};
}

inline const VfLoop &AsLoop(const VfNode &n) { return std::get<VfLoop>(n.v); }
inline const VfInst &AsInst(const VfNode &n) { return std::get<VfInst>(n.v); }
inline const VfMemBar &AsMemBar(const VfNode &n) { return std::get<VfMemBar>(n.v); }
inline VfLoop &AsLoop(VfNode &n) { return std::get<VfLoop>(n.v); }
inline VfInst &AsInst(VfNode &n) { return std::get<VfInst>(n.v); }
inline VfMemBar &AsMemBar(VfNode &n) { return std::get<VfMemBar>(n.v); }
inline bool IsLoop(const VfNode &n) { return n.kind == VfNodeKind::LOOP; }
inline bool IsInst(const VfNode &n) { return n.kind == VfNodeKind::INST; }
inline bool IsMemBar(const VfNode &n) { return n.kind == VfNodeKind::MEMBAR; }

}  // namespace pto::mocker::vf
