// vf_trace.hpp — A5 VF 运行期 for-循环拦截的事件流运行时。
// 配合 PtoLoopTracePass(给 for 插桩)使用:pass 插 __pto_trace_loop_{enter,iter,exit},
// VF body 走 a5_vf_stub.hpp::capture::rec() 追加 Op 事件。本头维护 thread-local 事件流,
// BuildVfInfo() 把它折叠成 VfInfo 嵌套树(纯函数,可喂合成事件流单测)。
// 生命周期:VfInfo/VfInst/MemInfo 中的字符串持 owning std::string(见 vf_info.hpp),折叠时从
//    Events() 拷出。故 ~ScopeSentinel 内 Reset() 清空 Events() 后,VfInfo 仍可安全活到
//    EndPtoInstr→PredictVfCycles 消费,无悬空引用。
#pragma once

#include "vf_info.hpp"

#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace pto::mocker::vf::trace {

enum class EvKind : uint8_t {
    LoopEnter,
    LoopIter,
    LoopExit,
    Op,
    MemBar,
};

struct Event {
    EvKind kind;
    uint64_t loopId = 0;    // loop 事件的 loopId(Op/MemBar 不用)
    std::string opName;     // Op/MemBar 的微指令名(loop 事件为空)
    VfInst inst;            // Op 的完整 operand 信息
};

inline std::vector<Event> &Events() {
    static thread_local std::vector<Event> e;
    return e;
}
inline bool &Armed() {
    static thread_local bool a = false;
    return a;
}

inline void Reset() { Events().clear(); }
inline void Arm(bool v) { Armed() = v; }

inline void RecordOp(VfInst inst) {
    if (Armed()) Events().push_back({EvKind::Op, 0, inst.opName, std::move(inst)});
}
inline void RecordOp(std::string name) {
    RecordOp(VfInst{std::move(name), {}, {}});
}
inline void RecordMemBar(std::string name) {
    if (Armed()) Events().push_back({EvKind::MemBar, 0, std::move(name)});
}

}  // namespace pto::mocker::vf::trace

// C ABI hook:PtoLoopTracePass 在 IR 层注入对这些符号的 call。前端会误判"未使用"不发定义,
// __attribute__((used)) 强制发射;配合 inline(COMDAT)→ 多 TU include 不冲突。
#define PTO_TRACE_HOOK __attribute__((used)) inline
extern "C" {
PTO_TRACE_HOOK void __pto_trace_loop_enter(uint64_t loopId, const char * /*file*/, int /*line*/, int /*col*/) {
    if (!pto::mocker::vf::trace::Armed()) return;
    pto::mocker::vf::trace::Events().push_back(
        {pto::mocker::vf::trace::EvKind::LoopEnter, loopId, {}});
}
PTO_TRACE_HOOK void __pto_trace_loop_iter(uint64_t loopId) {
    if (!pto::mocker::vf::trace::Armed()) return;
    pto::mocker::vf::trace::Events().push_back(
        {pto::mocker::vf::trace::EvKind::LoopIter, loopId, {}});
}
PTO_TRACE_HOOK void __pto_trace_loop_exit(uint64_t loopId) {
    if (!pto::mocker::vf::trace::Armed()) return;
    pto::mocker::vf::trace::Events().push_back(
        {pto::mocker::vf::trace::EvKind::LoopExit, loopId, {}});
}
// __VEC_SCOPE__ 区间 marker:由 a5_vf_stub.hpp 的 ScopeSentinel 构造/析构调用。运行期 no-op,
// 仅作 IR 里可被 pass 识别的地标 call,pass 据此判定哪些 for 在 scope 内(只给区间内 for 插桩)。
PTO_TRACE_HOOK void __pto_vf_scope_enter() {}
PTO_TRACE_HOOK void __pto_vf_scope_exit() {}
}
#undef PTO_TRACE_HOOK

namespace pto::mocker::vf::trace {

// BuildVfInfo:递归下降解析事件流。每个 loop 由 enter..(iter+body)..exit 描述;
// 每轮 body 由 parseBody 解析(停在属于本 loop 的下一条 iter/exit)。
// 均匀循环假设:同一 loop 各轮 body 结构须相同(VF 模板成立),否则报错不兜底。
namespace detail {

constexpr uint64_t kNoEnclosing = static_cast<uint64_t>(-1);

inline bool NodesEqual(const std::vector<VfNode> &a, const std::vector<VfNode> &b);

inline bool NodeEqual(const VfNode &a, const VfNode &b) {
    if (a.kind != b.kind) return false;
    switch (a.kind) {
        case VfNodeKind::INST:  return AsInst(a) == AsInst(b);
        case VfNodeKind::MEMBAR: return AsMemBar(a).name == AsMemBar(b).name;
        case VfNodeKind::LOOP: {
            const VfLoop &la = AsLoop(a), &lb = AsLoop(b);
            return la.count == lb.count && NodesEqual(la.body, lb.body);
        }
    }
    return false;
}

inline bool NodesEqual(const std::vector<VfNode> &a, const std::vector<VfNode> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!NodeEqual(a[i], b[i])) return false;
    return true;
}

// 解析器:游标在事件流上前进。ok=false 表示流非法(不兜底)。
struct Parser {
    const std::vector<Event> &ev;
    size_t i = 0;
    bool ok = true;
    std::string err;

    explicit Parser(const std::vector<Event> &e) : ev(e) {}

    // 解析一段 body(直到遇到属于 enclosingId 的 iter/exit;enclosingId==kNoEnclosing 时解析到流末尾)。
    std::vector<VfNode> parseBody(uint64_t enclosingId) {
        std::vector<VfNode> nodes;
        while (i < ev.size()) {
            const Event &e = ev[i];
            if (e.kind == EvKind::LoopIter || e.kind == EvKind::LoopExit) {
                if (enclosingId != kNoEnclosing && e.loopId == enclosingId) return nodes;
                ok = false;  // 不属于本层的 iter/exit 泄漏 → 非法(内层 loop 应已自行消费)
                err = "stray loop_iter/exit for loopId=" + std::to_string(e.loopId);
                return nodes;
            }
            if (e.kind == EvKind::Op) {
                nodes.push_back(MakeInst(e.inst));
                ++i;
            } else if (e.kind == EvKind::MemBar) {
                nodes.push_back(MakeMemBar(e.opName));
                ++i;
            } else if (e.kind == EvKind::LoopEnter) {
                std::vector<VfNode> loop = parseLoop();
                if (!ok) return nodes;
                nodes.push_back(MakeLoop(0, {}));  // 占位,下面替换
                nodes.back() = std::move(loop.front());
            } else {
                ok = false;
                err = "unexpected event kind in body";
                return nodes;
            }
        }
        return nodes;
    }

    // 游标在 LoopEnter(id);解析完整 loop,返回单节点 vector。
    // 计数语义:iter hook 插 header 顶部,每次进 header 都触发(含最后退出检查的空 body 迭代)。
    // 故真迭代数 = 非空 body 的迭代数;空 body 迭代(退出检查)丢弃,均匀性只比非空 body。
    std::vector<VfNode> parseLoop() {
        uint64_t id = ev[i].loopId;
        ++i;  // consume LoopEnter
        uint64_t count = 0;
        std::vector<VfNode> canonBody;  // 首个非空 body,作均匀性基准
        bool haveCanon = false;
        while (i < ev.size()) {
            const Event &e = ev[i];
            if (e.kind == EvKind::LoopIter && e.loopId == id) {
                ++i;  // consume iter
                std::vector<VfNode> body = parseBody(id);
                if (!ok) return {};
                if (body.empty()) continue;  // 退出检查迭代,丢弃
                ++count;
                if (!haveCanon) {
                    canonBody = std::move(body);
                    haveCanon = true;
                } else if (!NodesEqual(canonBody, body)) {
                    ok = false;
                    err = "non-uniform loop body for loopId=" + std::to_string(id) +
                          " (VF 模板应为均匀循环)";
                    return {};
                }
            } else if (e.kind == EvKind::LoopExit && e.loopId == id) {
                ++i;  // consume exit
                break;
            } else {
                ok = false;
                err = "malformed loop: expected iter/exit for loopId=" + std::to_string(id);
                return {};
            }
        }
        return {MakeLoop(count, std::move(canonBody))};  // count==0(全退出检查/零迭代)→ 空 body
    }
};

}  // namespace detail

struct BuildResult {
    bool ok = false;
    VfInfo info;
    std::string err;
};

// 把当前事件流折叠成 VfInfo。失败(非均匀循环/非法流)→ ok=false + err,不兜底。
inline BuildResult BuildVfInfo(std::string_view op, std::string_view shape) {
    BuildResult r;
    detail::Parser p(Events());
    std::vector<VfNode> tree = p.parseBody(detail::kNoEnclosing);
    if (!p.ok || p.i != Events().size()) {
        r.ok = false;
        r.err = p.ok ? ("unconsumed events at index " + std::to_string(p.i))
                     : p.err;
        return r;
    }
    r.ok = true;
    r.info.op = op;
    r.info.shape = shape;
    r.info.tree = std::move(tree);
    return r;
}

// 调试:把 VfInfo 树格式化成文本。
inline void FormatNodes(const std::vector<VfNode> &nodes, int depth, std::string &out) {
    std::string ind(depth * 2, ' ');
    for (const VfNode &n : nodes) {
        if (IsLoop(n)) {
            const VfLoop &lp = AsLoop(n);
            out += ind + "loop{n=" + std::to_string(lp.count) + "}\n";
            FormatNodes(lp.body, depth + 1, out);
        } else if (IsInst(n)) {
            out += ind + "op(" + AsInst(n).opName + ")\n";
        } else {
            out += ind + "membar(" + AsMemBar(n).name + ")\n";
        }
    }
}

inline std::string FormatTree(const VfInfo &vf) {
    std::string out = "program(" + vf.op + ")\n";
    FormatNodes(vf.tree, 1, out);
    return out;
}

}  // namespace pto::mocker::vf::trace
