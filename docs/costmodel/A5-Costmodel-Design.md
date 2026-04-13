# A5 平台 Costmodel 设计方案

## 1. 背景与动机

现有 costmodel 仅覆盖 A2/A3 平台（`include/pto/costmodel/a2a3/`），需要扩展支持 A5 平台。A5 与 A2A3 在指令集和向量编程模型上有显著差异，因此需要一套适配 A5 架构特性的 costmodel 设计。

## 2. A5 与 A2A3 的关键差异

### 2.1 向量编程模型：MemBase vs RegBase

| 维度 | A2A3 (membase) | A5 (regbase) |
|------|----------------|--------------|
| **指令签名** | `vadd(dst_ptr, src0_ptr, src1_ptr, repeats, strides...)` | `vadd(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING)` |
| **数据搬运** | 隐含在向量指令中，通过 stride 参数寻址 | 显式 `vlds(vreg, ptr, offset, dist)` / `vsts(vreg, ptr, offset, dist, preg)` |
| **编程模式** | 一步到位：运算即访存 | 三步：load -> compute -> store |
| **作用域** | 无显式 scope | `__VEC_SCOPE__ { ... }` 包裹向量操作序列 |
| **成本粒度** | 一条 CCE 调用 = 一次运算 | 一次运算 = vlds + vcompute + vsts 至少三条 CCE 调用 |

**示例对比：A2A3 的 TAdd vs A5 的 TAdd**

A2A3 (`include/pto/npu/a2a3/TAdd.hpp`):
```cpp
// 直接在内存地址上运算，stride/repeat 参数控制寻址
vadd(dst, src0, src1, repeats, dstBlkStride, src0BlkStride, src1BlkStride,
     dstRptStride, src0RptStride, src1RptStride);
```

A5 (`include/pto/npu/a5/TAdd.hpp` -> `TBinOp.hpp`):
```cpp
__VEC_SCOPE__ {
    RegTensor<T> vreg0, vreg1, vreg2;
    MaskReg preg;
    for (...) {
        preg = CreatePredicate<T>(sreg);
        vlds(vreg0, src0Ptr, offset, NORM);          // load src0 -> reg
        vlds(vreg1, src1Ptr, offset, NORM);          // load src1 -> reg
        vadd(vreg2, vreg0, vreg1, preg, MODE_ZEROING); // reg compute
        vsts(vreg2, dstPtr, offset, distValue, preg);  // store reg -> dst
    }
}
```

### 2.2 VFImplKind：向量实现变体

A5 平台每条 PTO 向量指令可以选择不同的底层实现策略，由 `VFImplKind` 枚举控制：

```cpp
enum VFImplKind : unsigned {
    VFIMPL_DEFAULT = 0,
    VFIMPL_1D_NO_POST_UPDATE = 1,  // 1D 展平循环，无地址自动递增
    VFIMPL_2D_NO_POST_UPDATE = 2,  // 2D 嵌套循环（行x列），无地址递增
    VFIMPL_1D_POST_UPDATE = 3,     // 1D 循环，带 POST_UPDATE 地址自增
    VFIMPL_2D_POST_UPDATE = 4,     // 2D 嵌套循环，带 POST_UPDATE
};
```

不同 `VFImplKind` 导致相同 PTO 指令产生不同数量和模式的 CCE 调用，例如：
- `1D` 模式：总 repeat = `ceil(row * col / elementsPerRepeat)`
- `2D` 模式：总 repeat = `row * ceil(col / elementsPerRepeat)`

### 2.3 VF 融合（VF Fusion）

VF 融合是 A5 平台的编译器优化特性：

- **定义**：编译器将同一 `__VEC_SCOPE__` 块内的 load-compute-store 序列融合成更高效的流水线操作
- **控制**：`#pragma no_simd_vf_fusion` 可在特定场景下禁用融合（如存在数据依赖时）
- **文档依据**：`docs/machine/abstract-machine.md` 明确提到 "The compiler applies fusions such as VF fusion when available"
- **代码依据**：`include/pto/npu/a5/TPartBinOps.hpp:144` 使用 `#pragma no_simd_vf_fusion` 禁用融合
- **性能影响**：VF 融合/不融合的时间差异非常大，costmodel **必须**感知融合状态

### 2.4 CCE 指令集差异

| 指令类别 | A2A3 | A5 |
|---------|------|-----|
| **向量 load/store** | (隐含在运算指令中) | `vlds`, `vsts`, `vsstb`, `vbr` |
| **向量运算** | `vadd(ptr, ptr, ptr, repeats, strides...)` | `vadd(reg, reg, reg, mask, mode)` |
| **向量标量运算** | `vadds(ptr, ptr, scalar, repeats, strides...)` | `vadds(reg, reg, scalar, mask, mode)` |
| **谓词/掩码** | `set_vector_mask(mask0, mask1)` | `plt_b8/b16/b32(scalar, POST_UPDATE)` |
| **同步** | `pipe_barrier` | `mem_bar(VST_VLD)` |
| **矩阵搬运** | `copy_matrix_cc_to_gm` | `copy_matrix_cc_to_ub`, `copy_matrix_cc_to_cbuf` |
| **特殊搬运** | (无) | `copy_cbuf_to_bt`, `copy_cbuf_to_fbuf` |
| **参数配置** | `set_l3d_rpt` | `set_loop3_para`, `set_channel_para`, `set_fmatrix`, `set_fmatrix_b` |
| **数据类型** | `QuantMode_t` (部分) | `QuantMode_t` (扩展), `AccToVecMode`, `ReluPreMode` |

---

## 3. CCE 调用膨胀分析

### 3.1 具体案例：TADD\<float, 64x128\>

以 `TADD<float, 64x128>` 为例，走一遍 A5 的代码展开路径：

```
参数推导：
  T = float (4 bytes)
  ElementsPerRepeat = CCE_VL / sizeof(float) = 256 / 4 = 64
  validRows = 64, validCols = 128
  repeatTimes = ceil(128 / 64) = 2

进入 TBinOps_2D_NoPostUpdate（TBinOp.hpp:68）：

  __VEC_SCOPE__ {
      for (i = 0; i < 64; ++i) {          // 外层 64 行
          for (j = 0; j < 2; ++j) {        // 内层 2 次 repeat
              plt_b32(sreg, POST_UPDATE);   // CCE 调用 #1  ← 谓词
              vlds(vreg0, ...);             // CCE 调用 #2  ← load src0
              vlds(vreg1, ...);             // CCE 调用 #3  ← load src1
              vadd(vreg2, vreg0, vreg1, preg, MODE_ZEROING);  // CCE 调用 #4  ← compute
              vsts(vreg2, ...);             // CCE 调用 #5  ← store dst
          }
      }
  }

  总 CCE 调用数 = 64 * 2 * 5 = 640 条
```

对比 A2A3 同 shape：
```
  vadd(dst, src0, src1, repeats=128, dstBlkStride, src0BlkStride, src1BlkStride,
       dstRptStride, src0RptStride, src1RptStride);
  总 CCE 调用数 = 1 条
```

**膨胀比：640 : 1**

### 3.2 各类指令的膨胀情况

| PTO 指令 | Shape | A2A3 CCE 数 | A5 CCE 数 | 膨胀倍数 |
|----------|-------|-------------|-----------|---------|
| TADD | float[64,64] | 1 | 320 | 320x |
| TADD | float[64,128] | 1 | 640 | 640x |
| TEXP | float[64,64] | 1 | 256 | 256x |
| TColSum | float[64,64] | ~5 | ~126 | ~25x |
| TRowSum | float[64,64] | ~10 | ~200+ | ~20x |
| TMov(Vec->Vec) | float[64,64] | ~1 | ~256 | 256x |
| TMATMUL | half[16,16]x[16,16] | 1 | 1 | 1x（矩阵不涉及 VF） |

一个典型 kernel（20 条 PTO 指令，float[64,128] tile）：
- **A2A3**: ~20 条 CCE trace 记录
- **A5**: ~6,000-12,000 条 CCE trace 记录

### 3.3 膨胀是否是问题？

**数据量本身不是致命问题**：几千到上万条 trace 记录在现代系统中完全可以承受。真正的挑战是 VF 融合的建模。

如果逐条累加 CCE 延迟：
```
scope 内单次迭代：plt_b32(1) + vlds(4) + vlds(4) + vadd(4) + vsts(4) = 17 cycles
640 条合计：64 * 2 * 17 = 2176 cycles（严重高估）

实际 VF 融合后：vf_startup(~5) + 128 * fused_pipeline(~6) = 773 cycles（约 3x 差异）
```

所以关键是：**evaluator 需要感知 VF scope 边界，在 scope 内应用融合规则，而不是逐条累加。**

---

## 4. 推荐方案：CCE Mock + VF Scope-Aware Evaluator

### 4.1 方案选型理由

CCE Mock 方案有一个 PTO 级方案无法比拟的优势：**一劳永逸**。

| 维度 | CCE Mock 方案 | PTO 级公式方案 |
|------|-------------|--------------|
| **新增 PTO 指令** | 零成本，只要用已有 CCE 内建函数就自动覆盖 | 每条 PTO 指令都要写对应的 cost 公式代码 |
| **CCE 指令种类** | 按种类建规则表（~30 条），覆盖所有组合 | 需要按 PTO 指令 x dtype x VFImplKind x 融合状态排列组合 |
| **拟合参数获取** | 只需 CCE 级延迟 + 融合系数（少量参数） | 需要每条 PTO 指令的端到端实测数据 |
| **维护成本** | PTO 实现变化不影响（只要 CCE 不变） | PTO 实现变化需同步更新公式 |

因此选择**保留 CCE Mock 方案**，通过增强 evaluator 来解决 VF 融合问题。

### 4.2 核心思路

**在 trace 中标记 `__VEC_SCOPE__` 边界，evaluator 在 scope 内识别 CCE 模式并应用融合延迟公式，而非逐条累加。**

`__VEC_SCOPE__` 内部的 CCE 调用模式是高度规律的：

```
__VEC_SCOPE__ {
    for (repeats) {
        [plt_*]             // 谓词，开销极小
        [vlds] * N_load     // load（N_load 取决于运算类型：一元=1，二元=2）
        [vcompute]          // 主计算指令（vadd/vmul/vexp/...）
        [vsts] * N_store    // store（通常 = 1）
    }
}
```

Evaluator 只需要做三件事：
1. 识别 scope 边界
2. 数 scope 内的 repeat 次数
3. 识别主计算指令，查融合延迟表

### 4.3 具体设计

#### 4.3.1 Trace 系统扩展：增加 scope 边界标记

在 `trace.hpp` 的 `CceCallRecord` 中增加 scope 边界标记：

```cpp
enum class CceCallKind : uint8_t {
    Normal,           // 普通 CCE 调用
    VecScopeBegin,    // __VEC_SCOPE__ 进入
    VecScopeEnd,      // __VEC_SCOPE__ 退出
};

struct CceCallRecord {
    std::string_view name;
    std::vector<TraceArg> args;
    CceCallKind kind = CceCallKind::Normal;
    bool vf_fused = true;  // 该 scope 是否启用 VF 融合（no_simd_vf_fusion 时为 false）
};
```

在 `cce_stub.hpp` 中用桩函数标记 scope 边界：

```cpp
// __VEC_SCOPE__ 展开后会调用某个 scope 进入/退出函数
// 在桩中记录边界
inline void __vec_scope_begin(bool vf_fused = true) {
    pto::mocker::RecordCceCall("__vec_scope_begin", {}, CceCallKind::VecScopeBegin, vf_fused);
}
inline void __vec_scope_end() {
    pto::mocker::RecordCceCall("__vec_scope_end", {}, CceCallKind::VecScopeEnd);
}
```

#### 4.3.2 CCE Stub：保持 trace 记录能力

A5 的 `cce_stub.hpp` 与 A2A3 类似，继续作为 trace 记录的核心。但由于 A5 的向量指令签名不同（regbase），stub 的函数签名需要适配 A5：

```cpp
// A5 的 vlds — regbase 签名
inline void vlds(auto &reg, auto ptr, auto offset, auto dist) {
    pto::mocker::RecordCceCall("vlds", {pto::mocker::MakeTraceArg("offset", offset)});
}
// A5 的 vlds with POST_UPDATE
inline void vlds(auto &reg, auto ptr, auto offset, auto dist, auto mode) {
    pto::mocker::RecordCceCall("vlds", {pto::mocker::MakeTraceArg("offset", offset)});
}

// A5 的 vadd — regbase 签名
inline void vadd(auto &dst, auto &src0, auto &src1, auto &preg, auto mode) {
    pto::mocker::RecordCceCall("vadd");
}

// A5 的 vsts — regbase 签名
inline void vsts(auto &reg, auto ptr, auto offset, auto dist, auto &preg) {
    pto::mocker::RecordCceCall("vsts");
}
inline void vsts(auto &reg, auto ptr, auto offset, auto dist, auto &preg, auto mode) {
    pto::mocker::RecordCceCall("vsts");
}

// 谓词
inline vector_bool plt_b32(uint32_t &val, auto mode) {
    pto::mocker::RecordCceCall("plt_b32");
    return {};
}

// 同步
inline void mem_bar(auto barrier_type) {
    pto::mocker::RecordCceCall("mem_bar");
}

// 矩阵搬运
inline void copy_matrix_cc_to_ub(auto dst, auto src, ...) {
    pto::mocker::RecordCceCall("copy_matrix_cc_to_ub", { ... });
}
```

其余编译占位符和 no-op 垫片与 A2A3 类似。

#### 4.3.3 VF Scope-Aware Evaluator：核心逻辑

在 `evaluator/` 中新增 VF scope 感知的评估逻辑：

```cpp
// VF 融合延迟表：按计算指令分类（而非按 PTO 指令分类）
struct VfFusionRule {
    std::string_view compute_op;    // 主计算指令名：vadd, vmul, vexp, ...
    uint64_t vf_startup;            // __VEC_SCOPE__ 启动开销
    uint64_t fused_pipeline;        // 融合后单次完整流水线延迟（含 load + compute + store）
    uint64_t unfused_sum;           // 非融合时单次迭代的串行延迟之和
};

// 示例规则表
inline constexpr VfFusionRule kA5VfFusionRules[] = {
    // compute_op    startup  fused  unfused
    { "vadd",        5,       6,     17 },
    { "vsub",        5,       6,     17 },
    { "vmul",        5,       6,     17 },
    { "vdiv",        5,       10,    26 },
    { "vexp",        5,       9,     23 },
    { "vln",         5,       9,     23 },
    { "vrelu",       5,       5,     13 },
    { "vabs",        5,       5,     13 },
    { "vcadd",       5,       8,     20 },   // 列归约用
    // ...
};
```

评估逻辑伪代码：

```cpp
CycleEstimate EvaluatePtoInstr(const PtoInstrRecord &instr, const ArchConfig &arch) {
    uint64_t total_cycles = 0;

    size_t i = 0;
    while (i < instr.cce_calls.size()) {
        const auto &call = instr.cce_calls[i];

        if (call.kind == CceCallKind::VecScopeBegin) {
            // 进入 VF scope —— 使用融合评估
            bool fused = call.vf_fused;
            auto [scope_cycles, next_i] = EvaluateVfScope(instr.cce_calls, i, arch, fused);
            total_cycles += scope_cycles;
            i = next_i;
        } else {
            // scope 外的普通 CCE 调用（如 mad, copy_*, set_flag 等）
            total_cycles += EvaluateCceCall(call, arch).cycles;
            ++i;
        }
    }

    return {total_cycles, true, {}};
}

std::pair<uint64_t, size_t> EvaluateVfScope(
    const std::vector<CceCallRecord> &calls, size_t begin,
    const ArchConfig &arch, bool fused)
{
    // 1. 找到 scope 结束位置
    size_t end = begin + 1;
    while (end < calls.size() && calls[end].kind != CceCallKind::VecScopeEnd) {
        ++end;
    }

    // 2. 统计 scope 内部的 CCE 调用
    size_t scope_inner_count = end - begin - 1;  // 排除 begin 和 end 标记

    // 3. 识别主计算指令（在 scope 内找第一个非 vlds/vsts/plt 的 CCE）
    std::string_view compute_op;
    for (size_t j = begin + 1; j < end; ++j) {
        const auto &c = calls[j];
        if (c.kind != CceCallKind::Normal) continue;
        if (c.name != "vlds" && c.name != "vsts" && c.name != "plt_b8" &&
            c.name != "plt_b16" && c.name != "plt_b32") {
            compute_op = c.name;
            break;
        }
    }

    // 4. 查融合规则
    const VfFusionRule *rule = FindVfFusionRule(compute_op);

    if (rule && fused) {
        // 融合模式：计算 repeat 次数，应用融合公式
        // repeat = scope 内主计算指令的出现次数
        size_t repeats = CountCceCalls(calls, begin, end, compute_op);
        uint64_t cycles = rule->vf_startup + repeats * rule->fused_pipeline;
        return {cycles, end + 1};
    } else {
        // 非融合模式（或未知计算指令）：回退到逐条累加
        uint64_t cycles = 0;
        for (size_t j = begin + 1; j < end; ++j) {
            cycles += EvaluateCceCall(calls[j], arch).cycles;
        }
        return {cycles, end + 1};
    }
}
```

#### 4.3.4 处理复杂 scope 内部结构

某些 PTO 指令的 `__VEC_SCOPE__` 内部不是简单的单层循环，例如列归约：

```cpp
// TColReduceOps.hpp — 列归约 scope 内有嵌套循环
__VEC_SCOPE__ {
    for (rptTimes) {                // 外层列 repeat
        vlds(dstReg, ...);         // 初始化 dst
        for (nLoop) {              // 内层行归约（每次处理 2 行）
            vlds(src0, ...);
            vlds(src1, ...);
            ReduceInstr(tmp, src0, src1, preg);   // 归约
            ReduceInstr(dst, dst, tmp, preg);     // 累加
        }
        vsts(dstReg, ...);
    }
}
```

对此类场景，scope-aware evaluator 的策略：
- 仍可统计 scope 内主计算指令的总次数（`ReduceInstr` 出现 `rptTimes * nLoop * 2` 次）
- 归约指令的融合延迟由其自身的 `VfFusionRule` 定义
- 非融合模式仍回退到逐条累加

### 4.4 融合延迟参数如何获取

融合规则表中的参数（`vf_startup`, `fused_pipeline`, `unfused_sum`）按**计算指令**分类，而非按 PTO 指令分类。这意味着：

- 只需对 ~15 种计算指令（vadd, vsub, vmul, vdiv, vmin, vmax, vexp, vln, vrelu, vabs, vnot, vsqrt, vrsqrt, vcadd, vcmax, ...）测定融合延迟
- 而非对 ~90 种 PTO 指令逐一测定
- 同一个计算指令被不同 PTO 指令复用时（如 TADD 和 TPartAdd 都用 vadd），共享同一条融合规则

实测方法：
1. 对每种计算指令，编写一个只包含该指令的 `__VEC_SCOPE__` micro-benchmark
2. 分别在融合（默认）和非融合（`#pragma no_simd_vf_fusion`）模式下测量 cycle
3. 反推出 `vf_startup` 和 `fused_pipeline` 参数

### 4.5 与 A2A3 共存的策略

```
用户代码
  |
  v
pto_instr.hpp  ->  MAP_INSTR_IMPL 宏
  |
  +-- __NPU_ARCH__ == 2201  ->  A2A3 路径
  |     |
  |     v
  |   a2a3/cce_stub.hpp (CCE trace)
  |     |
  |     v
  |   evaluator/trace_evaluator.hpp (CCE 逐条累加)
  |
  +-- __NPU_ARCH__ == 3101/3510  ->  A5 路径
        |
        v
      a5/cce_stub.hpp (CCE trace + scope 边界标记)
        |
        v
      evaluator/vf_scope_evaluator.hpp (scope 感知评估)
```

两条路径的差异仅在 evaluator 层：
- A2A3：`EvaluatePtoInstr` 逐条累加 CCE 延迟
- A5：`EvaluatePtoInstr` 识别 scope 边界，scope 内用融合公式，scope 外逐条累加

trace 收集、stub 框架、`MAP_INSTR_IMPL` 包装等基础设施完全复用。

---

## 5. 需要覆盖的 A5 CCE 指令清单

以下汇总 A5 PTO 指令所调用的 CCE 内建函数，用于构建 `cce_stub.hpp`：

**来自 `common.hpp` 的基础设施：**
- `RegTensor<T>` 类型（简化为 host 可编译版本）
- `MaskReg` = `vector_bool`
- `CreatePredicate<T>()` -> `plt_b8/b16/b32`
- `CeilDivision`

**来自 `TBinOp.hpp`（二元运算基础模板）：**
- `vlds(reg, ptr, offset, NORM)` / `vlds(..., POST_UPDATE)`
- `vsts(reg, ptr, offset, dist, preg)` / `vsts(..., POST_UPDATE)`
- 各运算指令（`vadd`, `vsub`, `vmul`, `vdiv`, `vmin`, `vmax`, `vand`, `vor` 等）
- `NORM`, `POST_UPDATE`, `MODE_ZEROING` 常量

**来自 `TMov.hpp`（数据搬运）：**
- `vlds`, `vsts`, `vsstb`（带 stride 的 store）
- `copy_matrix_cc_to_ub`, `copy_matrix_cc_to_cbuf`, `copy_matrix_cc_to_gm`
- `copy_cbuf_to_bt`, `copy_cbuf_to_fbuf`
- `set_loop3_para`, `set_channel_para`
- `set_fpc`, `set_quant_pre`

**来自 `TMatmul.hpp`（矩阵乘法）：**
- `mad` / `mad_mx`（矩阵乘累加）
- `load_cbuf_to_ca`, `load_cbuf_to_cb`（矩阵加载）

**来自 `TUnaryOp.hpp`（一元运算）：**
- `vabs`, `vexp`, `vln`, `vnot`, `vrelu`, `vrsqrt`, `vsqrt`（regbase 签名）
- `vbr`（broadcast）

**来自 `TCmp.hpp`（比较运算）：**
- 比较类 CCE 指令

**来自 `TColReduceOps.hpp`（列归约）：**
- `vcadd`, `vcmax`, `vcmin` 等归约专用指令

**来自 `TRowReduce.hpp`（行归约）：**
- `vcadd`, `vcmax`, `vcmin` 等归约专用指令

---

## 6. A5 Latency Rules 设计

### 6.1 CCE 级延迟规则（`arch_config.hpp` 中 `kA5Rules`）

用于 scope 外的逐条评估，以及非融合模式的回退：

```cpp
inline constexpr CceLatencyRule kA5Rules[] = {
    // === Scalar ===
    FixedRule("set_flag",         CoreType::Scalar, 1),
    FixedRule("wait_flag",        CoreType::Scalar, 1),
    FixedRule("wait_flag_dev",    CoreType::Scalar, 1),
    FixedRule("set_vector_mask",  CoreType::Scalar, 1),
    FixedRule("set_cmpmask",      CoreType::Scalar, 1),
    FixedRule("set_mask_count",   CoreType::Scalar, 1),
    FixedRule("set_mask_norm",    CoreType::Scalar, 1),
    FixedRule("set_mov_pad_val",  CoreType::Scalar, 1),
    FixedRule("set_padding",      CoreType::Scalar, 1),
    FixedRule("pipe_barrier",     CoreType::Scalar, 1),
    FixedRule("mem_bar",          CoreType::Scalar, 1),
    FixedRule("set_loop3_para",   CoreType::Scalar, 1),
    FixedRule("set_channel_para", CoreType::Scalar, 1),
    FixedRule("plt_b8",           CoreType::Scalar, 1),
    FixedRule("plt_b16",          CoreType::Scalar, 1),
    FixedRule("plt_b32",          CoreType::Scalar, 1),

    // === Vector (regbase 签名，scope 内逐条计价用) ===
    FixedRule("vlds",   CoreType::Vector, 4),
    FixedRule("vsts",   CoreType::Vector, 4),
    FixedRule("vsstb",  CoreType::Vector, 4),
    FixedRule("vbr",    CoreType::Vector, 1),
    FixedRule("vadd",   CoreType::Vector, 4),
    FixedRule("vsub",   CoreType::Vector, 4),
    FixedRule("vmul",   CoreType::Vector, 4),
    FixedRule("vdiv",   CoreType::Vector, 6),
    FixedRule("vmin",   CoreType::Vector, 4),
    FixedRule("vmax",   CoreType::Vector, 4),
    FixedRule("vand",   CoreType::Vector, 4),
    FixedRule("vor",    CoreType::Vector, 4),
    FixedRule("vadds",  CoreType::Vector, 4),
    FixedRule("vmuls",  CoreType::Vector, 4),
    FixedRule("vmins",  CoreType::Vector, 4),
    FixedRule("vmaxs",  CoreType::Vector, 4),
    FixedRule("vabs",   CoreType::Vector, 4),
    FixedRule("vaxpy",  CoreType::Vector, 4),
    FixedRule("vexp",   CoreType::Vector, 6),
    FixedRule("vln",    CoreType::Vector, 6),
    FixedRule("vrelu",  CoreType::Vector, 4),
    FixedRule("vnot",   CoreType::Vector, 4),
    FixedRule("vrsqrt", CoreType::Vector, 6),
    FixedRule("vsqrt",  CoreType::Vector, 6),
    FixedRule("vcadd",  CoreType::Vector, 6),
    FixedRule("vcmax",  CoreType::Vector, 6),
    FixedRule("vcmin",  CoreType::Vector, 6),

    // === Mte2 (Memory Load) ===
    BurstLenRule("copy_gm_to_ubuf_align_b8",  CoreType::Mte2, 8, 2, 1),
    BurstLenRule("copy_gm_to_ubuf_align_b16", CoreType::Mte2, 8, 2, 1),
    BurstLenRule("copy_gm_to_ubuf_align_b32", CoreType::Mte2, 8, 2, 1),
    BurstLenRule("copy_gm_to_cbuf",           CoreType::Mte2, 8, 2, 1),
    BurstLenRule("copy_ubuf_to_ubuf",         CoreType::Mte2, 4, 1, 1),

    // === Mte3 (Memory Store) ===
    BurstLenRule("copy_ubuf_to_gm_align_b8",  CoreType::Mte3, 8, 2, 1),
    BurstLenRule("copy_ubuf_to_gm_align_b16", CoreType::Mte3, 8, 2, 1),
    BurstLenRule("copy_ubuf_to_gm_align_b32", CoreType::Mte3, 8, 2, 1),
    BurstLenRule("copy_cbuf_to_gm",           CoreType::Mte3, 8, 2, 1),

    // === Fix (Matrix 搬运/转换) ===
    FixedRule("copy_matrix_cc_to_gm",   CoreType::Fix, 12),
    FixedRule("copy_matrix_cc_to_ub",   CoreType::Fix, 12),
    FixedRule("copy_matrix_cc_to_cbuf", CoreType::Fix, 12),
    FixedRule("copy_cbuf_to_bt",        CoreType::Fix, 8),
    FixedRule("copy_cbuf_to_fbuf",      CoreType::Fix, 8),

    // === Matrix ===
    MadRule("mad", CoreType::Matrix, 16, 16),
    MadRule("mad_mx", CoreType::Matrix, 16, 16),
};
```

> **注意**：以上延迟参数为初始估计值，最终值需通过硬件实测校准。

### 6.2 VF 融合延迟规则（新增）

```cpp
// 按 CCE 计算指令分类，而非按 PTO 指令分类
// 参数通过 micro-benchmark 实测获取
inline constexpr VfFusionRule kA5VfFusionRules[] = {
    // compute_op    vf_startup  fused_pipeline  unfused_sum
    { "vadd",        5,          6,              17 },
    { "vsub",        5,          6,              17 },
    { "vmul",        5,          6,              17 },
    { "vdiv",        5,          10,             26 },
    { "vmin",        5,          6,              17 },
    { "vmax",        5,          6,              17 },
    { "vand",        5,          6,              17 },
    { "vor",         5,          6,              17 },
    { "vadds",       5,          5,              14 },
    { "vmuls",       5,          5,              14 },
    { "vmins",       5,          5,              14 },
    { "vmaxs",       5,          5,              14 },
    { "vabs",        5,          5,              13 },
    { "vexp",        5,          9,              23 },
    { "vln",         5,          9,              23 },
    { "vrelu",       5,          5,              13 },
    { "vnot",        5,          5,              13 },
    { "vrsqrt",      5,          9,              23 },
    { "vsqrt",       5,          9,              23 },
    { "vcadd",       5,          8,              20 },
    { "vcmax",       5,          8,              20 },
    { "vcmin",       5,          8,              20 },
};
```

---

## 7. 架构设计

### 7.1 目录结构

```
include/pto/costmodel/
├── runtime_stub.hpp              # 运行时入口（复用）
├── pto_instr.hpp                 # PTO 指令包装（复用）
├── trace.hpp                     # Trace 收集系统（扩展：增加 scope 标记）
├── arch_config.hpp               # 架构配置（扩展：CCE 规则 + VF 融合规则）
├── common/                       # 通用兼容层（复用）
│   ├── qualifiers.hpp
│   ├── aclrt_stub.hpp
│   ├── runtime_util.hpp
│   └── arch_select.hpp
├── a2a3/                         # A2A3 CCE Mock（已有，不改动）
│   ├── README.md
│   └── cce_stub.hpp
├── a5/                           # A5 CCE Mock（新建）
│   ├── README.md
│   └── cce_stub.hpp              # A5 CCE 桩（含 scope 边界标记）
└── evaluator/                    # 评估器（扩展）
    ├── arg_reader.hpp
    ├── cce_evaluator.hpp
    ├── trace_evaluator.hpp       # 扩展：增加 VF scope 感知评估
    ├── vf_fusion_rules.hpp       # 新增：VF 融合规则表
    └── types.hpp
```

### 7.2 数据流

```
A5 用户代码
  |
  v
MAP_INSTR_IMPL -> PtoInstrScope("TADD")
  |
  v
TADD_IMPL -> TBinOps_2D_NoPostUpdate
  |
  v
__VEC_SCOPE__ {
  __vec_scope_begin(true)       -> trace: VecScopeBegin(vf_fused=true)
  for 64 rows x 2 repeats:
    plt_b32(...)                 -> trace: "plt_b32"
    vlds(vreg0, ...)            -> trace: "vlds"
    vlds(vreg1, ...)            -> trace: "vlds"
    vadd(vreg2, ...)            -> trace: "vadd"
    vsts(vreg2, ...)            -> trace: "vsts"
  __vec_scope_end()             -> trace: VecScopeEnd
}
  |
  v
CaptureCycleIntoTile(dst)
  |
  v
evaluator: 识别 VecScopeBegin -> 查 "vadd" 融合规则
  -> vf_startup(5) + 128 * fused_pipeline(6) = 773 cycles
```

---

## 8. 实施计划

### Phase 1：搭建 A5 CCE Stub

**目标**：创建 `a5/cce_stub.hpp`，使 A5 PTO 指令能在 host 侧编译通过并产生 trace。

**具体工作**：
1. 创建 `include/pto/costmodel/a5/` 目录
2. 编写 `cce_stub.hpp`：
   - 编译占位符（`QuantMode_t` 扩展版、`AccToVecMode`、`ReluPreMode`、`DistVST` 等）
   - 简化版 `RegTensor<T>`、`MaskReg`、`vector_bool` 等
   - A5 regbase 签名的 CCE trace 桩（`vlds`, `vsts`, `vadd` 等）
   - `__vec_scope_begin` / `__vec_scope_end` 边界标记桩
   - no-op 垫片（配置类指令）
3. 确认 `arch_select.hpp` 正确引入 A5 stub
4. 编写 `a5/README.md`

**验证标准**：TADD、TMov、TMatmul 在 `__COSTMODEL` 模式下编译通过并产生 trace。

### Phase 2：实现 VF Scope-Aware Evaluator

**目标**：增强 trace evaluator，支持 VF scope 感知评估。

**具体工作**：
1. 扩展 `trace.hpp`：在 `CceCallRecord` 中增加 scope 边界标记
2. 新增 `evaluator/vf_fusion_rules.hpp`：定义 `VfFusionRule` 表
3. 修改 `evaluator/trace_evaluator.hpp`：
   - A2A3 路径：保持原有逐条累加逻辑不变
   - A5 路径：新增 scope 感知评估分支
4. 完善 `kA5Rules`（CCE 级延迟规则）和 `kA5VfFusionRules`（VF 融合规则）

**验证标准**：
- A2A3 的评估结果不变
- A5 的评估结果正确反映 VF 融合效果
- 非融合 scope（`#pragma no_simd_vf_fusion`）回退到逐条累加

### Phase 3：参数校准与端到端验证

**目标**：校准延迟参数，确保估算结果与实测对齐。

**具体工作**：
1. 编写 micro-benchmark，测定各计算指令的融合/非融合延迟
2. 校准 `kA5VfFusionRules` 中的参数
3. 为核心 A5 PTO 指令编写 costmodel 测试用例
4. 对比实测数据验证精度

### Phase 4：扩展覆盖

**目标**：覆盖所有 A5 PTO 指令。

**具体工作**：
1. 补充遗漏的 CCE stub（TTrans, TCmp, TSort, TMrgSort, TGather, TScatter 等）
2. 补充对应的 VF 融合规则
3. 处理多 scope 指令（TPartBinOps 等）

---

## 9. 风险与注意事项

1. **`__VEC_SCOPE__` 展开方式**：`__VEC_SCOPE__` 是编译器宏，需要确认其在 costmodel host 编译环境下的展开方式，以及如何在展开后的代码中插入 scope 边界标记。如果 `__VEC_SCOPE__` 展开为 RAII 构造/析构，则可在构造/析构函数中插入标记。

2. **VF 融合规则参数校准**：初期可使用估计值，后续需通过 A5 硬件实测数据校准。校准工作量可控（~15 种计算指令）。

3. **A5 头文件编译兼容性**：A5 PTO 头文件比 A2A3 更复杂，host 兼容层可能需要额外处理。

4. **多 scope 场景**：某些 PTO 指令（如 TPartBinOps）包含多个 `__VEC_SCOPE__`，evaluator 需要正确处理 scope 嵌套和边界。

5. **与 A2A3 的 evaluator 兼容**：A2A3 的评估逻辑不能受影响。通过 `ArchConfig` 区分平台，A2A3 继续使用逐条累加，A5 使用 scope 感知评估。

## 10. 参考资料

- A2A3 Costmodel 实现：`include/pto/costmodel/a2a3/`
- A2A3 Costmodel 文档：`include/pto/costmodel/a2a3/README.md`
- A5 PTO 指令实现：`include/pto/npu/a5/`
- A5 公共类型定义：`include/pto/npu/a5/common.hpp`
- A5 二元运算模板：`include/pto/npu/a5/TBinOp.hpp`
- A5 一元运算模板：`include/pto/npu/a5/TUnaryOp.hpp`
- A5 列归约模板：`include/pto/npu/a5/TColReduceOps.hpp`
- A5 行归约实现：`include/pto/npu/a5/TRowReduce.hpp`
- A5 矩阵乘法：`include/pto/npu/a5/TMatmul.hpp`
- A5 数据搬运：`include/pto/npu/a5/TMov.hpp`
- 抽象机器模型：`docs/machine/abstract-machine.md`
- VFImplKind 枚举：`include/pto/common/type.hpp:167`
- 轻量级接口提案：`include/pto/costmodel/LIGHTWEIGHT_COSTMODEL_INTERFACE.md`
