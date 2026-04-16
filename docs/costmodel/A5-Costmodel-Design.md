# A5 平台 Costmodel 设计方案

## 1. 方案概述

采用纯分析模型，不运行 PTO 指令代码。通过在 `pto_instr_impl.hpp` 中用宏控制 include 路径，使 `XXX_IMPL` 指向 costmodel 侧的分析实现，替换原始 npu 实现。

与 A2A3 costmodel 的区别：A2A3 的 `XXX_IMPL` 调用原始代码 + CCE stub 录 trace，再评估 trace 得到 cycles。A5 的 `XXX_IMPL` 直接从 tile 参数推导 VF 结构，计算 cycles。

**调用流程不变**：`TADD(dst, src0, src1)` → `MAP_INSTR_IMPL(TADD, dst, src0, src1)` → `TADD_IMPL(dst, src0, src1)` → `CaptureCycleIntoTile(dst)`。唯一改变的是 `TADD_IMPL` 的实现。

---

## 2. 以 TAdd 为例说明

### 2.1 现有代码（不改动）

**`pto_instr.hpp`**（TADD 入口函数，不改动）：

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TADD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TADD, dst, src0, src1);   // 不改动
    return {};
}
```

**`npu/a5/TAdd.hpp`**（原始实现，不改动）：

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TADD_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    TAddCheck(dst, src0, src1);
    TAdd<...>(dst.data(), src0.data(), src1.data(), ...);
    // → 最终调用 vlds, vadd, vsts 等 CCE intrinsic
}
```

### 2.2 需要修改：`pto_instr_impl.hpp`

A5 已有 `__COSTMODEL` 分支（line 117-148），当前 include 原始 npu 头文件。改为 include costmodel 侧头文件：

```cpp
#ifdef PTO_NPU_ARCH_A5
#ifdef __COSTMODEL
// --- 修改前：include 原始 npu 实现 ---
// #include "pto/npu/a5/TAdd.hpp"
// #include "pto/npu/a5/TSub.hpp"
// ...

// --- 修改后：include costmodel 分析实现 ---
#include "pto/costmodel/a5/TAdd.hpp"
#include "pto/costmodel/a5/TSub.hpp"
#include "pto/costmodel/a5/TMul.hpp"
// ... 每个 PTO 指令一个 include
#else
// 正常模式：include 原始 npu 实现（不改动）
#include "pto/npu/a5/TAdd.hpp"
// ...
#endif
#endif
```

**这是唯一的已有文件修改。** 这一行改动使得 `TADD_IMPL` 指向 costmodel 侧的分析实现。

### 2.3 新增：`costmodel/a5/TAdd.hpp`

提供分析版 `TADD_IMPL`，签名与原始版本完全一致：

```cpp
// costmodel/a5/TAdd.hpp
#ifndef COSTMODEL_A5_TADD_HPP
#define COSTMODEL_A5_TADD_HPP

#include <pto/costmodel/a5/analytical_eval.hpp>

namespace pto {

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TADD_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::mocker::a5::EvaluateBinOp("TADD", dst, src0, src1);
}

} // namespace pto
#endif
```

### 2.4 新增：`costmodel/a5/analytical_eval.hpp`

所有 BinOp 类指令（TADD、TSUB、TMUL 等）共享同一个求值逻辑：

```cpp
// costmodel/a5/analytical_eval.hpp
#ifndef COSTMODEL_A5_ANALYTICAL_EVAL_HPP
#define COSTMODEL_A5_ANALYTICAL_EVAL_HPP

#include <pto/costmodel/a5/instr_traits.hpp>
#include <pto/costmodel/a5/vf_descriptor.hpp>
#include <pto/costmodel/trace.hpp>

namespace pto::mocker::a5 {

template <typename Dst, typename... Srcs>
void Evaluate(const char *name, Dst &dst, const Srcs &... srcs)
{
    // 1. 查指令描述
    auto &info = kInstrTraits.at(name);

    // 2. 推导 VF 结构
    auto vf = BuildVfDescriptor(info, dst, srcs...);
    // BuildVfDescriptor 内部：
    //   - IsContiguous(dst, srcs...) → 1D / 2D
    //   - CalcIters(rows, cols, epr, is_1d) → outer, inner, total
    //   - category->build_inner_body(cce_op) → [VLDS, VLDS, VADD, VSTS]

    // 3. VF 建模公式计算 cycles
    float cycles = VfModeling(vf);

    // 4. 记录到 trace，使 CaptureCycleIntoTile 能取到结果
    RecordCceCall("VF_COST", {MakeTraceArg("cycles", static_cast<uint64_t>(cycles))});
}

// BinOp 类指令的便捷入口
template <typename Dst, typename Src0, typename Src1>
void EvaluateBinOp(const char *name, Dst &dst, Src0 &src0, Src1 &src1) {
    Evaluate(name, dst, src0, src1);
}

// UnaryOp 类指令的便捷入口
template <typename Dst, typename Src>
void EvaluateUnary(const char *name, Dst &dst, Src &src) {
    Evaluate(name, dst, src);
}

} // namespace pto::mocker::a5
#endif
```

### 2.5 其他 BinOp 指令的 costmodel 文件

结构和 TAdd 完全一样，只换名字和类别入口：

```cpp
// costmodel/a5/TSub.hpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TSUB_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::mocker::a5::EvaluateBinOp("TSUB", dst, src0, src1);
}

// costmodel/a5/TMul.hpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TMUL_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    ::pto::mocker::a5::EvaluateBinOp("TMUL", dst, src0, src1);
}
```

区别仅在传入 `Evaluate` 的指令名不同，由 `kInstrTraits` 查表得到不同的 category 和 cce_op。

---

## 3. VF 结构获取

VF 结构是指 `__VEC_SCOPE__` 内 for 循环的完整描述信息，供 VF 建模公式使用。需要推导出以下四个要素：

```
VfDescriptor {
    nesting,       // 循环嵌套层数（1 或 2）
    iters,         // 各层循环次数 {outer, inner, total}
    inner_body,    // 单次迭代的 CCE 操作序列
    post_update,   // 是否使用 POST_UPDATE 模式
}
```

下面逐步说明每个要素的推导方法。以 `TADD(dst, src0, src1)` 为例，假设 `Tile<Vec, half, 4, 256>`，`validRow=4, validCol=256`。

### 3.1 指令类别 + CCE 操作码

**输入**：指令名（如 `"TADD"`），查 `kInstrTraits` 表。

**推导**：每条 PTO 指令在 `instr_traits.hpp` 中声明其所属类别和 CCE 计算操作码：

```cpp
inline const std::unordered_map<std::string_view, InstrInfo> kInstrTraits = {
    {"TADD",     {&BinOpCategory::inst(),   OpCode::VADD}},
    {"TSUB",     {&BinOpCategory::inst(),   OpCode::VSUB}},
    {"TMUL",     {&BinOpCategory::inst(),   OpCode::VMUL}},
    {"TABS",     {&UnaryOpCategory::inst(),  OpCode::VABS}},
    {"TEXP",     {&UnaryOpCategory::inst(),  OpCode::VEXP}},
    {"TROWSUM",  {&RowReduceCategory::inst(), OpCode::VCADD}},
    // ... 新增指令加一行
};
```

**示例**：`"TADD"` → `BinOpCategory` + `VADD`。

### 3.2 嵌套层级（1D / 2D）

**输入**：dst、src 的 tile shape（rows, cols, validCol）。

**推导**：与 `npu/a5/` 中的 `isContiguous` 判断逻辑一致（`TBinOp.hpp:179-182`）：

```cpp
bool IsContiguous(const auto &dst, const auto &... srcs) {
    // 所有 tile 的 validCol == cols，或所有 tile 都是单行
    bool all_cols_match = (dst.valid_col == dst.cols) && (... && (srcs.valid_col == srcs.cols));
    bool all_single_row = (dst.rows == 1) && (... && (srcs.rows == 1));
    return all_cols_match || all_single_row;
}
```

`isContiguous = true` → **1D**（单层循环），`false` → **2D**（嵌套循环）。

**依据**：原始代码根据 `isContiguous` 选择调用 `TBinOp1DSwitch` 或 `TBinOp2DSwitch`，这决定了 `__VEC_SCOPE__` 内是 1D 还是 2D 循环结构。

**示例**：`validCol=256 == cols=256`，三个 tile 都满足 → `isContiguous = true` → **1D**。

### 3.3 迭代次数

**输入**：tile 的 rows、cols + 数据类型的 ElementsPerRepeat（EPR）。

**推导**：所有标准 1D/2D 模式共用相同的迭代公式：

```cpp
IterationCounts CalcIters(unsigned rows, unsigned cols, unsigned epr, bool is_1d) {
    if (is_1d) {
        uint32_t total = CeilDivision(rows * cols, epr);
        return {.outer = 1, .inner = total, .total = total};
    }
    uint32_t outer = rows;
    uint32_t inner = CeilDivision(cols, epr);
    return {.outer = outer, .inner = inner, .total = outer * inner};
}

unsigned GetEPR(const auto &tile) {
    return CCE_VL / sizeof(tile.dtype);  // half: 256/2=128
}
```

**依据**——`npu/a5/` 下所有 2D 模式的循环结构完全一致：

| 指令 | 源码位置 | 循环结构 |
|------|---------|---------|
| TBinOp_2D | TBinOp.hpp:79 | `for(i < validRows) for(j < CeilDivision(validCols, EPR))` |
| TUnaryOp_2D | TUnaryOp.hpp:75 | `for(i < validRow) for(j < CeilDivision(validCol, EPR))` |
| TRowReduce | TRowReduce.hpp:265 | `for(i < rows) for(j < CeilDivision(cols, EPR))` |
| TGather | TGather.hpp:42 | `for(i < validRow) for(j < CeilDivision(validCol, batchSize))` |
| TScatter | TScatter.hpp:34 | `for(i < validRow) for(j < CeilDivision(validCol, batchSize))` |

通用规律：`outer = rows`，`inner = ceil(cols / chunk_size)`。

**示例**：`rows=4, cols=256, EPR=128, is_1d=true` → `total = ceil(4×256, 128) = 8`，即单层循环 8 次迭代。

另一个示例：若 `validCol=96, cols=128` → `isContiguous=false` → 2D → `outer=4, inner=ceil(96, 128)=1, total=4`。

### 3.4 内部体 CCE 操作序列

**输入**：类别模板 + 具体的 CCE 操作码。

**推导**：每种类别定义一个内部体模板，其中 `COMPUTE` 位由具体的 `cce_op` 填入：

```cpp
struct BinOpCategory : CategoryBase {
    std::vector<OpCode> build_inner_body(OpCode compute) const override {
        return {OpCode::VLDS, OpCode::VLDS, compute, OpCode::VSTS};
    }
};

struct UnaryOpCategory : CategoryBase {
    std::vector<OpCode> build_inner_body(OpCode compute) const override {
        return {OpCode::VLDS, compute, OpCode::VSTS};
    }
};

// RowReduce、ColReduce、Gather、Scatter ... 各自定义
```

**依据**：所有 BinOp 的 `__VEC_SCOPE__` 内部体都是 `vlds(src0), vlds(src1), Op::BinInstr, vsts(dst)`，不管走 1D 还是 2D、PostUpdate 还是 NoPostUpdate，这个序列不变。详见 `TBinOp.hpp` 中 `TBinOps_1D_*` 和 `TBinOps_2D_*` 的 for 循环体。

**示例**：`BinOpCategory` + `VADD` → `[VLDS, VLDS, VADD, VSTS]`。

### 3.5 完整推导示例

`TADD(dst, src0, src1)`，`Tile<Vec, half, 4, 256>`，`validRow=4, validCol=256`：

```
输入：
  指令名 = "TADD"
  dst:  rows=4, cols=256, validCol=256
  src0: rows=4, cols=256, validCol=256
  src1: rows=4, cols=256, validCol=256
  dtype = half

Step 1: 查表 kInstrTraits["TADD"] → BinOpCategory, VADD

Step 2: 推导嵌套
  isContiguous: validCol(256)==cols(256)，三个 tile 均满足 → true
  nesting = 1（1D）

Step 3: 计算迭代次数
  EPR = 256 / 2 = 128
  is_1d = true → total = ceil(4×256, 128) = 8
  iters = {outer=1, inner=8, total=8}

Step 4: 构造内部体
  BinOpCategory::build_inner_body(VADD) → [VLDS, VLDS, VADD, VSTS]

Step 5: 组装 VfDescriptor
  {
    nesting = 1,
    iters   = {1, 8, 8},
    inner_body = [VLDS, VLDS, VADD, VSTS],
    post_update = false
  }
  → 交由 VF 建模公式计算 cycles
```

---

## 4. 端到端流程

```
TADD(dst, src0, src1)                    // pto_instr.hpp，不改动
  → MAP_INSTR_IMPL(TADD, dst, src0, src1)  // 不改动
    → PtoInstrScope("TADD")                // 开 trace scope
    → TADD_IMPL(dst, src0, src1)            // 指向 costmodel/a5/TAdd.hpp
      → EvaluateBinOp("TADD", dst, src0, src1)
        → kInstrTraits["TADD"] → BinOpCategory + VADD
        → IsContiguous(dst, src0, src1) → 1D / 2D
        → CalcIters(rows, cols, epr, is_1d) → {outer, inner, total}
        → BinOpCategory::build_inner_body(VADD) → [VLDS, VLDS, VADD, VSTS]
        → VfModeling(VfDescriptor) → cycles
        → RecordCceCall("VF_COST", cycles)   // 记录到 trace
    → CaptureCycleIntoTile(dst)              // 评估 trace，取到 cycles
```

---

## 5. 目录结构

```
include/pto/costmodel/a5/
├── analytical_eval.hpp    # Evaluate() 求值逻辑（共享）
├── vf_descriptor.hpp      # VfDescriptor 构建 + 嵌套推导 + 迭代计算
├── instr_traits.hpp       # 指令描述表
├── category.hpp           # 类别模板
├── TAdd.hpp               # TADD_IMPL（分析版）
├── TSub.hpp               # TSUB_IMPL
├── TMul.hpp               # TMUL_IMPL
├── TAbs.hpp               # TABS_IMPL
├── TExp.hpp               # TEXP_IMPL
└── ...                    # 每个 PTO 指令一个文件

修改已有文件：
├── pto_instr_impl.hpp     # A5 __COSTMODEL 分支：include 路径从 npu/a5/ 改为 costmodel/a5/
```

---

## 6. 维护说明

| 操作 | 改动 |
|------|------|
| 新增 PTO 指令 | 1. 加 `costmodel/a5/TXxx.hpp`（模板文件）<br>2. `instr_traits.hpp` 加一行<br>3. `pto_instr_impl.hpp` 加一行 include |
| 新增指令类别 | `category.hpp` 加一种模板（罕见） |
| isContiguous 逻辑变化 | `vf_descriptor.hpp` 同步修改 |

---

## 7. 参考资料

- Costmodel 入口：`include/pto/costmodel/pto_instr.hpp`
- 架构实现选择：`include/pto/common/pto_instr_impl.hpp`
- A2A3 Costmodel：`include/pto/costmodel/a2a3/`
- A5 PTO 指令实现：`include/pto/npu/a5/`
