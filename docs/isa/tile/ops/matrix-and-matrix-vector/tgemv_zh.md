# TGEMV


## 指令示意图


![TGEMV tile operation](../../../../figures/isa/TGEMV.svg)

## 简介


`TGEMV` 是 cube 路径上的矩阵-向量乘指令。它不是 vector 指令，而是矩阵乘合同在 `m = 1` 条件下的专门形式：左输入仍走 `Left`，右输入仍走 `Right`，结果仍写入 `Acc`。

把 GEMV 单独列成一条指令，是为了让接口、用法和调度语义更直接，不必让读者总是从“一般 matmul 的退化情况”去倒推。

## 数学语义


设：

- `K = bMatrix.GetValidRow()`
- `N = bMatrix.GetValidCol()`

对 `0 <= j < N`：

$$ \mathrm{C}_{0,j} = \sum_{k=0}^{K-1} \mathrm{A}_{0,k} \cdot \mathrm{B}_{k,j} $$

这里输出只有一行，因此 `TGEMV` 可以理解为矩阵乘一条向量，但它仍然遵守 cube 路径的角色和布局约束。

## 机制


`TGEMV` 仍然使用：

- `Left` 作为左操作数，对应 L0A 路径；
- `Right` 作为右操作数，对应 L0B 路径；
- `Acc` 作为输出累加器。

和 `TMATMUL` 的主要区别，不在“是不是 cube 指令”，而在运行时合同里固定了 `m = 1`。因此它的 costmodel、角色限制和 target 边界都更接近 matmul，而不是向量算术。

## 汇编语法


PTO-AS 形式：参见 [汇编写法与操作数](../../../syntax-and-operands/assembly-model_zh.md)。

同步形式：

```text
%acc = tgemv %a, %b : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 1（SSA）


```text
%c = pto.tgemv %a, %b : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）


```text
pto.tgemv ins(%a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c : !pto.tile_buf<...>)
```

## C++ 内建接口


声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TGEMV(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, WaitEvents &... events);
```

## 输入与输出


- `aMatrix`：左操作数 tile，必须是 `Left`。
- `bMatrix`：右操作数 tile，必须是 `Right`。
- `cMatrix`：结果累加器 tile，必须是 `Acc`。

输出合同是：生成一行结果 `C[0, j]`。这条指令不会把普通 vector buffer 直接提升成 cube 合同。

## 约束


!!! warning "约束"
    ### 通用约束

    - 静态 shape 必须满足：
      - `TileLeft::Rows == TileRes::Rows`
      - `TileLeft::Cols == TileRight::Rows`
      - `TileRight::Cols == TileRes::Cols`
    - tile 角色必须满足：
      - `TileLeft::Loc == Left`
      - `TileRight::Loc == Right`
      - `TileRes::Loc == Acc`
    - 运行时要求：
      - `m = 1`
      - `k`、`n` 位于 `[1, 4095]`

    ### A2A3 约束

    `A2A3` 指 Ascend 910B 与 Ascend 910C。当前仓内实现公开支持的 `(CType, AType, BType)` 组合包括：

    - `(int32_t, int8_t, int8_t)`
    - `(float, half, half)`
    - `(float, float, float)`
    - `(float, bfloat16_t, bfloat16_t)`

    ### A5 约束

    `A5` 指 Ascend 950 PR 与 Ascend 950 DT。当前实现要求：

    - 累加器类型必须是 `int32_t` 或 `float`；
    - 若累加器为 `int32_t`，左右输入都必须是 `int8_t`；
    - 若累加器为 `float`，当前实现支持 `half`、`bfloat16_t`、`float` 和部分 fp8 输入对；
    - A5 的 `Right` 角色有独立布局 / fractal 约束，不能拿 A2A3 的右操作数布局直接套用。

## 不允许的情形


!!! danger "不允许的情形"
    - `m != 1`；
    - 角色不是 `Left` / `Right` / `Acc`；
    - 形状不满足 GEMV 兼容关系；
    - 在不支持的 target 上使用不支持的 dtype 组合。

## 性能与吞吐


仓内 A2A3 costmodel 对 `TGEMV` 与 `TMATMUL` 共用 `mad/mmad` 模型，只是 GEMV 固定 `m = 1`，因此公式可直接写成：

```text
cycles = 14 + ceil(N/16) * ceil(K / baskK) * repeat_cost
```

其中：

- `baskK = 32 / sizeof(left_element_type)`；
- int8、fp16 bucket 的 `repeat_cost = 1`；
- fp32 bucket 的 `repeat_cost = 2`。

当前仓库没有公开单列的 A5 latency / throughput 表。

## 示例


### 自动（Auto）


```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using A = TileLeft<half, 1, 16>;
  using B = TileRight<half, 16, 16>;
  using C = TileAcc<float, 1, 16>;
  A a;
  B b;
  C c;
  TGEMV(c, a, b);
}
```

### 手动（Manual）


```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using A = TileLeft<half, 1, 16>;
  using B = TileRight<half, 16, 16>;
  using C = TileAcc<float, 1, 16>;
  A a;
  B b;
  C c;
  TASSIGN(a, 0x1000);
  TASSIGN(b, 0x2000);
  TASSIGN(c, 0x3000);
  TGEMV(c, a, b);
}
```

## 相关页面


- [TGEMV_ACC](./tgemv-acc_zh.md)
- [TGEMV_BIAS](./tgemv-bias_zh.md)
- [TGEMV_MX](./tgemv-mx_zh.md)
- [矩阵与矩阵-向量指令集](../../matrix-and-matrix-vector_zh.md)

# pto.tgemv
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Summary
本节给出该指令/主题的核心语义与使用定位，和英文章节保持一致。

## Mechanism
本节说明执行机制与关键语义规则，细节与边界条件以英文版为准。

### 1. TGEMV (Tile-based GEMV)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### 2. TGEMV_ACC (Tile-based GEMV with Accumulation)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### 3. TGEMV_BIAS (Tile-based GEMV with Bias)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Syntax
本节列出语法形态（SSA / DPS / Assembly），用于与英文页逐项对照。

### AS Level 1 (SSA)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### AS Level 2 (DPS)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## C++ Intrinsic
本节给出 C++ 内建接口入口与参数语义说明。

## Inputs
本节定义输入操作数角色、数据来源与有效区域要求。

## Expected Outputs
本节定义输出结果及其在有效区域内的语义保证。

## Side Effects
本节说明除结果写回外是否存在额外可观察副作用。

## Constraints
本节列出类型、布局、shape、valid-region 与 profile 相关约束。

## Exceptions
本节描述非法输入、不支持组合与验证失败行为。

## Target-Profile Restrictions
本节给出 A2/A3、A5 及 CPU-SIM 的差异化限制与行为说明。

## Examples
本节提供 Auto/Manual 及 AS 形式示例，便于中英文对照复现。

### Auto
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

#### 1. TGEMV
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

#### 2. TGEMV_ACC
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

#### 3. TGEMV_BIAS
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Manual
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

#### 1. TGEMV
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

#### 2. TGEMV_ACC
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

#### 3. TGEMV_BIAS
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Auto Mode
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

# Auto mode: compiler/runtime-managed placement and scheduling.
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Manual Mode
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

# Manual mode: bind resources explicitly before issuing the instruction.
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

# Optional for tile operands:
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

# pto.tassign %arg0, @tile(0x1000)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

# pto.tassign %arg1, @tile(0x2000)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### PTO Assembly Form
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

# AS Level 2 (DPS)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Related Ops / Instruction Set Links
本节给出上下游指令与相关章节链接。
