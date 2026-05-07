# TGEMV_MX


## 指令示意图


![TGEMV_MX tile operation](../../../../figures/isa/TGEMV_MX.svg)

## 简介


`TGEMV_MX` 是 MX 路径下的矩阵向量乘版本。它和 `TMATMUL_MX` 共享同一套 scale Tile 思路，只是把乘法域收窄到 GEMV 形态。

从接口上看，这条指令同样支持普通、累加和 bias 三种变体。

## 数学语义


GEMV 基础乘法域可以写成：

$$ \mathrm{C}_{0,j} = \sum_{k=0}^{K-1} \mathrm{A}_{0,k} \cdot \mathrm{B}_{k,j} $$

和 `TMATMUL_MX` 一样，`aScaleMatrix` / `bScaleMatrix` 会参与 MX 路径下的重建 / 缩放；它们的精确作用由目标定义，而不是由这页单独给出一套独立数值规则。

## 汇编语法


PTO-AS 形式：参见 [汇编写法与操作数](../../../syntax-and-operands/assembly-model_zh.md)。

示意形式：

```text
%acc = tgemv.mx %a, %a_scale, %b, %b_scale : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 1（SSA）


```text
%acc = pto.tgemv.mx %a, %a_scale, %b, %b_scale : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）


```text
pto.tgemv.mx ins(%a, %a_scale, %b, %b_scale : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%acc : !pto.tile_buf<...>)
```

## C++ 内建接口


声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                              TileRightScale &bScaleMatrix, WaitEvents &... events);

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix,
                              TileRight &bMatrix, TileRightScale &bScaleMatrix, WaitEvents &... events);

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                              TileRightScale &bScaleMatrix, TileBias &biasData, WaitEvents &... events);
```

## 约束


!!! warning "约束"
    ### A5 真正支持的 MX 语义

    - `TGEMV_MX` 复用与 `TMATMUL_MX` 相同的 `CheckMadMxValid(...)` 约束，因此：
      - 结果累加器必须是 `float`
      - 输入必须是受支持的 fp4 或 fp8 组合
      - Left / Right / Acc 的位置与 fractal 方向必须合法
    - 运行时这里不再取 `m = aMatrix.GetValidRow()`，而是固定 GEMV 语义使用单行输出：
      - `k = aMatrix.GetValidCol()`
      - `n = bMatrix.GetValidCol()`
      - `k/n` 均必须落在 `[1, 4095]`
    - Bias 变体要求：
      - `biasData` 元素类型为 `float`
      - `biasData` 是单行 `TileType::Bias`

    ### 其他目标的现状

    - CPU 模拟器当前会忽略 `aScaleMatrix` / `bScaleMatrix`，退化为普通 `TGEMV` / `TGEMV_ACC` / `TGEMV_BIAS`。
    - Kirin9030 当前没有 MX 实现路径。

    因此，这条指令的真实 MX 数值语义目前仍以 A5 backend 为准。

## 示例


```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using A = TileLeft<float8_e5m2_t, 1, 64>;
  using B = TileRight<float8_e5m2_t, 64, 32>;
  using ScaleA = TileLeftScale<float8_e8m0_t, 1, 2>;
  using ScaleB = TileRightScale<float8_e8m0_t, 2, 32>;
  using C = TileAcc<float, 1, 32>;

  A a;
  B b;
  ScaleA scaleA;
  ScaleB scaleB;
  C c;
  TGEMV_MX(c, a, scaleA, b, scaleB);
}
```

## 相关页面


- [TMATMUL_MX](./tmatmul-mx_zh.md)
- [TGEMV](./tgemv_zh.md)
- [矩阵与矩阵向量指令集](../../matrix-and-matrix-vector_zh.md)

# pto.tgemv_mx
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Summary
本节给出该指令/主题的核心语义与使用定位，和英文章节保持一致。

## Mechanism
本节说明执行机制与关键语义规则，细节与边界条件以英文版为准。

## Syntax
本节列出语法形态（SSA / DPS / Assembly），用于与英文页逐项对照。

### AS Level 1 (SSA)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### AS Level 2 (DPS)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### IR Level 1 (SSA)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### IR Level 2 (DPS)
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
