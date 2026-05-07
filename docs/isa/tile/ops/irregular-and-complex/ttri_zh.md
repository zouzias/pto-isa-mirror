# TTRI


## 指令示意图


![TTRI tile operation](../../../../figures/isa/TTRI.svg)

## 简介


`TTRI` 生成一个三角掩码 Tile。它不读取源 Tile，而是根据目标 Tile 的有效形状和 `diagonal` 参数直接在 `dst` 里写出上三角或下三角的 0/1 模式。

这条指令常用于注意力 mask、三角区域约束或后续按位/乘法掩码场景。

## 数学语义


设 `R = dst.GetValidRow()`、`C = dst.GetValidCol()`，`d = diagonal`。

### 下三角形式 `isUpperOrLower = 0`


$$
\mathrm{dst}_{i,j} =
\begin{cases}
1 & j \le i + d \\
0 & \text{否则}
\end{cases}
$$

### 上三角形式 `isUpperOrLower = 1`


$$
\mathrm{dst}_{i,j} =
\begin{cases}
0 & j < i + d \\
1 & \text{否则}
\end{cases}
$$

`diagonal = 0` 表示主对角线；正值会把保留区域向右扩展，负值则会收缩。

## 汇编语法


PTO-AS 形式：参见 [汇编写法与操作数](../../../syntax-and-operands/assembly-model_zh.md)。

### AS Level 1（SSA）


```text
%dst = pto.ttri %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）


```text
pto.ttri ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口


声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileData, int isUpperOrLower, typename... WaitEvents>
PTO_INST RecordEvent TTRI(TileData &dst, int diagonal, WaitEvents &... events);
```

## 约束


!!! warning "约束"
    - `isUpperOrLower` 只能是：
      - `0`：下三角
      - `1`：上三角
    - `dst` 必须是 row-major Tile。
    - 支持的数据类型随目标略有差异：
      - CPU / A2A3：`int32_t`、`int16_t`、`uint32_t`、`uint16_t`、`half`、`float` 等
      - A5：额外覆盖 `int8_t`、`uint8_t`、`bfloat16_t`

## 示例


```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using MaskT = Tile<TileType::Vec, float, 16, 16>;
  MaskT mask;
  TTRI<MaskT, 0>(mask, 0);  // lower triangular
}
```

## 相关页面


- [TCMP](../elementwise-tile-tile/tcmp_zh.md)
- [不规则与复杂指令集](../../irregular-and-complex_zh.md)

# pto.ttri
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
