# TFILLPAD_INPLACE


## 指令示意图


![TFILLPAD_INPLACE tile operation](../../../../figures/isa/TFILLPAD_INPLACE.svg)

## 简介


`TFILLPAD_INPLACE` 是 `TFILLPAD` 的原位变体。它的结果语义与普通 `TFILLPAD` 相同，都是“保留源 valid region，剩余区域写入 pad 值”；区别在于这条接口面向“把目标 Tile 当成被原地修补的对象”的使用方式。

如果你只看最终结果，可以把它理解成“同尺寸 `TFILLPAD`”；如果你关心实现路径，它表达的是“允许 backend 走原位修补式填充”。

## 数学语义


设：

- `VR = src.GetValidRow()`
- `VC = src.GetValidCol()`

对 `dst` 的每个元素 `(i, j)`：

$$
\mathrm{dst}_{i,j} =
\begin{cases}
\mathrm{src}_{i,j} & \text{当 } i < VR \text{ 且 } j < VC \\
\mathrm{pad} & \text{否则}
\end{cases}
$$

其中 `pad` 由 `TileDataDst::PadVal` 决定。

## 汇编语法


PTO-AS 形式：参见 [汇编写法与操作数](../../../syntax-and-operands/assembly-model_zh.md)。

### AS Level 1（SSA）


```text
%dst = pto.tfillpad_inplace %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）


```text
pto.tfillpad_inplace ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口


声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD_INPLACE(DstTileData &dst, SrcTileData &src, WaitEvents &... events);
```

## 约束


!!! warning "约束"
    ### 通用约束

    - `dst.Rows == src.Rows`
    - `dst.Cols == src.Cols`
    - `TileDataDst::PadVal != PadValue::Null`
    - `src` 和 `dst` 的元素大小必须一致，并且当前实现只接受 `1`、`2` 或 `4` 字节元素
    - 如果 `dst.GetValidRow() == 0` 或 `dst.GetValidCol() == 0`，backend 会直接返回

    ### Backend 说明

    - A2/A3、A5 和 CPU 模拟器都把这条指令实现成与同尺寸 `TFILLPAD` 等价的结果语义。
    - “原位”更多是接口和实现路径上的语义，而不是另一套数学定义。

## 示例


```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16,
                     BLayout::RowMajor, 16, 16, SLayout::NoneBox,
                     TileConfig::fractalABSize, PadValue::Max>;

  TileT src;
  TileT dst;
  TFILLPAD_INPLACE(dst, src);
}
```

## 相关页面


- [TFILLPAD](./tfillpad_zh.md)
- [TFILLPAD_EXPAND](./tfillpad-expand_zh.md)
- [布局与重排指令集](../../layout-and-rearrangement_zh.md)

# pto.tfillpad_inplace
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
