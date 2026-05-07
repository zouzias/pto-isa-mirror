# pto.trowmax


`pto.trowmax` 属于[归约与扩展](../../reduce-and-expand_zh.md)指令集。

## 概述


对每一行按列取最大值。

## 机制


设：

- `R = src.GetValidRow()`
- `C = src.GetValidCol()`

则对 `0 <= i < R`：

$$ \mathrm{dst}_{i,0} = \max_{0 \le j < C} \mathrm{src}_{i,j} $$

它把 `(R, C)` 压成 `(R, 1)`，保留行索引、折叠列方向。lowering 过程中通常会引入 `tmp`。

## 语法


同步形式：

```text
%dst = trowmax %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）


```text
%dst = pto.trowmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）


```text
pto.trowmax ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口


```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWMAX(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &... events);
```

## 输入


- `src`：源 tile
- `tmp`：临时 tile
- `dst`：目标 tile

## 预期输出


- `dst[i,0]`：第 `i` 行所有列元素中的最大值

## 副作用


除产生目标 tile 外，没有额外架构副作用。

## 约束


!!! warning "约束"
    - `dst` 与 `src` 必须都为 `TileType::Vec`
    - `src` 必须使用标准 ND 布局：行主且非分形
    - `dst` 可以是 ND，或 `Cols == 1` 的 DN 布局
    - `dst` 与 `src` 元素类型必须一致
    - 运行时要求：
      - `src.GetValidRow() != 0`
      - `src.GetValidCol() != 0`
      - `src.GetValidRow() == dst.GetValidRow()`

    ### A2A3

    - 支持类型：`half`、`float`、`int32_t`、`int16_t`
    - 实现同时接受 ND 输出和 `Cols == 1` 的 DN 输出
    - 当前实现路径会把 `tmp` 传入后端调用，但文档不额外引入 checked implementation 没声明的 `tmp` 限制

## 异常与非法情形


!!! danger "异常与非法情形"
    - 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## 性能


当前仓内没有把 `trowmax` 单列成公开 cost table。它应视为行归约类路径，而不是普通二元逐元素算术。

## 示例


```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TROWMAX(dst, src, tmp);
}
```

## 相关页面


- 指令集总览：[归约与扩展](../../reduce-and-expand_zh.md)
- 上一条指令：[pto.tcolmax](./tcolmax_zh.md)
- 下一条指令：[pto.trowmin](./trowmin_zh.md)

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

### Manual
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
