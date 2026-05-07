# pto.trowargmax


`pto.trowargmax` 属于[归约与扩展](../../reduce-and-expand_zh.md)指令集。

## 概述


返回每一行最大值所在的列索引。

## 机制


设：

- `R = src.GetValidRow()`
- `C = src.GetValidCol()`

则对 `0 <= i < R`：

$$ \mathrm{dst}_{i,0} = \underset{0 \le j < C}{\operatorname{argmax}} \; \mathrm{src}_{i,j} $$

输出 tile 只保留一列，因此 `dst[i,0]` 表示“第 `i` 行最大值落在哪一列”。如果一行里有多个相同最大值，具体选哪一个索引由实现决定，可移植代码不能依赖固定 tie-breaking。

## 语法


同步形式：

```text
%dst = trowargmax %src : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 1（SSA）


```text
%dst = pto.trowargmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### IR Level 2（DPS）


```text
pto.trowargmax ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口


```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWARGMAX(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events);
```

## 输入


- `src`：源 tile
- `tmp`：可能参与归约中间阶段的临时 tile
- `dst`：目标索引 tile

## 预期输出


- `dst[i,0]`：第 `i` 行最大值所在列索引

## 副作用


除产生目标 tile 外，没有额外架构副作用。

## 约束


!!! warning "约束"
    - `dst` 与 `src` 必须为 `TileType::Vec`
    - 源元素类型支持：`half`、`float`
    - 目标元素类型支持：`uint32_t`、`int32_t`
    - 运行时要求：
      - `src.GetValidRow() != 0`
      - `src.GetValidCol() != 0`
      - `src.GetValidRow() == dst.GetValidRow()`

    ### A2A3

    - `src` 必须是标准 ND 布局：行主且非分形
    - `dst` 可为单列 DN，或 validCol 为 1 的 ND
    - 当 `srcValidCol <= elementPerRepeat` 时，`tmp` 可能不参与运算
    - 当 `srcValidCol > elementPerRepeat` 时，`tmp` 会用于分阶段归约
    - `tmp` 的行数应与 `src` 一致

    ### A5

    - A5 路径接口仍接收 `tmp`，但当前 checked implementation 中并不实际使用它

## 异常与非法情形


!!! danger "异常与非法情形"
    - 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## 性能


当前仓内没有把 `trowargmax` 单列成公开 cost table。它应视为索引归约路径，而不是普通数值归约。

## 示例


```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, uint32_t, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TROWARGMAX(dst, src, tmp);
}
```

## 相关页面


- 指令集总览：[归约与扩展](../../reduce-and-expand_zh.md)
- 上一条指令：[pto.trowmin](./trowmin_zh.md)
- 下一条指令：[pto.trowargmin](./trowargmin_zh.md)

## Summary
本节给出该指令/主题的核心语义与使用定位，和英文章节保持一致。

## Mechanism
本节说明执行机制与关键语义规则，细节与边界条件以英文版为准。

## Syntax
本节列出语法形态（SSA / DPS / Assembly），用于与英文页逐项对照。

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

# IR Level 2 (DPS)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Related Ops / Instruction Set Links
本节给出上下游指令与相关章节链接。
