# pto.tdequant


`pto.tdequant` 属于[不规则与复杂](../../irregular-and-complex_zh.md) tile 指令族。

## 概述


使用按行广播的 `scale` 与 `offset` tile，把整数量化 tile 反量化为浮点 tile。

## 机制


`pto.tdequant` 把量化整数源 tile 恢复为浮点数值表示。在当前仓内实现里，目标 tile 是浮点型，源 tile 是整型，`scale` / `offset` 为“每行一个值”的参数 tile，并沿列广播。

对目标 valid region 内的每个 `(r, c)`：

$$ \mathrm{dst}_{r,c} = \left(\mathrm{src}_{r,c} - \mathrm{offset}_r\right) \cdot \mathrm{scale}_r $$

## 语法


### AS Level 1（SSA）


```text
%dst = pto.tdequant %src, %scale, %offset : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）


```text
pto.tdequant ins(%src, %scale, %offset : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
              outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口


声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc, typename TileDataPara, typename... WaitEvents>
PTO_INST RecordEvent TDEQUANT(TileDataDst &dst, TileDataSrc &src, TileDataPara &scale, TileDataPara &offset,
                              WaitEvents &... events);
```

## 输入


| 操作数 | 角色 | 说明 |
|--------|------|------|
| `dst` | 目标 tile | 浮点输出 tile |
| `src` | 源 tile | 量化整数 tile |
| `scale` | 参数 tile | 每行一个 scale 值，并沿列广播 |
| `offset` | 参数 tile | 每行一个 offset 值，并沿列广播 |

## 预期输出


`dst` 在其 valid region 内保存反量化后的浮点结果。

## 副作用


除产生目标 tile 外，没有额外架构副作用。不会隐式建立与无关流量的 fence。

## 约束


!!! warning "约束"
    - `dst` 与 `src` 的 valid row / valid col 必须一致。
    - `scale` 和 `offset` 的行数必须与 `dst.GetValidRow()` 一致。
    - 当前仓内实现中，`dst` 为 `float` / `float32_t`。
    - 当前仓内实现中，`src` 为 `int8_t` 或 `int16_t`。
    - 当前仓内实现中，`scale` / `offset` 使用与 `dst` 相同的浮点元素类型。
    - 当前具体实现要求 row-major tile。

## 不允许的情形


!!! danger "不允许的情形"
    - 使用不支持的类型组合、layout 或不匹配的 valid region。
    - 把 `scale` / `offset` 当作按列变化的参数，除非目标 profile 另行文档化。

## Target-Profile 限制


| 特性 | CPU Simulator | A2/A3 | A5 |
|------|:-------------:|:-----:|:--:|
| `float <- int8_t` | Yes | Yes | Yes |
| `float <- int16_t` | Yes | Yes | Yes |
| row-major tile | Yes | Yes | Yes |

当前仓内在 CPU 仿真、A2/A3 和 A5 上都能找到具体实现。

## 示例


```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using DstTile = Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor>;
  using SrcTile = Tile<TileType::Vec, int8_t, 64, 64, BLayout::RowMajor>;
  using ParaTile = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor>;

  DstTile dst;
  SrcTile src;
  ParaTile scale;
  ParaTile offset;

  TDEQUANT(dst, src, scale, offset);
}
```

## 相关页面


- [不规则与复杂](../../irregular-and-complex_zh.md)
- [pto.tquant](./tquant_zh.md)

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

## See Also
本节给出上下游指令与相关章节链接。
