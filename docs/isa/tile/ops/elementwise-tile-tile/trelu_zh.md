# pto.trelu


`pto.trelu` 属于[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)指令集。

## 概述


对 tile 做逐元素 ReLU。

## 机制


对目标 tile 的 valid region 中每个 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \max(\mathrm{src}_{i,j}, 0) $$

它是 tile 版的逐元素 ReLU，用来把负值裁成 0。

## 语法


### PTO-AS


```text
%dst = trelu %src : !pto.tile<...>
```

### AS Level 1（SSA）


```text
%dst = pto.trelu %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）


```text
pto.trelu ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口


```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TRELU(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

## 输入


- `%src`：源 tile
- `%dst`：目标 tile

## 预期输出


- `%dst`：逐元素 ReLU 结果 tile

## 副作用


除产生目标 tile 外，没有额外架构副作用。

## 约束


!!! warning "约束"
    - 操作迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
    - `src` 与 `dst` 的 validRow / validCol 通常应兼容。

## 异常与非法情形


!!! danger "异常与非法情形"
    - 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制


### A2A3


- 支持类型：`half`、`float`、`int32_t`
- tile 必须是行主序向量 tile
- 静态 valid 边界必须合法
- 运行时通常要求 `src` 与 `dst` 的 `validRow/validCol` 一致

### A5


- 支持类型：`half`、`float`、`int32_t`
- tile 必须是行主序向量 tile
- 静态 valid 边界必须合法
- 运行时通常要求 `src` 与 `dst` 的 `validRow/validCol` 一致

## 性能


### A2A3


英文页把 `TRELU` 归入一元 tile 运算桶：

| 指标 | 数值 |
| --- | --- |
| 启动时延 | 13 |
| 完成时延 | 26（沿用一元 / 超越函数桶） |
| 每次 repeat 吞吐 | 1 |
| 流水间隔 | 18 |

### A5


当前手册未单列 `trelu` 的独立周期表，应视为目标 profile 相关。

## 示例


```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  TRELU(out, x);
}
```

## 相关页面


- 指令集总览：[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)
- 上一条指令：[pto.tnot](./tnot_zh.md)
- 下一条指令：[pto.tneg](./tneg_zh.md)

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

## Performance
本节说明性能路径、吞吐估算与形状/布局敏感因素。

### A2/A3 Throughput
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

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
