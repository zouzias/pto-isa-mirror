# pto.taddc


`pto.taddc` 属于[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)指令集。

## 概述


三输入逐元素加法：`src0 + src1 + src2`。

## 机制


对目标 tile 的 valid region 中每个 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} + \mathrm{src1}_{i,j} + \mathrm{src2}_{i,j} $$

它是一条三输入 tile 算术指令，重点在于把三路源 tile 直接合并进一次逐元素运算。

## 语法


### PTO-AS


```text
%dst = taddc %src0, %src1, %src2 : !pto.tile<...>
```

### AS Level 1（SSA）


```text
%dst = pto.taddc %src0, %src1, %src2 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）


```text
pto.taddc ins(%src0, %src1, %src2 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口


```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TADDC(TileData &dst, TileData &src0, TileData &src1, TileData &src2, WaitEvents &... events);
```

## 输入


- `%src0`、`%src1`、`%src2`：三个源 tile
- `%dst`：目标 tile
- 迭代域：`dst` 的 valid row / valid col

## 预期输出


- `%dst`：逐元素三输入加法结果 tile

## 副作用


除产生目标 tile 外，没有额外架构副作用。

## 约束


!!! warning "约束"
    - 迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
    - 三个源 tile 与目标 tile 应在 shape、layout 和 valid region 上兼容。

## 异常与非法情形


!!! danger "异常与非法情形"
    - 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制


- `pto.taddc` 在 CPU 仿真、A2/A3 和 A5 上保留一致的 PTO 可见语义，但具体支持子集仍取决于 profile。

## 性能


### A2A3


英文页当前把 `TADDC` 归到与二元算术同类的估算口径：

| 指标 | FP | INT |
| --- | --- | --- |
| 启动时延 | 14 | 14 |
| 完成时延 | 19 | 17 |
| 每次 repeat 吞吐 | 2 | 2 |
| 流水间隔 | 18 | 18 |

### A5


当前手册未单列 `taddc` 的独立周期表，应视为目标 profile 相关。

## 示例


```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT a, b, c, out;
  TADDC(out, a, b, c);
}
```

## 相关页面


- 指令集总览：[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)
- 上一条指令：[pto.tprelu](./tprelu_zh.md)
- 下一条指令：[pto.tsubc](./tsubc_zh.md)

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
