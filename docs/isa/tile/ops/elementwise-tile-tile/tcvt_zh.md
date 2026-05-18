# pto.tcvt

`pto.tcvt` 属于[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)指令集。

## 概述

按指定舍入模式，对 tile 做逐元素类型转换；部分形式还允许显式指定饱和模式。

## 机制

对目标 tile 的 valid region 中每个 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{cast}_{\mathrm{rmode},\mathrm{satmode}}\!\left(\mathrm{src}_{i,j}\right) $$

其中：

- `rmode` 控制舍入规则
- `satmode`（若显式给出）控制溢出时是否饱和

这条指令既覆盖 tile 内部的数值类型变化，也把“舍入 / 饱和是否显式暴露”做成了接口的一部分。

## 舍入模式

| 模式 | 行为 |
| --- | --- |
| `CAST_RINT` | 就近舍入，ties to even |
| `CAST_ROUND` | 就近舍入，ties away from zero |
| `CAST_FLOOR` | 向负无穷舍入 |
| `CAST_CEIL` | 向正无穷舍入 |
| `CAST_TRUNC` | 向 0 舍入 |

## 饱和模式

| 模式 | 行为 |
| --- | --- |
| `ON` | 开启饱和 |
| `OFF` | 关闭饱和 |

## 语法

### PTO-AS

```text
%dst = tcvt %src {rmode = #pto.round_mode<CAST_RINT>} : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tcvt %src {rmode = #pto.round_mode<CAST_RINT>} : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tcvt ins(%src {rmode = #pto.round_mode<CAST_RINT>}: !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataD, typename TileDataS, typename TmpTileData, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, TmpTileData &tmp, RoundMode mode,
                          SaturationMode satMode, WaitEvents &... events);

template <typename TileDataD, typename TileDataS, typename TmpTileData, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, TmpTileData &tmp, RoundMode mode, WaitEvents &... events);

template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, RoundMode mode, SaturationMode satMode,
                          WaitEvents &... events);

template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, RoundMode mode, WaitEvents &... events);
```

`tmp` 版本用于那些需要显式 scratch tile 的转换路径。

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `%src` | 源 tile | 在 `dst` valid region 上逐坐标读取 |
| `%dst` | 目标 tile | 保存转换后的元素值 |
| `mode` | 舍入模式 | `CAST_RINT` / `CAST_ROUND` / `CAST_FLOOR` / `CAST_CEIL` / `CAST_TRUNC` |
| `satMode` | 可选饱和模式 | `ON` / `OFF` |
| `tmp` | 可选临时 tile | 需要显式 scratch 的路径使用 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `%dst` | `!pto.tile<...>` | 逐元素转换后的结果 tile |

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

!!! warning "约束"
    - `src` 与 `dst` 必须在 shape 和 valid region 上兼容。
    - 源 / 目标类型对必须属于目标 profile 支持的集合。
    - 给定类型对必须支持所选 rounding mode。
    - 对于需要显式 scratch 的路径，调用方必须使用 `tmp` 版本。
    - 关闭饱和可能改变某些低精度整数转换路径的溢出语义。

## 不允许的情形

!!! danger "不允许的情形"
    - 使用目标 profile 不支持的类型对。
    - 使用该类型对不支持的 rounding mode。
    - 在关闭饱和时仍假设溢出会被 clamp。

## Target-Profile 限制

`pto.tcvt` 在 CPU 仿真、A2/A3 和 A5 上都保留 PTO 可见语义，但具体支持的类型对、是否需要 scratch、以及饱和关闭后的溢出处理仍然依赖 backend。

当前 checkout 中，fp16 → int8 的非饱和路径通过带 scratch 的 helper 实现，并且会按行做子分块处理。

## 示例

### 自动模式

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, half, 16, 16>;
  SrcT src;
  DstT dst;
  TCVT(dst, src, RoundMode::CAST_RINT);
}
```

### 显式饱和 / scratch

```cpp
using TmpT = Tile<TileType::Vec, int32_t, 16, 16>;
TmpT tmp;
TCVT(dst, src, tmp, RoundMode::CAST_TRUNC, SaturationMode::OFF);
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)
- 上一条指令：[pto.tsubc](./tsubc_zh.md)
- 下一条指令：[pto.tsel](./tsel_zh.md)
