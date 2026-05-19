# pto.tpow

`pto.tpow` 属于[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)指令集。

## 概述

以两个 tile 分别提供底数与指数，做逐元素幂运算。

## 机制

对目标 valid region 内的每个 `(r, c)`：

$$ \mathrm{dst}_{r,c} = \mathrm{pow}(\mathrm{base}_{r,c}, \mathrm{exp}_{r,c}) $$

当前 C++ 内建接口要求显式传入 `tmp` tile 作为 scratch。这个 `tmp` 属于实现辅助资源，不是文本汇编里额外暴露的语义操作数。

## 语法

### PTO Assembly Form

```text
%dst = tpow %base, %exp : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tpow %base, %exp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tpow ins(%base, %exp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <auto PrecisionType = PowAlgorithm::DEFAULT, typename DstTile, typename BaseTile, typename ExpTile,
          typename TmpTile, typename... WaitEvents>
PTO_INTERNAL RecordEvent TPOW(DstTile &dst, BaseTile &base, ExpTile &exp, TmpTile &tmp, WaitEvents &... events);
```

`PrecisionType` 支持：

- `PowAlgorithm::DEFAULT`
- `PowAlgorithm::HIGH_PRECISION`

## 输入

| 操作数 | 角色 | 说明 |
|--------|------|------|
| `dst` | 目标 tile | 保存结果 |
| `base` | 源 tile | 底数 tile |
| `exp` | 源 tile | 指数 tile |
| `tmp` | 临时 tile | C++ 接口要求的 scratch tile |

## 预期输出

`dst` 在 valid region 内保存逐元素幂运算结果。

## 副作用

除产生目标 tile 外，没有额外架构副作用。`tmp` 可能被 backend 当作 scratch 使用。

## 约束

!!! warning "约束"
    - `dst`、`base`、`exp` 的元素类型必须一致。
    - `dst`、`base`、`exp` 的 valid row / valid col 必须一致。
    - 当前具体实现要求三者都是 row-major 的 `TileType::Vec`。
    - `PowAlgorithm::DEFAULT` 支持 `float`、`half`、`int32_t`、`uint32_t`、`int16_t`、`uint16_t`、`int8_t`、`uint8_t`。
    - `PowAlgorithm::HIGH_PRECISION` 支持 `float`、`half`、`bfloat16_t`。

## 不允许的情形

!!! danger "不允许的情形"
    - 使用不支持的类型 / layout 组合。
    - 在 valid region 不一致时依赖结果语义。
    - 仅凭公共模板声明就假设 CPU 或 A2/A3 一定支持这条指令。

## Target-Profile 限制

当前手册树只对 A5 给出 `tpow` 的具体实现合同。本 checkout 虽然在 `pto_instr.hpp` 中暴露了公共模板，但没有为 CPU 仿真或 A2/A3 给出稳定的文档化实现保证。

因此，可移植代码应把 `pto.tpow` 视为 A5 专属能力，除非目标 profile 页面明确放宽这一点。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor>;
  TileT dst;
  TileT base;
  TileT exp;
  TileT tmp;

  TPOW(dst, base, exp, tmp);
  TPOW<PowAlgorithm::HIGH_PRECISION>(dst, base, exp, tmp);
}
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)
- 上一条指令：[pto.texp](./texp_zh.md)
- 下一条指令：[pto.tnot](./tnot_zh.md)
