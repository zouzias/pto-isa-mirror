# pto.tpows


`pto.tpows` 属于[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)指令集。

## 概述


底数来自 tile、指数来自标量的逐元素幂运算。

## 机制


对目标 valid region 内的每个 `(r, c)`：

$$ \mathrm{dst}_{r,c} = \mathrm{pow}(\mathrm{base}_{r,c}, \mathrm{exp}) $$

和 `pto.tpow` 一样，当前 C++ 接口要求一个 `tmp` scratch tile；这个 `tmp` 不是文本汇编里额外暴露的语义输入，而是实现辅助资源。

## 语法


### PTO Assembly Form


```text
%dst = tpows %base, %exp : !pto.tile<...>, dtype -> !pto.tile<...>
```

### AS Level 1（SSA）


```text
%dst = pto.tpows %base, %exp : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）


```text
pto.tpows ins(%base, %exp : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口


声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <auto PrecisionType = PowAlgorithm::DEFAULT, typename DstTile, typename BaseTile, typename TmpTile,
          typename... WaitEvents>
PTO_INTERNAL RecordEvent TPOWS(DstTile &dst, BaseTile &base, typename DstTile::DType exp, TmpTile &tmp,
                               WaitEvents &... events);
```

## 输入


| 操作数 | 角色 | 说明 |
|--------|------|------|
| `dst` | 目标 tile | 保存结果 |
| `base` | 源 tile | 底数 tile |
| `exp` | 标量 | 沿目标 valid region 广播的指数 |
| `tmp` | 临时 tile | C++ 接口要求的 scratch tile |

## 预期输出


`dst` 在 valid region 内保存逐元素幂结果。

## 副作用


除产生目标 tile 外，没有额外架构副作用。`tmp` 可能被 backend 当作 scratch 使用。

## 约束


!!! warning "约束"
    - `dst` 与 `base` 的元素类型必须一致。
    - `dst` 与 `base` 的 valid row / valid col 必须一致。
    - 当前具体实现要求两者都是 row-major 的 `TileType::Vec`。
    - 标量指数的类型为 `typename DstTile::DType`。
    - `PowAlgorithm::DEFAULT` 支持 `float`、`half`、`int32_t`、`uint32_t`、`int16_t`、`uint16_t`、`int8_t`、`uint8_t`。
    - `PowAlgorithm::HIGH_PRECISION` 支持 `float`、`half`、`bfloat16_t`。

## 不允许的情形


!!! danger "不允许的情形"
    - 使用不支持的类型 / layout 组合。
    - 在 valid region 不一致时依赖结果语义。
    - 仅凭公共模板声明就假设 CPU 或 A2/A3 一定支持这条指令。

## Target-Profile 限制


当前手册树只对 A5 给出 `tpows` 的具体实现合同。本 checkout 没有为 CPU 仿真或 A2/A3 给出稳定的文档化实现保证。

因此，可移植代码应把 `pto.tpows` 视为 A5 专属能力，除非目标 profile 页面明确放宽这一点。

## 示例


```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor>;
  TileT dst;
  TileT base;
  TileT tmp;

  TPOWS(dst, base, 2.0f, tmp);
  TPOWS<PowAlgorithm::HIGH_PRECISION>(dst, base, 2.0f, tmp);
}
```

## 相关页面


- 指令集总览：[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)
- 上一条指令：[pto.tmuls](./tmuls_zh.md)
- 下一条指令：[pto.tfmods](./tfmods_zh.md)

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

## Related Ops / Instruction Set Links
本节给出上下游指令与相关章节链接。
