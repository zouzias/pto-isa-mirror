# pto.trems


`pto.trems` 属于[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)指令集。

## 概述


对 tile 和标量逐元素执行余数运算。

## 机制


对目标 tile 的有效区域内每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \bmod \mathrm{scalar} $$

与 `tfmods` 相比，这里强调的是 remainder / 取余语义，而不是 `fmod`。

## 语法


同步形式：

```text
%dst = trems %src, %scalar : !pto.tile<...>, f32
```

### AS Level 1（SSA）


```text
%dst = pto.trems %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）


```text
pto.trems ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口


```cpp
template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TREMS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar,
                           TileDataTmp &tmp, WaitEvents &... events);
```

## 输入


- `src`：源 tile
- `scalar`：广播到所有元素的标量
- `dst`：目标 tile
- `tmp`：A2A3 侧实现可能需要的临时 tile
- 迭代域：`dst` 的 valid row / valid col

## 预期输出


- `dst`：逐元素余数结果 tile

## 副作用


除产生目标 tile 外，没有额外架构副作用。

## 约束


!!! warning "约束"
    - 除零行为由目标平台定义；CPU 模拟器在调试构建下会断言。
    - 操作迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。

## 异常与非法情形


!!! danger "异常与非法情形"
    - 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制


### A2A3


- `dst` 与 `src` 必须使用相同元素类型
- 支持元素类型：`float`、`int32_t`
- `dst` 与 `src` 必须是向量 tile且为行主序
- 运行时要求：`dst.GetValidRow() == src.GetValidRow() > 0` 且 `dst.GetValidCol() == src.GetValidCol() > 0`
- `tmp` 需要满足：
  - `tmp.GetValidCol() >= dst.GetValidCol()`
  - `tmp.GetValidRow() >= 1`
  - 数据类型与 `dst` 匹配
- 对 `int32_t` 输入，A2A3 还要求 `src` 与 `scalar` 在 `[-2^24, 2^24]` 范围内，以保证转成 float32 计算时精确

### A5


- `dst` 与 `src` 必须使用相同元素类型
- 支持元素类型：`float`、`int32_t`、`uint32_t`、`half`、`int16_t`、`uint16_t`
- `dst` 与 `src` 必须是向量 tile
- 静态 valid 边界必须合法
- 运行时要求：`dst.GetValidRow() == src.GetValidRow()` 且 `dst.GetValidCol() == src.GetValidCol()`
- `tmp` 形参在 A5 上接受但不参与约束与实现路径

## 性能


当前仓内没有为 `trems` 单列公开 cost bucket。若代码依赖具体延迟，应把它视为目标 profile 相关的余数路径。

## 示例


```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  Tile<TileType::Vec, float, 16, 16> tmp;
  TREMS(out, x, 3.0f, tmp);
}
```

## 相关页面


- 指令集总览：[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)
- 上一条指令：[pto.tfmods](./tfmods_zh.md)
- 下一条指令：[pto.tmaxs](./tmaxs_zh.md)

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
