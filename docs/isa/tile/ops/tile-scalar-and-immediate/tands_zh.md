# pto.tands


`pto.tands` 属于[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)指令集。

## 概述


对 tile 和标量逐元素做按位与。

## 机制


`pto.tands` 作用在 tile payload 上，而不是标量控制状态。对目标 tile 的有效区域内每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \;\&\; \mathrm{scalar} $$

标量会广播到整个 valid region。它本质上是 tile 版的按位掩码操作，适合做位域裁剪、标志位过滤和逐元素掩码化。

## 语法


同步形式：

```text
%dst = tands %src, %scalar : !pto.tile<...>, i32
```

### AS Level 1（SSA）


```text
%dst = pto.tands %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）


```text
pto.tands ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口


```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TANDS(TileDataDst &dst, TileDataSrc &src, typename TileDataDst::DType scalar, WaitEvents &... events);
```

## 输入


- `src`：源 tile
- `scalar`：广播到所有元素的标量掩码
- `dst`：目标 tile
- 迭代域：`dst` 的 valid row / valid col

## 预期输出


- `dst`：逐元素按位与结果 tile

## 副作用


除产生目标 tile 外，没有额外架构副作用。

## 约束


!!! warning "约束"
    - 操作迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
    - 这条指令面向整数元素类型，不适用于浮点 tile。

## 异常与非法情形


!!! danger "异常与非法情形"
    - 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。
    - 程序不能依赖文档合法域之外的行为。

## Target-Profile 限制


### A2A3


- 面向整型元素类型。
- `dst` 与 `src` 必须使用相同元素类型。
- `dst` 与 `src` 必须是向量 tile。
- 运行时要求：`src.GetValidRow() == dst.GetValidRow()` 且 `src.GetValidCol() == dst.GetValidCol()`。
- 手动模式下，不支持把源 tile 与目标 tile 绑定到同一片内存。

### A5


- 面向 `TEXPANDS` 与 `TAND` 支持的整型元素类型。
- `dst` 与 `src` 必须使用相同元素类型。
- `dst` 与 `src` 必须是向量 tile。
- 手动模式下，不支持把源 tile 与目标 tile 绑定到同一片内存。

## 性能


当前仓内没有为 `tands` 单列公开 cost bucket。若代码依赖具体延迟，应把它视为目标 profile 相关的 tile 按位逻辑路径。

## 示例


```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileDst = Tile<TileType::Vec, uint16_t, 16, 16>;
  using TileSrc = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileDst dst;
  TileSrc src;
  TANDS(dst, src, 0xffu);
}
```

## 相关页面


- 指令集总览：[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)
- 上一条指令：[pto.tmaxs](./tmaxs_zh.md)
- 下一条指令：[pto.tors](./tors_zh.md)

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
