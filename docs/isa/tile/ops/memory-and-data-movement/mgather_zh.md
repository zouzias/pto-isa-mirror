# pto.mgather


`pto.mgather` 属于[内存与数据搬运](../../memory-and-data-movement_zh.md)指令集。

## 概述


根据逐元素索引，从全局内存把离散位置的数据收集到 tile 里。

## 机制


对目标有效区域中的每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{mem}[\mathrm{idx}_{i,j}] $$

和连续 `TLOAD` 不同，这条指令不再沿一个矩形区域顺序搬运，而是通过索引 tile 指定每个元素应该从 GM 的哪个位置取值。

## 语法


同步形式：

```text
%dst = mgather %mem, %idx : !pto.memref<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）


```text
%dst = pto.mgather %mem, %idx : (!pto.partition_tensor_view<MxNxdtype>, pto.tile<...>)
-> !pto.tile<loc, dtype, rows, cols, blayout, slayout, fractal, pad>
```

### AS Level 2（DPS）


```text
pto.mgather ins(%mem, %idx : !pto.partition_tensor_view<MxNxdtype>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口


```cpp
template <typename TileDst, typename GlobalData, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MGATHER(TileDst &dst, GlobalData &src, TileInd &indexes, WaitEvents &... events);
```

## 输入


- `src`：源 `GlobalTensor`
- `indexes`：索引 tile，为每个元素给出 GM 位置
- `dst`：目标 tile

## 预期输出


- `dst`：按 `indexes` 指定位置 gather 得到的结果 tile

## 副作用


这条指令会从 GM 读取。索引越界的行为由目标 profile 定义。

## 约束


!!! warning "约束"
    - `dst/src` 元素类型必须属于：
      `uint8_t`、`int8_t`、`uint16_t`、`int16_t`、`uint32_t`、`int32_t`、`half`、`bfloat16_t`、`float`
    - AICore 目标上还支持 `float8_e4m3_t` 和 `float8_e5m2_t`
    - `indexes` 的元素类型必须是 `int32_t` 或 `uint32_t`
    - `dst` 和 `indexes` 必须是 row-major 的 `TileType::Vec`
    - `src` 必须是 GM 上的 `GlobalTensor`，且布局为 ND
    - `dst.Rows == indexes.Rows`
    - `indexes` 可以是 `[N,1]` 或 `[N,M]`
    - `dst` 的行宽必须满足 32B 对齐

## 异常与非法情形


!!! danger "异常与非法情形"
    - 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。
    - CPU 模拟器不会对 `indexes` 做强制越界检查。

## Target-Profile 限制


- `pto.mgather` 在 CPU 仿真、A2/A3 和 A5 上都保留相同的 PTO 可见语义，但具体支持子集和索引解释仍取决于 profile。

## 性能


当前手册未单列 `mgather` 的公开周期表。它的成本通常主要受访存离散程度和目标实现的 gather 路径影响。

## 示例


```text
%dst = pto.mgather %mem, %idx : (!pto.partition_tensor_view<MxNxdtype>, pto.tile<...>)
```

## 相关页面


- 指令集总览：[内存与数据搬运](../../memory-and-data-movement_zh.md)
- 下一条指令：[pto.mscatter](./mscatter_zh.md)

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
