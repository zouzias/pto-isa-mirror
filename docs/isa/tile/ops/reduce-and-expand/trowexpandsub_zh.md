# pto.trowexpandsub


`pto.trowexpandsub` 属于[归约与扩展](../../reduce-and-expand_zh.md)指令集。

## 概述


`pto.trowexpandsub` 在行扩展基础上执行减法组合，章节结构与英文页保持一致。

## 机制


数学定义与广播规则请以英文规范页为准。

## 语法


### AS Level 1（SSA）


```text
%dst = pto.trowexpandsub %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）


```text
pto.trowexpandsub ins(%src0 : !pto.tile_buf<...>, %src1 : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口


```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDSUB(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## 输入


- `src0`：主输入 tile
- `src1`：辅助输入 tile
- `dst`：输出 tile

## 预期输出


`dst` 按 `trowexpandsub` 语义写入结果。

## 副作用


除写入目标 tile 外，无额外架构副作用。

## 约束


!!! warning "约束"
    - Tile 类型、布局、数据类型与合法域约束请参考英文页。

## 异常


!!! danger "异常"
    - 非法操作数组合、不支持类型或布局将被 verifier / 后端拒绝。

## Target-Profile 限制


??? info "Target-Profile 限制"
    - A2/A3 与 A5 的实现差异、运行时检查请参考英文页。

## 示例


### Auto


```cpp
// Auto: 由编译器/运行时完成资源放置与调度
```

### Manual


```cpp
// Manual: 先显式绑定资源，再发射指令
```

### Auto Mode


```text
%dst = pto.trowexpandsub %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode


```text
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.trowexpandsub %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form


```text
%dst = trowexpandsub %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.trowexpandsub ins(%src0 : !pto.tile_buf<...>, %src1 : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## 自动模式说明


自动模式下由编译器/运行时负责资源放置与调度。

## 手动模式说明


手动模式下建议先完成资源绑定与同步编排，再发射该指令。

## 相关页面


- 指令集总览：[归约与扩展](../../reduce-and-expand_zh.md)
- 英文规范页：[trowexpandsub.md](./trowexpandsub.md)

## Summary
本节给出该指令/主题的核心语义与使用定位，和英文章节保持一致。

## Mechanism
本节说明执行机制与关键语义规则，细节与边界条件以英文版为准。

## Syntax
本节列出语法形态（SSA / DPS / Assembly），用于与英文页逐项对照。

### AS Level 1 (SSA)
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

# Auto mode: compiler/runtime-managed placement and scheduling.
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

# Manual mode: bind resources explicitly before issuing the instruction.
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Related Ops / Instruction Set Links
本节给出上下游指令与相关章节链接。
