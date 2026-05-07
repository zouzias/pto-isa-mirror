# pto.tadd


`pto.tadd` 属于[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)指令集。

## 概述


对两个源 tile 进行逐 lane 加法并写入目标 tile。迭代域由目标 tile 的有效区域定义。

## 机制


对目标有效区域内每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} + \mathrm{src1}_{i,j} $$

当源 tile 在某些 lane 上不覆盖目标有效域时，越界 lane 的读取行为以目标 profile 定义为准（详见英文页）。

## 语法


### Assembly Form（PTO-AS）


```text
%dst = tadd %src0, %src1 : !pto.tile<...>
```

### AS Level 1（SSA）


```mlir
%dst = pto.tadd %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）


```mlir
pto.tadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

### 微操作映射


`pto.tadd` 在后端可映射为读取两路源寄存器、执行加法、写回目标寄存器的流水序列。

## C++ 内建接口


```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TADD(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events);
```

## 输入


| 操作数 | 角色 | 说明 |
|---|---|---|
| `%src0` | 左操作数 | 第一路源 tile |
| `%src1` | 右操作数 | 第二路源 tile |
| `WaitEvents...` | 可选同步 | 发射前需要等待的事件 |

## 预期输出


| 结果 | 类型 | 说明 |
|---|---|---|
| `%dst` | `!pto.tile<...>` | 目标 tile 在有效域内写入 `src0 + src1` |

## 副作用


除写入目标 tile 外，无额外架构副作用；不会隐式栅栏不相关流量。

## 约束


!!! warning "约束"
    - `src0`、`src1`、`dst` 元素类型必须一致。
    - 布局组合必须受目标 profile 支持。
    - 迭代域为 `dst.GetValidRow() × dst.GetValidCol()`。

## 异常


!!! danger "异常"
    - 类型不匹配、布局不合法、目标后端不支持等情形会被 verifier 或后端拒绝。

## Target-Profile 限制


??? info "Target-Profile 限制"
    - A2/A3 与 A5 的支持类型、布局与行为细节请参考英文页：`tadd.md`。

## 性能


### A2/A3 吞吐


在 A2/A3 上通常映射为向量二元算子路径，吞吐模型与重复次数相关。

### Shape-Dependent 优化


不同 valid shape 与步幅会选择不同指令序列路径。

### 布局对吞吐的影响


RowMajor 一般更容易走连续快路径；其他布局可能进入通用路径。

### 吞吐估算示例


具体 cycle 估算公式与常量请参考英文页。

## 示例


### C++（Auto Mode）


```cpp
TADD(dst, src0, src1);
```

### C++（Manual Mode）


```cpp
TASSIGN(src0, 0x1000);
TASSIGN(src1, 0x2000);
TASSIGN(dst,  0x3000);
TADD(dst, src0, src1);
```

### MLIR（SSA）


```mlir
%result = pto.tadd %src0, %src1 : (!pto.tile<f32, 16, 16>, !pto.tile<f32, 16, 16>) -> !pto.tile<f32, 16, 16>
```

### MLIR（DPS）


```mlir
pto.tadd ins(%src0, %src1 : !pto.tile_buf<f32, 16, 16>, !pto.tile_buf<f32, 16, 16>)
         outs(%result : !pto.tile_buf<f32, 16, 16>)
```

## 相关页面 / 指令集链接


- 指令集总览：[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)
- 上一条指令：（无）
- 下一条指令：[pto.tabs](./tabs_zh.md)
- 英文规范页：[tadd.md](./tadd.md)

## Summary
本节给出该指令/主题的核心语义与使用定位，和英文章节保持一致。

## Mechanism
本节说明执行机制与关键语义规则，细节与边界条件以英文版为准。

## Syntax
本节列出语法形态（SSA / DPS / Assembly），用于与英文页逐项对照。

### Assembly Form (PTO-AS)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### AS Level 1 — SSA Form
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### AS Level 2 — DPS Form
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Micro-Operation Mapping
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

### Shape-Dependent Optimizations
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Layout Impact on Throughput
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Example Throughput Estimate
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Examples
本节提供 Auto/Manual 及 AS 形式示例，便于中英文对照复现。

### C++ — Auto Mode
本节给出 C++ 内建接口入口与参数语义说明。

### C++ — Manual Mode
本节给出 C++ 内建接口入口与参数语义说明。

### MLIR — SSA Form
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### MLIR — DPS Form
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Related Ops / Instruction Set Links
本节给出上下游指令与相关章节链接。
