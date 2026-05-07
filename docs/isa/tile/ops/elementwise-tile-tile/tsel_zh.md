# pto.tsel


`pto.tsel` 属于[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)指令集。

## 概述


基于掩码按元素在两个源 tile 之间做条件选择。

## 机制


对目标有效域中的每个 `(i,j)`：

$$
\mathrm{dst}_{i,j}=
\begin{cases}
\mathrm{src0}_{i,j}, & \text{mask}_{i,j}\neq 0 \\
\mathrm{src1}_{i,j}, & \text{otherwise}
\end{cases}
$$

掩码 tile 使用目标定义的压缩编码；`tmp` 作为谓词解包的临时工作 tile。

## 语法


同步形式：

```text
%dst = tsel %mask, %src0, %src1 : !pto.tile<...>
```

### AS Level 1（SSA）


```text
%dst = pto.tsel %mask, %src0, %src1 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）


```text
pto.tsel ins(%mask, %src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口


```cpp
template <typename TileData, typename MaskTile, typename TmpTile, typename... WaitEvents>
PTO_INST RecordEvent TSEL(TileData &dst, MaskTile &selMask, TileData &src0,
                          TileData &src1, TmpTile &tmp, WaitEvents &... events);
```

## 输入


| 操作数 | 角色 | 说明 |
|---|---|---|
| `%dst` | 目标 tile | 接收选择结果 |
| `%mask` | 谓词掩码 | 非零选 `src0`，否则选 `src1` |
| `%src0` | true 分支 | 掩码为真时取值 |
| `%src1` | false 分支 | 掩码为假时取值 |
| `%tmp` | 临时 tile | 谓词解包工作区 |

## 预期输出


| 结果 | 类型 | 说明 |
|---|---|---|
| `%dst` | `!pto.tile<...>` | 有效域内写入按掩码选择结果 |

## 副作用


除写入目标 tile 外，无额外架构副作用。

## 约束


!!! warning "约束"
    - `sizeof(TileData::DType)` 必须是 2 或 4 字节。
    - `dst/src0/src1` 必须同类型、同形状、RowMajor。
    - `tmp` 容量需满足目标谓词展开需求。

## 不允许的情况


!!! danger "不允许的情况"
    - 非 RowMajor 的 `dst/src0/src1`。
    - `dst/src0/src1` 声明形状不一致。

## Target-Profile 限制


??? info "Target-Profile 限制"
    - A2/A3 与 A5 支持类型、布局要求及 `tmp` 约束请参考英文页：`tsel.md`。

## 性能


### A2/A3 吞吐


`TSEL` 走向量二元路径时，其吞吐模型与 `TADD` 同量级，具体常量请参考英文页。

## 示例


### Auto


```cpp
TSEL(dst, mask, src0, src1, tmp);
```

### Manual


```cpp
TASSIGN(src0, 0x1000);
TASSIGN(src1, 0x2000);
TASSIGN(dst,  0x3000);
TASSIGN(mask, 0x4000);
TASSIGN(tmp,  0x5000);
TSEL(dst, mask, src0, src1, tmp);
```

### PTO Assembly Form


```text
%dst = tsel %mask, %src0, %src1 : !pto.tile<...>
pto.tsel ins(%mask, %src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## 相关页面 / 指令集链接


- 指令集总览：[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)
- 上一条指令：[pto.tcvt](./tcvt_zh.md)
- 下一条指令：[pto.trsqrt](./trsqrt_zh.md)
- 英文规范页：[tsel.md](./tsel.md)

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

## Cases That Are Not Allowed
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Target-Profile Restrictions
本节给出 A2/A3、A5 及 CPU-SIM 的差异化限制与行为说明。

## Performance
本节说明性能路径、吞吐估算与形状/布局敏感因素。

### A2/A3 Throughput
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Examples
本节提供 Auto/Manual 及 AS 形式示例，便于中英文对照复现。

## Related Ops / Instruction Set Links
本节给出上下游指令与相关章节链接。
