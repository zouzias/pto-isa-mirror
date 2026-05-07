# pto.tload


`pto.tload` 属于[内存与数据搬运](../../memory-and-data-movement_zh.md)指令集。

## 概述


将全局内存中的矩形区域加载到 tile。传输范围由 `dst.GetValidRow()` 与 `dst.GetValidCol()` 决定。

## 机制


`pto.tload` 发起从 GlobalTensor 到目标 tile 的 DMA 传输。其映射关系可表示为：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{r_0+i,\;c_0+j} $$

其中 `(r_0, c_0)` 是源张量基址偏移，具体地址计算还受 stride 与布局影响。

该操作是异步的，返回 `RecordEvent`，消费目标 tile 前需 `TSYNC` 或等价同步。

## 语法


### PTO Assembly Form


```text
%t0 = tload %sv[%c0, %c0] : (!pto.memref<...>, index, index) -> !pto.tile<...>
```

### AS Level 1（SSA）


```text
%dst = pto.tload %mem : !pto.partition_tensor_view<MxNxdtype> ->
!pto.tile<loc, dtype, rows, cols, blayout, slayout, fractal, pad>
```

### AS Level 2（DPS）


```text
pto.tload ins(%mem : !pto.partition_tensor_view<MxNxdtype>)
          outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口


```cpp
template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TLOAD(TileData &dst, GlobalData &src, WaitEvents &... events);
```

## 输入


| 操作数 | 说明 |
|---|---|
| `dst` | 目标 tile，传输范围由其 valid 区域决定 |
| `src` | 源 GlobalTensor |
| `events...` | 可选等待事件 |

## 预期输出


| 结果 | 类型 | 说明 |
|---|---|---|
| `RecordEvent` | `RecordEvent` | 表示加载完成；消费前需同步 |

## 副作用


读取 GM、写入 tile buffer；不隐式栅栏其他不相关流量。

## 约束


!!! warning "约束"
    - 传输大小为 `dst.GetValidRow() × dst.GetValidCol()`。
    - `sizeof(tile.dtype) == sizeof(gtensor.dtype)`。
    - 布局组合必须在目标 profile 支持列表内。

## Layout 兼容性


| TileType | ND→ND | DN→DN | NZ→NZ | ND→NZ | DN→ZN |
|---|:---:|:---:|:---:|:---:|:---:|
| `Vec` | Yes | Yes | Yes | No | No |
| `Mat` | Yes | Yes | Yes | Yes | Yes |
| `Acc` | Yes | No | Yes | No | No |

## 异常


!!! danger "异常"
    - 非法操作数组合、类型不支持、布局不合法或目标不支持会被 verifier/后端拒绝。

## Target-Profile 限制


??? info "Target-Profile 限制"
    - A2/A3 与 A5 在布局、数据类型、特定格式（含 MX）支持上存在差异，详见英文页：`tload.md`。

## 示例


```cpp
RecordEvent e = TLOAD(t, gin);
TSYNC(e);
```

## 另见


- 指令集总览：[内存与数据搬运](../../memory-and-data-movement_zh.md)
- 下一条指令：`tprefetch`
- 英文规范页：[tload.md](./tload.md)

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

## Layout Compatibility
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Exceptions
本节描述非法输入、不支持组合与验证失败行为。

## Target-Profile Restrictions
本节给出 A2/A3、A5 及 CPU-SIM 的差异化限制与行为说明。

## Examples
本节提供 Auto/Manual 及 AS 形式示例，便于中英文对照复现。

## See Also
本节给出上下游指令与相关章节链接。
