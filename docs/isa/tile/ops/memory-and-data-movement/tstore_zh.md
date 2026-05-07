# pto.tstore


`pto.tstore` 属于[内存与数据搬运](../../memory-and-data-movement_zh.md)指令集。

## 概述


把 tile 中的数据写回 `GlobalTensor`（GM）。普通写回、原子写回以及当前实现中面向 `TileType::Acc` 的量化写回都在这一类接口里。

## 机制


若用带基址偏移的二维视角表示，可写成：

$$ \mathrm{dst}_{r_0 + i,\; c_0 + j} = \mathrm{src}_{i,j} $$

真正的写回范围由源 tile 的 valid region 决定。

## 语法


同步形式：

```text
tstore %t1, %sv_out[%c0, %c0]
```

### AS Level 1（SSA）


```text
pto.tstore %src, %mem : (!pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### AS Level 2（DPS）


```text
pto.tstore ins(%src : !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

## C++ 内建接口


```cpp
template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, WaitEvents &... events);

template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, uint64_t preQuantScalar, WaitEvents &... events);

template <typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE_FP(GlobalData &dst, TileData &src, FpTileData &fp, WaitEvents &... events);
```

## 输入


- `src`：源 tile
- `dst`：目标 `GlobalTensor`
- `atomicType`：可选原子写回模式
- `preQuantScalar` / `fp`：当前实现中仅对 `TileType::Acc` 合法的量化 / fix-pipe 路径附加参数

## 预期输出


- 数据从 `src` 写回 `dst`
- 若使用原子模式，则在 GM 侧执行累加或比较类原子写回
- 若使用 `TSTORE_FP`，则通过 fix-pipe sideband state 路径写回

## 副作用


这条指令会写 GM。原子模式下，并发写入的结果还会依赖实现和内存一致性规则。

## 约束


!!! warning "约束"
    - 写回范围由 `src.GetValidRow()` / `src.GetValidCol()` 决定
    - 目标 `GlobalTensor` 的 shape / stride 必须允许这次写回

## Target-Profile 限制


### A2A3


- 源 tile 位置必须是 `TileType::Vec`、`TileType::Mat` 或 `TileType::Acc`
- 运行时要求：所有 `dst.GetShape(dim)` 和 `src.GetValidRow()/GetValidCol()` 都大于 0
- 对 `Vec/Mat`：
  - 支持类型：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`int64_t`、`uint64_t`、`half`、`bfloat16_t`、`float`
  - `sizeof(TileData::DType) == sizeof(GlobalData::DType)`
  - 布局必须匹配 ND / DN / NZ，或满足单行 / 单列特殊情形
  - `int64_t/uint64_t` 只支持 `ND→ND` 与 `DN→DN`
- 对 `TileType::Acc`：
  - 目标布局必须为 ND 或 NZ
  - 源 dtype 必须为 `int32_t` 或 `float`
  - 不量化时，目标 dtype 必须为 `__gm__ int32_t/float/half/bfloat16_t`
  - shape 受 `Cols <= 4095`、`Rows` 上限等约束

### A5


- 源 tile 位置必须为 `TileType::Vec` 或 `TileType::Acc`
- `Vec` 路径支持更宽的元素类型集合，包括部分 float8 / packed float4 形式
- `Vec` 路径要求 `sizeof(TileData::DType) == sizeof(GlobalData::DType)`
- 布局需匹配 ND / DN / NZ，且额外存在行宽 / 列高字节对齐约束
- `Acc` 路径与 A2A3 类似，但 `AtomicAdd` 还会进一步限制目标 dtype

## 异常与非法情形


!!! danger "异常与非法情形"
    - 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## 性能


当前仓内没有把 `tstore` 单列成完整公开周期表，但在 A2A3 的搬运带宽模型里，Vec → Vec / GM 路径通常按带宽估算。

## 示例


```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

template <typename T>
void example_auto(__gm__ T* out) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gout(out);
  TileT t;
  TSTORE(gout, t);
}
```

## 相关页面


- 指令集总览：[内存与数据搬运](../../memory-and-data-movement_zh.md)
- [一致性基线](../../../memory-model/consistency-baseline_zh.md)

## Summary
本节给出该指令/主题的核心语义与使用定位，和英文章节保持一致。

## Mechanism
本节说明执行机制与关键语义规则，细节与边界条件以英文版为准。

### Fix-Pipe Variant (`TSTORE_FP`)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Variants
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Variant 1: Standard Store
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Variant 2: Fix-Pipe Store (`TSTORE_FP`)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Syntax
本节列出语法形态（SSA / DPS / Assembly），用于与英文页逐项对照。

### PTO Assembly Form
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### AS Level 1 (SSA)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### AS Level 2 (DPS)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Inputs
本节定义输入操作数角色、数据来源与有效区域要求。

## Expected Outputs
本节定义输出结果及其在有效区域内的语义保证。

## Side Effects
本节说明除结果写回外是否存在额外可观察副作用。

## Constraints
本节列出类型、布局、shape、valid-region 与 profile 相关约束。

## Target-Profile Restrictions
本节给出 A2/A3、A5 及 CPU-SIM 的差异化限制与行为说明。

## Exceptions
本节描述非法输入、不支持组合与验证失败行为。

## Common Patterns
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Pattern 1: Basic Vector Tile Store
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Pattern 2: Atomic Accumulation
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Pattern 3: Fix-Pipe Quantized Store (Production Inference)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Pattern 4: Manual Mode with TASSIGN
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## See Also
本节给出上下游指令与相关章节链接。
