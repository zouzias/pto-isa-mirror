# pto.tmuls

`pto.tmuls` 属于[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)指令集。

## 概要

把一个标量逐元素乘到 tile 上。

## 数学语义

对目标 tile 有效区域中的每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \cdot \mathrm{scalar} $$

标量会被广播到整个 valid region。

## 语法

同步形式：

```text
%dst = pto.tmuls %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TMULS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar,
                           WaitEvents &... events);
```

## 约束

!!! warning "约束"
    - 操作迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。

## Target-Profile 限制

### A2A3

- `TileData::DType` 必须属于：`int32_t`、`int`、`int16_t`、`half`、`float16_t`、`float`、`float32_t`
- tile 位置必须是向量 tile
- 静态 valid 边界必须合法
- 运行时要求：`src0.GetValidRow() == dst.GetValidRow()` 且 `src0.GetValidCol() == dst.GetValidCol()`
- tile 布局必须是行主序

### A5

- `TileData::DType` 必须属于：`uint8_t`、`int8_t`、`uint16_t`、`int16_t`、`uint32_t`、`int32_t`、`half`、`float`、`bfloat16_t`
- tile 位置必须是向量 tile
- 静态 valid 边界必须合法
- 运行时要求：`src0.GetValidCol() == dst.GetValidCol()`
- tile 布局必须是行主序

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
    using TileT = Tile<TileType::Vec, float, 16, 16>;
    TileT src, dst;
    TMULS(dst, src, 2.0f);
}
```
