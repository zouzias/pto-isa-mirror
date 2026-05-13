# pto.tsubs

`pto.tsubs` 属于[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)指令集。

## 概要

把一个标量逐元素从 tile 上减去。

## 数学语义

对目标 tile 有效区域中的每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} - \mathrm{scalar} $$

标量会被广播到整个 valid region。

## 语法

同步形式：

```text
%dst = pto.tsubs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSUBS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar,
                           WaitEvents &... events);
```

## 约束

!!! warning "约束"
    - `dst` 与 `src0` 必须使用相同元素类型。
    - 标量类型必须匹配 `TileDataSrc::DType`。
    - 操作迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。

## Target-Profile 限制

### A2A3

- 数据类型必须属于：`int32_t`、`int`、`int16_t`、`half`、`float16_t`、`float`、`float32_t`
- tile 位置必须是向量 tile
- 运行时要求：`src0.GetValidRow() == dst.GetValidRow()` 且 `src0.GetValidCol() == dst.GetValidCol()`

### A5

- 数据类型必须属于：`int32_t`、`int`、`int16_t`、`half`、`float16_t`、`float`、`float32_t`
- tile 位置必须是向量 tile
- 静态 valid 边界必须合法
- 运行时要求：`src0.GetValidRow() == dst.GetValidRow()` 且 `src0.GetValidCol() == dst.GetValidCol()`

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
    using TileT = Tile<TileType::Vec, float, 16, 16>;
    TileT x, out;
    TSUBS(out, x, 1.0f);
}
```
