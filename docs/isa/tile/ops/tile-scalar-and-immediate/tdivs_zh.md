# pto.tdivs

`pto.tdivs` 属于[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)指令集。

## 概要

带标量的逐元素除法，既支持 tile / scalar，也支持 scalar / tile 两种方向。

## 数学语义

对目标 tile 有效区域中的每个元素 `(i, j)`：

- tile / scalar 形式：

  $$ \mathrm{dst}_{i,j} = \frac{\mathrm{src}_{i,j}}{\mathrm{scalar}} $$

- scalar / tile 形式：

  $$ \mathrm{dst}_{i,j} = \frac{\mathrm{scalar}}{\mathrm{src}_{i,j}} $$

标量会被广播到整个 valid region。

## 语法

```text
%dst = pto.tdivs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
%dst = pto.tdivs %scalar, %src : (dtype, !pto.tile<...>) -> !pto.tile<...>
```

## C++ 内建接口

```cpp
template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          typename... WaitEvents>
PTO_INST RecordEvent TDIVS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar,
                           WaitEvents &... events);

template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          typename... WaitEvents>
PTO_INST RecordEvent TDIVS(TileDataDst &dst, typename TileDataDst::DType scalar, TileDataSrc &src0,
                           WaitEvents &... events);
```

`PrecisionType` 可取：

- `DivAlgorithm::DEFAULT`
- `DivAlgorithm::HIGH_PRECISION`

## 约束

!!! warning "约束"
    - 操作迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
    - 除零行为由目标 profile 定义。
    - `HIGH_PRECISION` 只在 A5 可用，A3 上该选项会被忽略。

## Target-Profile 限制

### A2A3

- 数据类型必须属于：`int32_t`、`int`、`int16_t`、`half`、`float16_t`、`float`、`float32_t`
- tile 位置必须是向量 tile
- 静态 valid 边界必须合法
- 运行时要求：`src0.GetValidRow() == dst.GetValidRow()` 且 `src0.GetValidCol() == dst.GetValidCol()`
- tile 布局必须是行主序

### A5

- 数据类型必须属于：`uint8_t`、`int8_t`、`uint16_t`、`int16_t`、`uint32_t`、`int32_t`、`half`、`float`
- tile 位置必须是向量 tile
- 静态 valid 边界必须合法
- 运行时要求：`src0.GetValidRow() == dst.GetValidRow()` 且 `src0.GetValidCol() == dst.GetValidCol()`
- tile 布局必须是行主序
- tile / scalar 形式在 A5 backend 上通常会映射到“乘以倒数”的实现路径，`scalar == 0` 的行为遵循目标浮点异常约定

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
    using TileT = Tile<TileType::Vec, float, 16, 16>;
    TileT src, dst;
    TDIVS(dst, src, 2.0f);
    TDIVS<DivAlgorithm::HIGH_PRECISION>(dst, src, 2.0f);
}
```
