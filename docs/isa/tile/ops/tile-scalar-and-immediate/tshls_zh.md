# pto.tshls

## 概要

Tile 按标量逐元素左移。

## 数学语义

对目标 tile 有效区域中的每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \ll \mathrm{scalar} $$

## 汇编语法

同步形式：

```text
%dst = pto.tshls %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSHLS(TileDataDst &dst, TileDataSrc &src, typename TileDataDst::DType scalar,
                           WaitEvents &... events);
```

## 约束

!!! warning "约束"
    - `dst` 与 `src` 必须使用相同元素类型。
    - `dst` 与 `src` 必须是向量 tile。
    - 操作迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
    - 标量仅支持零和正值。

## Target-Profile 限制

### A2A3

- 支持的元素类型为 `int32_t`、`int`、`int16_t`、`uint32_t`、`unsigned int` 和 `uint16_t`
- 运行时要求：`src.GetValidRow() == dst.GetValidRow()` 且 `src.GetValidCol() == dst.GetValidCol()`

### A5

- 支持的元素类型为 `int32_t`、`int16_t`、`int8_t`、`uint32_t`、`uint16_t` 和 `uint8_t`
- 两个 tile 的静态有效边界都必须满足 `ValidRow <= Rows` 且 `ValidCol <= Cols`
- 运行时要求：`src.GetValidRow() == dst.GetValidRow()` 且 `src.GetValidCol() == dst.GetValidCol()`

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
    using TileT = Tile<TileType::Vec, uint16_t, 16, 16>;
    TileT dst, src;
    TSHLS(dst, src, 0x2);
}
```
