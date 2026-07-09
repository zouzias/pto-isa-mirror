# TCOLEXPAND

## 指令示意图

![TCOLEXPAND tile operation](../figures/isa/TCOLEXPAND.svg)

## 简介

将每个源列的第一个元素广播到目标列中。

## 数学语义

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。对于 `0 <= i < R` 和 `0 <= j < C`：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{0,j} $$

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPAND(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

## 约束

- Tile 类型：`dst` 和 `src` 必须为 `TileType::Vec`。
- Tile 布局：`src` 和 `dst` 均须使用 ND 布局（行主且非分形：`isRowMajor` 且 `SLayout::NoneBox`）。
- 数据类型：元素大小必须为 1、2 或 4 字节；`dst` 和 `src` 的元素类型必须一致。
- 运行期检查：
    - A2A3：若 `dst.GetValidRow()`、`dst.GetValidCol()`、`src.GetValidRow()`、`src.GetValidCol()` 中任意一个为零则提前返回。
    - A5：断言 `srcValidRow != 0 && srcValidCol != 0`。
    - 两个后端均断言 `src.GetValidCol() == dst.GetValidCol()`。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TCOLEXPAND(dst, src);
}
```
