# TMULADDDST

## 指令示意图

![TMULADDDST tile operation](../figures/isa/TMULADDDST.svg)

## 简介

三元逐元素运算：`src0 * src1 + dst`。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \* \mathrm{src1}_{i,j} + \mathrm{dst}_{i,j} $$

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TMULADDDST(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events);
```

## 约束

- **实现检查**:
    - `TileData::DType` 必须是以下之一：`half`、`float`。
    - Tile 布局必须是行主序（`TileData::isRowMajor`）。
- **通用约束**:
    - Tile 位置必须是向量（`TileData::Loc == TileType::Vec`）。
    - 静态有效边界：`TileData::ValidRow <= TileData::Rows` 且 `TileData::ValidCol <= TileData::Cols`。
    - 运行时：`dst`、`src0` 和 `src1` 的有效行列数必须相同。
    - 标量类型必须与 Tile 数据类型一致。
- 该操作在 `dst.GetValidRow()` / `dst.GetValidCol()` 上迭代。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT a, b, out;
  TMULADDDST(out, a, b);
}
```
