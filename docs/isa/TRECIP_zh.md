# TRECIP

## 指令示意图

![TRECIP tile operation](../figures/isa/TRECIP.svg)

## 简介

Tile 的逐元素倒数。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$ \mathrm{dst}_{i,j} = \frac{1}{\mathrm{src}_{i,j}} $$

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <auto PrecisionType = RecipAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          typename... WaitEvents>
PTO_INST RecordEvent TRECIP(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

`PrecisionType`可指定以下值：

* `RecipAlgorithm::DEFAULT`：普通算法，速度快但精度较低。
* `RecipAlgorithm::HIGH_PRECISION`：高精度算法，速度较慢。

## 约束

- **实现检查 (NPU)**:
    - `TileData::DType` 必须是以下之一：`float` 或 `half`。
    - Tile 位置必须是向量（`TileData::Loc == TileType::Vec`);
    - 静态有效边界：`TileData::ValidRow <= TileData::Rows` 且 `TileData::ValidCol <= TileData::Cols`。
    - 运行时：`src.GetValidRow() == dst.GetValidRow()` 且 `src.GetValidCol() == dst.GetValidCol()`。
    - Tile 布局必须是行主序（`TileData::isRowMajor`）。
    - A3 的 TRECIP 指令不支持将源 Tile 和目标 Tile 设置为相同的内存。
- **有效区域**:
    - 该操作使用 `dst.GetValidRow()` / `dst.GetValidCol()` 作为迭代域。
- **域 / NaN**:
    - 除零行为由目标定义；CPU 模拟器在调试构建中会断言。
- **高精度算法**
    - 仅在A5上有效，`PrecisionType`选项在A3上将被忽略。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  TRECIP(out, x);
  TRECIP<RecipAlgorithm::HIGH_PRECISION>(out, x);
}
```
