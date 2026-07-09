# TSQRT

## 指令示意图

![TSQRT tile operation](../figures/isa/TSQRT.svg)

## 简介

逐元素平方根。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$ \mathrm{dst}_{i,j} = \sqrt{\mathrm{src}_{i,j}} $$

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <auto PrecisionType = SqrtAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          typename... WaitEvents>
PTO_INST RecordEvent TSQRT(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

`PrecisionType` 可选值：

* `SqrtAlgorithm::DEFAULT`：普通算法，速度较快但精度较低。
* `SqrtAlgorithm::HIGH_PRECISION`：高精度算法，速度较慢但精度更高。

## 约束

- **实现检查 (NPU)**:
    - `TileData::DType` 必须是以下之一：`float` 或 `half`。
    - Tile 位置必须是向量（`TileData::Loc == TileType::Vec`);
    - 静态有效边界：`TileData::ValidRow <= TileData::Rows` 且 `TileData::ValidCol <= TileData::Cols`。
    - 运行时：`src.GetValidRow() == dst.GetValidRow()` 且 `src.GetValidCol() == dst.GetValidCol()`。
    - Tile 布局必须是行主序（`TileData::isRowMajor`）。
- **有效区域**:
    - 该操作使用 `dst.GetValidRow()` / `dst.GetValidCol()` 作为迭代域.
- **域 / NaN**:
    - 行为由目标定义（例如，对于负数输入）。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TSQRT(dst, src);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TSQRT(dst, src);
}
```

