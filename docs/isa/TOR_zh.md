# TOR

## 指令示意图

![TOR tile operation](../figures/isa/TOR.svg)

## 简介

两个 Tile 的逐元素按位或。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \;|\; \mathrm{src1}_{i,j} $$

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TOR(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## 约束

- **实现检查 (A2A3)**:
    - 支持的元素类型为 `uint8_t`、`int8_t`、`uint16_t`、`int16_t`、`uint32_t` 和 `int32_t`。
    - `dst`、`src0` 和 `src1` 必须使用相同的元素类型。
    - `dst`、`src0` 和 `src1` 必须是行主序。
    - 运行时：`src0.GetValidRow()/GetValidCol()` 和 `src1.GetValidRow()/GetValidCol()` 必须与 `dst` 一致。
- **实现检查 (A5)**:
    - 支持的元素类型为 `uint8_t`、`int8_t`、`uint16_t`、`int16_t`、`uint32_t` 和 `int32_t`。
    - `dst`、`src0` 和 `src1` 必须使用相同的元素类型。
    - `dst`、`src0` 和 `src1` 必须是行主序。
    - 运行时：`src0.GetValidRow()/GetValidCol()` 和 `src1.GetValidRow()/GetValidCol()` 必须与 `dst` 一致。
- **有效区域**:
    - 该操作使用 `dst.GetValidRow()` / `dst.GetValidCol()` 作为迭代域。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, int32_t, 16, 16>;
  TileT a, b, out;
  TOR(out, a, b);
}
```

