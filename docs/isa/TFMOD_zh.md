# TFMOD

## 指令示意图

![TFMOD tile operation](../figures/isa/TFMOD.svg)

## 简介

两个 Tile 的逐元素余数，余数符号与被除数相同。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$\mathrm{dst}_{i,j} = \mathrm{fmod}(\mathrm{src0}_{i,j}, \mathrm{src1}_{i,j})$$

## 汇编语法

同步形式：

```text
%dst = tfmod %src0, %src1 : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tfmod %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tfmod ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <auto PrecisionType = FmodAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0,
          typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TFMOD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

`PrecisionType` 可取以下值：

* `FmodAlgorithm::DEFAULT`：普通算法，速度较快但精度较低。
* `FmodAlgorithm::HIGH_PRECISION`：高精度算法，但速度较慢。

## 约束

- **实现检查 (A2A3)**:
    - `TileData::DType` 必须是 `float`。
    - Tile 布局必须是行主序（`TileData::isRowMajor`）。
    - 运行时：`src0`、`src1` 和 `dst` tiles 应具有相同的 `validRow/validCol`。
- **实现检查 (A5)**:
    - `TileData::DType` 必须是以下之一：`half`, `float`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`。
    - Tile 布局必须是行主序（`TileData::isRowMajor`）。
    - 运行时：`src0`、`src1` 和 `dst` tiles 应具有相同的 `validRow/validCol`。
- 该操作在 `dst.GetValidRow()` / `dst.GetValidCol()` 上迭代。
- 除零行为由目标定义；CPU 模拟器在调试构建中会断言。
- **高精度算法**:
    - 仅在 A5 上对 `float` 类型有效；`PrecisionType` 选项在 A2A3 上将被忽略。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, int32_t, 16, 16>;
  TileT out, a, b;
  TFMOD(out, a, b);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tfmod %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tfmod %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = tfmod %src0, %src1 : !pto.tile<...>
# AS Level 2 (DPS)
pto.tfmod ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
