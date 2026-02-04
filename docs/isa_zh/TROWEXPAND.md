# TROWEXPAND

## 简介

按行广播：将每行的 `src[i,0]` 广播到该行所有列，写入 `dst`。

## 计算流程图

![TROWEXPAND 计算流程图](figures/TROWEXPAND.svg)

## 数学解释

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。对于 `0 <= i < R` 和 `0 <= j < C`：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,0} $$

## 汇编语法

PTO-AS 形式：参见 `docs/grammar/PTO-AS.md`。

同步形式：

```text
%dst = trowexpand %src : !pto.tile<...> -> !pto.tile<...>
```
## C++ Intrinsic（内建接口）

在 `include/pto/common/pto_instr.hpp` 中声明：

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPAND(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events);
```

## 约束

实现检查（NPU）：

- Tile 类型：`dst` 和 `src` 必须为 `TileType::Vec`。
- Tile 布局：`src` 和 `dst` 的 ND 分形（`isRowMajor` 和 `SLayout::NoneBox`）。
- 数据类型：A2A3/A5 元素类型必须是以下之一：`int8_t` 或 `uint8_t` 或 `int16_t` 或 `uint16_t` 或 `int32_t` 或 `uint32_t` 或 `half` 或 `bfloat16_t` 或 `float`。
- 运行时有效检查：
  - A2A3：如果 `dstValidRow`、`dstValidCol`、`srcValidRow`、`srcValidCol` 中的任何一个为零，则提前返回。
  - A5：断言 `srcValidRow == dstValidRow` 并断言 `srcValidRow != 0 && srcValidCol != 0`。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TROWEXPAND(dst, src);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TROWEXPAND(dst, src);
}
```