# pto.tfmods

`pto.tfmods` 属于[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)指令集。

## 概要

对 tile 和标量逐元素执行 `fmod`。

## 数学语义

对目标 tile 有效区域中的每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{fmod}(\mathrm{src}_{i,j}, \mathrm{scalar}) $$

标量会被广播到整个 valid region。

## 语法

同步形式：

```text
%dst = pto.tfmods %src, %scalar : (!pto.tile<...>, f32) -> !pto.tile<...>
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TFMODS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar,
                            WaitEvents &... events);
```

## 约束

!!! warning "约束"
    - 除零行为由目标平台定义；CPU 模拟器在调试构建下会断言。
    - 操作迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。

## Target-Profile 限制

### A2A3

- `dst` 与 `src` 必须使用相同元素类型
- 支持元素类型：`float`、`float32_t`
- `dst` 与 `src` 必须是向量 tile 且为行主序
- 运行时要求：`dst.GetValidRow() == src.GetValidRow() > 0` 且 `dst.GetValidCol() == src.GetValidCol() > 0`

### A5

- `dst` 与 `src` 必须使用相同元素类型
- 支持元素类型是目标实现支持的 2 字节或 4 字节类型（包括 `half`、`float`）
- `dst` 与 `src` 必须是向量 tile
- 静态 valid 边界必须合法
- 运行时要求：`dst.GetValidRow() == src.GetValidRow()` 且 `dst.GetValidCol() == src.GetValidCol()`

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
    using TileT = Tile<TileType::Vec, float, 16, 16>;
    TileT x, out;
    TFMODS(out, x, 3.0f);
}
```
