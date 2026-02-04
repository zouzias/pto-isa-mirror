# TFILLPAD

## 简介

按有效区域将 `src` 拷贝到 `dst`，其余位置用编译期 `PadVal` 指定的 padding 值填充。

## 计算流程图

![TFILLPAD 计算流程图](figures/TFILLPAD.svg)

## 数学解释

设 `VR = src.GetValidRow()` 和 `VC = src.GetValidCol()`。对于每个目标元素 `(i, j)`：

$$
\mathrm{dst}_{i,j} =
\begin{cases}
\mathrm{src}_{i,j} & \text{if } i < VR \text{ and } j < VC \\
\mathrm{pad}       & \text{otherwise}
\end{cases}
$$

`pad` 由 `TileDataDst::PadVal` 和元素类型确定（例如，`+inf/-inf` 对于可用的浮点类型，
否则 `std::numeric_limits<T>::max()/min()`）。

## 汇编语法

PTO-AS 形式：参见 `docs/grammar/PTO-AS.md`。

同步形式（概念）：

```text
%dst = tfillpad %src : !pto.tile<...> -> !pto.tile<...>
```

## C++ Intrinsic（内建接口）

在 `include/pto/common/pto_instr_impl.hpp` 拉入的后端标头中实现：

```cpp
template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TFILLPAD(TileDataDst& dst, TileDataSrc& src);

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TFILLPAD_INPLACE(TileDataDst& dst, TileDataSrc& src);

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TFILLPAD_EXPAND(TileDataDst& dst, TileDataSrc& src);

template <typename TileData, PadValue PadVal = PadValue::Zero>
PTO_INTERNAL void TFILLPAD(TileData &dst, TileData &src);
```

## 约束

- `TileDataDst::PadVal != PadValue::Null`。
- `sizeof(TileDataDst::DType) == sizeof(TileDataSrc::DType)` 且元素大小必须为 `1`、`2` 或 `4` 字节。
- `TFILLPAD`：`TileDataDst::Rows/Cols` 必须匹配 `TileDataSrc::Rows/Cols`。
- `TFILLPAD_EXPAND`：`TileDataDst::Rows >= TileDataSrc::Rows` 和 `TileDataDst::Cols >= TileDataSrc::Cols`。
- `TFILLPAD(TileData &dst, TileData &src)`:`if TileData::TileType is Mat, layout only support (!TileData::isRowMajor && TileData::Slayout::RowMajor), and PadVal only support PadValue::Zero`

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example1() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::NoneBox, TileConfig::fractalABSize, PadValue::Min>;

  SrcT src;
  DstT dst;
  TFILLPAD(dst, src);
}

void example2() {
  using TileMatData = Tile<TileType::Mat, float, 16, 256, BLayout::ColMajor, 1, 224, SLayout::RowMajor, 512>;

  TileMatData matTile;
  TFILLPAD(matTile, matTile);
}
```