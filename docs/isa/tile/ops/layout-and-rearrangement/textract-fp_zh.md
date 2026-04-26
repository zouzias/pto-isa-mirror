# TEXTRACT_FP

## 指令示意图

![TEXTRACT_FP tile operation](../../../../figures/isa/TEXTRACT_FP.svg)

## 简介

`TEXTRACT_FP` 是 `TEXTRACT` 的向量量化版本：它从一个较大的源 Tile 中抽取子块，并结合 `fp` Tile 提供的量化参数，把结果写到目标 Tile。

和 `TINSERT_FP` 一样，当前真实 backend 主要把它用于“从 `Acc` Tile 的某个窗口提取结果，并按量化规则输出到 `Mat` Tile”。

## 机制

若只看位置关系，它可以概念化为：

$$ \mathrm{dst}_{i,j} = \mathrm{Convert}\!\left(\mathrm{src}_{indexRow + i,\; indexCol + j};\ \mathrm{fp}\right) $$

这里的 `indexRow/indexCol` 决定从源 Tile 的哪个子窗口开始提取。对带 `fp` 的路径来说，`indexCol` 还会影响当前使用的量化参数切片；backend 会用它去偏移 `fp` 所指向的配置地址。

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../../../../assembly/PTO-AS_zh.md)。

### AS Level 1（SSA）

```text
%dst = pto.textract_fp %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.textract_fp ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow,
                                 uint16_t indexCol, WaitEvents &... events);
```

## 约束

### 通用约束

- `indexRow + dst.Rows` 不能超过 `src.Rows`。
- `indexCol + dst.Cols` 不能超过 `src.Cols`。
- `fp` 的设计意图是承载缩放/量化参数，可移植代码应把它建成 `TileType::Scaling`。

### A2/A3 实现

- 这条路径基于 `CheckTMovAccToMat(...)`，因此：
  - `src` 必须来自 `Acc`
  - `dst` 必须是 `Mat`
  - `dst` fractal size 必须是 `512`
  - `dst` 列宽字节数必须是 `32` 的倍数
- `FpTileData::Loc` 必须是 `TileType::Scaling`。
- backend 会按 `indexCol` 对 `fp` 地址做偏移，再设置 FPC。

### A5 实现

- A5 也把这条指令实现为 `Acc -> Mat` 的量化提取。
- 额外要求：
  - `dst` 必须是 `TileType::Mat`
  - `dst` 必须使用 `BLayout::ColMajor + SLayout::RowMajor`
  - `src` 必须是 `float` 或 `int32_t` 的 `Acc`
- `FpTileData::Loc` 必须是 `TileType::Scaling`。
- backend 同样会依据 `indexCol` 偏移 `fp` 地址。

### CPU 模拟器

- CPU 模拟器当前接受 `TEXTRACT_FP` 接口，但会忽略 `fp` 参数，退化为普通 `TEXTRACT`。
- 因此，依赖 `fp` 数值的量化行为应以 NPU backend 为准。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using SrcT = TileAcc<float, 32, 32>;
  using DstT = Tile<TileType::Mat, int8_t, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
  using FpT = Tile<TileType::Scaling, uint64_t, 1, 32>;

  SrcT src;
  DstT dst;
  FpT fp;
  TEXTRACT_FP(dst, src, fp, 0, 16);
}
```

## 相关页面

- [TEXTRACT](./textract_zh.md)
- [TINSERT_FP](./tinsert-fp_zh.md)
- [TMOV_FP](./tmov-fp_zh.md)
