# TEXTRACT_FP

## 简介

带 `fp`（scale）Tile 的 `TEXTRACT` 变体（向量量化/缩放参数，行为实现定义）。

## 计算流程图

![TEXTRACT_FP 计算流程图](figures/TEXTRACT_FP.svg)

## See also

- TEXTRACT 基本指令：`docs/isa/TEXTRACT.md`。

## C++ Intrinsic（内建接口）

在 `include/pto/common/pto_instr.hpp` 中声明：

```cpp
PTO_INST RecordEvent TEXTRACT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp,
                            uint16_t indexRow, uint16_t indexCol, WaitEvents&... events);
```