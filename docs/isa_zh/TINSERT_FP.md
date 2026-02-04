# TINSERT_FP

## 简介

带 `fp`（scale）Tile 的 `TINSERT` 变体（向量量化/缩放参数，行为实现定义）。

## 计算流程图

![TINSERT_FP 计算流程图](figures/TINSERT_FP.svg)

## See also

- TINSERT 基本指令：`docs/isa/TINSERT.md`。

## C++ Intrinsic（内建接口）

在 `include/pto/common/pto_instr.hpp` 中声明：

```cpp
PTO_INST RecordEvent TINSERT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp,
                            uint16_t indexRow, uint16_t indexCol, WaitEvents&... events);
```