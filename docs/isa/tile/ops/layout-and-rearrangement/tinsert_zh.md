# TINSERT

## 指令示意图


![TINSERT tile operation](../../../../figures/isa/TINSERT.svg)

## 简介


在 (indexRow, indexCol) 偏移处将子 Tile 插入到目标 Tile 中。

## 数学语义


设 `R = src.GetValidRow()` 和 `C = src.GetValidCol()`。概念上，对于 `0 <= i < R` 和 `0 <= j < C`：

$$
\mathrm{dst}_{\mathrm{indexRow}+i,\;\mathrm{indexCol}+j} = \mathrm{src}_{i,j}
$$

## 汇编语法


PTO-AS 形式：参见 [汇编写法与操作数](../../../syntax-and-operands/assembly-model_zh.md)。

同步形式：

```text
%dst = tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）


```text
%dst = pto.tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）


```text
pto.tinsert ins(%src[%r0, %r1] : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口


声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TINSERT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

#ifdef PTO_NPU_ARCH_A5
template <TInsertMode mode, typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src, uint32_t indexRow = 0, uint32_t indexCol = 0, WaitEvents &... events);
#endif
```

## 约束


!!! warning "约束"
    - **A2/A3**:
        - 文档中列出的这些重载对应 `Acc -> Mat` 插入路径，包括普通形式、`reluMode` 形式、标量预量化形式以及向量预量化（`TINSERT_FP`）形式。
        - 运行时边界必须满足 `indexRow + src.Rows <= dst.Rows` 且 `indexCol + src.Cols <= dst.Cols`。
    - **A5**:
        - 除了上面的 `Acc -> Mat` 插入路径外，A5 还额外提供 `template <TInsertMode mode, ...> TINSERT(...)`，用于 `Vec -> Mat` 与 `Vec -> Vec` 插入变体。
        - `mode == TInsertMode::ND` 要求源向量 tile 为行优先，并以 ND 布局插入到矩阵 tile。
        - `mode == TInsertMode::ND_VEC` 要求源和目的都为行优先向量 tile。
        - NZ 系列模式（`NZ`、`NZ_PLUS_1`、`SPLIT2_NZ_PLUS_1`、`SPLIT4_NZ_PLUS_1`）要求源向量 tile 为 NZ 格式，目的为矩阵 tile。

## 示例


参见 `docs/isa/` 和 `docs/coding/tutorials/` 中的相关示例。

## 汇编示例（ASM）


### 自动模式


```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

### 手动模式


```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

### PTO 汇编形式


```text
%dst = tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tinsert ins(%src[%r0, %r1] : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

# pto.tinsert
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Summary
本节给出该指令/主题的核心语义与使用定位，和英文章节保持一致。

## Mechanism
本节说明执行机制与关键语义规则，细节与边界条件以英文版为准。

## Variants
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Variant 1: Standard Insert
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Variant 2: ReLU Insert
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Variant 3: Scalar-Quant Insert
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Variant 4: Fix-Pipe Insert (`TINSERT_FP`)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Variant 5: A5 Mode-Specific Insert
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Supported Tile-Type Pairs
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### A2/A3
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### A5
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Supported Element Types
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Syntax
本节列出语法形态（SSA / DPS / Assembly），用于与英文页逐项对照。

### PTO Assembly Form
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### AS Level 1 (SSA)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## C++ Intrinsic
本节给出 C++ 内建接口入口与参数语义说明。

## Inputs
本节定义输入操作数角色、数据来源与有效区域要求。

## Constraints
本节列出类型、布局、shape、valid-region 与 profile 相关约束。

## Common Patterns
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Pattern 1: Accumulator Insert into Matrix
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Pattern 2: Fix-Pipe Quantized Insert
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Pattern 3: Accumulator Scatter via Staged Inserts
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## See Also
本节给出上下游指令与相关章节链接。
