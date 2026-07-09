# TROWEXPANDEXPDIF

## 指令示意图

![TROWEXPANDEXPDIF tile operation](../figures/isa/TROWEXPANDEXPDIF.svg)

## 简介

行指数差运算：计算 exp(src0 - src1)，其中 src1 为每行标量。

## 数学语义

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。

### 模式 1

设 `s_i` 为从扩展操作数中获取的每行标量（每行一个值，ColMajor 布局）。

对于 `0 <= i < R` 和 `0 <= j < C`：

$$
\mathrm{dst}_{i,j} = \exp(\mathrm{src0}_{i,j} - s_i)
$$

### 模式 2

设 `b_i` 为从扩展操作数中获取的第 `i` 行的 32 字节块（RowMajor，每行 `32 / sizeof(T)` 个值）。该块在行内每 `32 / sizeof(T)` 个元素自然重复。

对于 `0 <= i < R` 和 `0 <= j < C`：

$$
\mathrm{dst}_{i,j} = \exp(\mathrm{src0}_{i,j} - b_i[\,j \bmod (32 / \mathit{sizeof}(T))\,])
$$

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDEXPDIF(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDEXPDIF(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp, WaitEvents &... events);
```

## 约束

- `TileDataDst::DType == TileDataSrc0::DType == TileDataSrc1::DType`
- `TileDataDst::DType`、`TileDataSrc0::DType`、`TileDataSrc1::DType` 必须是以下之一：`half`、`float`。
- Tile 形状/布局约束（编译时）：`TileDataDst::isRowMajor`。
- `src0` 或 `src1` 中必须恰好一个与 `dst` 的有效形状相同（即 `validRow == dst.validRow` 且 `validCol == dst.validCol`），该操作数为全尺寸操作数；另一个为**扩展操作数**（行广播源）。全尺寸操作数必须为 **RowMajor**（`isRowMajor == true`）。
- 模式 1：扩展操作数预期提供**每行一个标量**（ColMajor，`validCol == 1`）。
- 模式 2：扩展操作数预期提供**每行 32 字节数据**（RowMajor，`validCol == 32 / sizeof(T)`）。
- 确切的布局/分形约束是目标特定的；参见 `include/pto/npu/*/TRowExpand*.hpp` 下的后端头文件。

### 临时 Tile

C++ API 提供了显式传入 `TileDataTmp &tmp` 的重载。在 A2A3 上该重载仅支持**模式 1**（ColMajor 扩展操作数，每行标量）；在 A5 上它委托给 3 参数重载，支持两种模式。在 A2A3 上，`TROWEXPANDEXPDIF` 由 `TROWEXPANDSUB` 后接 `TEXP` 实现，因此 tmp Tile 用于 SUB 步骤的广播缓冲区。在 A5 上通过 `vexpdif`（`float`）或 `vsub` 后接 `vexp`（`half`）内联实现，tmp Tile 被忽略。

- **A2A3**：tmp Tile 作为 `TROWEXPANDSUB` 步骤的广播缓冲区使用。ColMajor 扩展操作数的每行标量值通过 `vbrcb` 指令广播到 tmp 缓冲区，为每行创建一个 32 字节块，然后在减法运算中作为扩展操作数使用。`vbrcb` 指令的 repeat stride 为 8 个块（256 字节），每个 repeat 处理 8 行。最小 tmp 大小计算：
    - **公共参数**：
        - `R = dst.GetValidRow()`，`T = TileDataDst::DType`。
    - 当 `R < 256` 时：
        $$ \text{tmpSize} = \left\lceil\frac{R}{8}\right\rceil \times 256 \text{ 字节} $$
    - 当 `R >= 256` 时：
        - 操作采用循环方式，每次循环最多 30 个 repeat（240 行）。tmp 缓冲区在各循环间复用，每次循环需要：
        $$ \text{tmpSize} = 30 \times 256 = 7680 \text{ 字节} $$
    - 对于任何模式 1 调用，一个紧凑的形状无关上界为 **8 KB**（8192 字节）。
    - 不带 `tmp` 的 3 参数重载支持模式 1 和模式 2。对于模式 1，使用内部 8 KB 缓冲区（`TMP_UB_OFFSET`）。对于模式 2，不需要广播缓冲区。
- **A5**：`tmp` Tile 被接受但不使用（`[[maybe_unused]]`）。A5 硬件通过 `vlds` 指令的广播模式原生支持行广播，因此不需要临时缓冲区。

## 示例

参见 `docs/isa/` 和 `docs/coding/tutorials/` 中的相关示例。
