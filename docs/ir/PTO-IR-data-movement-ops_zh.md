# 数据移动/布局

本文档描述数据移动和布局转换操作。

**操作总数：** 12

---

## 操作

### TEXTRACT

**数学解释：**

从概念上讲，从 `src` 中复制从 `(indexRow, indexCol)` 开始的窗口到 `dst`。确切的映射取决于布局。

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。对于 `0 <= i < R` 和 `0 <= j < C`：

 \mathrm{dst}_{i,j} = \mathrm{src}_{\mathrm{indexRow}+i,\; \mathrm{indexCol}+j}

**IR Level 1 (SSA)：**
```text
%dst = pto.textract %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.textract ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TEXTRACT_FP

**数学解释：**

除非另有说明，语义在有效区域上定义，目标相关的行为标记为实现定义。

**IR Level 1 (SSA)：**
```text
%dst = pto.textract_fp %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.textract_fp ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TIMG2COL

**数学解释：**

除非另有说明，语义在有效区域上定义，目标相关的行为标记为实现定义。

**IR Level 1 (SSA)：**
```text
%dst = pto.timg2col %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.timg2col ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TINSERT

**数学解释：**

设 `R = src.GetValidRow()` 和 `C = src.GetValidCol()`。从概念上讲，对于 `0 <= i < R` 和 `0 <= j < C`：


\mathrm{dst}_{\mathrm{indexRow}+i,\;\mathrm{indexCol}+j} = \mathrm{src}_{i,j}

**IR Level 1 (SSA)：**
```text
%dst = pto.tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tinsert ins(%src[%r0, %r1] : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TINSERT_FP

**数学解释：**

除非另有说明，语义在有效区域上定义，目标相关的行为标记为实现定义。

**IR Level 1 (SSA)：**
```text
%dst = pto.tinsert_fp %src, %fp, %idxrow, %idxcol : (!pto.tile<...>, !pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tinsert_fp ins(%src, %fp, %idxrow, %idxcol : !pto.tile_buf<...>, !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TFILLPAD

**数学解释：**

设 `VR = src.GetValidRow()` 和 `VC = src.GetValidCol()`。对于每个目标元素 `(i, j)`：


\mathrm{dst}_{i,j} =
\begin{cases}
\mathrm{src}_{i,j} & \text{如果 } i < VR \text{ 且 } j < VC \\
\mathrm{pad}       & \text{否则}
\end{cases}


`pad` 由 `TileDataDst::PadVal` 和元素类型确定（例如，对于浮点类型，当可用时为 `+inf/-inf`，
否则为 `std::numeric_limits<T>::max()/min()`）。

**IR Level 1 (SSA)：**
```text
%dst = pto.tfillpad %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tfillpad ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TFILLPAD_INPLACE

**数学解释：**

除非另有说明，语义在有效区域上定义，目标相关的行为标记为实现定义。

**IR Level 1 (SSA)：**
```text
%dst = pto.tfillpad_inplace %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tfillpad_inplace ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TFILLPAD_EXPAND

**数学解释：**

除非另有说明，语义在有效区域上定义，目标相关的行为标记为实现定义。

**IR Level 1 (SSA)：**
```text
%dst = pto.tfillpad_expand %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tfillpad_expand ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TMOV

**数学解释：**

从概念上讲，在有效区域上将元素从 `src` 复制或转换到 `dst`。确切的转换取决于所选模式和目标。

对于纯复制情况：

 \mathrm{dst}_{i,j} = \mathrm{src}_{i,j}

**IR Level 1 (SSA)：**
```text
%dst = pto.tmov.s2d %src  : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tmov ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TMOV_FP

**数学解释：**

从概念上讲，使用从 `fp` 派生的实现定义的量化/反量化配置转换每个元素：

 \mathrm{dst}_{i,j} = \mathrm{Convert}\!\left(\mathrm{src}_{i,j};\ \mathrm{fp}\right)

**IR Level 1 (SSA)：**
```text
%dst = pto.tmov.fp %src, %fp : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tmov.fp ins(%src, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TRESHAPE

**数学解释：**

除非另有说明，语义在有效区域上定义，目标相关的行为标记为实现定义。

**IR Level 1 (SSA)：**
```text
%dst = pto.treshape %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.treshape ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TTRANS

**数学解释：**

对于 2D tile，在有效转置域上：

 \mathrm{dst}_{i,j} = \mathrm{src}_{j,i} 

确切的形状/布局和转置域取决于目标（参见约束）。

**IR Level 1 (SSA)：**
```text
%dst = pto.ttrans %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.ttrans ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
