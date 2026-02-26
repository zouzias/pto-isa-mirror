# 轴归约/扩展操作

本文档描述行/列归约和广播操作。

**操作总数：** 23

---

## 操作

### TROWSUM

**数学解释：**

设 `R = src.GetValidRow()` 和 `C = src.GetValidCol()`。对于 `0 <= i < R`：

 \mathrm{dst}_{i,0} = \sum_{j=0}^{C-1} \mathrm{src}_{i,j}

**IR Level 1 (SSA)：**
```text
%dst = pto.trowsum %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.trowsum ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLSUM

**数学解释：**

设 `R = src.GetValidRow()` 和 `C = src.GetValidCol()`。对于 `0 <= j < C`：

 \mathrm{dst}_{0,j} = \sum_{i=0}^{R-1} \mathrm{src}_{i,j} 

`isBinary` 选择实现路径（二叉树累加 vs. 顺序累加）。

**IR Level 1 (SSA)：**
```text
%dst = pto.tcolsum %src : !pto.tile<...> -> !pto.tile<...>
%dst = pto.tcolsum %src, %tmp {isBinary = false} : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolsum ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
pto.tcolsum ins(%src, %tmp {isBinary = false} : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLPROD

**数学解释：**

设 `R = src.GetValidRow()` 和 `C = src.GetValidCol()`。对于 `0 <= j < C`：

 \mathrm{dst}_{0,j} = \prod_{i=0}^{R-1} \mathrm{src}_{i,j}

**IR Level 1 (SSA)：**
```text
%dst = pto.tcolprod %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolprod ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLMAX

**数学解释：**

设 `R = src.GetValidRow()` 和 `C = src.GetValidCol()`。对于 `0 <= j < C`：

 \mathrm{dst}_{0,j} = \max_{0 \le i < R} \mathrm{src}_{i,j}

**IR Level 1 (SSA)：**
```text
%dst = pto.tcolmax %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolmax ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWMAX

**数学解释：**

设 `R = src.GetValidRow()` 和 `C = src.GetValidCol()`。对于 `0 <= i < R`：

 \mathrm{dst}_{i,0} = \max_{0 \le j < C} \mathrm{src}_{i,j}

**IR Level 1 (SSA)：**
```text
%dst = pto.trowmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.trowmax ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWMIN

**数学解释：**

设 `R = src.GetValidRow()` 和 `C = src.GetValidCol()`。对于 `0 <= i < R`：

 \mathrm{dst}_{i,0} = \min_{0 \le j < C} \mathrm{src}_{i,j}

**IR Level 1 (SSA)：**
```text
%dst = pto.trowmin %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.trowmin ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPAND

**数学解释：**

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。对于 `0 <= i < R` 和 `0 <= j < C`：

 \mathrm{dst}_{i,j} = \mathrm{src}_{i,0}

**IR Level 1 (SSA)：**
```text
%dst = pto.trowexpand %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.trowexpand ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDDIV

**数学解释：**

对于有效区域中的每个元素 `(i, j)`：

 \mathrm{dst}_{i,j} = \frac{\mathrm{src0}_{i,j}}{\mathrm{src1}_{0,i}}

**IR Level 1 (SSA)：**
```text
%dst = pto.tcolexpanddiv %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolexpanddiv ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDMUL

**数学解释：**

对于有效区域中的每个元素 `(i, j)`：

 \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \cdot \mathrm{src1}_{0,i}

**IR Level 1 (SSA)：**
```text
%dst = pto.tcolexpandmul %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolexpandmul ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDSUB

**数学解释：**

对于有效区域中的每个元素 `(i, j)`：

 \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} - \mathrm{src1}_{0,i}

**IR Level 1 (SSA)：**
```text
%dst = pto.tcolexpandsub %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolexpandsub ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDADD

**数学解释：**

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。设 `s_i` 为从 `src1` 获取的每行标量（每行一个值）。

对于 `0 <= i < R` 和 `0 <= j < C`：


\mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} + s_i

**IR Level 1 (SSA)：**
```text
%dst = pto.trowexpandadd %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.trowexpandadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDMAX

**数学解释：**

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。设 `s_i` 为从 `src1` 获取的每行标量（每行一个值）。

对于 `0 <= i < R` 和 `0 <= j < C`：


\mathrm{dst}_{i,j} = \max(\mathrm{src0}_{i,j}, s_i)

**IR Level 1 (SSA)：**
```text
%dst = pto.trowexpandmax %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.trowexpandmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDMIN

**数学解释：**

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。设 `s_i` 为从 `src1` 获取的每行标量（每行一个值）。

对于 `0 <= i < R` 和 `0 <= j < C`：


\mathrm{dst}_{i,j} = \min(\mathrm{src0}_{i,j}, s_i)

**IR Level 1 (SSA)：**
```text
%dst = pto.trowexpandmin %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.trowexpandmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDEXPDIF

**数学解释：**

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。设 `s_i` 为从 `src1` 获取的每行标量（每行一个值）。

对于 `0 <= i < R` 和 `0 <= j < C`：


\mathrm{dst}_{i,j} = \exp(\mathrm{src0}_{i,j} - s_i)

**IR Level 1 (SSA)：**
```text
%dst = pto.trowexpandexpdif %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.trowexpandexpdif ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLMIN

**数学解释：**

设 `R = src.GetValidRow()` 和 `C = src.GetValidCol()`。对于 `0 <= j < C`：

 \mathrm{dst}_{0,j} = \min_{0 \le i < R} \mathrm{src}_{i,j}

**IR Level 1 (SSA)：**
```text
%dst = pto.tcolmin %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolmin ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPAND

**数学解释：**

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。对于 `0 <= i < R` 和 `0 <= j < C`：

 \mathrm{dst}_{i,j} = \mathrm{src}_{0,j}

**IR Level 1 (SSA)：**
```text
%dst = pto.tcolexpand %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolexpand ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDDIV

**数学解释：**

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。设 `s_j` 为从 `src1` 获取的每列标量（每列一个值）。

对于 `0 <= i < R` 和 `0 <= j < C`：


\mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} / s_j

**IR Level 1 (SSA)：**
```text
%dst = pto.tcolexpanddiv %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolexpanddiv ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDMUL

**数学解释：**

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。设 `s_j` 为从 `src1` 获取的每列标量（每列一个值）。

对于 `0 <= i < R` 和 `0 <= j < C`：


\mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \cdot s_j

**IR Level 1 (SSA)：**
```text
%dst = pto.tcolexpandmul %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolexpandmul ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDADD

**数学解释：**

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。设 `s_j` 为从 `src1` 获取的每列标量（每列一个值）。

对于 `0 <= i < R` 和 `0 <= j < C`：


\mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} + s_j

**IR Level 1 (SSA)：**
```text
%dst = pto.tcolexpandadd %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolexpandadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDMAX

**数学解释：**

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。设 `s_j` 为从 `src1` 获取的每列标量（每列一个值）。

对于 `0 <= i < R` 和 `0 <= j < C`：


\mathrm{dst}_{i,j} = \max(\mathrm{src0}_{i,j}, s_j)

**IR Level 1 (SSA)：**
```text
%dst = pto.tcolexpandmax %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolexpandmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDMIN

**数学解释：**

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。设 `s_j` 为从 `src1` 获取的每列标量（每列一个值）。

对于 `0 <= i < R` 和 `0 <= j < C`：


\mathrm{dst}_{i,j} = \min(\mathrm{src0}_{i,j}, s_j)

**IR Level 1 (SSA)：**
```text
%dst = pto.tcolexpandmin %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolexpandmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDSUB

**数学解释：**

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。设 `s_j` 为从 `src1` 获取的每列标量（每列一个值）。

对于 `0 <= i < R` 和 `0 <= j < C`：


\mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} - s_j

**IR Level 1 (SSA)：**
```text
%dst = pto.tcolexpandsub %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolexpandsub ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDEXPDIF

**数学解释：**

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。设 `s_j` 为从 `src1` 获取的每列标量（每列一个值）。

对于 `0 <= i < R` 和 `0 <= j < C`：


\mathrm{dst}_{i,j} = \exp(\mathrm{src0}_{i,j} - s_j)

**IR Level 1 (SSA)：**
```text
%dst = pto.tcolexpandexpdif %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcolexpandexpdif ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
