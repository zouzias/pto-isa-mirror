# Tile-标量 / Tile-立即数

本文档描述 tile 与标量值或立即常量之间的操作。

**操作总数：** 19

---

## 操作

### TEXPANDS

**数学解释：**

对于有效区域中的每个元素 `(i, j)`：

 \mathrm{dst}_{i,j} = \mathrm{scalar}

**IR Level 1 (SSA)：**
```text
%dst = pto.texpands %scalar : dtype -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.texpands ins(%scalar : dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TCMPS

**数学解释：**

对于有效区域中的每个元素 `(i, j)`：

 \mathrm{dst}_{i,j} = \left(\mathrm{src}_{i,j}\ \mathrm{cmpMode}\ \mathrm{scalar}\right) 

`dst` 的编码/类型是实现定义的（通常是类似掩码的 tile）。

**IR Level 1 (SSA)：**
```text
%dst = pto.tcmps %src, %scalar {cmpMode = #pto<cmp xx>} : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tcmps ins(%src, %scalar{cmpMode = #pto<cmp xx>}: !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TSELS

**数学解释：**

对于有效区域中的每个元素 `(i, j)`：


\mathrm{dst}_{i,j} =
\begin{cases}
\mathrm{src0}_{i,j} & \text{如果 } \mathrm{selectMode} = 1 \\
\mathrm{src1}_{i,j} & \text{否则}
\end{cases}

**IR Level 1 (SSA)：**
```text
%dst = pto.tsels %src0, %src1, %scalar : (!pto.tile<...>, !pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tsels ins(%src0, %src1, %scalar : !pto.tile_buf<...>, !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TMINS

**数学解释：**

对于有效区域中的每个元素 `(i, j)`：

 \mathrm{dst}_{i,j} = \min(\mathrm{src}_{i,j}, \mathrm{scalar})

**IR Level 1 (SSA)：**
```text
%dst = pto.tmins %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tmins ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TADDS

**数学解释：**

对于有效区域中的每个元素 `(i, j)`：

 \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} + \mathrm{scalar}

**IR Level 1 (SSA)：**
```text
%dst = pto.tadds %src, %scalar : (!pto.tile<...>,dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tadds ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TSUBS

**数学解释：**

对于有效区域中的每个元素 `(i, j)`：

 \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} - \mathrm{scalar}

**IR Level 1 (SSA)：**
```text
%dst = pto.tsubs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tsubs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TDIVS

**数学解释：**

对于有效区域中的每个元素 `(i, j)`：

- Tile/标量：

   \mathrm{dst}_{i,j} = \frac{\mathrm{src}_{i,j}}{\mathrm{scalar}} 

- 标量/Tile：

   \mathrm{dst}_{i,j} = \frac{\mathrm{scalar}}{\mathrm{src}_{i,j}}

**IR Level 1 (SSA)：**
```text
%dst = pto.tdivs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
%dst = pto.tdivs %scalar, %src : (dtype, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tdivs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
pto.tdivs ins(%scalar, %src : dtype, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TMULS

**数学解释：**

对于有效区域中的每个元素 `(i, j)`：

 \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \cdot \mathrm{scalar}

**IR Level 1 (SSA)：**
```text
%dst = pto.tmuls %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tmuls ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TFMODS

**数学解释：**

对于有效区域中的每个元素 `(i, j)`：

\mathrm{dst}_{i,j} = \mathrm{fmod}(\mathrm{src}_{i,j}, \mathrm{scalar})

**IR Level 1 (SSA)：**
```text
%dst = pto.tfmods %src, %scalar : !pto.tile<...>, f32
```

**IR Level 2 (DPS)：**
```text
pto.tfmods ins(%src, %scalar : !pto.tile_buf<...>, f32) outs(%dst : !pto.tile_buf<...>)
```

---

### TREMS

**数学解释：**

对于有效区域中的每个元素 `(i, j)`：

\mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \bmod \mathrm{scalar}

**IR Level 1 (SSA)：**
```text
%dst = pto.trems %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.trems ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TMAXS

**数学解释：**

对于有效区域中的每个元素 `(i, j)`：

 \mathrm{dst}_{i,j} = \max(\mathrm{src}_{i,j}, \mathrm{scalar})

**IR Level 1 (SSA)：**
```text
%dst = pto.tmaxs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tmaxs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TANDS

**数学解释：**

对于有效区域中的每个元素 `(i, j)`：

 \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \;\&\; \mathrm{scalar}

**IR Level 1 (SSA)：**
```text
%dst = pto.tands %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tands ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TORS

**数学解释：**

对于有效区域中的每个元素 `(i, j)`：

 \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \;|\; \mathrm{scalar}

**IR Level 1 (SSA)：**
```text
%dst = pto.tors %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tors ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TSHLS

**数学解释：**

对于有效区域中的每个元素 `(i, j)`：

 \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \ll \mathrm{scalar}

**IR Level 1 (SSA)：**
```text
%dst = pto.tshls %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tshls ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TSHRS

**数学解释：**

对于有效区域中的每个元素 `(i, j)`：

 \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \gg \mathrm{scalar}

**IR Level 1 (SSA)：**
```text
%dst = pto.tshrs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tshrs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TXORS

**数学解释：**

对于有效区域中的每个元素 `(i, j)`：

 \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \oplus \mathrm{scalar}

**IR Level 1 (SSA)：**
```text
%dst = pto.txors %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.txors ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TLRELU

**数学解释：**

对于有效区域中的每个元素 `(i, j)`：

 \mathrm{dst}_{i,j} = (\mathrm{src}_{i,j} > 0) ? \mathrm{src}_{i,j} : (\mathrm{src}_{i,j} \cdot \mathrm{slope})

**IR Level 1 (SSA)：**
```text
%dst = pto.tlrelu %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tlrelu ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TADDSC

**数学解释：**

对于有效区域中的每个元素 `(i, j)`：

 \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} + \mathrm{scalar} + \mathrm{src1}_{i,j}

**IR Level 1 (SSA)：**
```text
%dst = pto.taddsc %src0, %scalar, %src1 : (!pto.tile<...>, dtype, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.taddsc ins(%src0, %scalar, %src1 : !pto.tile_buf<...>, dtype, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSUBSC

**数学解释：**

对于有效区域中的每个元素 `(i, j)`：

 \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} - \mathrm{scalar} + \mathrm{src1}_{i,j}

**IR Level 1 (SSA)：**
```text
%dst = pto.tsubsc %src0, %scalar, %src1 : (!pto.tile<...>, dtype, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tsubsc ins(%src0, %scalar, %src1 : !pto.tile_buf<...>, dtype, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
