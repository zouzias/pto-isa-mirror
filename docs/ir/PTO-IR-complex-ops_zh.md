# 复杂操作

本文档描述复杂操作，包括排序、聚集和量化。

**操作总数：** 13

---

## 操作

### TPRINT

**数学解释：**

除非另有说明，语义在有效区域上定义，目标相关的行为标记为实现定义。

**IR Level 1 (SSA)：**
```text
pto.tprint %src : !pto.tile<...> | !pto.partition_tensor_view<MxNxdtype> -> ()
```

**IR Level 2 (DPS)：**
```text
pto.tprint ins(%src : !pto.tile_buf<...> | !pto.partition_tensor_view<MxNxdtype>)
```

---

### TMRGSORT

**数学解释：**

将排序的输入列表合并到 `dst` 中。排序顺序、元素格式（例如，值/索引对）和执行计数的含义取决于实现。

 \mathrm{dst} = \mathrm{merge}(\mathrm{src}_0, \mathrm{src}_1, \ldots)

**IR Level 1 (SSA)：**
```text
%dst = pto.tmrgsort %src, %blockLen : (!pto.tile<...>, dtype) -> !pto.tile<...>
%dst, %executed = pto.tmrgsort %src0, %src1, %src2, %src3 {exhausted = false}
 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> (!pto.tile<...>, vector<4xi16>)
```

**IR Level 2 (DPS)：**
```text
pto.tmrgsort ins(%src, %blockLen : !pto.tile_buf<...>, dtype)  outs(%dst : !pto.tile_buf<...>)
pto.tmrgsort ins(%src0, %src1, %src2, %src3 {exhausted = false} : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
outs(%dst, %executed : !pto.tile_buf<...>, vector<4xi16>)
```

---

### TSORT32

**数学解释：**

将 `src` 中的值排序到 `dst` 中，并在 `idx` 中生成索引映射。从概念上讲，对于每一行 `i`：

 \mathrm{dst}_{i,k} = \mathrm{src}_{i,\pi_i(k)} 

其中 $\pi_i$ 是行中索引的排列。排序顺序和稳定性由目标定义。

**IR Level 1 (SSA)：**
```text
%dst, %idx = pto.tsort32 %src : !pto.tile<...> -> (!pto.tile<...>, !pto.tile<...>)
```

**IR Level 2 (DPS)：**
```text
pto.tsort32 ins(%src : !pto.tile_buf<...>) outs(%dst, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>)
```

---

### TGATHER

**数学解释：**

基于索引的聚集（概念性）：

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。对于 `0 <= i < R` 和 `0 <= j < C`：

 \mathrm{dst}_{i,j} = \mathrm{src0}\!\left[\mathrm{indices}_{i,j}\right] 

确切的索引解释和边界行为是实现定义的。

掩码模式聚集是由 `pto::MaskPattern` 控制的实现定义的选择/归约。

**IR Level 1 (SSA)：**
```text
%dst = pto.tgather %src, %indices : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
%dst = pto.tgather %src {maskPattern = #pto.mask_pattern<P0101>}: !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tgather ins(%src, %indices : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
pto.tgather ins(%src, {maskPattern = #pto.mask_pattern<P0101>} : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCI

**数学解释：**

对于有效元素上的线性化索引 `k`：

- 升序：

   \mathrm{dst}_{k} = S + k 

- 降序：

   \mathrm{dst}_{k} = S - k 

线性化顺序取决于 tile 布局（实现定义）。

**IR Level 1 (SSA)：**
```text
%dst = pto.tci %scalar {descending = false} : dtype -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tci ins(%scalar {descending = false} : dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TTRI

**数学解释：**

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。设 `d = diagonal`。

下三角（`isUpperOrLower=0`）从概念上产生：


\mathrm{dst}_{i,j} = \begin{cases}1 & j \le i + d \\ 0 & \text{otherwise}\end{cases}


上三角（`isUpperOrLower=1`）从概念上产生：


\mathrm{dst}_{i,j} = \begin{cases}0 & j < i + d \\ 1 & \text{otherwise}\end{cases}

**IR Level 1 (SSA)：**
```text
%dst = pto.ttri %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.ttri ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPARTADD

**数学解释：**

对于目标有效区域中的每个元素 `(i, j)`：


\mathrm{dst}_{i,j} =
\begin{cases}
\mathrm{src0}_{i,j} + \mathrm{src1}_{i,j} & \text{如果两个输入在 } (i,j) \text{ 处都定义} \\
\mathrm{src0}_{i,j} & \text{如果只有 src0 在 } (i,j) \text{ 处定义} \\
\mathrm{src1}_{i,j} & \text{如果只有 src1 在 } (i,j) \text{ 处定义}
\end{cases}

**IR Level 1 (SSA)：**
```text
%dst = pto.tpartadd %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tpartadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPARTMUL

**数学解释：**

对于目标有效区域中的每个元素 `(i, j)`：


\mathrm{dst}_{i,j} =
\begin{cases}
\mathrm{src0}_{i,j} \cdot \mathrm{src1}_{i,j} & \text{如果两个输入在 } (i,j) \text{ 处都定义} \\
\mathrm{src0}_{i,j} & \text{如果只有 src0 在 } (i,j) \text{ 处定义} \\
\mathrm{src1}_{i,j} & \text{如果只有 src1 在 } (i,j) \text{ 处定义}
\end{cases}

**IR Level 1 (SSA)：**
```text
%dst = pto.tpartmul %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tpartmul ins(%src0, %src1 : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPARTMAX

**数学解释：**

对于目标有效区域中的每个元素 `(i, j)`：


\mathrm{dst}_{i,j} =
\begin{cases}
\max(\mathrm{src0}_{i,j}, \mathrm{src1}_{i,j}) & \text{如果两个输入在 } (i,j) \text{ 处都定义} \\
\mathrm{src0}_{i,j} & \text{如果只有 src0 在 } (i,j) \text{ 处定义} \\
\mathrm{src1}_{i,j} & \text{如果只有 src1 在 } (i,j) \text{ 处定义}
\end{cases}

**IR Level 1 (SSA)：**
```text
%dst = pto.tpartmax %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tpartmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPARTMIN

**数学解释：**

对于目标有效区域中的每个元素 `(i, j)`：


\mathrm{dst}_{i,j} =
\begin{cases}
\min(\mathrm{src0}_{i,j}, \mathrm{src1}_{i,j}) & \text{如果两个输入在 } (i,j) \text{ 处都定义} \\
\mathrm{src0}_{i,j} & \text{如果只有 src0 在 } (i,j) \text{ 处定义} \\
\mathrm{src1}_{i,j} & \text{如果只有 src1 在 } (i,j) \text{ 处定义}
\end{cases}

**IR Level 1 (SSA)：**
```text
%dst = pto.tpartmin %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tpartmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TGATHERB

**数学解释：**

对于有效区域中的每个元素：

 \mathrm{dst}_{i,j} = *\left(\mathrm{srcBase} + \mathrm{offset}_{i,j}\right) 

确切的边界行为是实现定义的。

**IR Level 1 (SSA)：**
```text
%dst = pto.tgatherb %src, %offsets : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tgatherb ins(%src, %offsets : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSCATTER

**数学解释：**

对于每个源元素 `(i, j)`，写入：

 \mathrm{dst}_{\mathrm{idx}_{i,j},\ j} = \mathrm{src}_{i,j} 

如果多个元素映射到同一目标位置，最终值是实现定义的（当前实现中最后写入者获胜）。

**IR Level 1 (SSA)：**
```text
%dst = pto.tscatter %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TQUANT

**数学解释：**

除非另有说明，语义在有效区域上定义，目标相关的行为标记为实现定义。

**IR Level 1 (SSA)：**
```text
%dst = pto.tquant %src, %qp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tquant ins(%src, %qp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
