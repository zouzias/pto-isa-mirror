# 内存操作（GM ↔ Tile）

本文档描述全局内存和 tile 之间的内存操作。

**操作总数：** 6

---

## 操作

### TLOAD

**数学解释：**

符号取决于 `GlobalTensor` 的形状/步幅和 `Tile` 的布局。从概念上讲（2D 视图，带基础偏移）：

 \mathrm{dst}_{i,j} = \mathrm{src}_{r_0 + i,\; c_0 + j}

**IR Level 1 (SSA)：**
```text
%dst = pto.tload %mem : !pto.partition_tensor_view<MxNxdtype> ->
!pto.tile<loc, dtype, rows, cols, blayout, slayout, fractal, pad>
```

**IR Level 2 (DPS)：**
```text
pto.tload ins(%mem : !pto.partition_tensor_view<MxNxdtype>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPREFETCH

**数学解释：**

除非另有说明，语义在有效区域上定义，目标相关的行为标记为实现定义。

**IR Level 1 (SSA)：**
```text
%dst = pto.tprefetch %src : !pto.global<...> -> !pto.tile<...>
```

**IR Level 2 (DPS)：**
```text
pto.tprefetch ins(%src : !pto.global<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSTORE

**数学解释：**

符号取决于 `GlobalTensor` 的形状/步幅和 `Tile` 的布局。从概念上讲（2D 视图，带基础偏移）：

 \mathrm{dst}_{r_0 + i,\; c_0 + j} = \mathrm{src}_{i,j}

**IR Level 1 (SSA)：**
```text
pto.tstore %src, %mem : (!pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

**IR Level 2 (DPS)：**
```text
pto.tstore ins(%src : !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

---

### TSTORE_FP

**数学解释：**

设 `R = src.GetValidRow()` 和 `C = src.GetValidCol()`。从概念上讲（2D 视图，带基础偏移），对于 `0 <= i < R` 和 `0 <= j < C`：

 \mathrm{dst}_{r_0 + i,\; c_0 + j} = \mathrm{Convert}\!\left(\mathrm{src}_{i,j};\ \mathrm{fp}\right)

**IR Level 1 (SSA)：**
```text
pto.tstore.fp %src, %fp, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

**IR Level 2 (DPS)：**
```text
pto.tstore.fp ins(%src, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

---

### MGATHER

**数学解释：**

对于目标有效区域中的每个元素 `(i, j)`：

 \mathrm{dst}_{i,j} = \mathrm{mem}[\mathrm{idx}_{i,j}]

**IR Level 1 (SSA)：**
```text
%dst = pto.mgather %mem, %idx : (!pto.partition_tensor_view<MxNxdtype>, pto.tile<...>)
-> !pto.tile<loc, dtype, rows, cols, blayout, slayout, fractal, pad>
```

**IR Level 2 (DPS)：**
```text
pto.mgather ins(%mem, %idx : !pto.partition_tensor_view<MxNxdtype>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### MSCATTER

**数学解释：**

对于源有效区域中的每个元素 `(i, j)`：

 \mathrm{mem}[\mathrm{idx}_{i,j}] = \mathrm{src}_{i,j} 

如果多个元素映射到同一目标位置，最终值是实现定义的（CPU 模拟器：按行主序迭代顺序的最后写入者获胜）。

**IR Level 1 (SSA)：**
```text
pto.mscatter %src, %idx, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

**IR Level 2 (DPS)：**
```text
pto.mscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```
