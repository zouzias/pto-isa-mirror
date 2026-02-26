# Memory (GM 鈫?Tile)

This document describes memory operations between global memory and tiles.

**Total Operations:** 6

---

## Operations

### TLOAD

**Math Interpretation:**

Notation depends on the `GlobalTensor` shape/stride and the `Tile` layout. Conceptually (2D view, with a base offset):

 \mathrm{dst}_{i,j} = \mathrm{src}_{r_0 + i,\; c_0 + j}

**IR Level 1 (SSA):**
```text
%dst = pto.tload %mem : !pto.partition_tensor_view<MxNxdtype> ->
!pto.tile<loc, dtype, rows, cols, blayout, slayout, fractal, pad>
```

**IR Level 2 (DPS):**
```text
pto.tload ins(%mem : !pto.partition_tensor_view<MxNxdtype>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPREFETCH

**Math Interpretation:**

Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

**IR Level 1 (SSA):**
```text
%dst = pto.tprefetch %src : !pto.global<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tprefetch ins(%src : !pto.global<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSTORE

**Math Interpretation:**

Notation depends on the `GlobalTensor` shape/stride and the `Tile` layout. Conceptually (2D view, with a base offset):

 \mathrm{dst}_{r_0 + i,\; c_0 + j} = \mathrm{src}_{i,j}

**IR Level 1 (SSA):**
```text
pto.tstore %src, %mem : (!pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

**IR Level 2 (DPS):**
```text
pto.tstore ins(%src : !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

---

### TSTORE_FP

**Math Interpretation:**

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. Conceptually (2D view, with a base offset), for `0 <= i < R` and `0 <= j < C`:

 \mathrm{dst}_{r_0 + i,\; c_0 + j} = \mathrm{Convert}\!\left(\mathrm{src}_{i,j};\ \mathrm{fp}\right)

**IR Level 1 (SSA):**
```text
pto.tstore.fp %src, %fp, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

**IR Level 2 (DPS):**
```text
pto.tstore.fp ins(%src, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

---

### MGATHER

**Math Interpretation:**

For each element `(i, j)` in the destination valid region:

 \mathrm{dst}_{i,j} = \mathrm{mem}[\mathrm{idx}_{i,j}]

**IR Level 1 (SSA):**
```text
%dst = pto.mgather %mem, %idx : (!pto.partition_tensor_view<MxNxdtype>, pto.tile<...>)
-> !pto.tile<loc, dtype, rows, cols, blayout, slayout, fractal, pad>
```

**IR Level 2 (DPS):**
```text
pto.mgather ins(%mem, %idx : !pto.partition_tensor_view<MxNxdtype>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### MSCATTER

**Math Interpretation:**

For each element `(i, j)` in the source valid region:

 \mathrm{mem}[\mathrm{idx}_{i,j}] = \mathrm{src}_{i,j} 

If multiple elements map to the same destination location, the final value is implementation-defined (CPU simulator: last writer wins in row-major iteration order).

**IR Level 1 (SSA):**
```text
pto.mscatter %src, %idx, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

**IR Level 2 (DPS):**
```text
pto.mscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

---


