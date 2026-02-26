# Data Movement / Layout

This document describes data movement and layout transformation operations.

**Total Operations:** 12

---

## Operations

### TEXTRACT

**Math Interpretation:**

Conceptually copies a window starting at `(indexRow, indexCol)` from `src` into `dst`. Exact mapping depends on layouts.

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. For `0 <= i < R` and `0 <= j < C`:

 \mathrm{dst}_{i,j} = \mathrm{src}_{\mathrm{indexRow}+i,\; \mathrm{indexCol}+j}

**IR Level 1 (SSA):**
```text
%dst = pto.textract %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.textract ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TEXTRACT_FP

**Math Interpretation:**

Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

**IR Level 1 (SSA):**
```text
%dst = pto.textract_fp %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.textract_fp ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TIMG2COL

**Math Interpretation:**

Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

**IR Level 1 (SSA):**
```text
%dst = pto.timg2col %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.timg2col ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TINSERT

**Math Interpretation:**

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. Conceptually, for `0 <= i < R` and `0 <= j < C`:


\mathrm{dst}_{\mathrm{indexRow}+i,\;\mathrm{indexCol}+j} = \mathrm{src}_{i,j}

**IR Level 1 (SSA):**
```text
%dst = pto.tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tinsert ins(%src[%r0, %r1] : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TINSERT_FP

**Math Interpretation:**

Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

**IR Level 1 (SSA):**
```text
%dst = pto.tinsert_fp %src, %fp, %idxrow, %idxcol : (!pto.tile<...>, !pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tinsert_fp ins(%src, %fp, %idxrow, %idxcol : !pto.tile_buf<...>, !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TFILLPAD

**Math Interpretation:**

Let `VR = src.GetValidRow()` and `VC = src.GetValidCol()`. For each destination element `(i, j)`:


\mathrm{dst}_{i,j} =
\begin{cases}
\mathrm{src}_{i,j} & \text{if } i < VR \text{ and } j < VC \\
\mathrm{pad}       & \text{otherwise}
\end{cases}


`pad` is determined by `TileDataDst::PadVal` and the element type (e.g., `+inf/-inf` for floating types when available,
otherwise `std::numeric_limits<T>::max()/min()`).

**IR Level 1 (SSA):**
```text
%dst = pto.tfillpad %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tfillpad ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TFILLPAD_INPLACE

**Math Interpretation:**

Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

**IR Level 1 (SSA):**
```text
%dst = pto.tfillpad_inplace %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tfillpad_inplace ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TFILLPAD_EXPAND

**Math Interpretation:**

Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

**IR Level 1 (SSA):**
```text
%dst = pto.tfillpad_expand %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tfillpad_expand ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TMOV

**Math Interpretation:**

Conceptually copies or transforms elements from `src` into `dst` over the valid region. Exact transformation depends on the selected mode and target.

For the pure copy case:

 \mathrm{dst}_{i,j} = \mathrm{src}_{i,j}

**IR Level 1 (SSA):**
```text
%dst = pto.tmov.s2d %src  : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tmov ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TMOV_FP

**Math Interpretation:**

Conceptually converts each element using an implementation-defined quantization/dequantization configuration derived from `fp`:

 \mathrm{dst}_{i,j} = \mathrm{Convert}\!\left(\mathrm{src}_{i,j};\ \mathrm{fp}\right)

**IR Level 1 (SSA):**
```text
%dst = pto.tmov.fp %src, %fp : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tmov.fp ins(%src, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TRESHAPE

**Math Interpretation:**

Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

**IR Level 1 (SSA):**
```text
%dst = pto.treshape %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.treshape ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TTRANS

**Math Interpretation:**

For a 2D tile, over the effective transpose domain:

 \mathrm{dst}_{i,j} = \mathrm{src}_{j,i} 

Exact shape/layout and the transpose domain depend on the target (see Constraints).

**IR Level 1 (SSA):**
```text
%dst = pto.ttrans %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.ttrans ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---


