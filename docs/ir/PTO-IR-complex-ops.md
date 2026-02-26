# Complex

This document describes complex operations including sorting, gathering, and quantization.

**Total Operations:** 13

---

## Operations

### TPRINT

**Math Interpretation:**

Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

**IR Level 1 (SSA):**
```text
pto.tprint %src : !pto.tile<...> | !pto.partition_tensor_view<MxNxdtype> -> ()
```

**IR Level 2 (DPS):**
```text
pto.tprint ins(%src : !pto.tile_buf<...> | !pto.partition_tensor_view<MxNxdtype>)
```

---

### TMRGSORT

**Math Interpretation:**

Merges sorted input lists into `dst`. Ordering, element format (e.g., value/index pairs), and the meaning of executed counts depend on the implementation.

 \mathrm{dst} = \mathrm{merge}(\mathrm{src}_0, \mathrm{src}_1, \ldots)

**IR Level 1 (SSA):**
```text
%dst = pto.tmrgsort %src, %blockLen : (!pto.tile<...>, dtype) -> !pto.tile<...>
%dst, %executed = pto.tmrgsort %src0, %src1, %src2, %src3 {exhausted = false}
 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> (!pto.tile<...>, vector<4xi16>)
```

**IR Level 2 (DPS):**
```text
pto.tmrgsort ins(%src, %blockLen : !pto.tile_buf<...>, dtype)  outs(%dst : !pto.tile_buf<...>)
pto.tmrgsort ins(%src0, %src1, %src2, %src3 {exhausted = false} : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
outs(%dst, %executed : !pto.tile_buf<...>, vector<4xi16>)
```

---

### TSORT32

**Math Interpretation:**

Sorts values from `src` into `dst` and produces an index mapping in `idx`. Conceptually, for each row `i`:

 \mathrm{dst}_{i,k} = \mathrm{src}_{i,\pi_i(k)} 

where $\pi_i$ is a permutation of the indices in the row. Sort order and stability are target-defined.

**IR Level 1 (SSA):**
```text
%dst, %idx = pto.tsort32 %src : !pto.tile<...> -> (!pto.tile<...>, !pto.tile<...>)
```

**IR Level 2 (DPS):**
```text
pto.tsort32 ins(%src : !pto.tile_buf<...>) outs(%dst, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>)
```

---

### TGATHER

**Math Interpretation:**

Index-based gather (conceptual):

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. For `0 <= i < R` and `0 <= j < C`:

 \mathrm{dst}_{i,j} = \mathrm{src0}\!\left[\mathrm{indices}_{i,j}\right] 

Exact index interpretation and bounds behavior are implementation-defined.

Mask-pattern gather is an implementation-defined selection/reduction controlled by `pto::MaskPattern`.

**IR Level 1 (SSA):**
```text
%dst = pto.tgather %src, %indices : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
%dst = pto.tgather %src {maskPattern = #pto.mask_pattern<P0101>}: !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tgather ins(%src, %indices : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
pto.tgather ins(%src, {maskPattern = #pto.mask_pattern<P0101>} : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCI

**Math Interpretation:**

For a linearized index `k` over the valid elements:

- Ascending:

   \mathrm{dst}_{k} = S + k 

- Descending:

   \mathrm{dst}_{k} = S - k 

The linearization order depends on the tile layout (implementation-defined).

**IR Level 1 (SSA):**
```text
%dst = pto.tci %scalar {descending = false} : dtype -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tci ins(%scalar {descending = false} : dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TTRI

**Math Interpretation:**

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Let `d = diagonal`.

Lower-triangular (`isUpperOrLower=0`) conceptually produces:


\mathrm{dst}_{i,j} = egin{cases}1 & j \le i + d \ 0 & 	ext{otherwise}\end{cases}


Upper-triangular (`isUpperOrLower=1`) conceptually produces:


\mathrm{dst}_{i,j} = egin{cases}0 & j < i + d \ 1 & 	ext{otherwise}\end{cases}

**IR Level 1 (SSA):**
```text
%dst = pto.ttri %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.ttri ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPARTADD

**Math Interpretation:**

For each element `(i, j)` in the destination valid region:


\mathrm{dst}_{i,j} =
\begin{cases}
\mathrm{src0}_{i,j} + \mathrm{src1}_{i,j} & \text{if both inputs are defined at } (i,j) \\
\mathrm{src0}_{i,j} & \text{if only src0 is defined at } (i,j) \\
\mathrm{src1}_{i,j} & \text{if only src1 is defined at } (i,j)
\end{cases}

**IR Level 1 (SSA):**
```text
%dst = pto.tpartadd %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tpartadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPARTMUL

**Math Interpretation:**

For each element `(i, j)` in the destination valid region:


\mathrm{dst}_{i,j} =
egin{cases}
\mathrm{src0}_{i,j} \cdot \mathrm{src1}_{i,j} & 	ext{if both inputs are defined at } (i,j) \
\mathrm{src0}_{i,j} & 	ext{if only src0 is defined at } (i,j) \
\mathrm{src1}_{i,j} & 	ext{if only src1 is defined at } (i,j)
\end{cases}

**IR Level 1 (SSA):**
```text
%dst = pto.tpartmul %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tpartmul ins(%src0, %src1 : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPARTMAX

**Math Interpretation:**

For each element `(i, j)` in the destination valid region:


\mathrm{dst}_{i,j} =
\begin{cases}
\max(\mathrm{src0}_{i,j}, \mathrm{src1}_{i,j}) & \text{if both inputs are defined at } (i,j) \\
\mathrm{src0}_{i,j} & \text{if only src0 is defined at } (i,j) \\
\mathrm{src1}_{i,j} & \text{if only src1 is defined at } (i,j)
\end{cases}

**IR Level 1 (SSA):**
```text
%dst = pto.tpartmax %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tpartmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPARTMIN

**Math Interpretation:**

For each element `(i, j)` in the destination valid region:


\mathrm{dst}_{i,j} =
\begin{cases}
\min(\mathrm{src0}_{i,j}, \mathrm{src1}_{i,j}) & \text{if both inputs are defined at } (i,j) \\
\mathrm{src0}_{i,j} & \text{if only src0 is defined at } (i,j) \\
\mathrm{src1}_{i,j} & \text{if only src1 is defined at } (i,j)
\end{cases}

**IR Level 1 (SSA):**
```text
%dst = pto.tpartmin %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tpartmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TGATHERB

**Math Interpretation:**

For each element in the valid region:

 \mathrm{dst}_{i,j} = *\left(\mathrm{srcBase} + \mathrm{offset}_{i,j}\right) 

Exact bounds behavior is implementation-defined.

**IR Level 1 (SSA):**
```text
%dst = pto.tgatherb %src, %offsets : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tgatherb ins(%src, %offsets : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSCATTER

**Math Interpretation:**

For each source element `(i, j)`, write:

 \mathrm{dst}_{\mathrm{idx}_{i,j},\ j} = \mathrm{src}_{i,j} 

If multiple elements map to the same destination location, the final value is implementation-defined (last writer wins in the current implementation).

**IR Level 1 (SSA):**
```text
%dst = pto.tscatter %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TQUANT

**Math Interpretation:**

Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

**IR Level 1 (SSA):**
```text
%dst = pto.tquant %src, %qp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tquant ins(%src, %qp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---


