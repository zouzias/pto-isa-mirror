# Axis Reduce / Expand

This document describes row/column reduction and broadcast operations.

**Total Operations:** 23

---

## Operations

### TROWSUM

**Math Interpretation:**

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= i < R`:

 \mathrm{dst}_{i,0} = \sum_{j=0}^{C-1} \mathrm{src}_{i,j}

**IR Level 1 (SSA):**
```text
%dst = pto.trowsum %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.trowsum ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLSUM

**Math Interpretation:**

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= j < C`:

 \mathrm{dst}_{0,j} = \sum_{i=0}^{R-1} \mathrm{src}_{i,j} 

`isBinary` selects the implementation path (binary-tree accumulation vs. sequential accumulation).

**IR Level 1 (SSA):**
```text
%dst = pto.tcolsum %src : !pto.tile<...> -> !pto.tile<...>
%dst = pto.tcolsum %src, %tmp {isBinary = false} : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolsum ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
pto.tcolsum ins(%src, %tmp {isBinary = false} : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLPROD

**Math Interpretation:**

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= j < C`:

 \mathrm{dst}_{0,j} = \prod_{i=0}^{R-1} \mathrm{src}_{i,j}

**IR Level 1 (SSA):**
```text
%dst = pto.tcolprod %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolprod ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLMAX

**Math Interpretation:**

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= j < C`:

 \mathrm{dst}_{0,j} = \max_{0 \le i < R} \mathrm{src}_{i,j}

**IR Level 1 (SSA):**
```text
%dst = pto.tcolmax %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolmax ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWMAX

**Math Interpretation:**

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= i < R`:

 \mathrm{dst}_{i,0} = \max_{0 \le j < C} \mathrm{src}_{i,j}

**IR Level 1 (SSA):**
```text
%dst = pto.trowmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.trowmax ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWMIN

**Math Interpretation:**

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= i < R`:

 \mathrm{dst}_{i,0} = \min_{0 \le j < C} \mathrm{src}_{i,j}

**IR Level 1 (SSA):**
```text
%dst = pto.trowmin %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.trowmin ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPAND

**Math Interpretation:**

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. For `0 <= i < R` and `0 <= j < C`:

 \mathrm{dst}_{i,j} = \mathrm{src}_{i,0}

**IR Level 1 (SSA):**
```text
%dst = pto.trowexpand %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.trowexpand ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDDIV

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \frac{\mathrm{src0}_{i,j}}{\mathrm{src1}_{0,i}}

**IR Level 1 (SSA):**
```text
%dst = pto.tcolexpanddiv %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolexpanddiv ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDMUL

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \cdot \mathrm{src1}_{0,i}

**IR Level 1 (SSA):**
```text
%dst = pto.tcolexpandmul %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolexpandmul ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDSUB

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} - \mathrm{src1}_{0,i}

**IR Level 1 (SSA):**
```text
%dst = pto.tcolexpandsub %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolexpandsub ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDADD

**Math Interpretation:**

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Let `s_i` be the per-row scalar taken from `src1` (one value per row).

For `0 <= i < R` and `0 <= j < C`:


\mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} + s_i

**IR Level 1 (SSA):**
```text
%dst = pto.trowexpandadd %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.trowexpandadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDMAX

**Math Interpretation:**

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Let `s_i` be the per-row scalar taken from `src1` (one value per row).

For `0 <= i < R` and `0 <= j < C`:


\mathrm{dst}_{i,j} = \max(\mathrm{src0}_{i,j}, s_i)

**IR Level 1 (SSA):**
```text
%dst = pto.trowexpandmax %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.trowexpandmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDMIN

**Math Interpretation:**

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Let `s_i` be the per-row scalar taken from `src1` (one value per row).

For `0 <= i < R` and `0 <= j < C`:


\mathrm{dst}_{i,j} = \min(\mathrm{src0}_{i,j}, s_i)

**IR Level 1 (SSA):**
```text
%dst = pto.trowexpandmin %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.trowexpandmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TROWEXPANDEXPDIF

**Math Interpretation:**

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Let `s_i` be the per-row scalar taken from `src1` (one value per row).

For `0 <= i < R` and `0 <= j < C`:


\mathrm{dst}_{i,j} = \exp(\mathrm{src0}_{i,j} - s_i)

**IR Level 1 (SSA):**
```text
%dst = pto.trowexpandexpdif %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.trowexpandexpdif ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLMIN

**Math Interpretation:**

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= j < C`:

 \mathrm{dst}_{0,j} = \min_{0 \le i < R} \mathrm{src}_{i,j}

**IR Level 1 (SSA):**
```text
%dst = pto.tcolmin %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolmin ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPAND

**Math Interpretation:**

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. For `0 <= i < R` and `0 <= j < C`:

 \mathrm{dst}_{i,j} = \mathrm{src}_{0,j}

**IR Level 1 (SSA):**
```text
%dst = pto.tcolexpand %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolexpand ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDDIV

**Math Interpretation:**

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Let `s_j` be the per-column scalar taken from `src1` (one value per column).

For `0 <= i < R` and `0 <= j < C`:


\mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} / s_j

**IR Level 1 (SSA):**
```text
%dst = pto.tcolexpanddiv %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolexpanddiv ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDMUL

**Math Interpretation:**

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Let `s_j` be the per-column scalar taken from `src1` (one value per column).

For `0 <= i < R` and `0 <= j < C`:


\mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \cdot s_j

**IR Level 1 (SSA):**
```text
%dst = pto.tcolexpandmul %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolexpandmul ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDADD

**Math Interpretation:**

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Let `s_j` be the per-column scalar taken from `src1` (one value per column).

For `0 <= i < R` and `0 <= j < C`:


\mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} + s_j

**IR Level 1 (SSA):**
```text
%dst = pto.tcolexpandadd %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolexpandadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDMAX

**Math Interpretation:**

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Let `s_j` be the per-column scalar taken from `src1` (one value per column).

For `0 <= i < R` and `0 <= j < C`:


\mathrm{dst}_{i,j} = \max(\mathrm{src0}_{i,j}, s_j)

**IR Level 1 (SSA):**
```text
%dst = pto.tcolexpandmax %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolexpandmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDMIN

**Math Interpretation:**

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Let `s_j` be the per-column scalar taken from `src1` (one value per column).

For `0 <= i < R` and `0 <= j < C`:


\mathrm{dst}_{i,j} = \min(\mathrm{src0}_{i,j}, s_j)

**IR Level 1 (SSA):**
```text
%dst = pto.tcolexpandmin %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolexpandmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDSUB

**Math Interpretation:**

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Let `s_j` be the per-column scalar taken from `src1` (one value per column).

For `0 <= i < R` and `0 <= j < C`:


\mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} - s_j

**IR Level 1 (SSA):**
```text
%dst = pto.tcolexpandsub %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolexpandsub ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCOLEXPANDEXPDIF

**Math Interpretation:**

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Let `s_j` be the per-column scalar taken from `src1` (one value per column).

For `0 <= i < R` and `0 <= j < C`:


\mathrm{dst}_{i,j} = \exp(\mathrm{src0}_{i,j} - s_j)

**IR Level 1 (SSA):**
```text
%dst = pto.tcolexpandexpdif %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcolexpandexpdif ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---


