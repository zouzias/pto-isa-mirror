# Tile-Scalar / Tile-Immediate

This document describes operations between tiles and scalar values or immediate constants.

**Total Operations:** 19

---

## Operations

### TEXPANDS

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \mathrm{scalar}

**IR Level 1 (SSA):**
```text
%dst = pto.texpands %scalar : dtype -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.texpands ins(%scalar : dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TCMPS

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \left(\mathrm{src}_{i,j}\ \mathrm{cmpMode}\ \mathrm{scalar}\right) 

The encoding/type of `dst` is implementation-defined (often a mask-like tile).

**IR Level 1 (SSA):**
```text
%dst = pto.tcmps %src, %scalar {cmpMode = #pto<cmp xx>} : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcmps ins(%src, %scalar{cmpMode = #pto<cmp xx>}: !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TSELS

**Math Interpretation:**

For each element `(i, j)` in the valid region:


\mathrm{dst}_{i,j} =
\begin{cases}
\mathrm{src0}_{i,j} & \text{if } \mathrm{selectMode} = 1 \\
\mathrm{src1}_{i,j} & \text{otherwise}
\end{cases}

**IR Level 1 (SSA):**
```text
%dst = pto.tsels %src0, %src1, %scalar : (!pto.tile<...>, !pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tsels ins(%src0, %src1, %scalar : !pto.tile_buf<...>, !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TMINS

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \min(\mathrm{src}_{i,j}, \mathrm{scalar})

**IR Level 1 (SSA):**
```text
%dst = pto.tmins %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tmins ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TADDS

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} + \mathrm{scalar}

**IR Level 1 (SSA):**
```text
%dst = pto.tadds %src, %scalar : (!pto.tile<...>,dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tadds ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TSUBS

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} - \mathrm{scalar}

**IR Level 1 (SSA):**
```text
%dst = pto.tsubs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tsubs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TDIVS

**Math Interpretation:**

For each element `(i, j)` in the valid region:

- Tile/scalar:

   \mathrm{dst}_{i,j} = \frac{\mathrm{src}_{i,j}}{\mathrm{scalar}} 

- Scalar/tile:

   \mathrm{dst}_{i,j} = \frac{\mathrm{scalar}}{\mathrm{src}_{i,j}}

**IR Level 1 (SSA):**
```text
%dst = pto.tdivs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
%dst = pto.tdivs %scalar, %src : (dtype, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tdivs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
pto.tdivs ins(%scalar, %src : dtype, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TMULS

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \cdot \mathrm{scalar}

**IR Level 1 (SSA):**
```text
%dst = pto.tmuls %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tmuls ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TFMODS

**Math Interpretation:**

For each element `(i, j)` in the valid region:

\mathrm{dst}_{i,j} = \mathrm{fmod}(\mathrm{src}_{i,j}, \mathrm{scalar})

**IR Level 1 (SSA):**
```text
%dst = pto.tfmods %src, %scalar : !pto.tile<...>, f32
```

**IR Level 2 (DPS):**
```text
pto.tfmods ins(%src, %scalar : !pto.tile_buf<...>, f32) outs(%dst : !pto.tile_buf<...>)
```

---

### TREMS

**Math Interpretation:**

For each element `(i, j)` in the valid region:

\mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \bmod \mathrm{scalar}

**IR Level 1 (SSA):**
```text
%dst = pto.trems %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.trems ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TMAXS

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \max(\mathrm{src}_{i,j}, \mathrm{scalar})

**IR Level 1 (SSA):**
```text
%dst = pto.tmaxs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tmaxs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TANDS

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \;\&\; \mathrm{scalar}

**IR Level 1 (SSA):**
```text
%dst = pto.tands %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tands ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TORS

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \;|\; \mathrm{scalar}

**IR Level 1 (SSA):**
```text
%dst = pto.tors %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tors ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TSHLS

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \ll \mathrm{scalar}

**IR Level 1 (SSA):**
```text
%dst = pto.tshls %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tshls ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TSHRS

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \gg \mathrm{scalar}

**IR Level 1 (SSA):**
```text
%dst = pto.tshrs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tshrs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TXORS

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \oplus \mathrm{scalar}

**IR Level 1 (SSA):**
```text
%dst = pto.txors %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.txors ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TLRELU

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = (\mathrm{src}_{i,j} > 0) ? \mathrm{src}_{i,j} : (\mathrm{src}_{i,j} \cdot \mathrm{slope})

**IR Level 1 (SSA):**
```text
%dst = pto.tlrelu %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tlrelu ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

---

### TADDSC

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} + \mathrm{scalar} + \mathrm{src1}_{i,j}

**IR Level 1 (SSA):**
```text
%dst = pto.taddsc %src0, %scalar, %src1 : (!pto.tile<...>, dtype, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.taddsc ins(%src0, %scalar, %src1 : !pto.tile_buf<...>, dtype, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSUBSC

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} - \mathrm{scalar} + \mathrm{src1}_{i,j}

**IR Level 1 (SSA):**
```text
%dst = pto.tsubsc %src0, %scalar, %src1 : (!pto.tile<...>, dtype, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tsubsc ins(%src0, %scalar, %src1 : !pto.tile_buf<...>, dtype, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---


