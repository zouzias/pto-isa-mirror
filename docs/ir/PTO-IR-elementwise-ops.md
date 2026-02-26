# Elementwise (Tile-Tile)

This document describes element-wise operations between two tiles.

**Total Operations:** 28

---

## Operations

### TADD

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} + \mathrm{src1}_{i,j}

**IR Level 1 (SSA):**
```text
%dst = pto.tadd %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TABS

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \left|\mathrm{src}_{i,j}\right|

**IR Level 1 (SSA):**
```text
%dst = pto.tabs %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tabs ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TAND

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \;\&\; \mathrm{src1}_{i,j}

**IR Level 1 (SSA):**
```text
%dst = pto.tand %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tand ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TOR

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \;|\; \mathrm{src1}_{i,j}

**IR Level 1 (SSA):**
```text
%dst = pto.tor %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tor ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSUB

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} - \mathrm{src1}_{i,j}

**IR Level 1 (SSA):**
```text
%dst = pto.tsub %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tsub ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TMUL

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \cdot \mathrm{src1}_{i,j}

**IR Level 1 (SSA):**
```text
%dst = pto.tmul %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tmul ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TMIN

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \min(\mathrm{src0}_{i,j}, \mathrm{src1}_{i,j})

**IR Level 1 (SSA):**
```text
%dst = pto.tmin %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TMAX

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \max(\mathrm{src0}_{i,j}, \mathrm{src1}_{i,j})

**IR Level 1 (SSA):**
```text
%dst = pto.tmax %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCMP

**Math Interpretation:**

Conceptually, for each element `(i, j)` in the valid region, define a predicate:

 p_{i,j} = \left(\mathrm{src0}_{i,j}\ \mathrm{cmpMode}\ \mathrm{src1}_{i,j}\right) 

The predicate mask is stored in `dst` using an implementation-defined packed layout.

**IR Level 1 (SSA):**
```text
%dst = pto.tcmp %src0, %src1{cmpMode = #pto<cmp xx>}: (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcmp ins(%src0, %src1{cmpMode = #pto<cmp xx>}: !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TDIV

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \frac{\mathrm{src0}_{i,j}}{\mathrm{src1}_{i,j}}

**IR Level 1 (SSA):**
```text
%dst = pto.tdiv %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tdiv ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSHL

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \ll \mathrm{src1}_{i,j}

**IR Level 1 (SSA):**
```text
%dst = pto.tshl %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tshl ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSHR

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \gg \mathrm{src1}_{i,j}

**IR Level 1 (SSA):**
```text
%dst = pto.tshr %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tshr ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TXOR

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \oplus \mathrm{src1}_{i,j}

**IR Level 1 (SSA):**
```text
%dst = pto.txor %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.txor ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TLOG

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \log(\mathrm{src}_{i,j})

**IR Level 1 (SSA):**
```text
%dst = pto.tlog %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tlog ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TRECIP

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \frac{1}{\mathrm{src}_{i,j}}

**IR Level 1 (SSA):**
```text
%dst = pto.trecip %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.trecip ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TPRELU

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = (\mathrm{src0}_{i,j} > 0) ? \mathrm{src0}_{i,j} : (\mathrm{src0}_{i,j} \cdot \mathrm{src1}_{i,j})

**IR Level 1 (SSA):**
```text
%dst = pto.tprelu %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tprelu ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TADDC

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} + \mathrm{src1}_{i,j} + \mathrm{src2}_{i,j}

**IR Level 1 (SSA):**
```text
%dst = pto.taddc %src0, %src1, %src2 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.taddc ins(%src0, %src1, %src2 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSUBC

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} - \mathrm{src1}_{i,j} + \mathrm{src2}_{i,j}

**IR Level 1 (SSA):**
```text
%dst = pto.tsubc %src0, %src1, %src2 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tsubc ins(%src0, %src1, %src2 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TCVT

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \mathrm{cast}_{\mathrm{rmode}}\!\left(\mathrm{src}_{i,j}\right) 

where `rmode` is a rounding policy (see `pto::RoundMode`).

**IR Level 1 (SSA):**
```text
%dst = pto.tcvt %src{rmode = #pto<round_mode xx>}: !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tcvt ins(%src{rmode = #pto<round_mode xx>}: !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSEL

**Math Interpretation:**

For each element `(i, j)` in the valid region:


\mathrm{dst}_{i,j} =
\begin{cases}
\mathrm{src0}_{i,j} & \text{if } \mathrm{mask}_{i,j}\ \text{is true} \\
\mathrm{src1}_{i,j} & \text{otherwise}
\end{cases}

**IR Level 1 (SSA):**
```text
%dst = pto.tsel %mask, %src0, %src1 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tsel ins(%mask, %src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TRSQRT

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \frac{1}{\sqrt{\mathrm{src}_{i,j}}}

**IR Level 1 (SSA):**
```text
%dst = pto.trsqrt %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.trsqrt ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TSQRT

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \sqrt{\mathrm{src}_{i,j}}

**IR Level 1 (SSA):**
```text
%dst = pto.tsqrt %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tsqrt ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TEXP

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \exp(\mathrm{src}_{i,j})

**IR Level 1 (SSA):**
```text
%dst = pto.texp %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.texp ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TNOT

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \sim\mathrm{src}_{i,j}

**IR Level 1 (SSA):**
```text
%dst = pto.tnot %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tnot ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TRELU

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = \max(\mathrm{src}_{i,j}, 0)

**IR Level 1 (SSA):**
```text
%dst = pto.trelu %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.trelu ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TNEG

**Math Interpretation:**

For each element `(i, j)` in the valid region:

 \mathrm{dst}_{i,j} = -\mathrm{src}_{i,j}

**IR Level 1 (SSA):**
```text
%dst = pto.tneg %src : !pto.tile<...> -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tneg ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TREM

**Math Interpretation:**

For each element `(i, j)` in the valid region:

\mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \bmod \mathrm{src1}_{i,j}

**IR Level 1 (SSA):**
```text
%dst = pto.trem %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.trem ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### TFMOD

**Math Interpretation:**

For each element `(i, j)` in the valid region:

\mathrm{dst}_{i,j} = \mathrm{fmod}(\mathrm{src0}_{i,j}, \mathrm{src1}_{i,j})

**IR Level 1 (SSA):**
```text
%dst = pto.tfmod %src0, %src1 : !pto.tile<...>
```

**IR Level 2 (DPS):**
```text
pto.tfmod ins(%src0, %src1 : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---


