# PTO SubTile proposal vs AscendC Basic API

## Context
We compare a **SubTile‑based PTO style** (2D subtile abstraction, no internal loops in PTO kernel; user owns loop) versus **AscendC Basic API** (raw buffers + manual stride + pipeline queues).
Scope: elementwise / broadcast / reduce examples.

## PTO SubTile (2D) — idea
- PTO kernel only defines a **2D subtile** (e.g., `Tile<TileType::Vec, T, kTRows, kTCols>`)
- PTO does **not** generate internal loops across tiles
- User implements the **outer loops** across N‑D tensor tiles

### Advantages
1) **Still smaller code for N‑D loops**
   - You keep explicit control of outer loops, but inner math stays short (`TLOAD/TADD/TSTORE` etc.)
   - Looping over N‑D shapes is written once by the user; inside each iteration, the PTO tile code is minimal.

2) **Consistent 2D “tile math” semantics**
   - Elementwise, broadcast, reduce can be expressed on the same 2D subtile interface
   - Easy to reason about *what happens inside one tile*, while leaving iteration to the caller

3) **Better correctness ergonomics than raw AscendC**
   - PTO still encapsulates sync between MTE2/VEC/MTE3
   - Tile + GlobalTensor capture shape/stride for that subtile so you avoid re‑deriving every stride in the kernel body

4) **Incremental path from PTO → AscendC**
   - You can start with SubTile for correctness and later replace the tile body with AscendC for perf tuning

### Limitations (vs AscendC Basic API)
1) **Less flexible stride/layout control**
   - AscendC lets you express arbitrary raw stride patterns (including non‑contiguous, exotic packing)
   - SubTile assumes a 2D view with fixed row‑major semantics inside tile (even if outer loop visits ND)

2) **Boundary/masked tails still need care**
   - SubTile helps structure; tails still need mask handling in tile (rows/cols less than tile size)

3) **Lower ceiling for micro‑optimizations**
   - AscendC allows manual pipeline scheduling, queue depth tuning, and direct UB layout choices
   - SubTile hides these details, so peak perf may be lower in hand‑tuned cases

4) **Limited support for unusual data formats**
   - AscendC can target special formats/strides (e.g., interleaved channels, block‑sparse layouts)
   - SubTile’s 2D abstraction is best for dense row‑major tensors

## Examples: how SubTile helps vs AscendC

### Elementwise (Add)
- **SubTile:**
  - Per‑tile code: `TLOAD(src0Tile); TLOAD(src1Tile); TADD(dstTile, src0Tile, src1Tile); TSTORE(dstTile);`
  - Outer loop across tiles handled by user
- **AscendC Basic:**
  - Must manage `TPipe/TQue`, `DataCopy` lengths/strides, and pipeline sync explicitly for every kernel

### Broadcast (Row/Col expand)
- **SubTile:**
  - Represent 2D tile and broadcast inside tile; outer loops handle ND
  - Cleaner broadcast logic, less boilerplate
- **AscendC Basic:**
  - Must compute explicit src/dst address jumps and stride conversions manually

### Reduce (Row/Col sum)
- **SubTile:**
  - Tile reduce primitive (`TCOLSUM`, `TROWSUM`) on 2D tile
  - Outer loop handles reduction across tiles/blocks
- **AscendC Basic:**
  - Must manually express reduction loop, mask, and partial sum buffering

## Summary
**PTO SubTile** keeps the kernel body short and safe (tile‑level correctness, built‑in sync), while allowing the user to own ND loops. This strikes a balance: **more structure than raw AscendC**, but still flexible enough to integrate with custom tiling strategies.

**AscendC Basic API** remains more powerful for exotic strides/layouts and extreme optimization, but has **much higher boilerplate** and **manual pipeline management** overhead.

**Recommendation:** use SubTile for readability + correctness + common dense cases; fall back to AscendC when you need raw stride freedom or deep performance tuning.

## Detailed limitations by case (SubTile‑only, no internal loop)

### Elementwise (eltwise)
**What SubTile can’t cover well:**
- **Non‑contiguous or irregular strides inside a tile** (e.g., gather‑style layout, NHWC with channel gaps, block‑sparse rows).
- **Interleaved / packed formats** where one “row” in memory does not map to a logical row (e.g., special vector‑packed formats).

**AscendC basic API can:**
- Express arbitrary `DataCopy` strides and offsets, even when the inner rows are discontiguous or interleaved.
- Manually compute element addresses and load/store with custom stride logic.

### Broadcast (row/col expand)
**What SubTile can’t cover well:**
- **Unequal or irregular broadcast patterns** (e.g., broadcast every k columns with gaps, or scatter‑style broadcast).
- **ND broadcast with format‑dependent stride** (e.g., broadcasting across blocked channels in a packed layout).

**AscendC basic API can:**
- Implement custom address arithmetic for irregular broadcast patterns.
- Handle blocked/packed layouts by explicitly computing destination offsets.

### Reduce (row/col sum/min/max)
**What SubTile can’t cover well:**
- **Reductions over non‑contiguous axes inside a tile** (e.g., reduce over a strided dimension or a packed format).
- **Multi‑stage reductions with custom partial accumulation** (e.g., reduce over K with intermediate buffering in UB using non‑standard layouts).

**AscendC basic API can:**
- Build custom reduction loops with manual stride, masking, and partial sum buffering.
- Reduce along any axis by explicit address stepping, even when data is not row‑major.

### General pattern (root cause)
- SubTile assumes **2D row‑major dense tiles** and hides the pipeline/sync for that case.
- AscendC basic API exposes **raw stride and address control**, so it can cover irregular/packed/non‑contiguous layouts that SubTile cannot.

## ST example snippets (SubTile vs AscendC Basic)

### 1) Elementwise Add (large 2D tensor)

#### Case A: **Large columns** → use **1D SubTile** + user loop on rows
```cpp
// Treat each row as a 1D subtile (no internal loop)
using Tile1D = Tile<TileType::Vec, T, 1, kTCols, BLayout::RowMajor, -1, -1>;

for (int r = 0; r < totalRows; ++r) {
    Tile1D src0Tile(1, kTCols);
    Tile1D src1Tile(1, kTCols);
    Tile1D dstTile(1, kTCols);

    // Move UB window per row
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    // Update GlobalTensor view for this row (stride stays kTCols)
    GlobalData src0Row(src0 + r * kTCols);
    GlobalData src1Row(src1 + r * kTCols);
    GlobalData dstRow(out  + r * kTCols);

    TLOAD(src0Tile, src0Row);
    TLOAD(src1Tile, src1Row);
    TADD(dstTile, src0Tile, src1Tile);
    TSTORE(dstRow, dstTile);
}
```

#### Case B: **Small cols (e.g., 128 = 64×2) + many rows** → use **2D SubTile** + loop on rows
```cpp
using Tile2D = Tile<TileType::Vec, T, kTRows, 128, BLayout::RowMajor, -1, -1>;

for (int r0 = 0; r0 < totalRows; r0 += kTRows) {
    int vRows = min(kTRows, totalRows - r0);
    Tile2D src0Tile(vRows, 128);
    Tile2D src1Tile(vRows, 128);
    Tile2D dstTile(vRows, 128);

    // Move UB window per subtile iteration
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    GlobalData src0Blk(src0 + r0 * 128);
    GlobalData src1Blk(src1 + r0 * 128);
    GlobalData dstBlk(out  + r0 * 128);

    TLOAD(src0Tile, src0Blk);
    TLOAD(src1Tile, src1Blk);
    TADD(dstTile, src0Tile, src1Tile);
    TSTORE(dstBlk, dstTile);
}
```

**AscendC Basic (tile‑level)**
```cpp
// inside user outer loop over tiles
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue0, inQueue1;
AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;

int dataSize = vRows * kTCols;
pipe.InitBuffer(inQueue0, 1, dataSize * sizeof(T));
pipe.InitBuffer(inQueue1, 1, dataSize * sizeof(T));
pipe.InitBuffer(outQueue, 1, dataSize * sizeof(T));

auto s0 = inQueue0.AllocTensor<T>();
auto s1 = inQueue1.AllocTensor<T>();
AscendC::DataCopy(s0, src0Global, dataSize);
AscendC::DataCopy(s1, src1Global, dataSize);
inQueue0.EnQue(s0); inQueue1.EnQue(s1);

auto a0 = inQueue0.DeQue<T>();
auto a1 = inQueue1.DeQue<T>();
auto d  = outQueue.AllocTensor<T>();
AscendC::Add(d, a0, a1, dataSize);
outQueue.EnQue(d);
inQueue0.FreeTensor(a0); inQueue1.FreeTensor(a1);

auto out = outQueue.DeQue<T>();
AscendC::DataCopy(dstGlobal, out, dataSize);
outQueue.FreeTensor(out);
```

---

### 2) Broadcast (row expand)

**PTO SubTile (tile‑level)**
```cpp
// src is 1 x vCols, dst is vRows x vCols
TLOAD(srcTile, srcGlobal);
TROWEXPAND(dstTile, srcTile);  // broadcast row to tile height
TSTORE(dstGlobal, dstTile);
```

**AscendC Basic (tile‑level)**
```cpp
// Copy one row into UB
AscendC::DataCopy(srcLocal, srcGlobal, vCols);
// Repeat into dstLocal with manual loop or vector broadcast primitive
for (int r = 0; r < vRows; ++r) {
    AscendC::DataCopy(dstLocal + r * kTCols, srcLocal, vCols);
}
AscendC::DataCopy(dstGlobal, dstLocal, vRows * kTCols);
```

---

### 3) Reduce (colsum)

**PTO SubTile (tile‑level)**
```cpp
TLOAD(srcTile, srcGlobal);
TCOLSUM(dstTile, srcTile, tmpTile, /*isBinary=*/false);
TSTORE(dstGlobal, dstTile);
```

**AscendC Basic (tile‑level, Add stride mode)**
```cpp
AscendC::DataCopy(srcLocal, srcGlobal, vRows * kTCols);

// init dstLocal with first row
AscendC::DataCopy(dstLocal, srcLocal, vCols);

// accumulate remaining rows using Add stride mode
SetMaskNorm();
SetVectorMask(0, vCols);
AscendC::Add(dstLocal, dstLocal, srcLocal + kTCols,
             /*repeat=*/vRows - 1,
             /*dstRep=*/1, /*src0Rep=*/1, /*src1Rep=*/1,
             /*dstStride=*/0, /*src0Stride=*/0, /*src1Stride=*/kTCols / 8);

AscendC::DataCopy(dstGlobal, dstLocal, vCols);
```

> Note: AscendC Basic can also use vector reduction ops, but still requires manual stride + buffer management.

---

### 4) Reduce (rowsum)

**PTO SubTile (tile‑level)**
```cpp
TLOAD(srcTile, srcGlobal);
TROWSUM(dstTile, srcTile, tmpTile, /*isBinary=*/false);
TSTORE(dstGlobal, dstTile);
```

**AscendC Basic (tile‑level)**
```cpp
AscendC::DataCopy(srcLocal, srcGlobal, vRows * kTCols);
// Example using row-wise reduce with vector reduction (conceptual)
SetMaskNorm();
SetVectorMask(0, vCols);
AscendC::WholeReduceSum(dstLocal, srcLocal,
                        /*repeat=*/vRows - 1,
                        /*dstRep=*/1, /*srcRep=*/1,
                        /*dstStride=*/1, /*srcStride=*/kTCols / 8);
AscendC::DataCopy(dstGlobal, dstLocal, vRows);
```


## Why SubTile can be “single‑intrinsic, no internal loops” (and its limitation)

**Key assumption:** SubTile is a **fixed 2D dense row‑major tile**, and the operation is defined to act on exactly that tile shape. This lets a **single intrinsic** handle the whole tile (e.g., `TADD`, `TROWEXPAND`, `TCOLSUM`) without an internal loop.

**Limitation of this design:**
- The tile must be **contiguous and regular** in memory (row‑major, fixed stride).
- The tile operation must match a **single intrinsic’s semantics** (no irregular stride, no per‑row variance, no gather/scatter).
- If the logical operation requires **irregular access inside the tile** (packed formats, strided sub‑rows, sparse patterns), then a single intrinsic is insufficient and you need **explicit loops** or AscendC Basic API.

**Example limitation (why loop would be needed):**
- Suppose each row has a different stride or only every k‑th element is valid. A single `TADD` assumes a uniform stride across all rows; it cannot encode “skip every 3 elements” without a loop. AscendC can implement this with manual address arithmetic.

## Additional restrictions for “no loop” SubTile

To avoid internal loops, the SubTile path relies on **counter/repeat mode** for **continuous/1D layout**. This imposes limits:

1) **Continuous / 1D layout requirement**
   - The tile must be representable as a **contiguous 1D stream** (counter mode).
   - If the layout is irregular (gaps, packing, non‑uniform stride), a single intrinsic cannot cover it.

2) **2D repeat (row axis) limit**
   - For **2D tiles using repeat on the row axis**, the maximum repeat count is **255**.
   - This matters when `vCols <= VL` (e.g., 256B vector length) and you rely on `repeat` to cover rows.

3) **Vector length constraint**
   - `vCols` must fit within a vector length (e.g., **VL = 256B**), otherwise you need multiple vector ops (loop or additional slicing).

**Implication:** The “no loop” SubTile implementation is only valid for **contiguous, regular tiles** with **repeat ≤ 255** and **vCols ≤ VL**. Outside these conditions, you must add loops or use AscendC Basic API for explicit stride/address control.
