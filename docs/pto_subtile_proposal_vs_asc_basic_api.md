# PTO SubTile proposal vs AscendC Basic API

## Context
We compare a **SubTile‑based PTO style** (2D subtile abstraction, no internal loops in PTO kernel; user owns loop) versus **AscendC Basic API** (raw buffers + manual stride + pipeline queues).
Scope: elementwise / broadcast / reduce examples.

---

### 1) Elementwise Add (large 2D tensor)

#### PTO SubTile — Case A (large cols): big tile load, per‑row 1D subtile TADD
```cpp
using BigTile = Tile<TileType::Vec, T, kTRows, kTCols, BLayout::RowMajor, -1, -1>;
BigTile src0Big(vRows, kTCols);
BigTile src1Big(vRows, kTCols);
BigTile dstBig(vRows, kTCols);
TASSIGN(src0Big, 0x0);
TASSIGN(src1Big, 0x10000);
TASSIGN(dstBig, 0x20000);

TLOAD(src0Big, src0Global);
TLOAD(src1Big, src1Global);

using Tile1D = Tile<TileType::Vec, T, 1, kTCols, BLayout::RowMajor, -1, -1>;
for (int r = 0; r < vRows; ++r) {
    Tile1D src0Sub(1, kTCols);
    Tile1D src1Sub(1, kTCols);
    Tile1D dstSub(1, kTCols);

    uint32_t off = r * kTCols * sizeof(T);
    TASSIGN(src0Sub, 0x0 + off);
    TASSIGN(src1Sub, 0x10000 + off);
    TASSIGN(dstSub, 0x20000 + off);

    TADD(dstSub, src0Sub, src1Sub);
}

TSTORE(dstGlobal, dstBig);
```

#### PTO SubTile — Case B (short cols): big tile load, col‑subtile loop (FP32 vCols=64, stride=128)
```cpp
using BigTile = Tile<TileType::Vec, T, kTRows, kTCols, BLayout::RowMajor, -1, -1>;
BigTile src0Big(vRows, kTCols);
BigTile src1Big(vRows, kTCols);
BigTile dstBig(vRows, kTCols);
TASSIGN(src0Big, 0x0);
TASSIGN(src1Big, 0x10000);
TASSIGN(dstBig, 0x20000);

TLOAD(src0Big, src0Global);
TLOAD(src1Big, src1Global);

// FP32: VL=256B => vCols=64, row stride=128
using Tile2D = Tile<TileType::Vec, T, kTRows, 128, BLayout::RowMajor, -1, -1>;
for (int c0 = 0; c0 < totalCols; c0 += 64) {
    int vCols = min(64, totalCols - c0);
    Tile2D src0Sub(vRows, vCols);
    Tile2D src1Sub(vRows, vCols);
    Tile2D dstSub(vRows, vCols);

    uint32_t off = c0 * sizeof(T);
    TASSIGN(src0Sub, 0x0 + off);
    TASSIGN(src1Sub, 0x10000 + off);
    TASSIGN(dstSub, 0x20000 + off);

    TADD(dstSub, src0Sub, src1Sub);
}

TSTORE(dstGlobal, dstBig);
```

#### AscendC Basic — Case A (large cols) with TQue enqueue/dequeue
```cpp
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

// Row loop + repeat across columns (repeatTimes = vCols / VL)
// VL in elements: FP16=128, FP32=64 (256B)
BinaryRepeatParams rp;
// keep same row, advance inside row by datablocks
rp.dstRepStride = 0;
rp.src0RepStride = 0;
rp.src1RepStride = 0;
// dataBlockStride: advance within row (32B blocks)
rp.dstBlkStride = 1; rp.src0BlkStride = 1; rp.src1BlkStride = 1;

uint64_t mask = VL; // continuous mask (VL elements)
int repeatTimes = vCols / VL;
for (int r = 0; r < vRows; ++r) {
    AscendC::Add(d + r * kTCols, a0 + r * kTCols, a1 + r * kTCols,
                 mask, /*repeatTimes=*/repeatTimes, rp);
}

outQueue.EnQue(d);
inQueue0.FreeTensor(a0); inQueue1.FreeTensor(a1);

auto out = outQueue.DeQue<T>();
AscendC::DataCopy(dstGlobal, out, dataSize);
outQueue.FreeTensor(out);
```

#### AscendC Basic — Case B (short cols, FP32 vCols=64, stride=128) with TQue
```cpp
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

BinaryRepeatParams rp;
rp.dstRepStride = 128 / 8;  // row stride in blocks
rp.src0RepStride = 128 / 8;
rp.src1RepStride = 128 / 8;
rp.dstBlkStride = 1; rp.src0BlkStride = 1; rp.src1BlkStride = 1;

for (int c0 = 0; c0 < totalCols; c0 += 64) {
    uint64_t mask = 64; // FP32 continuous mask
    AscendC::Add(d + c0, a0 + c0, a1 + c0, mask, /*repeatTimes=*/vRows, rp);
}

outQueue.EnQue(d);
inQueue0.FreeTensor(a0); inQueue1.FreeTensor(a1);

auto out = outQueue.DeQue<T>();
AscendC::DataCopy(dstGlobal, out, dataSize);
outQueue.FreeTensor(out);
```




**Remark (elementwise add case):**
- **AscendC** exposes mask control (`isSetMask`) and **block stride** via `BinaryRepeatParams`, so you can model small **3D block strides** (32B block, VL=8 blocks, repeat) directly.
- You can also set **repeat stride = 0** to implement *broadcast* / *reduce‑like* patterns at the vector level.
- **PTO SubTile** keeps the add primitive simple; for broadcast/reduce you must explicitly use higher‑level subtile intrinsics like **TROWEXPANDADD/TCOLEXPANDADD** and **TROWSUM/TCOLSUM** (see next sections).

---

### 2) Broadcast (row/col expand)

#### PTO SubTile — TROWEXPANDADD (row expand)
```cpp
// Step 1: expand one row to a single block (32B => FP32 8 elems)
TROWEXPAND(dstBlock8, srcRow); // dstBlock8 shape: 1 x 8

// Step 2: row-expand-add on 2D tile, vCols = 8 (one block)
// outer loop over col blocks to cover full tile
for (int c0 = 0; c0 < totalCols; c0 += 8) {
    // sub‑tile with vCols=8
    Tile2D srcSub(vRows, 8);
    Tile2D dstSub(vRows, 8);
    TASSIGN(srcSub, 0x0 + c0 * sizeof(T));
    TASSIGN(dstSub, 0x20000 + c0 * sizeof(T));

    TROWEXPANDADD(dstSub, srcSub, dstBlock8); // add expanded row block
}
```

#### AscendC Basic — row expand (BRCB then Add)
```cpp
// brcb: broadcast one row to vector block (32B / 8 FP32)
AscendC::Brcb(dstBlock8, srcRow, /*mask=*/8);

// loop over column blocks (vCols = 8) to cover full tile
for (int c0 = 0; c0 < totalCols; c0 += 8) {
    AscendC::Add(dstLocal + c0, srcLocal + c0, dstBlock8,
                 /*mask=*/8, /*repeatTimes=*/vRows, rp);
}
```

#### PTO SubTile — TCOLEXPANDADD (col expand)
```cpp
// 1D tensor bounded by vCols <= VL
using Tile1D = Tile<TileType::Vec, T, 1, VL, BLayout::RowMajor, -1, -1>;

// outer loop over col blocks to cover whole tile
for (int c0 = 0; c0 < totalCols; c0 += VL) {
    int vCols = min(VL, totalCols - c0);
    Tile1D srcSub(1, vCols);
    Tile1D dstSub(1, vCols);

    TASSIGN(srcSub, 0x0 + c0 * sizeof(T));
    TASSIGN(dstSub, 0x20000 + c0 * sizeof(T));

    TCOLEXPANDADD(dstSub, srcSub);
}
```

#### AscendC Basic — col expand (1D, vCols<=VL, loop over blocks)
```cpp
for (int c0 = 0; c0 < totalCols; c0 += VL) {
    int vCols = min(VL, totalCols - c0);
    uint64_t mask = vCols;
    AscendC::Add(dstLocal + c0, srcLocal + c0, srcLocal + c0,
                 mask, /*repeatTimes=*/1, rp); // or use vector broadcast op
}
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


## Design Suggestion: Explicit 3D Tile Configuration

For vector add (and similar ops), expose a **HW-valid 3D tile configuration** that maps directly to instruction parameters:

```
SubTile_Rx8x8_4B   (FP32)
SubTile_Rx8x16_2B  (FP16)
```

Where:
- **R** = repeat count (dynamic, up to 255)
- **8** = number of 32B blocks per vector (VL = 8 blocks = 256B)
- **8 or 16** = elements per block (32B / sizeof(T))
- **4B / 2B** = element size suffix

**Example for FP32 VADD:**
```cpp
// SubTile_Rx8x8_4B: R repeats × 8 blocks × 8 FP32 elems (= 64 elems per repeat)
using VAddTile = SubTile<float, /*repeat=*/vRows, /*blocks=*/8, /*blockElems=*/8>;
VAddTile src0(vRows);
VAddTile src1(vRows);
VAddTile dst(vRows);

TASSIGN(src0, 0x0);
TASSIGN(src1, 0x10000);
TASSIGN(dst, 0x20000);

TADD(dst, src0, src1);
```

This maps to HW as:
- `mask = 64` (continuous mode, 8 blocks × 8 elems)
- `repeatTimes = vRows`
- `repStride = 8` (blocks per row)
- `blkStride = 1` (contiguous blocks)

**Benefits:**
1. Clear 1:1 mapping to HW instruction params
2. No ambiguity about tile layout
3. Easy to validate at compile time (repeat ≤ 255, blocks ≤ 8, etc.)
4. User sees the 3D structure (repeat × VL × block) directly

**For larger tiles (vCols > VL):** user adds outer loop on column blocks, each iteration uses `SubTile_Rx8x8_4B`.

---

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
---

## Production References: 2D Tile Abstractions in Industry

### 1. LLVM SVE / RISC-V RVV (Scalable Vectors)
- **Dimensionality:** 1D scalable (`<vscale x N x T>`)
- **Abstraction:** Vector length is runtime-variable; predicate masks control elements
- **Key point:** 1D only; 2D requires user loops
- **Reference:** https://llvm.org/docs/RISCV/RISCVVectorExtension.html

### 2. MLIR Vector Dialect
- **Dimensionality:** N-D (`vector<4x8xf32>`)
- **Abstraction:** Multi-dimensional vector types + ops (`vector.contract`, `vector.transfer_read`)
- **Key point:** Provides 2D/3D tile abstraction; lowers to 1D LLVM vectors + loops
- **Reference:** https://mlir.llvm.org/docs/Dialects/Vector/

### 3. Intel AMX (Advanced Matrix Extensions)
- **Dimensionality:** 2D tiles (up to 16 rows × 64 bytes per tile)
- **Abstraction:** 8 tile registers; user configures shape via `LDTILECFG`
- **Ops:** `TILELOADD`, `TILESTORED`, `TDPBF16PS` (tile matmul)
- **Key point:** Fixed 2D tile registers; most similar to SubTile concept
- **Reference:** https://en.wikipedia.org/wiki/Advanced_Matrix_Extensions

### 4. ARM SME (Scalable Matrix Extension)
- **Dimensionality:** 2D ZA array (scalable tiles)
- **Abstraction:** ZA is a large 2D array register; ZA tiles are square sub-arrays
- **Ops:** `FMOPA` (outer product), `LDR`/`STR` for ZA vectors
- **Key point:** SVE + 2D tile; scalable matrix size
- **Reference:** https://stackoverflow.com/questions/76305243/arm-a-profile-architecture-what-does-za-stand-for

### Comparison Summary

| Reference | 2D Tile? | User Controls Shape? | Ops Level | Similar to SubTile? |
|-----------|----------|---------------------|-----------|---------------------|
| LLVM SVE/RVV | ❌ (1D) | ✅ (vscale) | Vector element | Partially (VL abstraction) |
| MLIR Vector | ✅ | ✅ | Vector/Tensor | Yes (2D abstraction) |
| Intel AMX | ✅ | ✅ (TILECFG) | Matrix | **Most similar** (fixed 2D) |
| ARM SME | ✅ | ✅ (scalable) | Matrix | Yes (2D ZA tiles) |
| **PTO SubTile** | ✅ | ✅ (Tile<R,C>) | Vector/Tile | — |

**Conclusion:** SubTile is closest to **Intel AMX** and **ARM SME** — explicit 2D tile shape with user-controlled dimensions. The `SubTile_Rx8x8_4B` naming mirrors AMXs tile config concept, while the scalable row dimension is similar to ARM SMEs approach.
