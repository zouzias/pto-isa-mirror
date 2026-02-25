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
