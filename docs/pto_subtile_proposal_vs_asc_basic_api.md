# PTO SubTile proposal vs AscendC Basic API

## Context
We compare a **SubTile‑based PTO style** (2D subtile abstraction, no internal loops in PTO kernel; user owns loop) versus **AscendC Basic (tile‑level)**

##### Case A (large cols) — load big tile, enqueue/dequeue, then per‑row Add only
```cpp
// setup queues once
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue0, inQueue1;
AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
int dataSize = vRows * kTCols;
pipe.InitBuffer(inQueue0, 1, dataSize * sizeof(T));
pipe.InitBuffer(inQueue1, 1, dataSize * sizeof(T));
pipe.InitBuffer(outQueue, 1, dataSize * sizeof(T));

// CopyIn
auto s0 = inQueue0.AllocTensor<T>();
auto s1 = inQueue1.AllocTensor<T>();
AscendC::DataCopy(s0, src0Global, dataSize);
AscendC::DataCopy(s1, src1Global, dataSize);
inQueue0.EnQue(s0); inQueue1.EnQue(s1);

// Compute: per‑row Add only (no extra DataCopy)
auto a0 = inQueue0.DeQue<T>();
auto a1 = inQueue1.DeQue<T>();
auto d  = outQueue.AllocTensor<T>();
SetMaskNorm();
SetVectorMask(0, kTCols);
AscendC::Add(d, a0, a1,
             /*repeat=*/vRows - 1,
             /*dstRep=*/1, /*src0Rep=*/1, /*src1Rep=*/1,
             /*dstStride=*/kTCols/8, /*src0Stride=*/kTCols/8, /*src1Stride=*/kTCols/8);

outQueue.EnQue(d);
inQueue0.FreeTensor(a0); inQueue1.FreeTensor(a1);

// CopyOut
auto out = outQueue.DeQue<T>();
AscendC::DataCopy(dstGlobal, out, dataSize);
outQueue.FreeTensor(out);
```

##### Case B (short cols, FP32 vCols=64, row stride=128) — load big tile, then col‑block Add only
```cpp
// setup queues once
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue0, inQueue1;
AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
int dataSize = vRows * kTCols;
pipe.InitBuffer(inQueue0, 1, dataSize * sizeof(T));
pipe.InitBuffer(inQueue1, 1, dataSize * sizeof(T));
pipe.InitBuffer(outQueue, 1, dataSize * sizeof(T));

// CopyIn
auto s0 = inQueue0.AllocTensor<T>();
auto s1 = inQueue1.AllocTensor<T>();
AscendC::DataCopy(s0, src0Global, dataSize);
AscendC::DataCopy(s1, src1Global, dataSize);
inQueue0.EnQue(s0); inQueue1.EnQue(s1);

// Compute: col blocks (width=64) with Add only
auto a0 = inQueue0.DeQue<T>();
auto a1 = inQueue1.DeQue<T>();
auto d  = outQueue.AllocTensor<T>();

for (int c0 = 0; c0 < totalCols; c0 += 64) {
    SetMaskNorm();
    SetVectorMask(0, 64);
    AscendC::Add(d + c0, a0 + c0, a1 + c0,
                 /*repeat=*/vRows - 1,
                 /*dstRep=*/1, /*src0Rep=*/1, /*src1Rep=*/1,
                 /*dstStride=*/128/8, /*src0Stride=*/128/8, /*src1Stride=*/128/8);
}

outQueue.EnQue(d);
inQueue0.FreeTensor(a0); inQueue1.FreeTensor(a1);

// CopyOut
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
