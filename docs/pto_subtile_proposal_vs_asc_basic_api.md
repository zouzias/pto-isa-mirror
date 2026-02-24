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
