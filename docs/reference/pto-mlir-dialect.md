# PTO MLIR Dialect (Design)

This repository uses an **MLIR-compatible textual form** for `.pto` files that mirrors PTO ISA instructions as
`pto.*` operations and uses MLIR-style SSA names and type spellings.

This page documents the **intended** PTO MLIR dialect surface syntax (types, ops, and synchronization modeling). It is
currently used by the demo toolchain (`python/pypto` + `ptoas`) as a constrained, parseable subset.

## Types (textual spellings)

- Tile: `!pto.tile<ROWSxCOLSxDT>`
  - Example: `!pto.tile<64x64xf32>`
- Global memory view: `!pto.memref<SPACE,ROWSxCOLSxDT>` (or `!pto.tensor<...>`)
  - Example: `!pto.memref<gm,64x64xf32>`
- Tile buffer (mutable register): `!pto.tilebuf<ROWSxCOLSxDT>`
  - Example: `!pto.tilebuf<64x64xf32>`
- Event token: `!pto.event` (opaque)

Notes:

- These types are treated as **opaque containers** by the demo pipeline; only `ROWS/COLS/DT` are used for codegen.
- `SPACE` is currently informational (`gm` in demos).

## Ops (demo subset)

### `pto.alloc_tile`

Allocate a mutable tile buffer (logical register/buffer).

```text
%t0 = pto.alloc_tile : !pto.tilebuf<64x64xf32>
```

### `pto.tload`

Load a full tile from global memory.

```text
pto.tload ins(%a[%c0, %c0] : !pto.memref<gm,64x64xf32>)
         outs(%t0 : !pto.tilebuf<64x64xf32>)
```

### `pto.tadd`

Elementwise add of two tiles.

```text
pto.tadd ins(%t0, %t1 : !pto.tilebuf<64x64xf32>, !pto.tilebuf<64x64xf32>)
        outs(%t2 : !pto.tilebuf<64x64xf32>)
```

### `pto.tstore`

Store a full tile to global memory.

```text
pto.tstore ins(%t2 : !pto.tilebuf<64x64xf32>)
          outs(%out[%c0, %c0] : !pto.memref<gm,64x64xf32>)
```

## SSA vs DPS/register form

Tile values are modeled as **virtual registers**:

- Each tile buffer SSA name (for example `%t2`) is allocated once by `pto.alloc_tile`.
- Most `pto.*` tile ops are **destination-passing** and print as `ins(...) outs(...)`.

This makes each IR op map one-to-one to a PTO ISA instruction form (destination + sources), while remaining parseable
as MLIR-like text.

## Synchronization model

PTO execution spans multiple pipelines (MTE/Vector/Cube/etc). For correctness, cross-pipeline dependencies need
synchronization.

In the MLIR textual form used by this repo:

- The default `.pto` form is **line-ordered and synchronous**.
- The demo assembler (`ptoas`) inserts conservative synchronization when it detects a transition between major pipelines
  (for example `pto.tload` → `pto.tadd`, and `pto.tadd` → `pto.tstore`).

Internally, `ptoas` lowers this to PTO Tile Lib’s event primitives (see `include/pto/npu/*/TSync.hpp`) by constructing
`Event<Op::TLOAD, Op::VECTOR>` and `Event<Op::VECTOR, Op::TSTORE_VEC>` objects (and similar tags) and passing them as wait
operands to the corresponding intrinsic calls.

## Relation to PTO ISA spec

The operation set corresponds directly to the instruction forms listed in `pto-isa-def.txt` and the instruction pages
under `docs/isa/`.
