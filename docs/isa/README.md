<p align="center">
  <img src="../figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />
</p>

# PTO ISA Manual And Reference

This directory is the canonical PTO ISA tree. It combines the architecture manual, the instruction set guides, the instruction set contracts, and the exact instruction-reference groupings in one place.

## Textual Assembly Inside PTO ISA

This tree is the canonical PTO ISA manual. Textual assembly spelling belongs to the PTO ISA syntax instruction set, not to a second parallel architecture manual.

- PTO ISA defines architecture-visible semantics, legality, state, ordering, target-profile boundaries, and the visible behavior of tile, vector, scalar, communication, and system scheduling operations.
- PTO-AS is the assembler-facing spelling used to write those operations and operands. It is part of how PTO ISA is expressed, not a separate ISA with different semantics.

If the question is "what does this legal PTO program mean across CPU, A2/A3, and A5?", stay in this tree. If the question is "what is the operand shape or textual spelling of this operation?", use the syntax-and-operands pages in this same tree.

## Start Here

## Model Layers

Reading order matches the manual chapter map: programming and machine models, then syntax and state, then memory, then opcode reference.

- [Programming model](programming-model/tiles-and-valid-regions.md)
- [Machine model](machine-model/execution-agents.md)
- [Syntax and operands](syntax-and-operands/assembly-model.md)
- [Type system](state-and-types/type-system.md)
- [Location intent and legality](state-and-types/location-intent-and-legality.md)
- [Memory model](memory-model/consistency-baseline.md)

<<<<<<< HEAD
## Data Movement / Layout
- [TEXTRACT](TEXTRACT.md) - Extract a sub-tile from a source tile.
- [TEXTRACT_FP](TEXTRACT_FP.md) - Extract with fp/scaling tile (vector-quantization parameters).
- [TIMG2COL](TIMG2COL.md) - Image-to-column transform for convolution-like workloads.
- [TINSERT](TINSERT.md) - Insert a sub-tile into a destination tile at an (indexRow, indexCol) offset.
- [TINSERT_FP](TINSERT_FP.md) - Insert with fp/scaling tile (vector-quantization parameters).
- [TFILLPAD](TFILLPAD.md) - Copy+pad a tile outside the valid region with a compile-time pad value.
- [TFILLPAD_INPLACE](TFILLPAD_INPLACE.md) - In-place fill/pad variant.
- [TFILLPAD_EXPAND](TFILLPAD_EXPAND.md) - Fill/pad while allowing dst to be larger than src.
- [TMOV](TMOV.md) - Move/copy between tiles, optionally applying implementation-defined conversion modes.
- [TMOV_FP](TMOV_FP.md) - Move/convert from an accumulator tile into a destination tile, using a scaling (`fp`) tile for vector quantization parameters.
- [TRESHAPE](TRESHAPE.md) - Reinterpret a tile as another tile type/shape while preserving the underlying bytes.
- [TTRANS](TTRANS.md) - Transpose with an implementation-defined temporary tile.
- [TSUBVIEW](TSUBVIEW.md) - Reinterpret a tile as a subtile of another tile.
- [TGET_SCALE_ADDR](TGET_SCALE_ADDR.md) - Bind the on-chip address of output tile to a scaled factor of that of input tile.
- [TCONCAT](TCONCAT.md) - Concatenate two tiles horizontally along the column dimension.
=======
- [Instruction overview](instruction-families/README.md)
- [Instruction set contracts](instruction-families/README.md)
- [Format of instruction descriptions](reference/format-of-instruction-descriptions.md)
- [Tile instruction reference](tile/README.md)
- [Vector instruction reference](vector/README.md)
- [Scalar and control reference](scalar/README.md)
- [Communication instruction reference](comm/README.md)
- [System scheduling instruction reference](system/README.md)
- [Common conventions](conventions.md)
>>>>>>> de8fc112 (Refactor duplicated kernel pipeline helpers)

## Supporting Reference

- [Reference notes](reference/README.md) (glossary, diagnostics, portability, source of truth)

## Cross-Core Communication

- [TALLOC](TALLOC.md) - Allocate a TPipe FIFO slot as a GlobalTensor view.
- [TPUSH](TPUSH.md) - Push a producer tile into a TPipe FIFO for Cube-Vector communication.
- [TPOP](TPOP.md) - Pop a consumer tile from a TPipe FIFO for Cube-Vector communication.
- [TFREE](TFREE.md) - Release FIFO space for a TPipe entry; no-op for TileData TPOP flow.

The grouped instruction set trees under `tile/`, `vector/`, `scalar/`, `comm/`, and `system/` are the canonical PTO ISA paths.

- `docs/isa/tile/ops/`
- `docs/isa/vector/ops/`
- `docs/isa/scalar/ops/`
- `docs/isa/comm/`
- `docs/isa/system/ops/`
