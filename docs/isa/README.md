<p align="center">
  <img src="../figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />
</p>

# PTO ISA Reference

This directory contains the per-instruction reference for the PTO Tile Lib ISA.

- Source of truth (C++ intrinsics): `include/pto/common/pto_instr.hpp`
- Common conventions (operands, events, modifiers): `docs/isa/conventions.md`

## Synchronization
- [TSYNC](tile/ops/sync-and-config/tsync.md) - Synchronize PTO execution (wait on events or insert a per-op pipeline barrier).

## Manual / Resource Binding
- [TASSIGN](tile/ops/sync-and-config/tassign.md) - Bind a Tile object to an implementation-defined on-chip address (manual placement).
- [SETHF32MODE](tile/ops/sync-and-config/sethf32mode.md) - Configure HF32 transform mode (implementation-defined).
- [SETTF32MODE](tile/ops/sync-and-config/settf32mode.md) - Configure TF32 transform mode (implementation-defined).
- [SETFMATRIX](tile/ops/sync-and-config/setfmatrix.md) - Set FMATRIX register(s) for IMG2COL-like ops.
- [SET_IMG2COL_RPT](tile/ops/sync-and-config/set-img2col-rpt.md) - Set IMG2COL repeat metadata from an IMG2COL configuration tile.
- [SET_IMG2COL_PADDING](tile/ops/sync-and-config/set-img2col-padding.md) - Set IMG2COL padding metadata from an IMG2COL configuration tile.
- [TSUBVIEW](tile/ops/sync-and-config/subview.md) - Reinterpret a tile as a subtile of another tile.
- [TGET_SCALE_ADDR](tile/ops/sync-and-config/get-scale-addr.md) - Bind the on-chip address of output tile to a scaled factor of that of input tile.

## Elementwise (Tile-Tile)
- [TADD](tile/ops/elementwise-tile-tile/tadd.md) - Elementwise add of two tiles.
- [TABS](tile/ops/elementwise-tile-tile/tabs.md) - Elementwise absolute value of a tile.
- [TAND](tile/ops/elementwise-tile-tile/tand.md) - Elementwise bitwise AND of two tiles.
- [TOR](tile/ops/elementwise-tile-tile/tor.md) - Elementwise bitwise OR of two tiles.
- [TSUB](tile/ops/elementwise-tile-tile/tsub.md) - Elementwise subtract of two tiles.
- [TMUL](tile/ops/elementwise-tile-tile/tmul.md) - Elementwise multiply of two tiles.
- [TMIN](tile/ops/elementwise-tile-tile/tmin.md) - Elementwise minimum of two tiles.
- [TMAX](tile/ops/elementwise-tile-tile/tmax.md) - Elementwise maximum of two tiles.
- [TCMP](tile/ops/elementwise-tile-tile/tcmp.md) - Compare two tiles and write a packed predicate mask.
- [TDIV](tile/ops/elementwise-tile-tile/tdiv.md) - Elementwise division of two tiles.
- [TSHL](tile/ops/elementwise-tile-tile/tshl.md) - Elementwise shift-left of two tiles.
- [TSHR](tile/ops/elementwise-tile-tile/tshr.md) - Elementwise shift-right of two tiles.
- [TXOR](tile/ops/elementwise-tile-tile/txor.md) - Elementwise bitwise XOR of two tiles.
- [TLOG](tile/ops/elementwise-tile-tile/tlog.md) - Elementwise natural logarithm of a tile.
- [TRECIP](tile/ops/elementwise-tile-tile/trecip.md) - Elementwise reciprocal of a tile.
- [TPRELU](tile/ops/elementwise-tile-tile/tprelu.md) - Elementwise PReLU (parametric ReLU) with a per-element slope tile.
- [TADDC](tile/ops/elementwise-tile-tile/taddc.md) - Elementwise ternary add: `src0 + src1 + src2`.
- [TSUBC](tile/ops/elementwise-tile-tile/tsubc.md) - Elementwise ternary op: `src0 - src1 + src2`.
- [TCVT](tile/ops/elementwise-tile-tile/tcvt.md) - Elementwise type conversion with a specified rounding mode.
- [TSEL](tile/ops/elementwise-tile-tile/tsel.md) - Select between two tiles using a mask tile (per-element selection).
- [TRSQRT](tile/ops/elementwise-tile-tile/trsqrt.md) - Elementwise reciprocal square root.
- [TSQRT](tile/ops/elementwise-tile-tile/tsqrt.md) - Elementwise square root.
- [TEXP](tile/ops/elementwise-tile-tile/texp.md) - Elementwise exponential.
- [TNOT](tile/ops/elementwise-tile-tile/tnot.md) - Elementwise bitwise NOT of a tile.
- [TRELU](tile/ops/elementwise-tile-tile/trelu.md) - Elementwise ReLU of a tile.
- [TNEG](tile/ops/elementwise-tile-tile/tneg.md) - Elementwise negation of a tile.
- [TREM](tile/ops/elementwise-tile-tile/trem.md) - Elementwise remainder of two tiles.
- [TFMOD](tile/ops/elementwise-tile-tile/tfmod.md) - Elementwise fmod of two tiles.

## Tile-Scalar / Tile-Immediate
- [TEXPANDS](tile/ops/tile-scalar-and-immediate/texpands.md) - Broadcast a scalar into a destination tile.
- [TCMPS](tile/ops/tile-scalar-and-immediate/tcmps.md) - Compare a tile against a scalar and write per-element comparison results.
- [TSELS](tile/ops/tile-scalar-and-immediate/tsels.md) - Select one of two source tiles using a scalar `selectMode` (global select).
- [TMINS](tile/ops/tile-scalar-and-immediate/tmins.md) - Elementwise minimum of a tile and a scalar.
- [TADDS](tile/ops/tile-scalar-and-immediate/tadds.md) - Elementwise add a scalar to a tile.
- [TSUBS](tile/ops/tile-scalar-and-immediate/tsubs.md) - Elementwise subtract a scalar from a tile.
- [TDIVS](tile/ops/tile-scalar-and-immediate/tdivs.md) - Elementwise division with a scalar (tile/scalar or scalar/tile).
- [TMULS](tile/ops/tile-scalar-and-immediate/tmuls.md) - Elementwise multiply a tile by a scalar.
- [TFMODS](tile/ops/tile-scalar-and-immediate/tfmods.md) - Elementwise remainder with a scalar: `fmod(src, scalar)`.
- [TREMS](tile/ops/tile-scalar-and-immediate/trems.md) - Elementwise remainder with a scalar: `remainder(src, scalar)`.
- [TMAXS](tile/ops/tile-scalar-and-immediate/tmaxs.md) - Elementwise max of a tile and a scalar: `max(src, scalar)`.
- [TANDS](tile/ops/tile-scalar-and-immediate/tands.md) - Elementwise bitwise AND of a tile and a scalar.
- [TORS](tile/ops/tile-scalar-and-immediate/tors.md) - Elementwise bitwise OR of a tile and a scalar.
- [TSHLS](tile/ops/tile-scalar-and-immediate/tshls.md) - Elementwise shift-left a tile by a scalar.
- [TSHRS](tile/ops/tile-scalar-and-immediate/tshrs.md) - Elementwise shift-right a tile by a scalar.
- [TXORS](tile/ops/tile-scalar-and-immediate/txors.md) - Elementwise bitwise XOR of a tile and a scalar.
- [TLRELU](tile/ops/tile-scalar-and-immediate/tlrelu.md) - Leaky ReLU with a scalar slope.
- [TADDSC](tile/ops/tile-scalar-and-immediate/taddsc.md) - Elementwise fused add with scalar and a second tile: `src0 + scalar + src1`.
- [TSUBSC](tile/ops/tile-scalar-and-immediate/tsubsc.md) - Elementwise fused op: `src0 - scalar + src1`.

## Axis Reduce / Expand
- [TROWSUM](tile/ops/reduce-and-expand/trowsum.md) - Reduce each row by summing across columns.
- [TCOLSUM](tile/ops/reduce-and-expand/tcolsum.md) - Reduce each column by summing across rows.
- [TCOLPROD](tile/ops/reduce-and-expand/tcolprod.md) - Reduce each column by multiplying across rows.
- [TCOLMAX](tile/ops/reduce-and-expand/tcolmax.md) - Reduce each column by taking the maximum across rows.
- [TROWMAX](tile/ops/reduce-and-expand/trowmax.md) - Reduce each row by taking the maximum across columns.
- [TROWMIN](tile/ops/reduce-and-expand/trowmin.md) - Reduce each row by taking the minimum across columns.
- [TROWARGMAX](tile/ops/reduce-and-expand/trowargmax.md) - Get the column index of the maximum element for each row.
- [TROWARGMIN](tile/ops/reduce-and-expand/trowargmin.md) - Get the column index of the minimum element for each row.
- [TROWEXPAND](tile/ops/reduce-and-expand/trowexpand.md) - Broadcast the first element of each source row across the destination row.
- [TROWEXPANDDIV](tile/ops/reduce-and-expand/trowexpanddiv.md) - Row-wise broadcast divide: divide each row of `src0` by a per-row scalar vector `src1`.
- [TROWEXPANDMUL](tile/ops/reduce-and-expand/trowexpandmul.md) - Row-wise broadcast multiply: multiply each row of `src0` by a per-row scalar vector `src1`.
- [TROWEXPANDSUB](tile/ops/reduce-and-expand/trowexpandsub.md) - Row-wise broadcast subtract: subtract a per-row scalar vector `src1` from each row of `src0`.
- [TROWEXPANDADD](tile/ops/reduce-and-expand/trowexpandadd.md) - Row-wise broadcast add: add a per-row scalar vector.
- [TROWEXPANDMAX](tile/ops/reduce-and-expand/trowexpandmax.md) - Row-wise broadcast max with a per-row scalar vector.
- [TROWEXPANDMIN](tile/ops/reduce-and-expand/trowexpandmin.md) - Row-wise broadcast min with a per-row scalar vector.
- [TROWEXPANDEXPDIF](tile/ops/reduce-and-expand/trowexpandexpdif.md) - Row-wise exp-diff: compute exp(src0 - src1) with per-row scalars.
- [TCOLMIN](tile/ops/reduce-and-expand/tcolmin.md) - Reduce each column by taking the minimum across rows.
- [TCOLEXPAND](tile/ops/reduce-and-expand/tcolexpand.md) - Broadcast the first element of each source column across the destination column.
- [TCOLEXPANDDIV](tile/ops/reduce-and-expand/tcolexpanddiv.md) - Column-wise broadcast divide: divide each column by a per-column scalar vector.
- [TCOLEXPANDMUL](tile/ops/reduce-and-expand/tcolexpandmul.md) - Column-wise broadcast multiply: multiply each column by a per-column scalar vector.
- [TCOLEXPANDADD](tile/ops/reduce-and-expand/tcolexpandadd.md) - Column-wise broadcast add with per-column scalar vector.
- [TCOLEXPANDMAX](tile/ops/reduce-and-expand/tcolexpandmax.md) - Column-wise broadcast max with per-column scalar vector.
- [TCOLEXPANDMIN](tile/ops/reduce-and-expand/tcolexpandmin.md) - Column-wise broadcast min with per-column scalar vector.
- [TCOLEXPANDSUB](tile/ops/reduce-and-expand/tcolexpandsub.md) - Column-wise broadcast subtract: subtract a per-column scalar vector from each column.
- [TCOLEXPANDEXPDIF](tile/ops/reduce-and-expand/tcolexpandexpdif.md) - Column-wise exp-diff: compute exp(src0 - src1) with per-column scalars.

## Memory (GM <-> Tile)
- [TLOAD](tile/ops/memory-and-data-movement/tload.md) - Load data from a GlobalTensor (GM) into a Tile.
- [TPREFETCH](tile/ops/memory-and-data-movement/tprefetch.md) - Prefetch data from global memory into a tile-local cache/buffer (hint).
- [TSTORE](tile/ops/memory-and-data-movement/tstore.md) - Store data from a Tile into a GlobalTensor (GM), optionally using atomic write or quantization parameters.
- [TSTORE_FP](tile/ops/memory-and-data-movement/tstore.md) - Store an accumulator tile into global memory using a scaling (`fp`) tile for vector quantization parameters.
- [MGATHER](tile/ops/memory-and-data-movement/mgather.md) - Gather-load elements from global memory into a tile using per-element indices.
- [MSCATTER](tile/ops/memory-and-data-movement/mscatter.md) - Scatter-store elements from a tile into global memory using per-element indices.

## Matrix Multiply
- [TGEMV_MX](tile/ops/matrix-and-matrix-vector/tgemv-mx.md) - GEMV with additional scaling tiles for mixed-precision / quantized matrix-vector compute.
- [TMATMUL_MX](tile/ops/matrix-and-matrix-vector/tmatmul-mx.md) - Matrix multiply (GEMM) with additional scaling tiles for mixed-precision / quantized matmul on supported targets.
- [TMATMUL](tile/ops/matrix-and-matrix-vector/tmatmul.md) - Matrix multiply (GEMM) producing an accumulator/output tile.
- [TMATMUL_ACC](tile/ops/matrix-and-matrix-vector/tmatmul-acc.md) - Matrix multiply with accumulator input (fused accumulate).
- [TMATMUL_BIAS](tile/ops/matrix-and-matrix-vector/tmatmul-bias.md) - Matrix multiply with bias add.
- [TGEMV](tile/ops/matrix-and-matrix-vector/tgemv.md) - General Matrix-Vector multiplication producing an accumulator/output tile.
- [TGEMV_ACC](tile/ops/matrix-and-matrix-vector/tgemv-acc.md) - GEMV with explicit accumulator input/output tiles.
- [TGEMV_BIAS](tile/ops/matrix-and-matrix-vector/tgemv-bias.md) - GEMV with bias add.

## Data Movement / Layout
- [TEXTRACT](tile/ops/layout-and-rearrangement/textract.md) - Extract a sub-tile from a source tile.
- [TEXTRACT_FP](tile/ops/layout-and-rearrangement/textract.md) - Extract with fp/scaling tile (vector-quantization parameters).
- [TIMG2COL](tile/ops/layout-and-rearrangement/timg2col.md) - Image-to-column transform for convolution-like workloads.
- [TINSERT](tile/ops/layout-and-rearrangement/tinsert.md) - Insert a sub-tile into a destination tile at an (indexRow, indexCol) offset.
- [TINSERT_FP](tile/ops/layout-and-rearrangement/tinsert.md) - Insert with fp/scaling tile (vector-quantization parameters).
- [TFILLPAD](tile/ops/layout-and-rearrangement/tfillpad.md) - Copy+pad a tile outside the valid region with a compile-time pad value.
- [TFILLPAD_INPLACE](tile/ops/layout-and-rearrangement/tfillpad-inplace.md) - In-place fill/pad variant.
- [TFILLPAD_EXPAND](tile/ops/layout-and-rearrangement/tfillpad-expand.md) - Fill/pad while allowing dst to be larger than src.
- [TMOV](tile/ops/layout-and-rearrangement/tmov.md) - Move/copy between tiles, optionally applying implementation-defined conversion modes.
- [TMOV_FP](tile/ops/layout-and-rearrangement/tmov.md) - Move/convert from an accumulator tile into a destination tile, using a scaling (`fp`) tile for vector quantization parameters.
- [TRESHAPE](tile/ops/layout-and-rearrangement/treshape.md) - Reinterpret a tile as another tile type/shape while preserving the underlying bytes.
- [TTRANS](tile/ops/layout-and-rearrangement/ttrans.md) - Transpose with an implementation-defined temporary tile.

## Complex
- [TPRINT](tile/ops/irregular-and-complex/tprint.md) - Debug/print elements from a tile (implementation-defined).
- [TMRGSORT](tile/ops/irregular-and-complex/tmrgsort.md) - Merge sort for multiple sorted lists (implementation-defined element format and layout).
- [TSORT32](tile/ops/irregular-and-complex/tsort32.md) - Sort a fixed-size 32-element block and produce an index mapping.
- [TGATHER](tile/ops/irregular-and-complex/tgather.md) - Gather/select elements using either an index tile or a compile-time mask pattern.
- [TCI](tile/ops/irregular-and-complex/tci.md) - Generate a contiguous integer sequence into a destination tile.
- [TTRI](tile/ops/irregular-and-complex/ttri.md) - Generate a triangular (lower/upper) mask tile.
- [TPARTADD](tile/ops/irregular-and-complex/tpartadd.md) - Partial elementwise add with implementation-defined handling of mismatched valid regions.
- [TPARTMUL](tile/ops/irregular-and-complex/tpartmul.md) - Partial elementwise multiply with implementation-defined handling of mismatched valid regions.
- [TPARTMAX](tile/ops/irregular-and-complex/tpartmax.md) - Partial elementwise max with implementation-defined handling of mismatched valid regions.
- [TPARTMIN](tile/ops/irregular-and-complex/tpartmin.md) - Partial elementwise min with implementation-defined handling of mismatched valid regions.
- [TGATHERB](tile/ops/irregular-and-complex/tgatherb.md) - Gather elements using byte offsets.
- [TSCATTER](tile/ops/irregular-and-complex/tscatter.md) - Scatter rows of a source tile into a destination tile using per-element row indices.
- [TQUANT](tile/ops/irregular-and-complex/tquant.md) - Quantize a tile (e.g. FP32 to FP8) producing exponent/scaling/max outputs.
