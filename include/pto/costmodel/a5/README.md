# A5 Mocker Coverage Notes

This directory contains the A5-specific host-side compatibility layer used by `__COSTMODEL`.

The key point is that A5 mocker coverage is currently a subset of full A5 PTO coverage. In
[`include/pto/common/pto_instr_impl.hpp`](../../common/pto_instr_impl.hpp), the A5 mocker branch exposes `31`
instruction headers, while the full A5 branch exposes `100`. The `__COSTMODEL` split exists because many A5
headers still depend on runtime, SIMT, register, and packing/unpacking features that are not modeled here yet.

## Illustration

```text
                           pto/common/pto_instr.hpp
                                      |
                                      v
                           pto/common/pto_instr_impl.hpp
                                      |
                   +------------------+------------------+
                   |                                     |
                   | PTO_NPU_ARCH_A5 && __COSTMODEL      | PTO_NPU_ARCH_A5 && !__COSTMODEL
                   |                                     |
                   v                                     v
          small mocker-safe include set          full A5 instruction include set
          (31 headers today)                     (100 headers today)
                   |                                     |
                   v                                     v
       headers that only need current           headers that still need more than the
       mocker surface:                          current mocker provides:
       - qualifiers.hpp                         - SIMT kernel launch/runtime pieces
       - compat.hpp                             - more register/vector helper symbols
       - cce_stub.hpp                           - more packing/unpacking modes
                                                 - more traceable A5 intrinsics
                                                 - some nontrivial host emulation
```

## What Exists Today

Current A5 mocker support is built from:

- [`compat.hpp`](./compat.hpp): placeholder FP4/FP8/vector types and a small set of A5 enums/constants.
- [`cce_stub.hpp`](./cce_stub.hpp): trace stubs for a limited A5 intrinsic set such as `vlds`, `vsts`, `vadd`,
  `vsub`, `vmul`, `vmin`, `vmax`, `vdiv`, `vcvt`, `vaxpy`, `vshl`, `vshr`, `vor`, `vxor`, plus a few memory/config
  helpers.
- [`../common/qualifiers.hpp`](../common/qualifiers.hpp): host-side replacements for storage qualifiers and
  `__cce_get_tile_ptr`.

This is enough for the current mocker A5 subset such as elementwise arithmetic, some scalar ops, `TLoad`, `TStore`,
`TLRelu`, `TRsqrt`, and `TUnaryOp`.

## Gaps That Still Block Full A5 Coverage

### 1. A5 headers not enabled in the mocker include branch yet

These headers are present in the full A5 include list but absent from the `__COSTMODEL` A5 branch today:

- Data movement / reshape / transform:
  `TSubView.hpp`, `TMov.hpp`, `TExtract.hpp`, `TInsert.hpp`, `TReshape.hpp`, `TTrans.hpp`, `TFillPad.hpp`,
  `TPrefetch.hpp`, `TGetScaleAddr.hpp`
- Compare / select / bit / conversion:
  `TCmp.hpp`, `TCmps.hpp`, `TSel.hpp`, `TSels.hpp`, `TBinSOp.hpp`, `TCvt.hpp`, `TFMod.hpp`, `TFModS.hpp`,
  `TRem.hpp`, `TRemS.hpp`
- Reduction / gather / scatter / sort:
  `TMrgSort.hpp`, `TSort32.hpp`, `TGather.hpp`, `TGatherB.hpp`, `TScatter.hpp`, `MGather.hpp`, `MScatter.hpp`,
  `TColSum.hpp`, `TColProd.hpp`, `TColMax.hpp`, `TColMin.hpp`, `TRowProd.hpp`, `TRowReduce.hpp`, `TRowReduceIdx.hpp`
- Expand / broadcast style ops:
  `TConcat.hpp`, `TColExpand.hpp`, `TColExpandAdd.hpp`, `TColExpandDiv.hpp`, `TColExpandExpdif.hpp`,
  `TColExpandMax.hpp`, `TColExpandMin.hpp`, `TColExpandMul.hpp`, `TColExpandSub.hpp`, `TRowExpand.hpp`,
  `TRowExpandAdd.hpp`, `TRowExpandDiv.hpp`, `TRowExpandExpdif.hpp`, `TRowExpandMax.hpp`, `TRowExpandMin.hpp`,
  `TRowExpandMul.hpp`, `TRowExpandSub.hpp`
- Matmul / conv helper style ops:
  `TMatmul.hpp`, `TPartAdd.hpp`, `TPartMul.hpp`, `TPartMax.hpp`, `TPartMin.hpp`, `TImg2col.hpp`,
  `TSetFmatrix.hpp`, `TSetImg2colRpt.hpp`, `TSetImg2colPadding.hpp`
- Quant / dequant / pack / activation / misc:
  `TPrelu.hpp`, `TQuant.hpp`, `TDeQuant.hpp`, `TPack.hpp`, `THistogram.hpp`, `TTri.hpp`, `Tci.hpp`,
  `TPush.hpp`, `TPop.hpp`
- Debug-only:
  `TPrint.hpp`

These `69` headers are the immediate backlog if the goal is to make the A5 mocker include surface match the full A5
surface.

### 2. Missing SIMT runtime coverage

`MGather.hpp` and `MScatter.hpp` are not simple vector-intrinsic wrappers. They use SIMT-only pieces such as:

- `__simt_vf__`
- `LAUNCH_BOUND(...)`
- `__cce_simt_get_TID_X()`
- `__cce_simt_get_TID_Y()`
- `cce::async_invoke`
- `cce::dim3`

Current A5 mocker files do not provide host-side replacements for that execution model, so these headers are not
mocker-safe today.

### 3. Missing register/vector helper symbols in `compat.hpp`

Several excluded A5 headers use register types and helper symbols that are not defined in
[`compat.hpp`](./compat.hpp). Common examples include:

- pack/unpack modes:
  `PK_B32`, `PK4_B32`, `PK_B16`, `UNPK4_B8`, `UNPK_B32`
- broadcast / interleave modes:
  `BRC_B32`, `E2B_B32`, `DINTLV_B32`
- predicates and patterns:
  `PAT_ALL`, `PAT_ALLF`, `PAT_H`, `PAT_VL8`, `PAT_VL32`, `pset_b8`, `pset_b16`, `pset_b32`
- partition / rounding variants:
  `PART_P0`, `PART_P1`, `ROUND_A`, `RS_DISABLE`
- register helpers and vector types:
  `vintlv`, `vector_f8e4m3`

Without these, headers like `TCvt.hpp`, `TPack.hpp`, `TQuant.hpp`, `TDeQuant.hpp`, `TGather.hpp`, and parts of
`TRowExpand*` cannot even be cleanly compiled in host mocker mode.

### 4. Current A5 stubs still have no-op holes

Some A5 mocker intrinsics compile today but emit no trace information at all because the stub is still empty:

- `wait_intra_block`
- `set_intra_block`
- `set_mte2_nz_para`
- `set_loop_size_outtol1`
- `set_loop1_stride_outtol1`
- `set_loop2_stride_outtol1`
- `set_pad_val_outtol1`
- `set_channel_para`
- `copy_gm_to_cbuf_multi_nd2nz`
- `copy_gm_to_cbuf_multi_dn2nz`
- `copy_gm_to_cbuf_align_v2`

Those should be upgraded from empty placeholders to `RecordCceCall(...)` stubs before relying on them for trace or
latency work.

### 5. Some advanced A5 paths need more than a stub name

For several excluded ops, adding only one more `RecordCceCall(...)` entry is not enough. They also need host-visible
metadata so the trace is meaningful:

- `TMatmul.hpp`: `mad` / `mad_mx` are available on device paths, but fuller A5 matmul coverage also needs the header to
  be mocker-safe and the trace args to preserve `m/k/n`, phase, and mode details.
- `TQuant.hpp` / `TDeQuant.hpp`: these rely on mixed register formats, pack/unpack modes, broadcast modes, and often
  multi-step register sequences rather than one simple intrinsic.
- `TGather.hpp` / `TScatter.hpp` / `THistogram.hpp`: these depend on richer predicate patterns and packed store/load
  modes.
- `TPush.hpp` / `TPop.hpp`: these are control/dataflow helpers and may need more runtime modeling than plain trace
  stubs.

## Practical Coverage Plan

If the goal is full A5 mocker coverage, the lowest-risk order is:

1. Expand `compat.hpp` with the missing A5 register constants, vector types, predicate creators, and helper functions.
2. Replace the current empty A5 placeholder stubs with traceable `RecordCceCall(...)` versions.
3. Admit the non-SIMT, non-quantized headers first:
   `TCmp*`, `TSel*`, `TMov`, `TExtract`, `TReshape`, `TTrans`, `TMatmul`.
4. Then add the pack/convert/quant family:
   `TCvt`, `TPack`, `TDeQuant`, `TQuant`.
5. Leave `MGather` / `MScatter` and any true SIMT launch-style headers for last, because they need the largest runtime
   model expansion.

## Bottom Line

The special `#ifdef __COSTMODEL` in the A5 branch is currently acting as a safety boundary. It prevents full A5
headers from entering host builds before the A5 mocker has the runtime, register, and intrinsic coverage needed to
parse and trace them correctly.
