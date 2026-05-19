# gather (v3)

Auto-mode A3 prototype. Unpack and accumulate per-expert FFN outputs back into per-token rows, with **softmax routing weights** when `kTopK > 1`. Generic over `kTopK ∈ {1, 2, 4, 8, 16}`.

## What's new in v3

- **kTopK == 1**: replaces the per-row `TLOAD → TSTORE` loop with a chunked bulk-**TSCATTER** row reorder. Idx tile built on-chip from `A_id` + `TCI` ramp + `TCOLEXPANDADD`.
- **kTopK > 1**: drops the `reordered_scratch` GM round-trip. Builds the inverse map `r_inv[k, t] = r` in UB via one **TSCATTER**, then fuses pass 2 (reorder) and pass 3 (weighted combine) into a single H-chunked loop where the per-k reorder is a UB-side **TGATHER**.

The `weights_scratch` GM buffer is retained — the per-k weight column needs a strided `TLOAD` from GM, and the Tile type system hard-codes `RowStride` from layout + Cols, so UB-side column slicing of a `(kT, kPadded)` softmax tile is not possible.

## What it does

```
weights = softmax(outVal, axis=1)       # only computed when kTopK > 1
for r in [0, kT·kTopK):
    t = A_id[r]
    k = rank_id[r]
    if kTopK == 1:
        C[t] = B[r]                     # row reorder only
    else:
        # r_inv[k, t] = r built once via TSCATTER in UB
        pass

if kTopK > 1:
    for h-chunk col in [0, kH, kChunkH):
        bChunk = B[:, col:col+kChunkH]
        C[:, col:col+kChunkH] = sum_k weights[:, k] * bChunk[r_inv[k, :], :]
```

## Algorithm (kTopK == 1)

```cpp
// Hoisted once:
TLOAD     aIdCol   : (kT, 1) ColMajor int32         <- A_id
TMULS     baseCol  = aIdCol * kChunkH                // (kT, 1)
TROWEXPAND baseTile : (kT, kChunkH) int32            // broadcast across kChunkH
TCI       rampRow  : (1, kChunkH) int32 = [0..kChunkH)
TCOLEXPANDADD idxTile = baseTile + rampRow           // idx[r, h] = A_id[r]*kChunkH + h

// Per H-chunk:
for col in [0, kH, kChunkH):
    TLOAD    bChunk : (kT, kChunkH) fp32  <- B[:, col:col+kChunkH]
    TSCATTER cChunk, bChunk, idxTile
    TSTORE   cChunk -> C[:, col:col+kChunkH]
```

For `kT=256, kH=64`: 2 H-chunks of 32 cols each, ~6 bulk DMAs total. Replaces the v2 path of 256 single-row TLOAD/TSTORE pairs (each 256 B) + 256 `pipe_barrier(PIPE_ALL)`.

## Algorithm (kTopK > 1)

```cpp
// Pass 0: build r_inv in UB.
TCI       iotaRow  : (1, kPackedRows) int32 = [0..kPackedRows)
TLOAD     aIdRow   : (1, kPackedRows) int32
TLOAD     rIdRow   : (1, kPackedRows) int32
TMULS     scaledRow = rIdRow * kT
TADD      slotRow  = scaledRow + aIdRow              // rank_id*kT + A_id
TSCATTER  rInvFlat (kTopK, kT), iotaRow, slotRow     // rInvFlat[rank_id[r], A_id[r]] = r

// Pass 1: softmax(outVal) -> weights_scratch GM.
TLOAD          valTile     <- outVal                 // (kT, kPadded)
TROWMAX        maxTile     <- valTile
TROWEXPANDSUB  tmpTile     <- valTile - maxTile
TEXP           expTile     <- exp(tmpTile)
TROWSUM        sumTile     <- expTile
TROWEXPANDDIV  weightTile  <- expTile / sumTile
TSTORE         weights_scratch <- weightTile

// Pass 2: per H-chunk fused reorder + weighted combine.
TCI rampRow : (1, kChunkH) int32 = [0..kChunkH)
for col in [0, kH, kChunkH):
    TLOAD     bChunkBig : (kPackedRows, kChunkH) fp32 <- B[:, col:col+kChunkH]
    TEXPANDS  accChunk = 0                            // (kT, kChunkH)
    for k in [0, kTopK):
        TSUBVIEW   rInvKRow : view of rInvFlat row k   // (1, kT)
        TRESHAPE   rInvKCol : (kT, 1) ColMajor         // same memory
        TMULS      rInvKColScaled = rInvKCol * kChunkH
        TROWEXPAND baseK    : (kT, kChunkH) int32
        TCOLEXPANDADD idxK = baseK + rampRow           // idx[t, h] = r_inv[k, t]*kChunkH + h
        TGATHER    gathK, bChunkBig, idxK, tmpK        // (kT, kChunkH) <- bChunkBig[idxK]
        TLOAD      weightK : (kT, 1) ColMajor          <- weights_scratch[:, k] (strided)
        TROWEXPANDMUL scaledK = gathK * weightK
        TADD       accChunk += scaledK
    TSTORE accChunk -> C[:, col:col+kChunkH]
```

## H-chunk sizing

`kChunkH` shrinks with `kTopK` to keep `bChunkBig` (`kPackedRows × kChunkH × 4 B`) under the ~192 KB UB ceiling:

| `kTopK` | `kChunkH` | `bChunkBig` size |
|---|---|---|
| 1     | 32 | 32 KB |
| 2     | 32 | 64 KB |
| 4     | 16 | 64 KB |
| 8, 16 | 8  | 64 KB / 128 KB |

fp32 32-B alignment requires `kChunkH % 8 == 0`; all values above satisfy that.

## I/O buffers

| Buffer | Shape | dtype | Notes |
|---|---|---|---|
| `B` (input)         | `(kT·kTopK + 16, kH)` | fp32 | first `kT·kTopK` rows consulted |
| `A_id` (input)      | `(kT·kTopK + 16)`     | int32 | first `kT·kTopK` consulted |
| `rank_id` (input)   | `(kT·kTopK + 16)`     | int32 | only used when `kTopK > 1` |
| `outVal` (input)    | `(kT, kPadded)`       | fp32 | cols `kTopK..kPadded-1` host-padded with `-1e30` |
| `weights_scratch` (GM scratch) | `(kT, kPadded)` | fp32 | only used when `kTopK > 1` |
| `C` (output)        | `(kT, kH)`            | fp32 | final output |

`kPadded = max(8, kTopK)`. The v2 `reordered_scratch` buffer is **gone**.

### Why outVal columns `kTopK..kPadded-1` need `-1e30` padding

`TROWMAX`, `TROWEXPANDSUB`, `TEXP`, `TROWSUM` all read the full tile width. For `kTopK ∈ {1, 2, 4}` (where `kPadded = 8 > kTopK`), the unused columns must not corrupt the softmax. Filling them with `-1e30` makes the pipeline neutralize them:

```
-1e30 - real_max  ≈ -inf
exp(-inf)         =  0
0 contributes nothing to the row sum
0 / real_sum      =  0   →  padding cols of weights_scratch end up at 0.0
```

`rank_id` from scatter only takes values in `[0, kTopK)`, so we never look up a padding column at gather time.

## Target platform

A3 / Ascend 910B1. Vec target (`--cce-aicore-arch=dav-c220-vec`).

## Auto-mode constraints honored

- Single AICORE (`<<<1, nullptr, stream>>>`).
- All tiles declared once outside loops; auto allocator pins their UB addresses.
- `GlobalTensor` reconstructed per H-chunk iter with runtime offset.
- No `TASSIGN` aliasing, no `Tile::data()` in kernel, no `*_IMPL` calls, no raw CCE intrinsics, no `Event<>`, no manual sync, no `TPipe` / `TPUSH` / `TPOP`, no double buffering, no A5-only ops.
- No `pipe_barrier(PIPE_ALL)` per row — bulk operations let auto-mode insert the RAW edges.

## How to build and run

```bash
bash run.sh -r npu -v Ascend910B1
```

`scripts/gen_data.py` writes a self-contained `(B, A_id, rank_id, outVal, C_golden)` set. The C++ driver memsets `C` to zero, fires the kernel, and compares against `golden_C.bin` at `1e-3` abs tolerance (softmax adds `TEXP` / divide rounding on top of fp32 accumulation).

## Sweeping kTopK / kT / kH

Three places to update together:

- `scripts/gen_data.py`     : `kT`, `kH`, `kTopK`
- `main.cpp`                : `constexpr int kT/kH/kTopK`
- `gather_kernel.cpp`       : `namespace gather_cfg { constexpr unsigned ... }`

`kPadded` and `kChunkH` are derived from `kTopK` in the kernel — they update automatically when `kTopK` is patched.

## Known risks (v3)

- **First in-tree auto-mode A3 use of indexed `TGATHER` and `TSCATTER`.** The mask-pattern `TGATHER` variant is exercised by the TopK kernel; the indexed forms used here are not. Build / sync behavior unverified.
- **`TSUBVIEW + TRESHAPE` to view a `(1, kT)` prefix as `(kT, 1)` ColMajor** is a layout reinterpret (same memory, same kT consecutive int32s). The pattern follows the float↔uint32 type-pun in the TopK kernel but exchanges layout instead of dtype.
- `TROWEXPAND` of a `(kT, 1)` int32 ColMajor source to `(kT, kChunkH)` int32 RowMajor uses the general (non-`vbrcb`) code path since dst Cols > elemPerBlock — should be auto-callable but unverified end-to-end.

## Known limitations

- Single AICORE; no `block_idx` work split.
- fp32 only.
- No double-buffering.
- Pass-1 softmax chain `TROWMAX → TROWEXPANDSUB → TEXP → TROWSUM → TROWEXPANDDIV` is the same composition as the in-tree FA softmax macro plus the final divide. The macro is manual-mode and inserts `pipe_barrier(PIPE_V)` between phases; this auto-mode kernel relies on `__PTO_AUTO__` to insert the RAW edges.

## Pattern sources

- Softmax recipe: [tests/npu/a2a3/src/st/testcase/tfa/pto_macro_fa_softmax.hpp:54-60](../../../../tests/npu/a2a3/src/st/testcase/tfa/pto_macro_fa_softmax.hpp#L54-L60).
- GlobalTensor-with-runtime-offset / static tile / no manual sync: [moe_top1_unpermute_kernel.cpp](../../../../kernels/automode/a2a3/moe_top1_unpermute/moe_top1_unpermute_kernel.cpp).
- `TSUBVIEW + TRESHAPE` layout reinterpret pattern (originally used for float↔uint32 type pun): [topk_kernel.cpp Phase 5](../../../../kernels/automode/a2a3/topk/topk_kernel.cpp) and [known_good_kernel_examples.md §A12](../../../../docs_for_ai/known_good_kernel_examples.md).
