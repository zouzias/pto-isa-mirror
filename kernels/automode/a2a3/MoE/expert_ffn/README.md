# expert_ffn

Auto-mode A3 prototype. Per-expert fused FFN over packed-by-expert tokens.
**Rewrite** of the earlier element-wise gating placeholder — this version does
the real two-matmul MoE FFN body while keeping the ReLU intermediate local to
the expert/tile.

## What it does

For each expert `e` with `count[e] > 0`, applied to its packed slice `A[start[e] : start[e]+count[e]]`:

```
for each expert-local A_s tile:
    B_s = 0
    for each F_l1 panel:
        Y_t = relu(A_s @ W1[e, :, f:f+F_l1]).astype(fp16)
        B_s += Y_t @ W2[e, f:f+F_l1, :]
    store B_s to GM
```

| Buffer | Shape | dtype | Notes |
|---|---|---|---|
| `A` (input)            | `(kT·kTopK + 16, kH)` | fp16 | first `kT·kTopK` rows valid; trailing 16 = overspill pad |
| `expert_count` (input) | `kE` | int32 | per-expert row count in `A` |
| `expert_start` (input) | `kE` | int32 | prefix sum of `expert_count` |
| `W1` (input)           | `(kE, kH, kF)` | fp16 | per-expert up-projection |
| `W2` (input)           | `(kE, kF, kH)` | fp16 | per-expert down-projection |
| `Y_scratch` (ABI only) | `(kT·kTopK + 16, kF)` | fp16 | currently ignored by the fused kernel |
| `B` (output)           | `(kT·kTopK + 16, kH)` | fp32 | first `kT·kTopK` rows are the answer |

The +16 trailing pad is still allocated for ABI compatibility with the full
pipeline. The fused kernel uses dynamic valid rows, so validation only checks
the first `kT·kTopK` rows of `B`.

## Target platform

A3 / Ascend 910B1. Cube target (`--cce-aicore-arch=dav-c220-cube`). One shared
TU; one `__global__ AICORE` function (`runExpertFfn`) launched by
`launchExpertFfnFp16`.

## Auto-mode constraints honored

- Single AICORE; no `block_idx` work split.
- Static tile maxima with dynamic valid rows for the per-expert tail.
- `A_s` loads full rows of size `H`; `W1_t` loads full `H` rows and an
  `F_l1` column panel; `W2_t` loads the matching `F_l1` rows and full `H`
  columns.
- ReLU + fp32→fp16 is fused into `TMOV<..., ReluPreMode::NormalRelu>` from the
  Stage-1 accumulator to the L1 `Y_t` Mat tile. No vector hop and no GM scratch
  hop.
- The selected local working set is capped at `2^17` bytes:
  `A_s + W1_t + W2_t + Y_t + fp32 B_s footprint`.
- No `TASSIGN` literal addresses, no `#ifndef __PTO_AUTO__` manual-sync, no
  `Tile::data()` in kernel, no `*_IMPL` calls, no raw CCE intrinsics, no
  `Event<>`, no `TPipe`/`TPUSH`/`TPOP`, no double buffering, no A5-only ops.

## Tail Handling

The per-expert loop advances by the selected `M` tile, but sets dynamic valid rows for the last tile:

```cpp
currentM = min(M, count[e] - m0)
```

So each `A_s` tile belongs to exactly one expert and the kernel no longer
relies on expert-to-expert overspill overwrites.

## How to build and run

```bash
bash run.sh -r npu -v Ascend910B1
```

`scripts/gen_data.py` synthesizes a plausible `(count, start, A)` directly (no
scatter dependency); the C++ driver runs the fused kernel and compares
`B[0 : kT·kTopK]` against the numpy reference at `1e-2` abs tolerance (fp16
input → fp32 accumulator, distribution scale `[-3.3, 3.3]`).

## Sweeping kTopK / kE / kT / kH / kF

Three places to update together:

- `scripts/gen_data.py`     : `kT`, `kH`, `kF`, `kE`, `kTopK`, `kTileM`
- `main.cpp`                : `constexpr int kT/kH/kF/kE/kTopK/kTileM`
- `expert_ffn_kernel.cpp`   : `namespace expert_ffn_cfg { constexpr unsigned ... }`

Larger `kH` / `kF` are tiled over the H and F dimensions, but the selected
panel sizes must still divide the aligned dimensions.

## Known limitations (v1)

- `kTileM = 16` — below §11.9's proven `kTileM = 128`. Within the cube
  M-alignment rule, but un-tested at this exact size. The local-budget chooser
  may shrink from larger requested tile heights.
- `kH = kF = 64`. Larger shapes are now tiled over `H_l0`, `F_l1`, and `F_l0`,
  but still require divisibility by the selected aligned panel sizes.
- Single AICORE; no block_idx parallelism (per-expert parallelism or per-tile parallelism is a separate milestone).
- Only `ReluPreMode::NormalRelu` (no GELU/SiLU/LeakyReLU — the enum on this layer is `NoRelu` / `NormalRelu`).
- fp16 inputs / weights, fp32 accumulator; no other dtype paths.
