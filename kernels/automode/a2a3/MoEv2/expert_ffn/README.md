# expert_ffn

Auto-mode A3 prototype. Per-expert fused FFN over packed-by-expert tokens.
**Rewrite** of the earlier element-wise gating placeholder — this version does
the real two-matmul MoE FFN body while keeping the ReLU intermediate local to
the expert/tile.

## What it does

For each expert `e` with `count[e] > 0`, applied to its packed slice `A[start[e] : start[e]+count[e]]`:

```
for each expert-local A_s tile:
    for each N_l1 output-column panel:
        B_s[:, n:n+N_l1] = 0
        for each F_l1 panel:
            Y_t = relu(A_s @ W1[e, :, f:f+F_l1]).astype(fp16)
            B_s[:, n:n+N_l1] += Y_t @ W2[e, f:f+F_l1, n:n+N_l1]
        store that B_s panel to GM
```

| Buffer | Shape | dtype | Notes |
|---|---|---|---|
| `A` (input)            | `(kT·kTopK + kTileM, kH)` | fp16 | first `kT·kTopK` rows valid; trailing `kTileM` = overspill pad |
| `expert_count` (input) | `kE` | int32 | per-expert row count in `A` |
| `expert_start` (input) | `kE` | int32 | prefix sum of `expert_count` |
| `W1` (input)           | `(kE, kH, kF)` | fp16 | per-expert up-projection |
| `W2` (input)           | `(kE, kF, kH)` | fp16 | per-expert down-projection |
| `Y_scratch` (ABI only) | `(kT·kTopK + kTileM, kF)` | fp16 | GM scratch for the ReLU intermediate |
| `B` (output)           | `(kT·kTopK + kTileM, kH)` | fp32 | first `kT·kTopK` rows are the answer |

The trailing `kTileM` pad is still allocated for ABI compatibility with the full
pipeline. The fused kernel uses dynamic valid rows, so validation only checks
the first `kT·kTopK` rows of `B`.

## Target platform

A3 / Ascend 910B1. Cube target (`--cce-aicore-arch=dav-c220-cube`). One shared
TU; one `__global__ AICORE` function (`runExpertFfn`) launched by
`launchExpertFfnFp16`.

## Auto-mode constraints honored

- `kE` AI cores in parallel: `block_idx == expert index` (`get_block_idx()`). Each block handles one expert's token slice independently. Pattern confirmed in `kernels/automode/a2a3/flash_atten/fa_performance_kernel.cpp`.
- Static tile maxima with dynamic valid rows for the per-expert tail.
- `A_s` loads full logical rows but may panel the aligned `H` dimension as
  `H_l1`; `W1_t` loads the matching `H_l1 x F_l1` panel; `W2_t` loads the
  matching `F_l1 x N_l1` output-column panel.
- ReLU + fp32→fp16 is fused into the Stage-1 accumulator store:
  `TSTORE<..., ReluPreMode::NormalRelu>` writes `Y_t` to GM `Y_scratch`.
  Stage 2 reloads that tile for the second GEMM.
- The selected local working set is capped at `2^18` bytes:
  `A_s + W1_t + W2_t + Y_t + fp32 B_s panel footprint`.
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

- `kTileM = 64` — raises row residency versus the original 16-row prototype while
  keeping the local-budget chooser conservative.
- Larger shapes are tiled over `N_l1`, `H_l1`, `H_l0`, `F_l1`, and `F_l0`.
- No intra-expert token-level parallelism yet; each block serializes over its expert's M-tiles.
- Only `ReluPreMode::NormalRelu` (no GELU/SiLU/LeakyReLU — the enum on this layer is `NoRelu` / `NormalRelu`).
- fp16 inputs / weights, fp32 accumulator; no other dtype paths.
