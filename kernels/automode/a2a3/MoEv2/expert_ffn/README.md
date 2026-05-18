# expert_ffn

Auto-mode A3 prototype. Per-expert two-stage FFN over packed-by-expert tokens. **Rewrite** of the earlier element-wise gating placeholder — this version does the real two-matmul MoE FFN body.

## What it does

For each expert `e` with `count[e] > 0`, applied to its packed slice `A[start[e] : start[e]+count[e]]`:

```
Y_pre = A_chunk @ W1[e]                   # fp16 @ fp16 -> fp32  (Stage 1 GEMM1)
Y     = relu(Y_pre).astype(fp16)          # fused in Stage 1's TSTORE FixPipe
B     = Y @ W2[e]                         # fp16 @ fp16 -> fp32  (Stage 2 GEMM2)
```

| Buffer | Shape | dtype | Notes |
|---|---|---|---|
| `A` (input)            | `(kT·kTopK + 16, kH)` | fp16 | first `kT·kTopK` rows valid; trailing 16 = overspill pad |
| `expert_count` (input) | `kE` | int32 | per-expert row count in `A` |
| `expert_start` (input) | `kE` | int32 | prefix sum of `expert_count` |
| `W1` (input)           | `(kE, kH, kF)` | fp16 | per-expert up-projection |
| `W2` (input)           | `(kE, kF, kH)` | fp16 | per-expert down-projection |
| `Y_scratch` (GM scratch) | `(kT·kTopK + 16, kF)` | fp16 | post-ReLU fp16 intermediate |
| `B` (output)           | `(kT·kTopK + 16, kH)` | fp32 | first `kT·kTopK` rows are the answer |

The +16 trailing pad absorbs the last non-empty expert's last-tile overspill writes. Validation only checks the first `kT·kTopK` rows of `B`.

## Target platform

A3 / Ascend 910B1. Cube target (`--cce-aicore-arch=dav-c220-cube`). One shared TU; two `__global__ AICORE` functions (`runFfnStage1Gemm1Relu`, `runFfnStage2Gemm2`); two stream-serialised launches inside `launchExpertFfnFp16`.

## Auto-mode constraints honored

- Single AICORE per stage; no `block_idx` work split.
- Static tile shapes (`M = ceil(kTileM/16)*16 = 16`, `K`/`N` rounded to `blockAlign`).
- 5 cube tiles per stage (`TileMatA`, `TileMatB`, `TileLeft`, `TileRight`, `TileAcc`), all declared **outside** both loops (single auto-allocator analysis pin — same pattern as §A18 / §11.9).
- ReLU + fp32→fp16 fused into Stage 1's `TSTORE<..., ReluPreMode::NormalRelu>` (proven by §A17 / §11.8). No vector hop.
- ACL stream-order semantics guarantee Stage 2 only starts after Stage 1's `TSTORE`-to-`Y_scratch` is fully drained to GM — no within-kernel cross-GEMM auto-sync.
- No `TASSIGN` literal addresses, no `#ifndef __PTO_AUTO__` manual-sync, no `Tile::data()` in kernel, no `*_IMPL` calls, no raw CCE intrinsics, no `Event<>`, no `TPipe`/`TPUSH`/`TPOP`, no double buffering, no A5-only ops.

## Overspill scheme

The per-expert loop writes `kTileM = 16` rows per iteration even when `count[e]` is not a multiple of 16:

```cpp
for (m0 = 0; m0 < count; m0 += kTileM) { ... }   // ceil(count/kTileM) iters
```

For `count[e] = 4`, the kernel writes rows `[start, start+16)` — 12 of those are overspill into the territory of the next non-empty expert `e'` (whose `start[e'] = start[e]+count[e]`). When `e'` is processed, it overwrites those 12 rows with the correct data. The last non-empty expert's overspill lands in the trailing 16-row pad.

The arithmetic: max overspill per expert is `kTileM - 1 = 15` rows; `kTileM = 16` trailing pad is sufficient.

## How to build and run

```bash
bash run.sh -r npu -v Ascend910B1
```

`scripts/gen_data.py` synthesizes a plausible `(count, start, A)` directly (no scatter dependency); the C++ driver runs both stages and compares `B[0 : kT·kTopK]` against the numpy reference at `1e-2` abs tolerance (fp16 input → fp32 accumulator, distribution scale `[-3.3, 3.3]`).

## Sweeping kTopK / kE / kT / kH / kF

Three places to update together:

- `scripts/gen_data.py`     : `kT`, `kH`, `kF`, `kE`, `kTopK`, `kTileM`
- `main.cpp`                : `constexpr int kT/kH/kF/kE/kTopK/kTileM`
- `expert_ffn_kernel.cpp`   : `namespace expert_ffn_cfg { constexpr unsigned ... }`

Larger `kH` / `kF` (≥ 128) approach the L0B 64 KB ceiling for fp16 weights and need Split-K or Split-N inside each stage — **postponed** to a later milestone.

## Known limitations (v1)

- `kTileM = 16` — below §11.9's proven `kTileM = 128`. Within the cube M-alignment rule, but un-tested at this exact size. Falls back to 128 by changing one constant if needed.
- `kH = kF = 64`. Larger shapes require Split-K/Split-N (separate milestone; see [pto_auto_mode_hw_optimization_guide.md §1.3](../../../../../docs_for_ai/pto_auto_mode_hw_optimization_guide.md)).
- Single AICORE per stage; no block_idx parallelism (per-expert parallelism or per-tile parallelism is a separate milestone).
- Only `ReluPreMode::NormalRelu` (no GELU/SiLU/LeakyReLU — the enum on this layer is `NoRelu` / `NormalRelu`).
- fp16 inputs / weights, fp32 accumulator; no other dtype paths.
