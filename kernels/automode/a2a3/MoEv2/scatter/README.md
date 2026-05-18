# scatter

Auto-mode A3 prototype. Pack tokens by expert assignment for downstream per-expert FFN. Generic over `kTopK ∈ {1, 2, 4, 8, 16}`.

## What it does

For `X ∈ R^{kT × kH}` and `expert_id ∈ Z^{kT × kTopK}`, group `(t, k)` pairs by expert id, preserve original order within each expert, and emit:

- `A       [kT·kTopK + 16, kH]`  — packed token rows; row order is expert-major.
- `A_id    [kT·kTopK + 16]`      — back-map: `A_id[r]` is the token id that row `r` of `A` came from.
- `rank_id [kT·kTopK + 16]`      — back-map: `rank_id[r]` is the **top-k slot index** (0..kTopK-1) of the `(t, k)` pair that produced row `r`. Used by the gather kernel to look up `softmax_weight[t, rank_id[r]]` when `kTopK > 1`.
- `expert_count [kE]`            — histogram of expert assignments.
- `expert_start [kE]`            — prefix sum of `expert_count` (where each expert's chunk starts in `A`).

The trailing 16 rows are an **overspill landing pad** for the downstream `expert_ffn`'s last-tile writes. They are not initialized by this kernel.

## Target platform

A3 / Ascend 910B1. Vec target (`--cce-aicore-arch=dav-c220-vec`).

## Auto-mode constraints honored

- Single AICORE (`<<<1, nullptr, stream>>>`).
- Static row tile declared once outside both loops; auto allocator pins its UB address.
- Per-expert `count` / `start` / `counter` held in small local `int32_t[kE]` arrays — stack-resident, no UB tile. (Stack budget is `0x8000 = 32 KB` via `-mllvm -cce-aicore-stack-size`; `3 · kE · 4 B = 384 B` at `kE = 32` is negligible.)
- `pipe_barrier(PIPE_ALL)` at the start of each pack iteration (same hardware-confirmed pattern as the row loops in `topk_kernel.cpp` and the existing `expert_ffn` placeholder).
- GlobalTensor reconstructed per iteration with `(base + runtime offset)`.
- No TASSIGN aliasing, no `Tile::data()` in kernel, no `*_IMPL` calls, no raw CCE intrinsics, no `Event<>`, no manual sync, no `TPipe` / `TPUSH` / `TPOP`, no double buffering.

## How to build / run

```bash
bash run.sh -r npu -v Ascend910B1
```

Sequence:
1. `scripts/gen_data.py` writes inputs and golden binaries.
2. CMake reconfigures and builds.
3. `./scatter` loads inputs, fires the kernel, copies outputs back, and validates.

## Inputs / outputs

| File | Shape | dtype |
|---|---|---|
| `input/input_X.bin` | `kT × kH` | half (fp16) |
| `input/input_expert_id.bin` | `kT × kTopK` | int32 |
| `output/golden_A.bin` | `(kT·kTopK + 16) × kH` | half |
| `output/golden_A_id.bin` | `(kT·kTopK + 16)` | int32 |
| `output/golden_rank_id.bin` | `(kT·kTopK + 16)` | int32 |
| `output/golden_expert_count.bin` | `kE` | int32 |
| `output/golden_expert_start.bin` | `kE` | int32 |

Validation compares the **first `kT·kTopK` rows** of `A` and `A_id` only — the 16-row trailing pad is "don't care" (it's the next kernel's scratch zone).

Since scatter does only bitwise copies, `A` is compared with exact `memcmp` (no fp tolerance).

## Sweeping kTopK / kE / kT / kH

Three places to update together:

- `scripts/gen_data.py`     : `kT`, `kH`, `kE`, `kTopK`
- `main.cpp`                : `constexpr int kT/kH/kE/kTopK`
- `scatter_kernel.cpp`      : `namespace scatter_cfg { constexpr unsigned ... }`

Allowed `kTopK`: 1, 2, 4, 8, 16. Allowed `kE`: 16, 32. Allowed `kT`: 128, 256, 512.

## Known limitations (v1)

- Single AICORE; no block_idx work split.
- fp16 X / A only (cube-compatible dtype for the downstream FFN's TMATMUL).
- Two GM scalar reads of `expert_id[t·kTopK + k]` per `(t, k)` pair (one in pass 1, one in pass 3). Could cache in a small local array, but stack-size pressure at `kT·kTopK = 4096` (kTopK=16) would be `16 KB` — fits but tight.
- No double-buffering.
