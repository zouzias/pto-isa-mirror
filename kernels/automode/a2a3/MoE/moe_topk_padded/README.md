# moe_topk_padded

Auto-mode A3 prototype. Per-row top-K selection over MoE router logits, generic over `kTopK ∈ {1, 2, 4, 8, 16}`.

## What it does

For each token row `t` of `logits ∈ R^{kT × kE}`, produces the descending top-K values and matching expert indices.

```
outVal[t, :] = sort_descending(logits[t, :])[:kTopK]
outIdx[t, :] = argsort_descending(logits[t, :])[:kTopK]
```

## Target platform

A3 / Ascend 910B1. Vec target (`--cce-aicore-arch=dav-c220-vec`).

## Auto-mode constraints honored

- Single AICORE (`<<<1, nullptr, stream>>>`).
- Static tile shapes; `pipe_barrier(PIPE_ALL)` at the start of each row iteration (same hardware-confirmed pattern as [topk_kernel.cpp:109](../../../topk/topk_kernel.cpp#L109)).
- All tiles declared inside the row loop (per-iter liveness isolation).
- No `TASSIGN` aliasing tricks, no `Tile::data()` in kernel, no `*_IMPL` calls, no raw CCE intrinsics, no `Event<>`, no manual sync, no `TPipe`/`TPUSH`/`TPOP`, no double buffering.

## Why the "padded k" trick

`pto_tile.hpp` asserts the output tile's `Cols * sizeof(DType) % 32 == 0`. For `kTopK ∈ {1, 2, 4}` (4 / 8 / 16 bytes) that fails. We declare the output tile with static `Cols = max(8, kTopK)` so the alignment assertion passes, then set its dynamic valid region to `(1, kTopK)`. `TGATHER` and `TSTORE` only emit valid-region elements, so only `kTopK` values per row land in GM.

For `kTopK = 8` and `kTopK = 16` the natural width is already 32-byte-aligned, so `kGatherWidth == kTopK` and there's no padding.

## How to build

```bash
bash run.sh -r npu -v Ascend910B1
```

The script:
1. Runs `scripts/gen_data.py` (numpy reference, writes input + golden binaries).
2. Reconfigures and rebuilds in `build/`.
3. Runs `./moe_topk_padded`, which loads inputs, fires the kernel, copies outputs back, and compares against golden.

## How to compare against the Python reference

`scripts/gen_data.py` emits:

- `input/input_logits.bin`  (kT × kE float32)
- `input/input_idx.bin`     (kE uint32, identity row [0..kE-1])
- `output/golden_val.bin`   (kT × kTopK float32)
- `output/golden_idx.bin`   (kT × kTopK uint32)

The C++ driver writes the kernel result to `output/output_{val,idx}.bin` and runs `ResultCmp` against the golden. Prints `test data success` / `test data failed`.

## Sweeping kTopK

Three constants must move together for any kTopK change:

- `scripts/gen_data.py`            : `kTopK = ...`
- `main.cpp`                       : `constexpr int kTopK = ...;`
- `moe_topk_padded_kernel.cpp`     : `constexpr int kTopK = ...;` inside `launchMoeTopkPadded<T>`

Allowed values: 1, 2, 4, 8, 16. Then `bash run.sh -r npu -v Ascend910B1`.

## Known limitations (v1)

- Single AICORE; no block_idx work split.
- Float32 only (`TYPE_COEF = 1`).
- `kE` must be exactly 32 (one TSORT32 block; no merge / tail).
- `kTopK` must be in `{1, 2, 4, 8, 16}`.
- No double-buffering.

Tiebreaker order between equal logit values follows NumPy's stable argsort; the kernel's TSORT32-based ordering may differ on ties, in which case the comparison still passes at the value level but fails on the index. If observed, the test driver prints which rows have index mismatches but value matches.
