# router_topk_small

## Status

**Skeleton only.** Files (CMake, host main, kernel signatures + tile types,
Python golden, compare script, run.sh) are final. The body of the K-pass
argmax + suppression loop is `TODO(body)` and will be filled in the
implementation pass.

## Purpose

Prepare a row-wise small-K TopK kernel for future MoE work where
`topK > 1`. The existing in-tree TopK ([§A12](../../../../docs_for_ai/known_good_kernel_examples.md))
is tuned for large K and large columns; for routing we expect
`E ≤ 32` and `K ≤ 10` per row, so a different algorithm may be cheaper.

```text
input :  scores       [T, E] float32
output:  topk_values  [T, K] float32   (descending per row)
output:  topk_indices [T, K] uint32    (descending per row)
```

This kernel is **not** integrated into the MoE pipeline yet. It is a
standalone experiment to identify the right small-K shape and to discover
any pre-PR-852 quirks in `TROWARGMAX` before we depend on it in
`moe_router_top1` and `moe_top1_full`.

## Target platform

A3 only (`PTO_NPU_ARCH_A2A3`), vec arch (`--cce-aicore-arch=dav-c220-vec`).

## Tested shape

```text
T = 256
E = 16
K = 4
```

The kernel template parameter `kK` is set at compile time. Future variants
may need `K = 1, 2, 8`; that requires recompiling with the new template
arg, not a runtime change.

## Algorithmic choice (open)

Three candidate algorithms — final pick lands in the implementation pass.

### (a) K passes of `TROWARGMAX` + `TSCATTER` suppression

```text
scratch := scores                              [T, E]
for k = 0..K-1:
    TROWARGMAX(rowVal[T,1], rowIdx[T,1], scratch, tmp)
    outVal[:, k] := rowVal
    outIdx[:, k] := rowIdx
    TSCATTER(scratch, rowIdx, -inf_tile)       # per-row scatter
```

Pros: uses a native primitive for the hot path (argmax). Cons: TSCATTER's
per-row semantics not yet exercised in any auto-mode kernel in tree; needs
verification.

### (b) K passes of `TROWARGMAX` + ramp-mask suppression (currently sketched)

```text
build colId[T, E] = row-broadcast of [0..E-1]  (once, FP32)
scratch := scores
for k = 0..K-1:
    TROWARGMAX(rowVal[T,1], rowIdx[T,1], scratch, tmp)
    outVal[:, k] := rowVal
    outIdx[:, k] := rowIdx
    # Per-row mask: 1 where colId == rowIdx, 0 elsewhere
    # Broadcast rowIdx[T,1] -> [T,E], compare to colId, multiply by big neg
    rowIdxBroad := broadcast(rowIdx, [T, E])
    mask := (colId == rowIdxBroad)              # 0/1 in FP32
    scratch -= MASK_BIG * mask                  # drop picked column to -inf
```

Pros: pure vec ops, no scatter. Cons: needs broadcast + element-wise
compare, which means more vec passes per loop iteration. Whether `TEQ` (or
equivalent) exists as a user-facing wrapper is open.

### (c) Pre-sort the row with `TSORT32`, `TGATHER` first K

Reuses the §A12 TopK shape verbatim, just specialised to E ≤ 32 and
emitting only K (not E) results. Slightly heavier than algorithms (a)/(b)
for K << E, but zero novel auto-mode surface.

Pros: every primitive proven by §A12. Cons: a 32-element sort to extract
4 elements is asymptotically wasteful.

### Recommendation (to revisit at body time)

Start with **(b)** because it touches no new auto-mode surface beyond
`TROWARGMAX`. If TROWARGMAX hits the §2.7 PR-852 bug, drop to **(c)**.

## Risks

- **`TRowReduceIdxOps.hpp` PR-852 sync bug** — `TROWARGMAX`'s implementation
  is one of the four headers flagged in
  [auto_mode_bad_patterns.md §2.7](../../../../docs_for_ai/auto_mode_bad_patterns.md).
  This branch does NOT have PR-852 merged. If the first run produces wrong
  values or indices, that is the most likely cause; fall back to algorithm
  (c) and log the occurrence in
  [compile_error_logbook.md §E2](../../../../docs_for_ai/compile_error_logbook.md).

- **Suppression instruction availability** — algorithms (a) and (b) need
  TSCATTER or TEQ-style ops that may or may not be auto-mode-callable
  from kernel code. Until validated by build, treat both as Unknown.

## How to build / run

```bash
bash run.sh -r npu -v Ascend910B1
```

`run.sh` regenerates input/output through `scripts/gen_data.py`, builds with
`cmake … --cce-aicore-arch=dav-c220-vec --cce-enable-pto-passes`, runs the
kernel, dumps device output to `output/output_topk_*.bin`, and then the host
driver itself prints `test success` / `test failed` after comparing against
the goldens via `PtoTestCommon::ResultCmp` (matches §A18 moe_segmented_ffn_top1).
`scripts/compare_outputs.py` is a stand-alone diagnostic tool — run it
manually when you want a detailed mismatch report.

## How to compare against the Python reference

```python
order = np.argsort(-scores, axis=1, kind="stable")
topk_values  = np.take_along_axis(scores, order[:, :K], axis=1)
topk_indices = order[:, :K]
```

See `scripts/gen_data.py` for the exact generator and seed (np.random.seed(23)).

## Known limitations (skeleton phase)

- **Body not yet written.** The K-pass argmax + suppression loop is a TODO;
  the kernel will not produce correct output until the implementation pass.
- Fixed shape `T=256 E=16 K=4` for the current main.cpp / golden. Other
  shapes need a recompile.
- Single AICORE; no `block_idx` work split.
- FP32 scores only. FP16 path is a template instantiation away but not
  exercised in main.cpp.
- No interaction with the MoE pipeline; this is a standalone experiment.
