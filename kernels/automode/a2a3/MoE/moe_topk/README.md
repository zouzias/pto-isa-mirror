# moe_topk

Auto-mode A3 MoE kernel — Stage 2 of 3.

## What it does

Per-token top-K expert selection from router logits:

```
outVal[t, :], outIdx[t, :] = top_k(logits[t, :], k=kTopK)   descending
```

- **logits** `(kRows=256, kCols=32)` float32 — router logits (one row per token)
- **idx** `(kCols=32,)` uint32 — identity row `[0..31]` (shared across all token rows)
- **outVal** `(kRows=256, kTopK=2)` float32 — top-K values, descending
- **outIdx** `(kRows=256, kTopK=2)` uint32 — matching expert indices

The output feeds into `expert_ffn` to route each token to its assigned expert(s).

## Target platform

A3 (`PTO_NPU_ARCH_A2A3`). Vec path: `--cce-aicore-arch=dav-c220-vec`.

## Pipeline per row

With `kCols=32` and `kTopK=2` (float, TYPE_COEF=1):

```
kPackedCols = 32 * 2 = 64
blockLen    = 64
```

1. **TSORT32** — one 32-element block sort → 64 packed `(val, idx)` elements
2. Main merge loop **skipped** (blockLen×4 = 256 > kPackedCols = 64)
3. Tail block **skipped** (blockLen == kPackedCols)
4. **TGATHER P0101** — extract top-2 float values (positions 0, 2)
5. **TRESHAPE + TGATHER P1010** — extract top-2 uint32 indices (positions 1, 3)
6. **TSTORE** outVal + outIdx

TSORT32 alone fully sorts the 32-element row; no merge step is needed.

## Auto-mode constraints

- Single AICORE; no `block_idx` work split.
- `pipe_barrier(PIPE_ALL)` at row-loop start (hardware-confirmed requirement for cross-iter auto-sync gap; same as `topk` kernel).
- Tiles declared **inside** the row loop (per-iter liveness isolation; topk lesson).
- No manual sync beyond the confirmed `pipe_barrier`.

## Known assumptions / risks

| Item | Status |
|------|--------|
| TSORT32 valid for kCols=32 (one full 32-element block) | Inferred from topk (kCols=1280); exact behavior at boundary is assumed |
| OutValTile/OutIdxTile width=2 (8 bytes for float32) | **Assumption**: tile allocator pads to alignment. If not, first failure point. Workaround: increase kTopK to 8 |
| MaskPattern::P0001 for half dtype | Included via `if constexpr` (not tested; float path is primary) |

## How to build and run

```bash
bash run.sh -r npu -v Ascend910B1
```

Expected output: `test value success` + `test index success` + `test success`.

## Known limitations

- Float dtype only (TYPE_COEF=1). Half path included but untested.
- kCols must be exactly 32 for the single-block-sort property.
- Not connected to `router_matmul` input or `expert_ffn` output; standalone.
- No claim of build/run success until user provides compiler output.
