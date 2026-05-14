# router_matmul

Auto-mode A3 MoE kernel — Stage 1 of 3.

## What it does

Computes MoE router logits:

```
logits[t, e] = X[t, :] @ W_router[:, e]   for all tokens t, experts e
```

- **X** `(kT=256, kH=64)` float16 — token feature vectors
- **W_router** `(kH=64, kE=32)` float16 — router projection matrix
- **logits** `(kT=256, kE=32)` float32 — raw scores (one per token per expert)

This is a pure GEMM with FP16 inputs and FP32 accumulator. The output feeds into `moe_topk` to select the top-K experts per token.

## Target platform

A3 (`PTO_NPU_ARCH_A2A3`). Cube path: `--cce-aicore-arch=dav-c220-cube`.

## Auto-mode constraints

- Single AICORE; no `block_idx` work split.
- No manual sync (`set_flag`, `wait_flag`, `pipe_barrier`).
- No `TASSIGN`, `TPipe`, `TPUSH`, `TPOP`.
- Tiles declared outside the token loop (add_tile_array pattern); GlobalTensor views recomputed per iteration.
- `W_router` bGlobal view constructed once; reloaded every token-tile iteration (same as moe_segmented_gemm_relu bMatTile reload, confirmed-working).

## Shapes and tile sizes

| Dimension | Value | Note |
|-----------|-------|------|
| kT        | 256   | total tokens, must be kTileM-multiple |
| kH        | 64    | d_model |
| kE        | 32    | num_experts |
| kTileM    | 128   | cube M dimension |
| Iterations | 2    | kT / kTileM |

Alignment: `blockAlign = C0_SIZE_BYTE / sizeof(half) = 16`. kH=64, kE=32, kTileM=128 are all 16-aligned.

## How to build and run

```bash
bash run.sh -r npu -v Ascend910B1
```

This runs `gen_data.py` (creates `input/` and `output/` directories), builds, and executes. Expected output: `test data success`.

## How to compare against Python reference

```bash
python scripts/gen_data.py   # regenerate golden
```

Or manually inspect `output/golden_logits.bin` vs `output/output_logits.bin` using numpy.

## Known limitations

- kT must be an exact multiple of kTileM=128.
- No softmax or temperature scaling (raw logits only).
- Not connected to `moe_topk`; standalone prototype.
- No claim of build/run success until user provides compiler output.
