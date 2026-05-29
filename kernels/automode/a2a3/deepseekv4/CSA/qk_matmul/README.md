# qk_matmul — sparse_attn Q @ K^T stage

Auto-mode A3 **cube** kernel scaffold. Stage 2 of the FlashAttention-style
sparse-attention pipeline (`sparse_attn_kernel` at
[deepseek/kernel.py:277-352](../../../../../../deepseek/kernel.py)).

## Pipeline position

This leaf runs once per pipelined inner-block iteration (kernel.py:321
`for t in T.Pipelined(num_blocks)`), consuming `q_shared` (loaded once before
the loop) and `kv_shared` (refreshed by `gather_kv` each iteration). It
produces `acc_s` for the same iteration, which `online_softmax` then updates.
The outer-loop driver is not in this scaffold; this binary runs **one
iteration** of the pipeline.

## Op (kernel.py:328-330)

```python
T.gemm(q_shared, kv_shared, acc_s, transpose_B=True, policy=FullRow)
for i, j in T.Parallel(h, block):
    acc_s[i, j] *= scale
```

We deliberately fuse the post-GEMM scale into this leaf (host writes
`softmax_scale` to a side-car FP32 file and the launcher passes it as an
argument). The masking step at kernel.py:326-327 (`acc_s = -inf where
idx==-1`) is deferred to `online_softmax` — that leaf can apply a large
negative bias per masked column without crossing the cube/vector boundary.

## I/O

| Name | Shape | dtype | Source |
|------|-------|-------|--------|
| `q` | `(H, D)` | BF16 | kernel.py:302 (`q_shared`) |
| `kv_block` | `(BLOCK, D)` | BF16 | kernel.py:303 (`kv_shared` — one gathered block) |
| `acc_s` | `(H, BLOCK)` | FP32 | kernel.py:308 (`acc_s` after scale) |
| `softmax_scale` | scalar | FP32 | kernel.py:286 (`(1/d)**0.5` if `scale is None`) |

## Memory-budget-first plan

Per [CLAUDE.md §Memory-budget-first kernel planning](../../../../../../CLAUDE.md):

```text
Memory budgets:
- L1 custom budget:    BF16 Q panel + BF16 K^T panel (~H*D*2 + BLOCK*D*2 bytes)
- L0A custom budget:   one BF16 [H, D] fractal panel
- L0B custom budget:   one BF16 [BLOCK, D] fractal panel (acts as K^T)
- L0C custom budget:   one FP32 [H, BLOCK] fractal accumulator
- UB custom budget:    none (no vector ops on cube core)

Live tiles by memory level:
- L1:  qL1 (BF16, H x D), kL1 (BF16, BLOCK x D)
- L0A: qL0A (BF16, H x D fractal)
- L0B: kL0B (BF16, BLOCK x D fractal, transposed view)
- L0C: cL0C (FP32, H x BLOCK fractal accumulator)
- UB:  N/A on cube core

Smallest hardware operation:
- cube operation shape: 16x16x16 fractal MMA (BF16 x BF16 -> FP32)
- vector operation shape: N/A in this leaf

Loop tiling plan:
- tile-and-loop dimensions: single tile for tiny shape (H=4..16, BLOCK<=64, D<=128)
- inferred tile sizes: kTileM = H, kTileN = BLOCK, kTileK = D (one-shot)
- compile-time unroll/peel strategy: none for prototype
- tail handling only where needed: none — sizes are multiples of 16 after
  the head-padding at kernel.py:359

Test-shape plan:
- tiny debug shape:        H=4,  D=32,  BLOCK=32        (DEFAULT — single fractal)
- medium tiling shape:     H=16, D=64,  BLOCK=64        (one fractal panel each)
- model-inspired realistic: H=16, D=128, BLOCK=64       (matches kernel.py defaults)
- tail shape:              n/a (multi-tile tiling deferred)
```

## Auto-mode constraints

- A3 only; cube path: `--cce-aicore-arch=dav-c220-cube` (see
  [CMakeLists.txt](CMakeLists.txt)).
- Single AICORE. No `block_idx` work split.
- No `TASSIGN`, `TPipe`, `TPUSH`/`TPOP`, no raw `set_flag`/`wait_flag`,
  no `Tile::data()` pointer casts.
- Reason in 16x16 fractals in L0A/L0B/L0C; **do not** treat tiles as flat
  row-major matrices.
- `transpose_B=True`: anchor on the TileLeft / TileRight semantics in
  `docs_for_ai/tile_type_reference.md` — the B matrix uses `TileRight`.

## Build & run

```bash
bash run.sh -r npu -v Ascend910B1
```

## Comparison policy

- Tolerance-based: relative <= `1e-2` on FP32 output (BF16 GEMM with FP32
  accumulator + one scale multiply).

## Known limitations

- One block per binary; outer pipelined loop deferred.
- Masking (`idx==-1` → `-inf`) deferred to `online_softmax`.
- Tiny default `H=4` is below the kernel.py:359 head-padded minimum (16).
  Real binaries should be built with `--cases "16,128,512,64"` for the
  realistic shape.

## Status

`Unknown` whether this builds — kernel body is a pseudocode skeleton, not a
real implementation. **No claim of compile / run success** until user runs
`run.sh` on the compiler server and reports the output.
