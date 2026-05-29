# hc_mix_gemm — `mixes = (x_flat @ hc_fn^T) * rsqrt(...)`

Auto-mode A3 **cube** kernel that produces the HC mixing logits consumed by
`hc_split_sinkhorn`. This is the first op inside `Block.hc_pre`
([model.py:677-679](../../../../../deepseek/model.py)):

```python
x_flat = x.flatten(2).float()                                  # [B*S, HC_MULT*DIM]
rsqrt  = torch.rsqrt(x_flat.square().mean(-1, keepdim=True)
                     + self.norm_eps)                          # [B*S, 1]
mixes  = F.linear(x_flat, hc_fn) * rsqrt                       # [B*S, MIX_HC]
```

## Reference (PyTorch / TileLang)

- PyTorch path: `F.linear(x_flat, hc_fn)` at
  [model.py:679](../../../../../deepseek/model.py).
- The HC `hc_fn` parameter is declared FP32 (per
  [model.py:666-672](../../../../../deepseek/model.py) — `with set_dtype(torch.float32)`).
- `x_flat` is the flatten-over-(hc_mult, dim) view of the HC-replicated state
  ([model.py:677](../../../../../deepseek/model.py)).
- There is no dedicated TileLang kernel for this matmul; it dispatches through
  the generic `linear()` path
  ([kernel.py linear()](../../../../../deepseek/kernel.py)).

## I/O

| Name | Shape | dtype | Source line |
|------|-------|-------|-------------|
| `x_flat` | `(M = B*S, K = HC_MULT*DIM)` | FP32 | model.py:677 |
| `hc_fn`  | `(N = MIX_HC, K = HC_MULT*DIM)` | FP32 | model.py:667-668 |
| `rsqrt`  | `(M,)` | FP32 | model.py:678 (precomputed on host for this scaffold) |
| `mixes`  | `(M, N = MIX_HC)` | FP32 | model.py:679 |

`MIX_HC = (2 + HC_MULT) * HC_MULT`. With model defaults (`HC_MULT=4`),
`MIX_HC = 24`, `K = 4 * 4096 = 16384`.

Note: `rsqrt` is supplied pre-computed (`mean(x_flat^2) + eps -> rsqrt`).
The scaffold leaves the per-row `* rsqrt` multiply as a **vector epilogue**
inside the cube path's tail — implementer should fold it in once the cube
core works. Computing `rsqrt` inline would require a vector-side reduction
of `x_flat^2` over the inner dim, which is not the cube core's job.

## Memory-budget-first plan

Per [CLAUDE.md §Memory-budget-first kernel planning](../../../../../CLAUDE.md):

```text
Memory budgets:
- L1 custom budget:   FP32 A panel + FP32 B panel + slack (target <= 256 KB)
- L0A custom budget:  one FP32 A fractal-tile [kTileM, kTileK]   (cube)
- L0B custom budget:  one FP32 B fractal-tile [kTileN, kTileK]   (cube)
- L0C custom budget:  one FP32 C fractal-tile [kTileM, kTileN]   (cube accumulator)
- UB custom budget:   one FP32 C tile staged for vector epilogue (* rsqrt) + write-back

Live tiles by memory level:
- L1:  Atile (FP32, M x K_inner), Btile (FP32, N x K_inner)
- L0A: ATile (FP32, kTileM x kTileK fractal)
- L0B: BTile (FP32, kTileN x kTileK fractal)
- L0C: CTile (FP32, kTileM x kTileN fractal accumulator)
- UB:  CTile_fp32 staged for GM write, plus rsqrt[m0:m0+kTileM]

Smallest hardware operation:
- cube operation shape: 16x16x16 fractal MMA (FP32 x FP32 -> FP32)
                        Assumption: FP32 cube MMA available on A3; cross-check
                        docs_for_ai/tile_type_reference.md before commit.
- vector operation shape: VL FP32 (per-row multiply by rsqrt scalar)

Loop tiling plan:
- tile-and-loop dimensions: M tiled by kTileM=16, N tiled by kTileN=24 (= MIX_HC,
  single N tile), K reduced by kTileK=64 (K-inner loop over HC_MULT*DIM=16384)
- inferred tile sizes: kTileM=16, kTileN=24, kTileK=64  (Assumption — pending
  profile / L1/L0 fit check; N=24 is tiny relative to typical cube N=128)
- compile-time unroll/peel: none for prototype (single AICORE, simple loop)
- tail handling: M tail when (B*S) % kTileM != 0 (auto-mode valid-region);
  K is a known multiple of 64 (DIM=4096, HC_MULT=4 -> K=16384 = 256 * 64)
  so no K tail in the realistic case. For the tiny case (DIM=64), K=256,
  also a clean multiple of 64.

Test-shape plan:
- tiny debug shape:     B=1, S=16,  DIM=64,   HC_MULT=4 -> M=16,  K=256,   N=24
- medium tiling shape:  B=1, S=64,  DIM=256,  HC_MULT=4 -> M=64,  K=1024,  N=24
- model-inspired realistic: B=1, S=128, DIM=4096, HC_MULT=4 -> M=128, K=16384, N=24
- tail shape: B=1, S=37 (forces M tail), DIM=64, HC_MULT=4 -> M=37, K=256, N=24
```

## Auto-mode constraints

- A3 only; cube path: `--cce-aicore-arch=dav-c220-cube` (see [CMakeLists.txt](CMakeLists.txt)).
- Single AICORE. No `block_idx` work split.
- No `TASSIGN`, `TPipe`, `TPUSH`/`TPOP`, no raw `set_flag`/`wait_flag`,
  no `Tile::data()` pointer casts.
- `hc_fn` is reloaded every M-tile iteration — same pattern as
  [MoE/router_matmul](../../../../MoE/router_matmul/) (bMatTile reload).
- Reason in 16x16 fractals in L0A/L0B/L0C; **do not** treat tiles as flat
  row-major matrices.
- N = MIX_HC = 24 with HC_MULT=4. Pad to 32 if the cube fractal granularity
  requires it (`Assumption`: 24 may not be a clean multiple of 16; the
  16-aligned `N_padded = 32` is the likely target — cross-check
  `docs_for_ai/tile_type_reference.md` before committing tile shapes).

## Build & run

```bash
bash run.sh -r npu -v Ascend910B1
```

Family-level case manifest at `../build/generated_cases.{h,json}` is
regenerated by `../scripts/generate_cases.py` before the per-leaf build.

## Comparison policy

Tolerance-based: relative <= `1e-3` on FP32 output (FP32 GEMM vs FP32 numpy
reference; small ULP drift expected from cube fractal accumulation order
over `K = HC_MULT * DIM` summands).

## Known limitations

- `rsqrt` multiply is staged as a vector epilogue placeholder (see body
  comments in `hc_mix_gemm_kernel.cpp`). Implementer must fold it in.
- Realistic K=16384 is large; the K-inner loop count is 256 with kTileK=64.
- HC_MULT is assumed fixed to 4 by the model
  ([model.py:651](../../../../../deepseek/model.py)).

## Status

`Unknown` whether this builds — kernel body is a pseudocode skeleton, not a
real implementation. **No claim of compile / run success** until user runs
`run.sh` on the compiler server and reports the output.
