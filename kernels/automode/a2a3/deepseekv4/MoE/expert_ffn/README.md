# expert_ffn — DeepSeek-V4 per-expert SwiGLU FFN

Auto-mode A3 **hybrid (cube + vector)** kernel: SwiGLU FFN over the
expert-grouped packed rows produced by `scatter/`.
([deepseek/model.py:597-607](../../../../../deepseek/model.py)).

```python
gate = self.w1(x).float()                 # [count_e, INTER_DIM]
up   = self.w3(x).float()                 # [count_e, INTER_DIM]
if self.swiglu_limit > 0:
    up   = torch.clamp(up,   min=-swiglu_limit, max=swiglu_limit)
    gate = torch.clamp(gate,                    max=swiglu_limit)
x = F.silu(gate) * up                     # [count_e, INTER_DIM]
if weights is not None:
    x = weights * x                       # per-token routing weight
return self.w2(x.to(dtype))               # [count_e, DIM]
```

Mirrors the top-level
[MoE/expert_ffn/expert_ffn_kernel.cpp](../../../MoE/expert_ffn/expert_ffn_kernel.cpp)
fused-FFN structure; the prototype keeps the same:

- two-GEMM pipeline (`W1 → SiLU/clamp → ⊗ W3 → ⊗ weights → W2`),
- per-expert outer loop over `(e, count_e)` from `expert_count` /
  `expert_start`,
- 16-row token tile (`kTileM=16`) and the `chooseN/F/H` panel-sizing
  helpers,

with these DeepSeek-V4 specific additions:

1. **Two parallel input projections (`W1` for gate, `W3` for up).** The
   top-level kernel fuses gate+up into a single `W1`-shaped weight.
   This scaffold splits them to match the model definition exactly
   (model.py:592, 594).
2. **`swiglu_limit` clamp.** When `> 0`, clamp `up` to
   `[-swiglu_limit, swiglu_limit]` and clamp `gate` to `(-inf, swiglu_limit]`
   before the SiLU. Configurable via `-DDSMOE_SWIGLU_LIMIT=<float>`
   (default `0`, no clamp).
3. **Optional per-token routing weight.** When `weights != nullptr`,
   multiply the intermediate by `weights[m]` before `W2`. Default
   behavior is to skip (the gather-side weighted-sum can absorb this).

## I/O

| Name | Shape | dtype | Notes |
|------|-------|-------|-------|
| `A`            | `(T*N_ACTIVATED + 16, DIM)` | BF16 | scatter output (packed-by-expert) |
| `W1`           | `(N_ROUTED, INTER_DIM, DIM)` | BF16 | gate projection |
| `W3`           | `(N_ROUTED, INTER_DIM, DIM)` | BF16 | up projection |
| `W2`           | `(N_ROUTED, DIM, INTER_DIM)` | BF16 | output projection |
| `weights` (opt)| `(T*N_ACTIVATED + 16)`      | FP32 | per-row routing weight |
| `expert_count` | `(N_ROUTED)`                | INT32 | from scatter |
| `expert_start` | `(N_ROUTED)`                | INT32 | from scatter |
| `B`            | `(T*N_ACTIVATED + 16, DIM)` | FP32 | out; per-expert FFN result, FP32 acc |

Shape constants (from `../build/generated_cases.h`):
`kDsmoeT, kDsmoeDim, kDsmoeInterDim, kDsmoeNRouted, kDsmoeNActivated`.

## Memory-budget-first plan

Per [CLAUDE.md §Memory-budget-first kernel planning](../../../../../CLAUDE.md);
mirrors `MoE/expert_ffn/`:

```text
Memory budgets:
- L1 custom budget:    A_s (M×H_l1 BF16) + W1_t (H_l1×F_l1 BF16) +
                       W3_t (H_l1×F_l1 BF16) + W2_t (F_l1×N_l1 BF16) +
                       Y_t (M×F_l1 BF16) + B_s (M×N_l1 BF16)
                       ≤ 2^17 bytes (128 KB)
- L0A custom budget:   M × H_l0 BF16 + M × F_l0 BF16  ≤ 64 KB
- L0B custom budget:   H_l0 × F_l1 BF16 + F_l0 × N_l1 BF16 ≤ 64 KB
- L0C custom budget:   M × F_l1 FP32 (Y acc) + M × N_l1 FP32 (B acc)
                       ≤ 2^14 bytes (16 KB)
- UB custom budget:    routing-weight broadcast tile (M × 1 FP32) +
                       small SiLU/clamp scratch ≤ a few KB

Live tiles by memory level:
- L1:  A_s, W1_t, W3_t, W2_t, Y_t (post-SiLU intermediate), B_s
- L0A: A panel (M × H_l0), Y panel (M × F_l0)
- L0B: W1 panel (H_l0 × F_l1), W2 panel (F_l0 × N_l1)
- L0C: Y_acc (M × F_l1 FP32), B_acc (M × N_l1 FP32)
- UB:  per-row routing weight (M × 1 FP32) when used; small SiLU scratch

Smallest hardware operation:
- cube operation shape: 16×16×16 fractal MMA (BF16 × BF16 → FP32)
- vector operation shape: SiLU + clamp + (W) broadcast multiply over
  (M × F_l1) tiles in UB; mirrors `gate_softmax` activation composition.

Loop tiling plan:
- tile-and-loop dimensions: outer over (e, m0) per expert segment; inner
  over (n0 ∈ DIM, f1 ∈ INTER_DIM, h1 ∈ DIM); innermost over (h0, f0)
  for the L1→L0 K split.
- inferred tile sizes: M=16, H_l1/F_l1/N_l1 chosen by chooseN/F/H helpers
  (same as MoE/expert_ffn).
- compile-time unroll/peel strategy: none; rely on auto-mode.
- tail handling only where needed: last m0 tile shrinks to count_e mod M.

Test-shape plan:
- tiny debug shape:         T=64,  DIM=128,  INTER_DIM=256,  N_ROUTED=8, N_ACTIVATED=2
- model-inspired realistic: T=256, DIM=4096, INTER_DIM=4096, N_ROUTED=8, N_ACTIVATED=2
```

## Auto-mode constraints

- A3 only; cube path: `--cce-aicore-arch=dav-c220-cube`.
- Single AICORE; no `block_idx` split.
- Mirror the `MoE/expert_ffn` panel-sizing helpers (`chooseN`, `chooseF`,
  `chooseH`, `chooseH0`) byte-for-byte; only the dtype is BF16 instead
  of FP16.
- Reason in 16×16 fractals in L0A/L0B/L0C.
- No `TASSIGN`, `Tile::data()` in kernel, `*_IMPL`, raw CCE intrinsics.

## Risks (`Assumption` / `Unknown`)

- `Assumption`: BF16×BF16→FP32 cube path is available on A3 (same as
  the top-level `MoE/expert_ffn`'s FP16 path, with `half → bfloat16_t`).
- `Assumption`: `swiglu_limit > 0` clamp can be composed via `TMINS` /
  `TMAXS` (or a `TCLAMP` if available) in UB after the `TMOV` from L0C.
- `Assumption`: per-row routing-weight multiply applied AFTER SiLU and
  BEFORE `W2` (matches model.py:605-606: `x = weights * x`).
- `Unknown`: realistic shape `INTER_DIM=4096, DIM=4096` may need more
  conservative panel sizes than the top-level `kF=64, kH=64` defaults.
  Recompute the budget once the BF16 weight footprint is fixed.
- `Unknown`: FP4 weight loading (`ModelArgs.expert_dtype == "fp4"`,
  model.py:624-625) — out of scope for this prototype; assume BF16
  weights are pre-dequantized on the host.

## Build & run

```bash
bash run.sh -r npu -v Ascend910B1
```

## Comparison policy

- Tolerance-based on `B` (FP32 output): relative ≤ `1e-2`, absolute
  ≤ `1e-3` (BF16 GEMM with FP32 accumulator, two cascaded GEMMs + SiLU).

## Status

`Unknown` whether this builds. **No claim of compile / run success**
until user runs `run.sh` on the compiler server and reports the output.
