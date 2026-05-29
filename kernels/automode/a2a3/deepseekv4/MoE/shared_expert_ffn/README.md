# shared_expert_ffn — DeepSeek-V4 always-on shared expert FFN

Auto-mode A3 **hybrid (cube + vector)** kernel: the same SwiGLU FFN as
`expert_ffn/`, but applied to **all tokens** instead of just the routed
subset ([deepseek/model.py:628, 644](../../../../../deepseek/model.py)).

```python
self.shared_experts = Expert(args.dim, args.moe_inter_dim,
                             swiglu_limit=args.swiglu_limit)
...
y += self.shared_experts(x)        # x: [T, DIM]; no routing, no weight
```

The kernel body is structurally identical to `expert_ffn/`; the
differences are at the driver level:

- **No `expert_count` / `expert_start` arrays** — the whole row range
  `[0, T)` is the single segment.
- **No `weights` input** — `Expert(weights=None)` (model.py:597, 605).
- **Single weight set** — `W1, W2, W3` are 2-D (no `N_ROUTED` axis).

Could share an implementation with `expert_ffn/` in a future fused build:
both kernels would dispatch to a parameterized FFN backend with the
routing-axis size set to 1.

## I/O

| Name | Shape | dtype | Notes |
|------|-------|-------|-------|
| `X`  | `(T, DIM)`        | BF16 | input tokens (all of them) |
| `W1` | `(INTER_DIM, DIM)`| BF16 | gate proj |
| `W3` | `(INTER_DIM, DIM)`| BF16 | up proj |
| `W2` | `(DIM, INTER_DIM)`| BF16 | out proj |
| `Y`  | `(T, DIM)`        | BF16 | out; same dtype as `x.dtype` at model.py:607 |

Shape constants (from `../build/generated_cases.h`):
`kDsmoeT, kDsmoeDim, kDsmoeInterDim`.

> Note: the model casts the FFN result back to `x.dtype` at the end
> (model.py:607). For this prototype, `Y` is BF16; if the upstream
> uses FP32 hidden states, the cast can happen in the gather/post-add
> stage.

## Memory-budget-first plan

Same as `expert_ffn/` (single-expert path, M=kTileM=16; outer loop is
just one expert's count = T):

```text
Memory budgets:
- L1 custom budget:    A_s + W1_t + W3_t + W2_t + Y_t + B_s ≤ 2^17 bytes
- L0A custom budget:   M × H_l0 BF16 + M × F_l0 BF16  ≤ 64 KB
- L0B custom budget:   H_l0 × F_l1 BF16 + F_l0 × N_l1 BF16 ≤ 64 KB
- L0C custom budget:   M × F_l1 FP32 (Y acc) + M × N_l1 FP32 (B acc) ≤ 16 KB
- UB custom budget:    small SiLU/clamp scratch ≤ a few KB
                       (no routing-weight broadcast tile)

Live tiles by memory level:  same as expert_ffn, minus the routing-weight tile.

Smallest hardware operation:  cube 16×16×16 + vector SiLU/clamp.

Loop tiling plan:
- tile-and-loop dimensions: m0 over [0, T) by kTileM=16; inner panels
  as in expert_ffn.

Test-shape plan:
- tiny debug shape:         T=64,  DIM=128,  INTER_DIM=256
- model-inspired realistic: T=256, DIM=4096, INTER_DIM=4096
```

## Auto-mode constraints

Identical to `expert_ffn/`. Mirror the panel-sizing helpers
verbatim; drop the per-expert outer loop (start=0, count=T).

## Risks (`Assumption` / `Unknown`)

- `Assumption`: SwiGLU FFN math matches `expert_ffn/` exactly; the only
  axis change is the missing expert dimension on `W1/W3/W2`.
- `Assumption`: the output dtype matches the input dtype (BF16). The
  PyTorch source casts back via `x.to(dtype)` at model.py:607; for
  the prototype we keep both ends BF16. If the upstream is FP32, the
  driver can up-cast `Y` post-hoc.
- `Unknown`: whether `swiglu_limit` is non-zero in the
  `shared_experts` path. The constructor at model.py:628 passes the
  same `swiglu_limit` value as the routed experts, so we follow suit.

## Build & run

```bash
bash run.sh -r npu -v Ascend910B1
```

## Comparison policy

- Tolerance-based on `Y`: relative ≤ `1e-2`, absolute ≤ `1e-3` (BF16
  GEMM × 2 with FP32 accumulator).

## Status

`Unknown` whether this builds. **No claim of compile / run success**
until user runs `run.sh` on the compiler server and reports the output.
