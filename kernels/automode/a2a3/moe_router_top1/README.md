# moe_router_top1

## Status

**Skeleton only.** CMake, host driver, kernel signatures + tile types,
Python golden, and compare script are final. The bodies of both stage
kernels (cube GEMM and vec TROWARGMAX) are `TODO(body)`.

## Purpose

The missing router stage of the top-1 MoE pipeline. The previously-proven
six MoE milestones all assume `expert_id[t]` is supplied; this kernel
produces it from `X` and `W_router`:

```text
logits  = X @ W_router        (cube GEMM, FP16×FP16 -> FP32 acc)
expert_id[t] = argmax_e logits[t, e]
```

## Where this sits in the MoE pipeline

```text
X, W_router
    ↓
moe_router_top1                (THIS)    — produces expert_id
    ↓
moe_top1_permute               (§A13)
    ↓
moe_segmented_ffn_top1         (§A18)
    ↓
moe_top1_unpermute             (§A14)
```

## Target platform

A3 only (`PTO_NPU_ARCH_A2A3`), cube arch (`--cce-aicore-arch=dav-c220-cube`).
The TU compiles with the cube flag; stage 2 is vec-only but compiles fine
under the cube build (per §A18).

## Tested shape

```text
T = 256    tokens (rows of X)
H = 64     hidden dim
E = 16     experts   <-- chosen to match FP16 cube blockAlign of 16; avoids
                         the validN<N assumption that E=4 would force.
kTileM = 128
```

All other MoE folders previously used `E = 4`. This kernel and
`moe_top1_full` use `E = 16` because the cube GEMM minimum N for FP16
is 16. Downstream MoE kernels (`moe_top1_permute`, `moe_top1_unpermute`,
`moe_segmented_ffn_top1`) are E-templated and accept the new value once
the test inputs are regenerated.

## Composition

Two `__global__ AICORE` kernels in one TU, fired back-to-back on the
same stream (mirrors the §A18 split-kernel pattern):

```text
Stage 1 (cube) — runRouterStage1Gemm
    X[T, H] @ W_router[H, E] -> logits[T, E] (FP32)

Stage 2 (vec)  — runRouterStage2Argmax
    TROWARGMAX(expert_id[T, 1], logits[T, E], tmp)
```

ACL stream-order semantics guarantee stage 2 only starts after stage 1's
TSTORE has fully drained to GM. There is **no** within-kernel cross-stage
auto-sync involved.

## Why split, not fused

Fused single-kernel router would mix cube `TMATMUL` and vec `TROWARGMAX`
in one `__global__ AICORE` body. No in-tree kernel has done that
combination. The split-kernel form reuses two patterns that are each
already proven (§A16 / §A18 cube GEMM; §A12 vec TROWARGMAX precedent in
the manual TopK). Fusing them is on the table for a future iteration but
out of scope for v1.

## Risks

- **`TRowReduceIdxOps.hpp` PR-852 sync bug** — TROWARGMAX is implemented in
  this header, which is one of the four flagged in
  [auto_mode_bad_patterns.md §2.7](../../../../docs_for_ai/auto_mode_bad_patterns.md).
  This branch does NOT have PR-852 merged. **Diagnostic shape**: if
  `output_logits.bin` matches the golden but `output_expert_id.bin` does
  not, the bug is in stage 2. If logits also mismatch, debug stage 1 first.

- **Cube N=E=16 sits exactly on the FP16 alignment boundary**. The `validN`
  parameter equals `N`, so the validN<N assumption is not exercised here.

## How to build / run

```bash
bash run.sh -r npu -v Ascend910B1
```

The host driver runs the comparison in-process (via `PtoTestCommon::ResultCmp`)
and prints `test success` / `test failed` at exit — matches §A18.
`scripts/compare_outputs.py` is a stand-alone diagnostic for detailed
mismatch reports.

## How to compare against the Python reference

```python
logits   = (X.astype(np.float32) @ W_router.astype(np.float32)).astype(np.float32)
expert_id_ref = np.argmax(logits, axis=1).astype(np.uint32)
```

See `scripts/gen_data.py` for the exact generator (np.random.seed(29)) and
input bounds. FP16 inputs are integer-valued in `[-4, 4]` to keep the
GEMM bit-exact.

## Known limitations (skeleton phase)

- **Bodies not yet written.** Both stage kernels have `TODO(body)`; the
  kernel will not produce correct output until the implementation pass.
- Fixed shape `T=256 H=64 E=16` for the current main.cpp / golden. Other
  shapes need a recompile.
- Single AICORE per stage; no `block_idx` work split.
- FP16 inputs only; FP32 input variant deferred.
- Top-1 only; topK > 1 deferred (and would integrate `router_topk_small`).
- No softmax (`argmax(softmax(x)) == argmax(x)` for top-1 routing).
