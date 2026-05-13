# moe_top1_full

## Status

**Skeleton only.** CMake, host driver, kernel signatures + tile types,
Python golden, compare script, and run.sh are final. The bodies of the
five stage kernels are `TODO(body)`. Stage bodies 1 / 3 / 4 / 5 are
byte-for-byte ports of existing milestones; stage 2 is the only genuinely
new composition.

## Purpose

The first **integrated** auto-mode A3 top-1 MoE forward path. Takes
`X, W_router, W1, W2`; produces `Y`. No host-supplied `expert_id`. No
multi-folder pipeline — one project, one executable.

```text
Input  : X[T, H]  W_router[H, E]  W1[E, H, F]  W2[E, F, N]
Output : Y[T, N]            (with N = H by the §A18 design choice)

Steps:
  1. logits  = X @ W_router
  2. expert_id[t]  = argmax_e logits[t, e]
  3. permute tokens by expert
  4. for each expert: hidden = ReLU(packed_tokens_seg @ W1[e]);
                       packed_output_seg = hidden @ W2[e]
  5. unpermute packed_output back to original token order -> Y
```

## Where this sits

```text
X, W_router, W1, W2
    ↓
moe_top1_full        (THIS)        — integrated top-1 MoE forward
    ↓
Y
```

This is the consolidated deliverable. The six previously-passing MoE
milestone folders (`moe_top1_permute/`, `moe_segmented_*/`,
`moe_top1_unpermute/`) remain in tree as bisected proof points for each
sub-pattern; this project composes them.

## Target platform

A3 only (`PTO_NPU_ARCH_A2A3`), cube arch (`--cce-aicore-arch=dav-c220-cube`).

## Tested shape

```text
T = 256    tokens
H = 64     hidden dim (and also the N dim of GEMM2)
F = 64     FFN intermediate dim
E = 16     experts (cube-aligned)
kTileM = 128

T_PADDED_MAX = T + kE * kTileM = 256 + 16 * 128 = 2304   (host scratch upper bound)
```

`E = 16` (not `4`) — matches the FP16 cube alignment of 16 for the router
GEMM N dim, avoiding the validN<N assumption. The downstream FFN works
at any compile-time `kE`; the §A18 shape with `kE = 4` is a strict
subset of this.

## Architecture (5 stages on one stream)

Hybrid composition per the design choice locked in before skeleton work:
**fuse router + permute into one kernel; keep FFN + unpermute as the
proven §A18 split-kernel pattern.**

```text
Stage 1 (cube) — runMoeTop1Stage1RouterGemm
    X @ W_router -> logits[T, E] FP32
    [port of moe_router_top1 stage 1; cube tile budget = 5]

Stage 2 (vec)  — runMoeTop1Stage2ArgmaxPermute     <-- NEW composition
    TROWARGMAX logits -> expert_id
    on-device histogram + padded prefix-sum -> expert_count, expert_start
    pack pass writes token_to_packed and packed_tokens
    [fuses moe_router_top1 stage 2 with §A13 permute]

Stage 3 (cube) — runMoeTop1Stage3Ffn1Gemm1Relu
    Per-expert microtile loop. GEMM1 + fused ReLU + FP32->FP16 TSTORE.
    [byte-for-byte §A18 stage 1]

Stage 4 (cube) — runMoeTop1Stage4Ffn2Gemm2
    Per-expert microtile loop. GEMM2 on ffn_scratch.
    [byte-for-byte §A18 stage 2]

Stage 5 (vec)  — runMoeTop1Stage5Unpermute
    Row gather: Y[t] = packed_output[token_to_packed[t]].
    [byte-for-byte §A14]
```

All five fire on the same ACL stream. There is **no** within-kernel
cross-stage auto-sync involved.

## Why this composition (and not full fusion)

- Cube `TMATMUL` and vec `TROWARGMAX` have not been mixed inside one
  `__global__ AICORE` body anywhere in tree. Stage 1 must be cube and
  stage 2 must do a vec argmax, so they must be separate kernels.
- The §A18 split-kernel FFN composition is the safest known shape. A
  fused single-kernel FFN was attempted previously; the hang observed
  during that experiment was later attributed to hardware. Even with
  that clarification, full fusion would be a new auto-mode surface; the
  hybrid form here only fuses argmax with permute (both vec, both
  proven) and leaves the rest as the §A18 / §A14 known-good shapes.

## Scratch buffers (host-allocated)

The host allocates worst-case device scratch sized for `T_PADDED_MAX`:

| Buffer            | Shape                  | DType   | Notes |
|-------------------|------------------------|---------|-------|
| `packed_tokens`   | `[T_PADDED_MAX, kH]`   | half    | Produced by stage 2; consumed by stage 3.
| `expert_count`    | `[kE]`                 | int32   | PADDED counts.
| `expert_start`    | `[kE]`                 | int32   | PADDED starts.
| `token_to_packed` | `[T]`                  | int32   | Per-token packed position.
| `ffn_scratch`     | `[T_PADDED_MAX, kF]`   | half    | Stage 3 -> Stage 4 hand-off.
| `packed_output`   | `[T_PADDED_MAX, kH]`   | float32 | Stage 4 -> Stage 5.

`T_PADDED_MAX = T + kE * kTileM` is a coarse upper bound (each expert
gets up to `kTileM - 1` padding slots in the worst case). For the
tested shape that is 2304 rows; tightening it is a future optimisation.

## Risks

- **`TRowReduceIdxOps.hpp` PR-852 sync bug** — stage 2 uses TROWARGMAX,
  which is in the header flagged by
  [auto_mode_bad_patterns.md §2.7](../../../../docs_for_ai/auto_mode_bad_patterns.md).
  Diagnostic shape: if `output_logits.bin` matches the golden but
  `output_expert_id.bin` does not, suspect §2.7.
- **Five-kernel stream serialization** is a more complex composition than
  any prior milestone (§A18 had two). Auto-sync per kernel is the same
  as in each parent §A; the cross-kernel ordering is ACL-stream-level.

## Out of scope (per project instruction)

```text
- topK > 1 dispatch / weighted combine
- per-expert capacity / drop / fallback
- multi-core block_idx work split
- dynamic tail handling without host-padded scratch
- activations beyond ReLU
- bias path
- BF16 / INT8 / TF32
- backward pass
- performance characterization
- fused single-kernel form (FFN previously attempted; was hardware-hung once)
```

## How to build / run

```bash
bash run.sh -r npu -v Ascend910B1
python scripts/compare_outputs.py
```

## How to compare against the Python reference

```python
logits = X @ W_router                                  # FP32
expert_id = np.argmax(logits, axis=1)                  # uint32

for t in range(T):
    e = int(expert_id[t])
    hidden_fp32 = X[t] @ W1[e]
    hidden_fp16 = np.maximum(hidden_fp32, 0.0).astype(np.float16)
    Y[t]        = hidden_fp16.astype(np.float32) @ W2[e]
```

See `scripts/gen_data.py` for the seed (np.random.seed(31)) and input
bounds. FP16 inputs are integer-valued in `[-4, 4]` to keep all GEMMs
bit-exact through the FP16-exact integer regime.

## Known limitations (skeleton phase)

- **Bodies not yet written.** Five `TODO(body)` stages. The kernel will not
  produce correct output until the implementation pass.
- Fixed shape `T=256 H=64 F=64 E=16 kTileM=128`.
- Single AICORE per stage; no `block_idx` work split.
- Top-1 only.
- Worst-case `T_PADDED_MAX` is coarse.
