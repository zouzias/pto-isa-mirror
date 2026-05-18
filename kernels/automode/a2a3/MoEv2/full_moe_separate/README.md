# full_moe_separate

End-to-end MoE forward pass driven by **five separate kernel launches** with explicit `aclrtSynchronizeStream` between each stage. Sister folder of [full_moe_combined](../full_moe_combined/), which runs the same pipeline through **one host wrapper launcher** and only one final sync.

## What it does

```
logits   = X @ W_router                                  (router_matmul)
top_idx, top_val = topk(logits, kTopK)                   (moe_topk_padded)
A, A_id, rank_id, count, start = pack(X, top_idx)        (scatter)
B        = FFN(A, W1, W2, count, start)                  (expert_ffn, cube + cube)
weights  = softmax(top_val, axis=1)
C[t]     = Σ_k weights[t, k] * B[r where A_id[r]=t,      (gather)
                                  rank_id[r]=k]
```

For `kTopK == 1` the gather softmax is degenerate (weight = 1.0) and the kernel takes the direct row-reorder fast path.

## Differences vs full_moe_combined

| | full_moe_separate | full_moe_combined |
|---|---|---|
| Host syncs | one after **each** of the 5 launches | one **final** sync only |
| outVal padding bridge | host-side memcpy (compact → padded) | device-side `outval_pad` kernel |
| Single-launcher wrapper | no — main.cpp calls each launcher explicitly | yes (`launchFullMoeCombined`) |
| Number of kernel `.so` libs | 5 (copies of the originals) | 6 (5 originals + `outval_pad`) |

The KERNEL CODE for the 5 stages is identical between the two folders (both use local copies of the originals in `./kernels/`). The difference is purely host orchestration.

## Target platform

A3 / Ascend 910B1. Stages compiled with mixed targets:

| Stage | Target |
|---|---|
| router_matmul | `--cce-aicore-arch=dav-c220-cube` |
| moe_topk_padded | `--cce-aicore-arch=dav-c220-vec` |
| scatter | `--cce-aicore-arch=dav-c220-vec` |
| expert_ffn | `--cce-aicore-arch=dav-c220-cube` |
| gather | `--cce-aicore-arch=dav-c220-vec` |

## How to build and run

```bash
bash run.sh -r npu -v Ascend910B1
```

Sequence:
1. `scripts/gen_data.py` writes `input/input_*.bin` and `output/golden_C.bin`.
2. CMake reconfigures and builds the 5 kernel `.so` files + the host executable.
3. `./full_moe_separate` loads inputs, copies to device, fires 5 stages with sync between each, copies `C` back, validates against golden.

Validation tolerance is `5e-2` abs — generous because the full pipeline stacks fp16 matmul rounding, fp32 accumulation, fp32→fp16 cast between the two FFN GEMMs, softmax, and the final weighted sum.

## Sweeping kT / kH / kF / kE / kTopK

Use the local `sweep.sh` to patch constants across `main.cpp`, `scripts/gen_data.py`, and `kernels/*_kernel.cpp` in this folder only (does NOT touch the original 5 MoE/* kernel folders):

```bash
# One shape
bash sweep.sh 256 64 64 32 2 npu Ascend910B1

# kTopK axis
for k in 1 2 4 8 16; do bash sweep.sh 256 64 64 32 $k npu Ascend910B1; done
```

Allowed values: `kTopK ∈ {1, 2, 4, 8, 16}`, `kE ∈ {16, 32}`, `kT ∈ {128, 256, 512}`, `kH = kF ∈ {64, 128}`. `kH = kF = 256` needs Split-K — postponed.

`git checkout -- .` reverts the patched files.

## Pipeline details

### Stage 1 — router_matmul (cube)

`logits = X @ W_router`. Inputs fp16, fp32 accumulator. Standard cube TMATMUL loop over `kTileM = 128` token tiles. See [../router_matmul/](../router_matmul/) for the kernel.

### Stage 2 — moe_topk_padded (vec)

Per-row top-K of `logits`. Outputs `outVal` (kT × kTopK fp32, descending sorted) and `expert_id` (kT × kTopK uint32). See [../moe_topk_padded/](../moe_topk_padded/).

### Host bridge — outVal pad

The gather kernel's softmax operates on a (kT, kPadded) tile where `kPadded = max(8, kTopK)` and cols `kTopK..kPadded-1` must hold `-1e30` so they neutralize the per-row softmax (`exp(-1e30 - real_max) = 0`). `moe_topk_padded` writes the compact `(kT, kTopK)` form, so this host bridge:

1. Pre-fills the host-side padded buffer with `-1e30`.
2. Reads compact form back to host.
3. Memcpys the first `kTopK` floats of each row from compact into the padded buffer's first `kTopK` columns (the `kTopK..kPadded-1` cols stay `-1e30`).
4. Copies the padded buffer back to device.

This is one D2H + one H2D round-trip per pipeline run. The full_moe_combined folder avoids it by running a small device-side `outval_pad` kernel.

### Stages 3–5

`scatter` → `expert_ffn` (two cube launches internally) → `gather`. See each folder's README for the kernel-side detail.

## Known limitations

- All kernels are single-AICORE; no `block_idx` parallelism.
- `kH = kF = 256` requires Split-K in `expert_ffn` (postponed).
- For `kTopK == 1` the gather softmax path is skipped (fast path); the rest of the pipeline doesn't change.

## Pattern sources

- [../router_matmul/](../router_matmul/) (cube GEMM)
- [../moe_topk_padded/](../moe_topk_padded/) (top-K with kGatherWidth padding)
- [../scatter/](../scatter/) (histogram + prefix + pack with `rank_id`)
- [../expert_ffn/](../expert_ffn/) (two-stage cube FFN with NormalRelu fuse)
- [../gather/](../gather/) (v2 gather: softmax, reorder to `D[t,k,h]`, vector weighted combine)
