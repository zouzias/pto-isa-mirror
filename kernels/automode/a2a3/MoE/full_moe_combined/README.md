# full_moe_combined

End-to-end MoE forward pass driven by **one host wrapper launcher** (`launchFullMoeCombined`) that fires six `__global__ AICORE` kernels on the same stream — no intermediate `aclrtSynchronizeStream` calls, only one final sync. Sister folder of [full_moe_separate](../full_moe_separate/), which runs the same pipeline through five explicit per-stage syncs.

## What it does

```
launchFullMoeCombined(...)  // one host call from main.cpp
  → launchRouterMatmulFp16     (cube)
  → launchMoeTopkPadded<float> (vec)
  → launchOutValPad<float>     (vec)   // device-side compact -> padded bridge
  → launchScatterFp16          (vec)
  → launchExpertFfnFp16        (fused cube)
  → launchGather<float>        (vec)
aclrtSynchronizeStream(stream)         // one final sync only
```

For `kTopK == 1` the gather softmax is degenerate (weight = 1.0) and `gather` takes its fast path; the output is bit-equivalent to an unweighted sum.

## Differences vs full_moe_separate

| | full_moe_separate | full_moe_combined |
|---|---|---|
| Host syncs | one after each of 5 launches | one final sync only |
| outVal padding bridge | host-side memcpy (D2H + H2D round-trip per run) | device-side `outval_pad` kernel (one extra `__global__` on the stream) |
| Single-launcher wrapper | no — main.cpp calls each launcher explicitly | yes (`launchFullMoeCombined` static function in main.cpp) |
| Number of kernel `.so` libs | 5 | 6 (5 originals + `outval_pad`) |

The KERNEL CODE for the 5 stages is identical between the two folders. The 6th kernel (`outval_pad`) is only present in this folder — it does:

```
for t in 0..kT:
    TLOAD  rowTile     <- outVal_compact[t*kTopK : t*kTopK + kTopK]
    TSTORE outVal_padded[t*kPadded : t*kPadded + kTopK] <- rowTile
            (cols kTopK..kPadded-1 stay -1e30 from host pre-init)
```

This avoids the host bridge in `full_moe_separate` and lets the entire pipeline execute back-to-back on one ACL stream.

## Target platform

A3 / Ascend 910B1. Stages compiled with mixed targets:

| Stage | Target |
|---|---|
| router_matmul | `--cce-aicore-arch=dav-c220-cube` |
| moe_topk_padded | `--cce-aicore-arch=dav-c220-vec` |
| **outval_pad** | `--cce-aicore-arch=dav-c220-vec` (combined-only) |
| scatter | `--cce-aicore-arch=dav-c220-vec` |
| expert_ffn | `--cce-aicore-arch=dav-c220-cube` |
| gather | `--cce-aicore-arch=dav-c220-vec` |

## How to build and run

```bash
bash run.sh -r npu -v Ascend910B1
```

Sequence:
1. `scripts/gen_data.py` writes inputs and `golden_C.bin`.
2. CMake reconfigures and builds 6 kernel `.so` files + the host executable.
3. `./full_moe_combined` loads inputs, copies to device (including the `-1e30`-pre-filled padded outVal seed), calls `launchFullMoeCombined` ONCE, syncs ONCE, copies `C` back, validates against golden.

Validation tolerance: `5e-2` abs — same as full_moe_separate; the pipeline error budget stacks the same way regardless of orchestration model.

## Sweeping kT / kH / kF / kE / kTopK

```bash
bash sweep.sh 256 64 64 32 2 npu Ascend910B1

# kTopK axis
for k in 1 2 4 8 16; do bash sweep.sh 256 64 64 32 $k npu Ascend910B1; done
```

Patches local copies in this folder only (kernels/*.cpp, main.cpp, scripts/gen_data.py). Does NOT touch the original 5 MoE kernel folders. `git checkout -- .` reverts.

## Why a host wrapper instead of one giant `__global__`

The literal interpretation of "one kernel that does everything" would be a single `__global__ AICORE` function compiled with the mixed `dav-c220` target containing all of `TMATMUL` + `TSORT32` + `TROWMAX` + `TEXP` + ... — but there is **no in-tree precedent** for a single auto-mode `__global__` that mixes cube and vec instructions, and the UB/L1/L0 budget for stitching all of router_matmul + topk + scatter + ffn + gather into one function in one body is likely to exceed the auto-mode allocator's capacity at our smallest v1 shape, let alone the sweep matrix.

The host wrapper used here gives you the same observable end-state ("one host call, no intermediate sync, one final sync") while keeping each individual `__global__` within the proven envelope of [each sub-folder's confirmed-built kernel](../). The user-visible interface is still "one launcher call from main.cpp" — the cube/vec split is hidden behind the wrapper.

If a single-`__global__` variant becomes interesting later, the kernel sources in `./kernels/` are already a complete checklist of every instruction it would need to fuse.

## Known limitations

- Single AICORE per `__global__`; no `block_idx` parallelism.
- `kH = kF = 256` requires Split-K in `expert_ffn` (postponed).
- For `kTopK == 1` the gather softmax path is skipped (fast path); the rest of the pipeline doesn't change.

## Pattern sources

- [../router_matmul/](../router_matmul/), [../moe_topk_padded/](../moe_topk_padded/), [../scatter/](../scatter/), [../expert_ffn/](../expert_ffn/), [../gather/](../gather/) — the five stages.
- [../moe_top1_permute/](../../moe_top1_permute/), [../moe_top1_unpermute/](../../moe_top1_unpermute/) — the row TLOAD-TSTORE pattern that `outval_pad_kernel.cpp` follows.
