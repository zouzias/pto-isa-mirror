# MoE — auto-mode A3 kernel stack

Five independently buildable kernels that together implement a top-`kTopK` MoE forward pass on Ascend 910B1 in auto mode. Each sub-folder runs standalone via `bash run.sh -r npu -v Ascend910B1`; `sweep.sh` patches shape constants and runs all five at one shape; `sweep_all.sh` walks a curated 20-config matrix.

## Pipeline

```
                       ┌────────────────┐
   X      ─────────────►│ router_matmul │──► logits
  (fp16)                │   (cube)       │   (fp32)
                       └────────────────┘
                                │
                                ▼
                       ┌────────────────┐
                       │ moe_topk_padded│──► expert_id  [kT, kTopK] int32
                       │    (vec)       │──► outVal     [kT, kTopK] fp32 (descending sorted)
                       └────────────────┘                                  │
                                │                                          │
                                ▼                                          │
              X  ───►  ┌────────────────┐  ──► A             [kT·kTopK+16, kH] fp16
                       │   scatter      │  ──► A_id          [kT·kTopK+16]     int32
                       │    (vec)       │  ──► rank_id       [kT·kTopK+16]     int32
                       │                │  ──► expert_count  [kE]              int32
                       │                │  ──► expert_start  [kE]              int32
                       └────────────────┘
                                │
                                ▼
              W1, W2 ──►┌────────────────┐
                       │  expert_ffn    │──► B               [kT·kTopK+16, kH] fp32
                       │ (cube + cube,  │
                       │  Stage1+Stage2)│
                       └────────────────┘
                                │                                          │
                                ▼                                          ▼
                       ┌────────────────────────────────────────────────────┐
                       │              gather (vec)                          │
                       │   Pass 1 (kTopK > 1): softmax(outVal) -> weights   │
                       │   Pass 2: C[A_id[r]] += weights[A_id[r],rank_id[r]]│
                       │                              * B[r]                │
                       └────────────────────────────────────────────────────┘
                                │
                                ▼
                                C  [kT, kH] fp32
```

The +16 trailing rows on `A`, `A_id`, `rank_id`, `B`, and `Y_scratch` are the **overspill landing pad** for `expert_ffn`'s last-tile writes; `gather` ignores them.

For `kTopK == 1` softmax is degenerate (single-value softmax = 1.0), so `gather` takes a **fast path** that skips Pass 1 entirely and reduces to the unweighted v1 permute / accumulation.

### End-to-end variants

Two folders chain all five stages into a complete MoE forward pass — same Python golden, same C output, different orchestration:

- **[full_moe_separate](full_moe_separate/)** — main.cpp calls each of the 5 launchers individually with an explicit `aclrtSynchronizeStream` after each, plus a host-side memcpy bridge to pad outVal from `(kT, kTopK)` to `(kT, kPadded)` between the topk and gather stages.
- **[full_moe_combined](full_moe_combined/)** — main.cpp calls **one** wrapper launcher (`launchFullMoeCombined`) that fires six `__global__ AICORE` kernels on the same stream with **no** intermediate sync. The pad bridge is replaced by a small device-side `outval_pad` kernel. Only one final `aclrtSynchronizeStream` before reading C back to host.

Both folders include their own `sweep.sh` for shape-axis testing (kT / kH / kF / kE / kTopK), patching local kernel copies under `./kernels/` rather than the original sub-folders.

## Sub-folders

| Folder | Target | Kind | Status |
|---|---|---|---|
| [router_matmul](router_matmul/) | cube | GEMM: `logits = X @ W_router` | confirmed-built (pre-existing) |
| [moe_topk](moe_topk/) | vec | top-1 only (legacy; do not touch) | confirmed-built (pre-existing) |
| [moe_topk_padded](moe_topk_padded/) | vec | generic top-K with `kGatherWidth = max(8, kTopK)` + valid-region crop | confirmed-built v1 (kT=256, kE=32, kTopK=1) |
| [scatter](scatter/) | vec | pack tokens by expert; emits `A`, `A_id`, `rank_id`, `count`, `start` | confirmed-built v1 (kTopK=1); `rank_id` output added later |
| [expert_ffn](expert_ffn/) | cube ×2 | two-stage GEMM1+ReLU / GEMM2 with overspill | confirmed-built v1 |
| [gather](gather/) | vec | softmax-weighted scatter-add; `if kTopK==1` skips softmax | confirmed-built v1 (kTopK=1 = unweighted fast path); softmax path tested via sweep |
| [full_moe_separate](full_moe_separate/) | mixed | end-to-end pipeline; 5 launches with sync between each + host-side outVal pad bridge | first version of end-to-end |
| [full_moe_combined](full_moe_combined/) | mixed | end-to-end pipeline; one host wrapper fires 6 kernels on stream + device-side `outval_pad` + one final sync | first version of end-to-end with stream-orchestrated chain |

"Confirmed-built v1" = user ran `bash run.sh -r npu -v Ascend910B1` at the v1 shape and saw `test data success`. Other shapes have not been observed yet — that's what `sweep_all.sh` is for.

## v1 shape (the one confirmed working)

| `kT` | `kH` | `kF` | `kE` | `kTopK` | `kTileM` (expert_ffn) | `kTileM` (router_matmul) |
|---|---|---|---|---|---|---|
| 256 | 64 | 64 | 32 | 1 | 16 | 128 |

## How to run

### One folder standalone

```bash
cd <folder>
bash run.sh -r npu -v Ascend910B1
```

Each folder has its own `scripts/gen_data.py` (writes inputs + golden), `main.cpp` (loads inputs, fires the kernel, writes outputs, validates), and prints `test data success` / `test data failed`.

### All 5 folders at one shape

```bash
cd MoE
bash sweep.sh <kT> <kH> <kF> <kE> <kTopK> <RUN_MODE> <SOC_VERSION>
# e.g.:
bash sweep.sh 256 64 64 32 1 npu Ascend910B1
```

`sweep.sh` patches the shape constants in-place across all 5 folders (`gen_data.py`, `main.cpp`, `<name>_kernel.cpp`), then loops: build → run → record PASS/FAIL. Constants stay patched after the run — `git checkout -- .` reverts.

### Overnight matrix (20 configs)

```bash
cd MoE
bash sweep_all.sh -r npu -v Ascend910B1
# or just: bash sweep_all.sh   (defaults to npu / Ascend910B1)
```

Writes `sweep_all.log` (streaming) and `sweep_all.summary` (final matrix). Covers `kTopK ∈ {1,2,4,8,16}`, `kE ∈ {16,32}`, `kT ∈ {128,256,512}`, `kH=kF ∈ {64,128}`, and a handful of cross-axis combinations. `kH=kF=256` is **skipped** — that needs Split-K, postponed.

## Constraints inherited across all sub-folders

- Single AICORE per `__global__` (no `block_idx` work split anywhere yet).
- Static tile shapes; auto-mode requires compile-time tile dims, which is why each sweep is a full rebuild.
- fp16 inputs / weights, fp32 accumulators on cube paths ([known_good_kernel_examples.md §A6](../../../../docs_for_ai/known_good_kernel_examples.md)).
- ReLU fused into TSTORE FixPipe inside `expert_ffn` ([§A17 / §11.8](../../../../docs_for_ai/assumptions_to_verify.md)) — no vector hop between GEMM1 and GEMM2.
- No `TASSIGN` aliasing, no `Tile::data()` in kernels, no `*_IMPL` calls, no raw CCE intrinsics, no `Event<>`, no manual sync, no `TPipe`/`TPUSH`/`TPOP`, no double buffering.

## What's not yet built

- **End-to-end glue** chaining all 5 stages in one host driver. Each stage is currently validated independently against its own golden.
- **Split-K / Split-N inside expert_ffn** for `kH=kF ≥ 256` (L0B = 64 KB ceiling).
- **Multi-AICORE** parallelism (per-expert or per-tile).
- **kTopK > 1 on real hardware** — code is in place; only sweep_all.sh will confirm.

## Pointers

- Hardware mental model + buffer capacities: [docs_for_ai/pto_auto_mode_hw_optimization_guide.md §1.3](../../../../docs_for_ai/pto_auto_mode_hw_optimization_guide.md)
- Cross-arch differences: [docs_for_ai/a3_a5_differences.md §14](../../../../docs_for_ai/a3_a5_differences.md)
- Resolved-by-experiments log (incl. v1 entry): [docs_for_ai/assumptions_to_verify.md §11](../../../../docs_for_ai/assumptions_to_verify.md)
- Reference patterns we mirrored:
  - [moe_top1_permute](../moe_top1_permute/) → `scatter`
  - [moe_top1_unpermute](../moe_top1_unpermute/) → `gather`
  - [moe_segmented_ffn_top1](../moe_segmented_ffn_top1/) → `expert_ffn`
  - [moe_topk](moe_topk/) → `moe_topk_padded` (k-padded generalization)
  - [topk](../topk/) → underlying TSORT32/TMRGSORT machinery
