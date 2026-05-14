# MoE — auto-mode A3 kernel stack

Five independently buildable kernels that together implement a top-`kTopK` MoE forward pass on Ascend 910B1 in auto mode. Each sub-folder runs standalone via `bash run.sh -r npu -v Ascend910B1`; `sweep.sh` patches shape constants and runs all five at one shape; `sweep_all.sh` walks a curated 20-config matrix.

## Pipeline

```
                       ┌────────────────┐
   X      ─────────────►│ router_matmul │──► logits
  (fp16)                │   (cube)      │   (fp32)
                       └────────────────┘
                                │
                                ▼
                       ┌────────────────┐
                       │ moe_topk_padded│──► expert_id
                       │    (vec)       │    [kT, kTopK] int32
                       └────────────────┘
                                │
                                ▼
              X  ───►  ┌────────────────┐  ──► A             [kT·kTopK + 16, kH] fp16
                       │   scatter      │  ──► A_id          [kT·kTopK + 16]     int32
                       │    (vec)       │  ──► expert_count  [kE]                int32
                       │                │  ──► expert_start  [kE]                int32
                       └────────────────┘
                                │
                                ▼
              W1, W2 ──►┌────────────────┐
                       │  expert_ffn    │──► B             [kT·kTopK + 16, kH] fp32
                       │ (cube + cube,  │
                       │  Stage1+Stage2)│
                       └────────────────┘
                                │
                                ▼
                       ┌────────────────┐
                       │   gather       │──► C             [kT, kH] fp32
                       │    (vec)       │
                       └────────────────┘
```

The +16 trailing rows on `A`, `A_id`, `B`, and `Y_scratch` are the **overspill landing pad** for `expert_ffn`'s last-tile writes; `gather` ignores them.

## Sub-folders

| Folder | Target | Kind | Status |
|---|---|---|---|
| [router_matmul](router_matmul/) | cube | GEMM: `logits = X @ W_router` | confirmed-built (pre-existing) |
| [moe_topk](moe_topk/) | vec | top-1 only (legacy; do not touch) | confirmed-built (pre-existing) |
| [moe_topk_padded](moe_topk_padded/) | vec | generic top-K with `kGatherWidth = max(8, kTopK)` + valid-region crop | confirmed-built v1 (kT=256, kE=32, kTopK=1) |
| [scatter](scatter/) | vec | pack tokens by expert; emits `A`, `A_id`, `count`, `start` | confirmed-built v1 |
| [expert_ffn](expert_ffn/) | cube ×2 | two-stage GEMM1+ReLU / GEMM2 with overspill | confirmed-built v1 |
| [gather](gather/) | vec | `C[A_id[r]] += B[r]` (zero-init by host) | confirmed-built v1 |

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
