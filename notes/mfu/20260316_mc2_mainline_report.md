# 2026-03-16 MC2 Mainline MFU Report

## Objective

Verify that the current dense MC2 path is really fused into the training mainline, measure the
real training MFU on `resume 6500 -> 6506`, and record the result of the sparse `topk_idx /
topk_weight` experiment, the dispatcher metadata overlap optimization, and the later
`probs` communication overlap experiment. Also verify a first real PTO-ISA training-path
integration on the MC2 local reorder hotspot.

## Environment

- Repos:
  - `MindSpeed`: `/home/llx/MindSpeed`
  - `LLM`: `/home/llx/lzm_Mindspeed-LLm`
- Resume checkpoint:
  - `/home/llx/pto-isa/tmp_ckpts/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_load6500_sparse`
- Launch wrapper:
  - `/home/llx/pto-isa/tools/run_native_baseline_resume.sh`
- Fixed ports for the final dense rerun:
  - `MASTER_PORT=38243`
  - `HCCL_IF_BASE_PORT=56300`
- Common runtime knobs:
  - `ENABLE_MOE_ALLTOALL_MC2=1`
  - `ENABLE_MOE_ALLTOALL_OVERLAP_COMM=0`
  - `ENABLE_MOE_GROUPED_GEMM=1`
  - `ENABLE_MOE_PERMUTATION_ASYNC_COMM=1`
  - `ENABLE_MOE_PERMUTE_FUSION=1`
  - `MOE_TOKEN_DISPATCHER_TYPE=alltoall_seq`
  - `MINDSPEED_MOE_PERMUTE_ALIGN_SMALLER_DTYPE=1`
  - `LOG_PARAMS_NORM=0`
  - `ENABLE_TENSORBOARD=0`
  - `LOG_TIMERS_TO_TENSORBOARD=0`
  - `SAVE_CHECKPOINT=0`

## What Was Done

1. Reverted the sparse MC2 router-dispatcher experiment:
   - `MindSpeed 7c0091ec` `Revert "Add sparse MC2 router-dispatcher path"`
2. Restored the small `mc2_*` helper symbols in
   `/home/llx/MindSpeed/mindspeed/core/transformer/moe/moe_feature/mc2moe/utils.py`
   so the current dense MC2 codepath remains importable in the present dirty worktree.
3. Re-ran real training on the current dense MC2 mainline.
4. Kept the sparse experiment result for comparison.
5. Moved MC2 legacy dispatcher metadata D2H synchronization out of `preprocess()` and finalized it
   only when the metadata is actually consumed.
6. Delayed `global_probs` A2A wait/reorder until right before `activation_func_with_probs`, so
   the `probs` communication can overlap with MC2 `mm1`.
7. Added a first PTO-ISA operator for MC2 local chunk reorder:
   - PTO op:
     - `/home/llx/pto-isa/demos/baseline/moe_grouped_ffn/csrc/host/my_mc2_reorder.cpp`
   - MindSpeed integration:
     - `/home/llx/MindSpeed/mindspeed/core/transformer/moe/moe_feature/mc2moe/utils.py`
   - Wrapper logging support:
     - `/home/llx/pto-isa/tools/run_native_baseline_resume.sh`

### New Dispatcher Sync Optimization

- Commit:
  - `MindSpeed 3525f80b` `Overlap MC2 dispatcher metadata sync`
- Main idea:
  - Keep `send_counts/recv_counts/input_splits/output_splits` on asynchronous CPU copies first.
  - Remove the eager `torch.cuda.current_stream().synchronize()` in
    `legacy_a2a_token_dispatcher.preprocess()`.
  - Finalize the metadata only at `before_ep_alltoall` / `before_finish`, so the D2H metadata copy
    overlaps with `permute` and shared-expert pre-forward work.

## Mainline Integration Evidence

The current dense rerun log clearly shows the training graph is using the MC2 MoE mainline path:

- `use MoELayer, <class 'mindspeed.core.transformer.moe.moe_feature.adaptor.MindSpeedAlltoAllMC2MoeLayerAdaptor'>`
- `use GroupedMLP, <class 'mindspeed.core.transformer.moe.moe_feature.adaptor.MindSpeedGmmMC2Experts'>`
- `successfully loaded checkpoint ... at iteration 6500`
- Iterations `6501-6506` completed

Source log:

- `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260316_mc2_dense_report_current_rerun/train.log`

Relevant lines:

- `1320-1516`: MC2 MoE module/adaptor selected
- `3779`: checkpoint successfully loaded
- `3925-3931`: iteration and MFU lines

## Results

### Current Dense MC2 Mainline

- Log:
  - `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260316_mc2_dense_report_current_rerun/train.log`
- Iterations:
  - `6503`: `2112.1 ms`, `38.45% MFU`
  - `6504`: `2124.7 ms`, `38.22% MFU`
  - `6505`: `2129.8 ms`, `38.13% MFU`
  - `6506`: `2118.4 ms`, `38.33% MFU`
- Median over `6503-6506`:
  - `2121.55 ms / 38.275% MFU`

### Dense MC2 Mainline After Dispatcher Metadata Overlap

- Commit:
  - `MindSpeed 3525f80b` `Overlap MC2 dispatcher metadata sync`
- Log:
  - `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260316_mc2_async_meta_shortcheck/train.log`
- Iterations:
  - `6503`: `2101.3 ms`, `38.65% MFU`
  - `6504`: `2108.9 ms`, `38.51% MFU`
  - `6505`: `2098.8 ms`, `38.69% MFU`
  - `6506`: `2097.8 ms`, `38.71% MFU`
- Median over `6503-6506`:
  - `2100.05 ms / 38.67% MFU`

### `shared_expert fuse` Recheck On Top Of Async Metadata

- Log:
  - `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260316_mc2_async_meta_sharedfuse/train.log`
- Median over `6503-6506`:
  - `2102.90 ms / 38.62% MFU`
- Conclusion:
  - Stable, but still slightly worse than async-metadata-only mainline.

### Dense MC2 Mainline After `probs` Communication Overlap

- Main idea:
  - Do not wait on `AsyncAllToAllWithBackward` for `permuted_probs` inside
    `legacy_a2a_token_dispatcher.token_permutation()`.
  - Carry the pending handle across the dispatcher boundary.
  - Finalize `wait + reorder` only immediately before `activation_func_with_probs` in
    `mc2moe/experts.py`, so `probs` communication overlaps with MC2 `mm1`.
- Log:
  - `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260316_mc2_probs_overlap_shortcheck/train.log`
- Iterations:
  - `6503`: `2068.6 ms`, `39.26% MFU`
  - `6504`: `2074.8 ms`, `39.14% MFU`
  - `6505`: `2070.4 ms`, `39.22% MFU`
  - `6506`: `2067.4 ms`, `39.28% MFU`
- Median over `6503-6506`:
  - `2069.5 ms / 39.24% MFU`

### Dense MC2 Mainline After `probs` Overlap + Metadata DtoH Side-Stream

- Main idea:
  - Keep the previous delayed `global_probs` wait/reorder.
  - Move dispatcher metadata DtoH (`send_counts/recv_counts/input_splits/output_splits` and
    `num_global_tokens_per_local_expert_cpu`) onto `cuda_dtoh_stream` instead of issuing those
    copies on the current stream inside `preprocess()`.
  - Only synchronize the side stream when the counts are actually needed before the fused
    `npu_alltoallv_gmm` call.
- Log:
  - `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260316_mc2_probs_overlap_dtohstream_shortcheck_fix1/train.log`
- Iterations:
  - `6503`: `2058.1 ms`, `39.46% MFU`
  - `6504`: `2067.4 ms`, `39.28% MFU`
  - `6505`: `2067.7 ms`, `39.27% MFU`
  - `6506`: `2054.8 ms`, `39.52% MFU`
- Median over `6503-6506`:
  - `2062.75 ms / 39.37% MFU`
- Status:
  - Real-training positive result on the standard short-window `resume 6500 -> 6506` check.
  - A longer validation run of the previous `probs`-overlap-only mainline was interrupted after
    enough steady-state data had been collected in order to free the machine for this new test.

### Full Long-Window Reference For Non-PTO Dense MC2 Mainline

- Log:
  - `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260316_mc2_probs_overlap_dtohstream_500step_retry2/train.log`
- Median over `6503-7000`:
  - `2060.90 ms / 39.40% MFU`
- Important note:
  - This is the current best framework-side MC2 reference, but it is **not** a PTO-ISA-attributed
    result. The corresponding launch used `ENABLE_PTO_MOE_GROUPED_FFN=0`.

### PTO-ISA MC2 Reorder Path: Functional Validation

- PTO op:
  - `npu::pto_mc2_reorder_chunks(Tensor input, Tensor split_sizes, Tensor sorted_idxs) -> (Tensor, Tensor)`
- Functional checks:
  - Forward max diff vs Python reference: `0.0`
  - Backward grad max diff vs Python reference: `0.0`
- Integration switch:
  - `ENABLE_PTO_MOE_MC2_REORDER=1`
  - `PTO_MOE_MC2_SO_PATH=/home/llx/pto-isa/demos/baseline/moe_grouped_ffn/build/lib/libop_extension.so`

### PTO-ISA MC2 Reorder Path: Real Training Short Check

- Log:
  - `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260316_pto_mc2_reorder_shortcheck_rerun/train.log`
- Iterations:
  - `6503`: `2054.0 ms`, `39.54% MFU`
  - `6504`: `2063.0 ms`, `39.36% MFU`
  - `6505`: `2062.0 ms`, `39.38% MFU`
  - `6506`: `2049.9 ms`, `39.61% MFU`
- Median over `6503-6506`:
  - `2058.00 ms / 39.46% MFU`
- Interpretation:
  - This is the first real training result where the MC2 local reorder hotspot is routed through a
    PTO-ISA operator instead of the previous Python-side implementation.

### PTO-ISA MC2 Reorder Path: Current Long-Run Progress

- Log:
  - `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260316_pto_mc2_reorder_500step/train.log`
- Current completed steady-state interval:
  - `6503-6536`
- Median over `6503-6536`:
  - `2056.30 ms / 39.49% MFU`
- Current interpretation:
  - On the completed steady-state window so far, the PTO-ISA reorder path is slightly ahead of the
    non-PTO long-window reference (`39.40%`).

### PTO-ISA Attribution Evidence

- Updated wrapper:
  - `/home/llx/pto-isa/tools/run_native_baseline_resume.sh`
- Dry-run command file with explicit PTO exports:
  - `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260316_pto_mc2_reorder_dryrun/launch.sh`
- Recorded PTO exports:
  - `ENABLE_PTO_MOE_MC2_REORDER=1`
  - `PTO_MOE_MC2_SO_PATH=/home/llx/pto-isa/demos/baseline/moe_grouped_ffn/build/lib/libop_extension.so`
- Note:
  - Earlier PTO short checks were launched with environment prefixes before the wrapper recorded
    those exports into `launch.sh`. The wrapper has now been fixed so future PTO runs keep the
    attribution in their command files.

### Partial Long-Window Validation For `probs` Overlap Mainline

- Log:
  - `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260316_mc2_probs_overlap_500step/train.log`
- Completed steady-state interval before manual stop:
  - `6503-6774` (`272` steady steps)
- Median over `6503-6774`:
  - `2070.1 ms / 39.23% MFU`
- Note:
  - This run was stopped manually after `272` steady-state iterations so the machine could be reused
    to validate the newer metadata-side-stream candidate above.

### Negative Experiment: Device-Metadata Reorder

- Main idea:
  - Force MC2 reorder metadata onto NPU and build row indices on device.
- Log:
  - `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260316_mc2_device_meta_shortcheck/train.log`
- Median over `6503-6506`:
  - `2276.95 ms / 35.665% MFU`
- Conclusion:
  - Strong positive microbench, but clear end-to-end regression in real training. This direction
    was reverted immediately and is not part of the current mainline.

### Sparse `topk_idx/topk_weight` Experiment

- Commit:
  - `MindSpeed 4dd91ead` `Add sparse MC2 router-dispatcher path`
- Reverted by:
  - `MindSpeed 7c0091ec`
- Log:
  - `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260316_mc2_sparse_topk_resume6500_shortcheck_fix2/train.log`
- Median over `6503-6506`:
  - `2245.90 ms / 36.155% MFU`

### Previous Dense MC2 Stable Reference

- Commit:
  - `MindSpeed 20b36cb0` `Simplify MC2 seq dispatcher preprocess`
- Log:
  - `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260315_mc2_seq_preprocess_shortcheck_rerun/train.log`
- Median over `6503-6506`:
  - `2116.75 ms / 38.365% MFU`

## Comparison

- Current dense MC2 mainline vs sparse experiment:
  - `-124.35 ms/iter`
  - `+2.120 MFU points`
- Current dense MC2 mainline vs previous dense stable reference:
  - `+4.80 ms/iter`
  - `-0.090 MFU points`
- Async metadata mainline vs current dense MC2 mainline:
  - `-21.50 ms/iter`
  - `+0.395 MFU points`
- Async metadata mainline vs previous dense stable reference:
  - `-16.70 ms/iter`
  - `+0.305 MFU points`
- `probs` overlap mainline vs async metadata mainline:
  - `-30.55 ms/iter`
  - `+0.570 MFU points`
- `probs` overlap mainline vs current dense MC2 mainline:
  - `-52.05 ms/iter`
  - `+0.965 MFU points`
- PTO reorder long-run-progress vs non-PTO long-window reference:
  - `-4.60 ms/iter`
  - `+0.090 MFU points`
- `probs` overlap + metadata DtoH side-stream vs `probs` overlap mainline:
  - `-6.75 ms/iter`
  - `+0.130 MFU points`
- `probs` overlap + metadata DtoH side-stream vs current dense MC2 mainline:
  - `-58.80 ms/iter`
  - `+1.095 MFU points`

## Conclusion

1. The current dense MC2 path is really fused into the training mainline and has been verified by a
   real `resume 6500 -> 6506` training run.
2. The current best real-training result is:
   - `2062.75 ms / 39.37% MFU`
3. The sparse `topk_idx/topk_weight` path is functionally trainable, but it is not performance
   competitive right now:
   - `2245.90 ms / 36.155% MFU`
4. `shared_expert fuse` remains stable but does not beat the new async-metadata mainline:
   - `2102.90 ms / 38.62% MFU`
5. For the current codebase, the best verified short-window mainline is now the dense MC2 path
   with:
   - dispatcher metadata overlap
   - `probs` communication overlapped with MC2 `mm1`
   - metadata DtoH issued on a side stream and finalized only when the fused GMM path actually
     needs CPU counts
6. The device-metadata reorder experiment is not a viable mainline direction despite a strong
   isolated microbench win. End-to-end training is the only accepted criterion.
