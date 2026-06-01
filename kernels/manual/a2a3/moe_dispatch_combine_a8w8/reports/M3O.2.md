# M3O.2 Task Report

## Header

- Task: M3O.2 Dispatch/count/prefix stage-local split
- Owner: codex
- End state: review_ready
- Related issue: DCL-2026-06-01-130

## Changes

- Changed files: `kernel/moe_dispatch_combine_a8w8_kernel.cpp`,
  `kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp`, `host/main.cpp`, `TASKS.md`, `reports/M3O.2.md`
- Out-of-scope files touched: none

## Acceptance

- Acceptance source: `DESIGN.md` section 14.2, subsection `M3O.2 Dispatch/count/prefix stage-local 分活`
- Acceptance result: review_ready
- Verification summary: dispatch route pack no longer rescans prior routes inside the worker shard to compute row
  ordinal. Workers now write per-worker/per-expert counts, the merge phase derives `workerExpertPrefix`, and route
  pack writes rows using `expertBase + workerExpertPrefix + localOrdinal`.
- Dependency scan: pass for removed route-rescan helpers. Static checks were intentionally not run per user
  instruction.

## Result

- Correctness: pass for both required M3O cases on devices 4-5.
- Perf/timeline observation: not used as acceptance. The required runs kept the fixed benchmark policy
  `warmup_iters=0` and `measure_iters=1`.
- Blocked reason, if any: none.

## Evidence

- Build: `cmake --build kernels/manual/a2a3/moe_dispatch_combine_a8w8/build --target moe_dispatch_combine_a8w8 -j16`
  exited 0.
- Required small case: `ffn-v3-small`, `-pes 2 -M 16 -K 128 -N 128 -topK 2 -expertPerPe 2 --max-output-size 32`,
  `--first-device 4 --ndevices 7`, exited 0. Both ranks reported `pass=true` and `final_output.err_count=0`.
  Log: `/tmp/m3o2-small-final2-20260601121653.log`.
- Required large case: `ffn-v3-4097`, `-pes 2 -M 4097 -K 128 -N 128 -topK 2 -expertPerPe 2 --max-output-size 8194`,
  `--first-device 4 --ndevices 7`, exited 0. Both ranks reported `pass=true` and `final_output.err_count=0`.
  Log: `/tmp/m3o2-large-final2-20260601121708.log`.
- Hang regression probe: debug-stop stage 11 passed after removing the redundant parallel prefix rewrite. Log:
  `/tmp/m3o2-probe-stage11-fix2-20260601121406.log`.
- Source scan found no remaining matches for `M3NCountPreviousRoutesInShard`,
  `M3NDispatchPreviousWorkerRows(`, or `M3NBuildWorkerExpertPrefixShard`.
- Host report evidence prints `count_prefix_workers`, `worker_expert_count=true`, `worker_expert_prefix=true`,
  `route_shard_rescan=false`, and
  `dispatch_pack_row_formula=expertBase_plus_workerExpertPrefix_plus_localOrdinal`.

## Debug Note

- Root cause of the in-progress hang: `M3NMergeLocalRouteCounts` already wrote the full worker prefix table
  serially. The later `M3NBuildWorkerExpertPrefixShard` pass redundantly rewrote the same prefix rows from active AIV
  workers and introduced a hard-sync hang before debug-stop stage 11. Removing that redundant pass restored stage 11
  and full small/large correctness while preserving the no-rescan row formula.

## Delta And Handoff

- Design/task issue found: no; issue ID: none.
- User decision needed: no.
- Downstream notes: M3O.2 removes the shard route rescan hotspot and reports the worker/expert count-prefix path.
  Current fused GMM active AIC evidence remains block0-only; M3O.3 must address real multi-AIC tile scheduling.
- Next task: M3O.3.

## Init Quant Checkpoint

- Scope: follow-up checkpoint for `init_quant.md` Task 1/2 and debug-stop 16/17 boundary verification.
- Fresh run: 2026-06-01 17:17, logs under `/tmp/a8w8_init_quant_verify_20260601_171712`.
- Build: `cmake --build kernels/manual/a2a3/moe_dispatch_combine_a8w8/build --target moe_dispatch_combine_a8w8 --clean-first -j16`
  exited 0.
- Verified on devices 4-7:
  - `ffn-v3-small` debug-stop 12: pass, route count / source-local expanded row / payload sample match.
  - `ffn-v3-4097` debug-stop 11/12/13/14/15/16/17: pass.
  - Large debug-stop 11 reports `dispatch_active_aiv_workers=4`, `init_quant_worker_mask=15`.
  - Large debug-stop 14 reports `init_quant_expanded_row_contract=capacity_clipped`,
    `init_quant_token_matrix_full_match=true`, `init_quant_expanded_row_match=true`.
  - Large debug-stop 16/17 reports `init_quant_gmm1_input_match=true` and
    `init_quant_dispatch_ready_match=true`.
- Current blocker: fused direct-pack PTO Vec quant is not enabled. A direct PTO Vec attempt and a scalar-max +
  PTO Vec quant/store attempt both timed out on `ffn-v3-small` debug-stop 12, so the verified path keeps scalar
  quant and prints `route_quant_scalar_fallback_reason=mixed_fused_direct_pack_pto_vec_probe_timeout`.
