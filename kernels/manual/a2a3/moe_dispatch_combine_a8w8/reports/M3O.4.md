# M3O.4 Task Report

## Header

- Task: M3O.4 Activation/SwiGLU/requant and restore evidence
- Owner: codex
- End state: blocked
- Related issue: DCL-2026-06-01-132

## Changes

- Changed files: `include/moe_dispatch_combine_a8w8_types.hpp`,
  `kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp`, `host/main.cpp`, `TASKS.md`,
  `reports/M3O.4.md`
- Out-of-scope files touched: none

## Acceptance

- Acceptance source: `DESIGN.md` section 14.2, subsection
  `M3O.4 Activation/SwiGLU/requant 与 restore 证据修正`
- Acceptance result: blocked
- Verification summary: restore worker evidence is implemented and verified on both required M3O cases. Activation
  stage-local AIV sharding is not accepted: probes showed the AIV activation shard path can hang after entering the real
  row loop / post-compute sync path, so the correctness path remains the coarse scalar activation implementation and
  reports `activation_active_aiv_workers=1`.
- Dependency scan: source scan found no Catlass/AscendC fallback and no discarded scoreboard/prod-status/min-status
  route. Static checks were intentionally not run per user instruction.

## Result

- Correctness: pass for both required M3O cases on devices 4-5 in the recovered coarse activation path.
- Restore evidence: pass. Both required cases report 8 active restore AIV workers, per-worker token ranges, route counts,
  skipped route counts, and non-overlap evidence.
- Activation evidence: blocked. The valid path still reports `activation_active_aiv_workers=1`; this does not satisfy the
  M3O.4 activation worker acceptance requirement.
- Additional diagnosis: a non-overlap completion-edge experiment that made AIC wait
  `kM3N4ActivationToGmm2Flag` after AIV activation did not resolve the hang. Both the scalar row probe path and the
  existing PTO vector activation helper path timed out before completion, so the blocker is earlier than the
  activation-to-GMM2 consumer edge.

## Evidence

- Build: `cmake --build kernels/manual/a2a3/moe_dispatch_combine_a8w8/build --target moe_dispatch_combine_a8w8 -j16`
  exited 0.
- Required small case: `ffn-v3-small`, `-pes 2 -M 16 -K 128 -N 128 -topK 2 -expertPerPe 2 --max-output-size 32`,
  `--first-device 4 --ndevices 7`, exited 0. Both ranks reported `pass=true` and `final_output.err_count=0`.
  Log: `/tmp/m3o4-small-recovered2-20260601-132906.log`.
- Required large case: `ffn-v3-4097`, `-pes 2 -M 4097 -K 128 -N 128 -topK 2 -expertPerPe 2 --max-output-size 8194`,
  `--first-device 4 --ndevices 7`, exited 0. Both ranks reported `pass=true` and `final_output.err_count=0`.
  Log: `/tmp/m3o4-large-recovered-20260601-132929.log`.
- Required small case evidence includes `activation_active_aiv_workers=1`, `restore_active_aiv_workers=8`,
  `restore_total_token_count=16`, `restore_total_route_count=32`, `restore_total_skipped_route_count=0`, and
  `restore_worker_ranges=0:0-2,1:2-4,2:4-6,3:6-8,4:8-10,5:10-12,6:12-14,7:14-16`.
- Required large case evidence includes `activation_active_aiv_workers=1`, `restore_active_aiv_workers=8`,
  `restore_total_token_count=4097`, `restore_total_route_count=8194`, `restore_total_skipped_route_count=0`, and
  `restore_worker_ranges=0:0-513,1:513-1025,2:1025-1537,3:1537-2049,4:2049-2561,5:2561-3073,6:3073-3585,7:3585-4097`.
- Activation probe summary: debug-stop 41 and 42 passed, while later probes around the real row loop / post-compute sync
  path hung or were unsafe. Probe 44, which skips activation compute, and probe 46, which limits to the zero-quant path,
  passed. The current source keeps the correctness path on coarse scalar activation.
- Follow-up diagnosis on June 1, 2026:
  - Red signal: `/tmp/m3o4-red-small-20260601-140016.log` exited 42 because correctness passed but
    `activation_active_aiv_workers=1`, proving the acceptance failure remains observable.
  - Non-overlap AIV activation plus `ActivationToGmm2` wait experiment:
    `/tmp/m3o4-small-aiv-shard-20260601-140732.log` timed out with exit 124.
  - Scalar activation probes: debug-stop 45 (`/tmp/m3o4-probe-45-20260601-141214.log`) and debug-stop 47
    (`/tmp/m3o4-probe-47-20260601-141409.log`) timed out, while 41/42/44/46 returned 0.
  - PTO helper experiment: full small (`/tmp/m3o4-small-pto-shard-20260601-141807.log`) and debug-stop 43
    (`/tmp/m3o4-probe-43-pto-20260601-142346.log`) timed out. The current source was restored to the coarse
    activation path afterward.
  - Recovery sanity: rebuild exited 0 and `/tmp/m3o4-small-recovered-sanity-20260601-142734.log` exited 0 with
    `final_output.err_count=0` and `activation_active_aiv_workers=1`.

## Delta And Handoff

- Design/task issue found: yes; issue ID: M3O4-ACTIVATION-SHARD-HANG.
- User decision needed: no.
- Downstream notes: M3O.5 can proceed because combine worker policy is independent of activation sharding. M3O.6 must
  not reopen GMM1->activation or activation->GMM2 overlap until the activation shard hang is resolved or the design is
  explicitly changed.
- Next task: M3O.5.
