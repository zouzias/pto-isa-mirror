# M3O.1 Task Report

## Header

- Task: M3O.1 Report truth and stale claim cleanup
- Owner: codex
- End state: review_ready
- Related issue: DCL-2026-06-01-129

## Changes

- Changed files: `host/main.cpp`, `TASKS.md`, `reports/M3O.1.md`
- Out-of-scope files touched: none

## Acceptance

- Acceptance source: `DESIGN.md` section 14.2, subsection `M3O.1 Report 真实性与 stale claim 清理`
- Acceptance result: review_ready
- Verification summary: report fields now distinguish launch/requested facts from active worker evidence.
  `mixed_aic_blocks` and `mixed_aiv_blocks` were renamed to launch-count fields, route quant and GMM multi-block
  wording now uses claim/requested fields, and active fields are printed for GMM, dispatch, activation, combine, and
  restore.
- Dependency scan: pass for source/CMake/script stale field scan; static checks were intentionally not run per user
  instruction.

## Result

- Correctness: pass for both required M3O cases on devices 4-5.
- Perf/timeline observation: not-applicable for M3O.1 performance acceptance. E2E numbers were recorded only as
  context after correctness passed.
- Blocked reason, if any: none.

## Evidence

- Build: `cmake --build kernels/manual/a2a3/moe_dispatch_combine_a8w8/build --target moe_dispatch_combine_a8w8 -j16`
  exited 0.
- Required small case: `ffn-v3-small`, `-pes 2 -M 16 -K 128 -N 128 -topK 2 -expertPerPe 2 --max-output-size 32`,
  `--first-device 4 --ndevices 7`, exited 0. Both ranks reported `pass=true` and `final_output.err_count=0`.
- Required large case: `ffn-v3-4097`, `-pes 2 -M 4097 -K 128 -N 128 -topK 2 -expertPerPe 2 --max-output-size 8194`,
  `--first-device 4 --ndevices 7`, exited 0. Both ranks reported `pass=true` and `final_output.err_count=0`.
- Active evidence in the required runs includes `mixed_aic_launch_blocks=24`, `mixed_aiv_launch_blocks=48`,
  `gmm1_active_aic_blocks=1`, `gmm2_active_aic_blocks=1`, `dispatch_active_aiv_workers=1`,
  `combine_active_aiv_workers=1`, and `restore_active_aiv_workers=8`; large activation reported
  `activation_active_aiv_workers=1`.
- Stale field scan found no remaining source/CMake/script matches for the old misleading fields:
  `mixed_aic_blocks`, `mixed_aiv_blocks`, `route_quant_impl=`, `gmm_multiblock=true`, `scoreboard_async`,
  `scoreboardMinStatus`, or `producerStatus`.

## Delta And Handoff

- Design/task issue found: no; issue ID: none.
- User decision needed: no.
- Downstream notes: current fused GMM is truthfully reported as active block0-only with
  `gmm*_active_aic_blocks=1`; M3O.3 must replace that evidence with real active multi-AIC scheduler counters.
- Next task: M3O.2.
