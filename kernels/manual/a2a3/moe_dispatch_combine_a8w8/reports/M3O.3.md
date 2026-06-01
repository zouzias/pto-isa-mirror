# M3O.3 Task Report

## Header

- Task: M3O.3 GMM1/GMM2 active multi-AIC tile scheduler
- Owner: codex
- End state: review_ready
- Related issue: DCL-2026-06-01-131

## Changes

- Changed files: `kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp`, `host/main.cpp`,
  `include/moe_dispatch_combine_a8w8_layout.hpp`, `TASKS.md`, `reports/M3O.3.md`
- Out-of-scope files touched: none

## Acceptance

- Acceptance source: `DESIGN.md` section 14.2, subsection `M3O.3 GMM1/GMM2 active 多 AIC tile scheduler`
- Acceptance result: review_ready
- Verification summary: fused GMM1/GMM2 tile tasks now use block-stride AIC scheduling
  `taskId = blockIdx; taskId < taskCount; taskId += blockNum`. The scalar tile helper no longer filters to block0,
  and report fields are device evidence from peer debug counters rather than launch-count inference.
- Dependency scan: pass for no Catlass/AscendC fallback, no discarded scoreboard fields, no stale
  `gmm*_active_aic_blocks=1` host claim, and no block0-only guard inside the scalar GMM tile helper.

## Result

- Correctness: pass for both required M3O cases on devices 4-5.
- Active multi-AIC evidence: required large case reports `gmm1_active_aic_blocks=24` and
  `gmm2_active_aic_blocks=24` on both ranks.
- Perf/timeline observation: large E2E dropped to about 3.61s in the final run; this is recorded as context only and
  not used as the sole pass/fail criterion.
- Blocked reason, if any: none.

## Evidence

- Build: `cmake --build kernels/manual/a2a3/moe_dispatch_combine_a8w8/build --target moe_dispatch_combine_a8w8 -j16`
  exited 0.
- Required small case: `ffn-v3-small`, `-pes 2 -M 16 -K 128 -N 128 -topK 2 -expertPerPe 2 --max-output-size 32`,
  `--first-device 4 --ndevices 7`, exited 0. Both ranks reported `pass=true` and `final_output.err_count=0`.
  Log: `/tmp/m3o3-small-final3-20260601123923.log`.
- Required large case: `ffn-v3-4097`, `-pes 2 -M 4097 -K 128 -N 128 -topK 2 -expertPerPe 2 --max-output-size 8194`,
  `--first-device 4 --ndevices 7`, exited 0. Both ranks reported `pass=true` and `final_output.err_count=0`.
  Log: `/tmp/m3o3-large-final3-20260601123941.log`.
- Large rank evidence includes:
  `gmm1_active_aic_blocks=24`, `gmm2_active_aic_blocks=24`, `gmm1_tile_task_count=65/66`,
  `gmm2_tile_task_count=65/66`, and task distribution across blocks 0-23.
- First/last task evidence is printed for both GMM stages, including
  `gmm*_first_task_block_idx`, `gmm*_first_task_id`, `gmm*_first_task_expert`,
  `gmm*_first_task_row_begin`, `gmm*_first_task_n_base`, and matching last-task fields.

## Delta And Handoff

- Design/task issue found: no; issue ID: none.
- User decision needed: no.
- Downstream notes: M3O.3 only changes fused AIC GMM task scheduling and evidence. It does not reopen fine-grained
  GMM1->activation, activation->GMM2, or GMM2->combine overlap; those remain M3O.6 scope.
- Next task: M3O.4.
