# M3N.7 Task Report

## Header

- Task: M3N.7 SwiGLU to GMM2 sync-group rotation
- Owner: codex/current-session
- End state: accepted (reviewer code-level review, DCL-2026-05-30-117)
- Related issue: none

## Changes

- Changed files: `DESIGN.md`, `TASKS.md`, `host/main.cpp`, `include/moe_dispatch_combine_a8w8_layout.hpp`,
  `kernel/moe_dispatch_combine_a8w8_kernel.cpp`, `kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp`,
  `reports/M3N.6.md`, `reports/M3N.7.md`
- Out-of-scope files touched: none

## Acceptance

- Acceptance source: `DESIGN.md` section 14.4N, subsection
  `M3N.7 SwiGLU->GMM2 轮动（sync-group 粒度 + intersect 回 GMM2 tile）`
- Acceptance result: review_ready
- Verification summary: Activation now publishes `activationSyncGroupReady[syncIdx]` as each sync group finishes. AIC
  GMM2 consumes sync groups with a GM-poll ready wait, intersects `swigluGroupDesc` expert/row ranges back to the
  existing M2 GMM2 tile task plan, runs only matching GMM2 tasks, and then publishes `gmm2GroupReady[expert]` for the
  covered experts. M3N.2 activation worker splitting, sync-group row order, and GMM tile task schema are unchanged.
- Dependency scan: pass

## Result

- Correctness: pass
- Perf/timeline observation: structural/counter evidence only; no performance win is claimed.
- Blocked reason, if any: none.

## Evidence

- DCL-115 risk handling: M3N.7 does not allocate a third `pto::Event` edge on cross-core flag ids 6-10. It uses
  GM-poll ready on `activationSyncGroupReady[syncIdx]` and reports `m3n7_sync_transport=gm_poll_ready` plus
  `m3n7_flag_reuse_risk_avoided=true`.
- Regression root cause: first GM-poll version timed out at debug-stop stage 4 because AIC polled the ready slot without
  invalidating the GM cache line. The fix adds `dcci(..., SINGLE_CACHE_LINE)` before each AIC-side ready load.
- A3/CANN 8.5 build: pass.
- Balanced debug stop `--m2-fused-debug-stop-stage 3`: pass, confirming the M3N.6 activation path still exits.
- Balanced debug stop `--m2-fused-debug-stop-stage 4`: pass after the cache-line invalidation fix.
- Balanced debug stop `--m2-fused-debug-stop-stage 5`: pass.
- 4-rank `balanced` full fused int8: final output mismatches 0 on all ranks. Representative fields include
  `m3n7_activation_gmm2_overlap_enabled=true`, `activation_gmm2_overlap_granularity=sync_group`,
  `m3n7_activation_sync_group_producer_counter=2`, `m3n7_gmm2_sync_group_consumer_counter=2`,
  `m3n7_gmm2_intersect_task_count=4`, `activation_sync_group_ready_count=2`,
  `gmm2_group_ready_count=2`, and `m3n7_gmm2_start_before_last_activation_ready=true`.
- 4-rank `zero-token` full fused int8: final output mismatches 0 on all ranks; zero-token ranks keep producer/consumer
  counters at 2/2 and can report `m3n7_gmm2_intersect_task_count=0`, proving empty sync groups do not permanently wait.
- 4-rank `expertPerRank=4` multi-group full fused int8: final output mismatches 0 on all ranks, producer/consumer
  counters are 3/3, `activation_sync_group_ready_count=3`, `gmm2_group_ready_count=4`, and the last consumed group
  reports expert/row/tile preview fields.
- Dependency scan for forbidden CATLASS/CUTLASS/AscendC CrossCore/DataAsFlag/direct launch strings returned no matches.

## Delta And Handoff

- Design/task issue found: yes. DCL-115 flag-id reuse risk was real; `wait_flag_dev` keys by id, so C2V/V2C direction
  does not provide an independent namespace. The M3N.7 design was updated to use GM-poll ready for this edge.
- User decision needed: no.
- Downstream notes: M3N.8 can consume `gmm2GroupReady[expert]` for expert-granularity GMM2->combine rotation. Current
  M3N.7 still keeps Combine V1 owner-segment behavior and does not claim payload-level async overlap.
- Next task: M3N.8.
