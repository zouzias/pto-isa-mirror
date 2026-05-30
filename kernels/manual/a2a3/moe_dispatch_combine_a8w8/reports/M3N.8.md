# M3N.8 Task Report

## Header

- Task: M3N.8 GMM2 to combine expert-granularity rotation
- Owner: codex/current-session
- End state: review_ready
- Related issue: DCL-115 flag-id reuse risk follow-up

## Changes

- Changed files: `DESIGN.md`, `TASKS.md`, `host/main.cpp`, `kernel/moe_dispatch_combine_a8w8_kernel.cpp`,
  `kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp`, `reports/M3N.8.md`
- Out-of-scope files touched: none

## Acceptance

- Acceptance source: `DESIGN.md` section 14.4N, subsection `M3N.8 GMM2->combine 轮动（expert 粒度）`
- Acceptance result: review_ready
- Verification summary: Combine now consumes `gmm2GroupReady[expert]` with GM-poll ready, builds the existing
  M2.7a return-segment map one expert at a time, and shards that expert's continuous owner segments across the
  M3N.3 combine workers. The full GMM2->combine C2V wait is skipped when M3N.8 is enabled.
- Dependency scan: pass

## Result

- Correctness: pass
- Perf/timeline observation: structural/counter evidence only; no performance win is claimed.
- Blocked reason, if any: none.

## Evidence

- DCL-115 risk handling: M3N.8 does not allocate another `pto::Event` edge on cross-core flag ids 6-10. It reuses the
  M3N.7 `gmm2GroupReady[expert]` GM ready slot and reports `m3n8_sync_transport=gm_poll_ready` plus
  `m3n8_flag_reuse_risk_avoided=true`.
- A3/CANN 8.5 build: pass.
- Dependency scan for forbidden CATLASS/CUTLASS/AscendC CrossCore/DataAsFlag/direct launch strings returned no matches.
- 4-rank `zero-token` full fused int8: `m2.outputC` mismatches are 0 on all ranks. Representative fields include
  `m3n8_gmm2_expert_ready_counter=2`, `m3n8_combine_expert_consumer_counter=2`,
  `m3n8_per_expert_wait_scope=true`, `m3n8_full_gmm2_to_combine_cv_wait=false`,
  `combine_mode=continuous_segment`, and `combine_cv_wait_count=0`.
- 4-rank `balanced` with `expertPerRank=4`: `m2.outputC` mismatches are 0 on all ranks. Representative fields include
  `m3n8_gmm2_expert_ready_counter=4`, `m3n8_combine_expert_consumer_counter=4`,
  `m3n8_combine_segment_consumer_counter=16`, `m3n8_per_expert_wait_scope=true`,
  `combine_mode=continuous_segment`, `combine_syncall_count=10`, and `combine_cv_wait_count=0`.

## Delta And Handoff

- Design/task issue found: yes. The original M3N.8 text said `pto::Event`/`TSYNC_CVID`, but M3N.6/M3N.7 review found
  flag ids 6-10 are already at risk. The design is refined so M3N.8 must use GM-poll ready unless a later task proves a
  new stale-free flag protocol.
- User decision needed: no.
- Limitation: M3N.8 is expert-group / continuous-segment overlap, not article-level full async. The balanced runs can
  still report `m3n8_combine_start_before_last_gmm2_ready=false`; that field is a timing observation, not the primary
  proof. The structural proof is the per-expert GM-poll path plus equal GMM2-ready and combine-consumer counters.
- Next task: M3N.9.
