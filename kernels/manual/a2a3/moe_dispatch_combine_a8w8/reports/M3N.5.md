# M3N.5 Task Report

## Header

- Task: M3N.5 dispatch to GMM1 expert-granularity rotation
- Owner: codex/current-session
- End state: accepted (reviewer code-level review, DCL-2026-05-30-113)
- Related issue: none

## Changes

- Changed files: `host/main.cpp`, `kernel/protocol_core.hpp`, `kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp`, `TASKS.md`
- Out-of-scope files touched: none

## Acceptance

- Acceptance source: `DESIGN.md` section 14.4N, subsection `M3N.5 dispatch→GMM1 轮动（expert 粒度）`
- Acceptance result: review_ready
- Verification summary: Dispatch to GMM1 now rotates at local-expert granularity. AIV gathers one local expert, publishes `dispatchGroupReady[expert]`, and signals a per-expert PTO cross-core event. AIC waits on the matching expert event and runs only GMM1 tasks for that expert before moving to the next expert. M3N.1 dispatch worker partitioning, row order, and scoreboard semantics are unchanged.
- Dependency scan: pass

## Result

- Correctness: pass
- Perf/timeline observation: structural evidence only; no performance win is claimed
- Blocked reason, if any: none

## Evidence

- A3/CANN 8.5 build: pass.
- Balanced debug stops `1/2/3/4/5/56`: all return 0 with `pass=true` on 4 ranks.
- 4-rank `balanced` full fused int8, overlap off and overlap on: final output mismatches 0 on all ranks.
- Representative M3N.5 report fields: `m3n5_dispatch_gmm1_overlap_enabled=true`, `dispatch_overlap_granularity=expert`, `m3n5_dispatch_expert_ready_count=2`, `m3n5_gmm1_start_before_last_dispatch_ready=true`, and `dispatch_stage_overlap_enabled=true`.
- 4-rank `zero-token` full fused int8: final output mismatches 0; zero-token ranks report nonzero `m3n5_zero_token_expert_skip_count`, proving no permanent wait on empty experts.
- Dependency scan for forbidden CATLASS/CUTLASS/AscendC CrossCore/DataAsFlag/direct launch strings returned no matches.

## Delta And Handoff

- Design/task issue found: no
- User decision needed: no
- Downstream notes: M3N.6 can build on this by replacing the GMM1→SwiGLU full-open wait with sync-group granularity. M3N.5 does not change activation/GMM2/combine rotation, does not change return segment schema, and does not claim payload async performance.
- Next task: M3N.6.
