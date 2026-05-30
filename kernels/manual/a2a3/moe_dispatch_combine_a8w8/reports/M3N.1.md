# M3N.1 Task Report

## Header

- Task: M3N.1 Dispatch internal multi-AIV split
- Owner: codex/current-session
- End state: review_ready
- Related issue: none

## Changes

- Changed files: `host/main.cpp`, `kernel/moe_dispatch_combine_a8w8_kernel.cpp`, `kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp`
- Out-of-scope files touched: none

## Acceptance

- Acceptance source: `DESIGN.md` section 14.4N, subsection `M3N.1 Dispatch 内部多 AIV 切分`
- Acceptance result: review_ready
- Verification summary: Dispatch now uses multiple AIV workers inside the existing `SYNCALL<Mix>` BSP framework. Route/count/pack and count publish/wait are token/rank sharded, gather is local-expert sharded, and the compact dispatch ledger/scoreboard finalization remains on the main AIV after parallel gather. A 4-rank `balanced` dispatch debug run showed zero dispatch mismatches and worker evidence `route_pack_workers=8`, `count_sync_workers=8`, `dispatch_gather_workers=2`, `dispatch_worker_count=8`, with non-overlap evidence enabled. A full fused `balanced` int8 run passed final output correctness on all ranks with `dispatch_stage_overlap_enabled=false`.
- Dependency scan: pass

## Result

- Correctness: pass
- Perf/timeline observation: pass for worker/counter shape evidence; no stage-overlap performance claim
- Blocked reason, if any: none

## Delta And Handoff

- Design/task issue found: no; issue ID: none
- User decision needed: no
- Downstream notes: M3N.2 and M3N.3 remain independent Phase 1 splits. M3N.4 must not start until M3N.1, M3N.2, and M3N.3 are accepted or explicitly allowed by review.
- Next task: M3N.2 or M3N.3.
