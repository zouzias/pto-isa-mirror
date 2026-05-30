# M3N.3 Task Report

## Header

- Task: M3N.3 Combine internal multi-AIV owner-segment split
- Owner: codex/current-session
- End state: review_ready
- Related issue: none

## Changes

- Changed files: `host/main.cpp`, `include/moe_dispatch_combine_a8w8_layout.hpp`, `kernel/moe_dispatch_combine_a8w8_kernel.cpp`, `kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp`
- Out-of-scope files touched: none

## Acceptance

- Acceptance source: `DESIGN.md` section 14.4N, subsection `M3N.3 Combine 内部多 AIV owner-segment 切分（V1 continuous）`
- Acceptance result: review_ready
- Verification summary: Combine now builds the existing M2.7a return segment map on the main AIV, then active AIV workers shard GMM2 epilogue plus continuous owner-segment return by owner segment id inside the current hard `SYNCALL<Mix>` BSP framework. The M2.7a `ReturnSegmentPlan/OwnerSegment` schema, capacity, destination offset, row range, and hidden chunk semantics are unchanged. AIC now participates in the two combine subphase `SYNCALL<Mix>` points, fixing the initial post-shard hang. Per-worker segment counters use cache-line-isolated scratch and are summarized back into the host-visible compact counter block.
- Dependency scan: pass

## Result

- Correctness: pass
- Perf/timeline observation: pass for worker/segment/counter shape evidence; no GMM2 to combine stage-overlap performance claim
- Blocked reason, if any: none

## Evidence

- A3/CANN 8.5 build: pass.
- 4-rank `balanced` full fused int8: final output mismatches 0 on all ranks; representative counters show `combine_return_aiv_workers=8`, `combine_worker_segment_counts=0:1,1:1,2:1,3:1,4:1,5:1,6:1,7:1`, `combine_mapped_segment_count=8`, `combine_actual_write_count=8`, `combine_skipped_segment_count=0`, `combine_payload_parallel=true`, `return_payload_source=gmm2Out_segment_owner_shard`.
- 2-rank `skewed` full fused int8: final output mismatches 0 on both ranks; rank 0 shows 9 mapped/actual segments split across 8 workers with skipped 0, and rank 1 shows 2 mapped/actual segments split across workers 0 and 1 with skipped 0.

## Delta And Handoff

- Design/task issue found: no; issue ID: none
- User decision needed: no
- Downstream notes: M3N.4 can start after reviewer acceptance or explicit approval to proceed from the three Phase 1 split reports. M3N.3 does not claim GMM2/combine stage overlap, Sub-Tile/stride V2, async SDMA stride, or a performance win.
- Next task: M3N.4.
