# M3N.6 Task Report

## Header

- Task: M3N.6 GMM1 to SwiGLU sync-group rotation
- Owner: codex/current-session
- End state: accepted (reviewer code-level review, DCL-2026-05-30-115)
- Related issue: none

## Changes

- Changed files: `kernel/protocol_core.hpp`, `kernel/moe_dispatch_combine_a8w8_kernel.cpp`,
  `kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp`, `host/main.cpp`, `TASKS.md`
- Out-of-scope files touched: none

## Acceptance

- Acceptance source: `DESIGN.md` section 14.4N, subsection `M3N.6 GMM1->SwiGLU 轮动（sync-group 粒度）`
- Acceptance result: accepted
- Verification summary: GMM1 to activation now rotates at `swigluSyncGroups` sync-group granularity. AIC runs GMM1 for
  the experts covered by the current sync group, stores `gmm1SyncGroupReady[syncIdx]`, and signals a PTO C2V event.
  AIV waits the matching syncIdx, runs GMM1 epilogue for that sync group, then shards activation/requant only over the
  `dequantSum[syncIdx]..dequantSum[syncIdx+1]` row range. M3N.2 activation worker splitting and sync-group row order are
  unchanged.
- Dependency scan: pass

## Result

- Correctness: pass
- Perf/timeline observation: structural/counter evidence only; no performance win is claimed.
- Blocked reason, if any: none.

## Evidence

- A3/CANN 8.5 build: pass.
- Regression root cause: the first implementation let only AIC block 0 call `M3N6SignalGmm1SyncGroupReady`, which caused
  the balanced stage-3 debug-stop and full run to timeout. The fix keeps GM evidence writes on block 0 but lets all AIC
  blocks participate in the C2V `pto::Event` signal after `M3N4AicAllDoneCoarseSync`.
- Balanced debug stop `--m2-fused-debug-stop-stage 3`: pass after the fix.
- 4-rank `balanced` full fused int8: final output mismatches 0 on all ranks. Representative fields include
  `m3n6_gmm1_activation_overlap_enabled=true`, `gmm1_activation_overlap_granularity=sync_group`,
  `m3n6_gmm1_sync_group_producer_counter=2`, `m3n6_activation_sync_group_consumer_counter=2`,
  `gmm1_sync_group_ready_count=2`, `activation_sync_group_ready_count=2`, and `swiglu_sync_group_count=2`.
- 4-rank `zero-token` full fused int8: final output mismatches 0 on all ranks; zero-token experts still publish/signal
  and do not permanently wait.
- 4-rank `expertPerRank=4` multi-group full fused int8: final output mismatches 0 on all ranks, producer/consumer counters
  are 3/3, sync-group ready counts are 3, and at least one rank records
  `m3n6_activation_start_before_last_gmm1_ready=true`.
- Dependency scan for forbidden CATLASS/CUTLASS/AscendC CrossCore/DataAsFlag/direct launch strings returned no matches.

## Delta And Handoff

- Design/task issue found: yes. The reviewer recorded DCL-115: M3N.5 V2C and M3N.6 C2V reuse cross-core flag ids
  6-10, and `wait_flag_dev` keys by id rather than direction. M3N.6 remains accepted because it consumes all M3N.5
  V2C signals, crosses the coarse AIC sync, then produces M3N.6 C2V signals. M3N.7 and M3N.8 avoid expanding this risk
  by using GM-poll ready rather than adding another `pto::Event` edge on ids 6-10.
- User decision needed: no.
- Downstream notes: M3N.7 replaced the Activation->GMM2 edge with GM-poll ready on
  `activationSyncGroupReady[syncIdx]` and intersects each sync-group row range back to M2.GMM tile tasks.
- Next task: M3N.7.
