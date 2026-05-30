# M3N.11 Task Report

## Header

- Task: M3N.11 Combine V2 Sub-Tile/stride remote write
- Owner: codex/current-session
- End state: review_ready
- Related issue: DCL-115 flag-id reuse risk boundary

## Changes

- Changed files: `TASKS.md`, `reports/M3.md`, `reports/M3N.11.md`, `host/main.cpp`,
  `include/moe_dispatch_combine_a8w8_types.hpp`, `kernel/moe_dispatch_combine_a8w8_kernel.cpp`
- Out-of-scope files touched: none

## Acceptance

- Acceptance source: `DESIGN.md` section 14.4N, subsection `M3N.11 Combine V2 Sub-Tile/stride remote write`
- Acceptance result: review_ready
- Verification summary: the active GMM2 epilogue + return flow now consumes the existing M2.7a
  `subTileReturnPlan/subTileOwnerSegments/subTileReady` schema, waits `subTileReady[tileId]` by GM-poll before each
  owner segment, and sends each owner segment in fixed 16-row sub-tiles. Remote owner segments use synchronous strided
  PTO `TPUT` through 2D `GlobalTensor` source/destination strides; local owner segments use the existing local copy path.
  `OwnerSegment` fields, layout capacity, and destination offsets are unchanged.
- Dependency scan: pass

## Result

- Correctness: pass for the A3/CANN 8.5 targeted 2-rank skewed overlap-on run and the full M3 fused off/on regression
  suite.
- Perf/timeline observation: structural/counter evidence only. Sub-Tile sync `TPUT` is enabled, but full async combine is
  not claimed.
- Blocked reason, if any: async SDMA overlap for strided Sub-Tile return remains blocked by the primitive gap
  `TPUT_ASYNC requires flat-contiguous-1d`.

## Evidence

- A3/CANN 8.5 build: pass.
- Targeted 2-rank skewed overlap-on run: final output mismatch 0 on both ranks. Evidence includes
  `combine_mode=subtile_stride`, `subtile_rows=16`, `subtile_stride_width=256`,
  `subtile_ready_wait_count>0`, `subtile_remote_tput_count>0`, `subtile_local_copy_count>0`,
  `subtile_stride_sync_tput_enabled=true`, and `primitive_gap=TPUT_ASYNC_flat_contiguous_1d`.
- Full M3 off/on suite: exit 0 with all printed correctness/perf reports passing for small, balanced, skewed, and
  zero-token cases across overlap off/on.
- Timeout probe: stage 109 still records `m3n9_timeout_dump_present=true`,
  `timeout_dump_stage_name=dispatch_to_gmm1`, producer status array, and `scoreboardMinStatus`.
- Dependency scan for `TPUT_ASYNC(...)` and M3N.11 cross-core flag/Event additions returned no matches.

## Delta And Handoff

- Design/task issue found: yes. The DCL-115 flag-id 6-10 risk remains contained: M3N.11 does not add a new cross-core
  event edge and reuses the GM-poll ready style already required by M3N.7/M3N.8.
- User decision needed: no.
- Downstream notes: this is synchronous Sub-Tile/stride return correctness and evidence. M3N.12 can consume the printed
  `subtile_*` counters for timeline comparison, but should still report async SDMA stride as blocked unless the PTO
  primitive changes.
- Next task: M3N.12.
