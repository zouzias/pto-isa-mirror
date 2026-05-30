# M3N.2 Task Report

## Header

- Task: M3N.2 Activation/SwiGLU/requant internal multi-AIV split + pipe evidence
- Owner: codex/current-session
- End state: review_ready
- Related issue: none

## Changes

- Changed files: `host/main.cpp`, `kernel/moe_dispatch_combine_a8w8_kernel.cpp`, `kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp`
- Out-of-scope files touched: none

## Acceptance

- Acceptance source: `DESIGN.md` section 14.4N, subsection `M3N.2 Activation/SwiGLU/requant 内部多 AIV 切分 + pipe`
- Acceptance result: review_ready
- Verification summary: Activation now reuses the dense AIV worker assignment and processes M2.5-fixed `swigluGroupDesc` row/tile ranges with grid-stride workers. `gmm2PerTokenScale` initialization is sharded, per-row SwiGLU and dynamic requant numerical order is unchanged, and activation ready publication remains inside the current hard `SYNCALL<Mix>` BSP framework. A 4-rank `balanced` full fused int8 run passed final output correctness on all ranks and reported `activation_aiv_workers=8`, `activation_payload_parallel=true`, `swiglu_group_tile_ranges=0:0-2,1:2-4`, `activation_tiles_processed=4`, `activation_tiles_skipped=0`, and `activation_worker_ranges_nonoverlap=true`.
- Dependency scan: pass

## Result

- Correctness: pass
- Perf/timeline observation: pass for worker/tile/counter shape evidence; `activation_pipe_stages=1`, `activation_pipe_prefill=false`, `activation_pipe_drain=false` are honest single-buffer pipe evidence
- Blocked reason, if any: none

## Delta And Handoff

- Design/task issue found: no; issue ID: none
- User decision needed: no
- Downstream notes: M3N.3 remains the last Phase 1 internal split before M3N.4 can restructure the execution skeleton. M3N.2 does not claim GMM1/activation or activation/GMM2 stage overlap.
- Next task: M3N.3.
