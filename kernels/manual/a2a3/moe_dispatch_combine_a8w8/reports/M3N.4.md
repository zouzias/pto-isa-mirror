# M3N.4 Task Report

## Header

- Task: M3N.4 BSP to stream execution skeleton
- Owner: codex/current-session
- End state: review_ready
- Related issue: none

## Changes

- Changed files: `include/pto/npu/a2a3/TSync.hpp`, `host/main.cpp`, `kernel/protocol_core.hpp`, `kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp`, `TASKS.md`
- Out-of-scope files touched: none

## Acceptance

- Acceptance source: `DESIGN.md` section 14.4N, subsection `M3N.4 BSP->流水执行骨架（signal 全开 = 等价 BSP）`
- Acceptance result: review_ready
- Verification summary: The fused mixed AIC/AIV path now reports `exec_model=aic_aiv_stream` and uses six `pto::Event` cross-core full-open handshake sites for dispatch->GMM1, GMM1->epilogue, epilogue->activation, activation->GMM2, GMM2->combine, and combine->restore. The signal range remains full-open, so no expert/sync-group ready granularity is narrowed in this task. M3N.1-M3N.3 stage-internal `SYNCALL<Mix>` subphase barriers remain as coarse/internal correctness sync; AIC-only and AIV-only coarse sync points preserve BSP equivalence where a producing domain must become globally visible before the next same-domain stage.
- Dependency scan: pass

## Result

- Correctness: pass
- Perf/timeline observation: pass for structural evidence only; no payload async overlap or performance win is claimed
- Blocked reason, if any: none

## Evidence

- A3/CANN 8.5 build: pass.
- Balanced debug stops `1/2/3/4/5/56`: all return 0 with `pass=true`.
- 4-rank `balanced` full fused int8, overlap off: final output mismatches 0 on all ranks; representative report fields include `exec_model=aic_aiv_stream`, `m3n4_stream_skeleton_enabled=true`, `m3n4_signal_all_open=true`, `m3n4_handshake_mode=pto_event_cross_core_full_open`, `m3n4_handshake_site_count=6`, `syncall_count=20`, `cv_wait_count=6`, and `overlap_on_payload_async_claim=false`.
- 4-rank `balanced` full fused int8, overlap on: final output mismatches 0 on all ranks with the same full-open stream skeleton fields and `m3_overlap_execution=signal_full_open_stream_skeleton`.
- Dependency scan for forbidden CATLASS/CUTLASS/AscendC CrossCore/DataAsFlag/direct launch strings returned no matches.

## Delta And Handoff

- Design/task issue found: yes; issue ID: none. During verification, CV handshakes alone were not BSP-equivalent for V->V stage edges because non-producing AIV lanes could enter activation/restore before the main AIV finished epilogue/finalize work. The fix adds AIV-only coarse sync before the two V->V ready signals.
- User decision needed: no
- Downstream notes: M3N.5 can start after reviewer acceptance or explicit approval. M3N.4 does not shrink dispatch->GMM1 readiness to expert granularity, does not shrink activation/GMM2 readiness to sync-group granularity, does not change row order/scoreboard/return segment schema, and does not claim performance.
- Next task: M3N.5.
