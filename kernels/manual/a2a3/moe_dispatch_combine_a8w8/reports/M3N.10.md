# M3N.10 Task Report

## Header

- Task: M3N.10 Dispatch-GMM scoreboard async
- Owner: codex/current-session
- End state: review_ready
- Related issue: DCL-115 flag-id reuse risk follow-up

## Changes

- Changed files: `TASKS.md`, `reports/M3.md`, `reports/M3N.10.md`, `host/main.cpp`,
  `include/moe_dispatch_combine_a8w8_types.hpp`, `kernel/control_metadata.hpp`,
  `kernel/moe_dispatch_combine_a8w8_kernel.cpp`, `kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp`
- Out-of-scope files touched: none

## Acceptance

- Acceptance source: `DESIGN.md` section 14.4N, subsection `M3N.10 Dispatch-GMM scoreboard async`
- Acceptance result: review_ready
- Verification summary: Dispatch->GMM1 no longer uses the M3N.5 expert `pto::Event` path when overlap is on.
  AIV dispatch workers fill the existing M2.2c `scoreboardTaskMap/producerStatus` ledger by local-expert dependency
  domain, a main-AIV control role aggregates each domain into `scoreboardMinStatus[localExpert]`, and AIC GMM1 workers
  GM-poll only the matching domain before running that expert's GMM1 tile work. `--overlap-mode off` keeps the old
  M3N.4 full-open fallback and reports scoreboard async disabled.
- Dependency scan: pass

## Result

- Correctness: pass for the A3/CANN 8.5 M3 fused off/on regression suite (`small`, `balanced`, `skewed`, `zero-token`)
  and the timeout-dump probe.
- Perf/timeline observation: structural/counter evidence only; no performance win is claimed.
- Blocked reason, if any: none for M3N.10 delivery. Formal accepted state still needs reviewer code-level review.

## Evidence

- A3/CANN 8.5 build: pass.
- Targeted 4-rank balanced overlap-on run: `scoreboard_async_enabled=true`,
  `scoreboard_worker_poll_scope=scoreboardMinStatus[dependencyDomain]`,
  `scoreboard_async_global_min_task_id=false`, nonzero producer/worker poll counters, and final output mismatch 0.
- Full M3 off/on suite: 26 `[CorrectnessReport]` and 26 `[PerfReport]` sections; failure scan found no `pass=false`,
  final-output error, timeout failure, or nonzero semantic mismatch. The suite contains 13 scoreboard-disabled reports
  for overlap off and 13 scoreboard-enabled reports for overlap on.
- Timeout probe: stage 109 still records `m3n9_timeout_dump_present=true`,
  `timeout_dump_stage_name=dispatch_to_gmm1`, `timeout_dump_debug_stop_stage=109`,
  `timeout_dump_producer_status_array=2,0,0,0,0,0`, `timeout_dump_scoreboard_min_status=2`, and
  `timeout_dump_scoreboard_domain=0`.
- Dependency scan for forbidden CATLASS/CUTLASS/AscendC CrossCore/DataAsFlag/direct launch strings returned no matches.

## Delta And Handoff

- Design/task issue found: yes. M3N.10 closes the DCL-115 follow-up for dispatch->GMM1 by replacing the M3N.5
  flag-id 6-10 expert-ready edge with GM-poll scoreboard domains when overlap is on. M3N.5 expert-ready remains a
  fallback/debug path for overlap-off or unsupported scoreboard paths.
- User decision needed: no.
- Downstream notes: M3N.11 can build on the current M3N.8 GMM2->combine expert GM-poll path. M3N.10 does not implement
  Sub-Tile/stride return or timeline timestamps.
- Next task: M3N.11.
