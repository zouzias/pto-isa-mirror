# M3N.12 Task Report

## Header

- Task: M3N.12 Kernel timestamp and timeline output
- Owner: codex/current-session
- End state: review_ready
- Related issue: DCL-115 flag-id reuse risk boundary

## Changes

- Changed files: `TASKS.md`, `reports/M3.md`, `reports/M3N.12.md`, `host/main.cpp`,
  `include/moe_dispatch_combine_a8w8_types.hpp`, `kernel/control_metadata.hpp`,
  `kernel/moe_dispatch_combine_a8w8_kernel.cpp`, `kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp`
- Out-of-scope files touched: none

## Acceptance

- Acceptance source: `DESIGN.md` section 14.4N, subsection `M3N.12 Kernel timestamp 与 timeline 输出`
- Acceptance result: review_ready
- Verification summary: the fused kernel now records diagnostic timestamp rows into the pre-reserved
  `workspace.timelineScratch` area when `--timeline 1` is enabled. Host-side output prints `[Timeline]` records with
  the same `case_name`, `seed`, `run_id`, and shape fields as `[PerfReport]`, and uses
  `timeline_granularity=stage_group_tile_subtile`.
- Dependency scan: pass

## Result

- Correctness: pass for timeline-enabled small overlap on/off runs and for the full M3 fused regression suite with
  timeline disabled.
- Perf/timeline observation: pass for structural timeline evidence. Rows cover stage, GMM tile, SwiGLU group, and owner
  segment granularities; no performance win is claimed.
- Blocked reason, if any: none for M3N.12 delivery.

## Evidence

- A3/CANN 8.5 clean build: pass.
- Small overlap-on timeline run: pass, with route/count/dispatch/GMM1/SwiGLU/GMM2/combine/restore timeline rows and
  wait-source evidence including scoreboard and GM-poll paths.
- Small overlap-off timeline run: pass, with syncall wait-source evidence and explicit skipped rows for unavailable
  work.
- Full M3 suite: exit 0 for the fused small, balanced, skewed, and zero-token regression set with timeline disabled.
- Dependency scan found no new M3N.12 `set_flag`/`wait_flag`/cross-core event edge; `waitSource=kPtoEvent` is only a
  timeline label for existing paths.

## Delta And Handoff

- Design/task issue found: yes. The DCL-115 flag-id 6-10 risk remains contained: M3N.12 records timestamps and
  wait-source labels but adds no new cross-core flag/event edge. Later timeline expansion should continue using the
  pre-reserved timeline area rather than changing payload, scoreboard, or Sub-Tile segment layout.
- User decision needed: no.
- Downstream notes: M4.1 can consume `--timeline 1` as the minimal timestamp-dump switch after reviewer acceptance.
  M3N.12 does not change the M3N.11 async SDMA-stride primitive gap.
- Next task: reviewer code-level review of M3N.10-M3N.12, then M4.1.
