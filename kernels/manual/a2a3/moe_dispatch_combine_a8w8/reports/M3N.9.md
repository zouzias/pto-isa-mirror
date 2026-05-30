# M3N.9 Task Report

## Header

- Task: M3N.9 timeout dump and overlap off/on regression
- Owner: codex/current-session
- End state: review_ready
- Related issue: DCL-115 flag-id reuse risk follow-up

## Changes

- Changed files: `DESIGN.md`, `TASKS.md`, `host/main.cpp`, `kernel/control_metadata.hpp`,
  `kernel/moe_dispatch_combine_a8w8_kernel.cpp`, `kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp`,
  `scripts/run_a3.sh`, `reports/M3.md`, `reports/M3N.6.md`, `reports/M3N.9.md`
- Out-of-scope files touched: none

## Acceptance

- Acceptance source: `DESIGN.md` section 14.4N, subsection `M3N.9 Timeout dump 与 overlap 回归`
- Acceptance result: review_ready
- Verification summary: M3N.9 adds a device timeout-dump probe at fused debug-stop stage 109, host-side dump printing,
  and script wiring through `--m3-timeout-probe`. The full M3 off/on regression suite now covers the DCL-115 risk
  boundary: M3N.9 does not add a new `pto::Event` edge on cross-core flag ids 6-10, M3N.7/M3N.8 remain GM-poll ready,
  and `--overlap-mode off` disables the M3N.5-M3N.8 narrowed-wait paths.
- Dependency scan: pass

## Result

- Correctness: pass for the full M3 fused off/on regression suite and timeout probe.
- Perf/timeline observation: `[PerfReport]` E2E samples are still emitted by the existing run path. No performance win is
  claimed.
- Blocked reason, if any: none for M3N.9. M3N.10-M3N.12 remain pending, and M3N.8/M3N.9 still require review before
  stage M3 can close.

## Evidence

- DCL-115 risk review: no new cross-core flag ids or event helpers were added for M3N.9. `protocol_core.hpp` still has
  the M3N.5 V2C and M3N.6 C2V users of ids 6-10 only; M3N.7/M3N.8 consume `activationSyncGroupReady` /
  `gmm2GroupReady` by GM-poll.
- Timeout dump fields: `m3n9_timeout_dump_present`, `timeout_dump_rank`, `timeout_dump_expert`,
  `timeout_dump_token_owner_rank`, `timeout_dump_expert_owner_rank`, `timeout_dump_stage`,
  `timeout_dump_stage_name`, `timeout_dump_signal_id`, and `timeout_dump_debug_stop_stage` are printed from
  `scoreboardTimeoutCounters`.
- Full M3 suite: `--m3-suite 1` runs small, balanced, skewed, and zero-token cases for both `--overlap-mode off` and
  `--overlap-mode on`; the command exited 0, emitted `[CorrectnessReport]` / `[PerfReport]` pass evidence, and a
  targeted failure scan found no `pass=false`, nonzero mismatch, nonzero final-output error count, or timeout process
  failure.
- `--overlap-mode off` evidence includes `m3_overlap_requested=false`, M3N.5-M3N.8 enabled fields false, and
  `m3n4_signal_all_open=true`.
- `--overlap-mode on` evidence includes M3N.5-M3N.8 enabled fields true, `m3n7_sync_transport=gm_poll_ready`,
  `m3n8_sync_transport=gm_poll_ready`, and final `m2.outputC` mismatches 0.
- Timeout probe evidence includes `m2_fused_debug_stop_stage=109`, `overlap_timeout_count=1`,
  `m3n9_timeout_dump_present=true`, `timeout_dump_stage_name=dispatch_to_gmm1`, and
  `timeout_dump_debug_stop_stage=109`.
- Regression root cause fixed during M3N.9 cleanup: M3N dispatch shard workers must not write adjacent
  `expandedRowIdx` entries in the same 64-byte cache line. `M3NDispatchWorkerCount` now reduces worker count until
  route-shard boundaries are cache-line aligned, and payload/scale cache invalidation is per written row.
- Hygiene fix: `ClearDeviceStateM2` clears `outputC` before each fused iteration, so repeated off/on runs cannot inherit
  stale final-output data.

## Delta And Handoff

- Design/task issue found: yes. DCL-115 remains a real M3N.6 risk: M3N.6 is safe only because M3N.5 V2C events are
  consumed before M3N.6 C2V events are produced, separated by the coarse AIC sync. M3N.9 keeps M3N.7/M3N.8 on GM-poll
  ready and adds no new event edge on ids 6-10.
- User decision needed: no.
- Limitation: the debug-stop stage 109 probe is a deterministic diagnostic hook, not a real hardware hang injection.
  M3N.9 does not implement scoreboard async, Sub-Tile/stride return, or timeline timestamps.
- Next task: M3N.10.
