# M3O.5 Task Report

## Header

- Task: M3O.5 Combine stage-local worker policy and V1/V2 choice
- Owner: codex
- End state: review_ready
- Related issue: DCL-2026-06-01-133

## Changes

- Changed files: `kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp`,
  `kernel/moe_dispatch_combine_a8w8_kernel.cpp`, `include/moe_dispatch_combine_a8w8_types.hpp`, `TASKS.md`,
  `reports/M3O.5.md`
- Out-of-scope files touched: none

## Acceptance

- Acceptance source: `DESIGN.md` section 14.2, subsection
  `M3O.5 Combine stage-local worker policy 与 V1/V2 选择`
- Acceptance result: review_ready
- Verification summary: combine now uses a dense combine-local worker assignment instead of reusing dispatch worker
  assignment. The synchronous sub-tile stride return path remains the selected mode for both required cases; asynchronous
  strided return is still honestly reported as the `TPUT_ASYNC_flat_contiguous_1d` primitive gap.
- Dependency scan: source scan found no Catlass/AscendC fallback and no discarded scoreboard/prod-status/min-status
  route. Static checks were intentionally not run per user instruction.

## Result

- Correctness: pass for both required M3O cases on devices 4-7.
- Combine worker evidence: pass. Both required cases report `combine_active_aiv_workers=8`,
  `combine_owner_segment_workers=8`, non-worker0 segment distribution, and `combine_worker_ranges_nonoverlap=true`.
- Combine mode: `subtile_stride` for both required cases. This preserves the existing V2 synchronous sub-tile return path
  because the current return layout already has tile/sub-tile owner segments, while full async strided return remains
  blocked by the public primitive gap.
- Blocked reason, if any: none for M3O.5.

## Evidence

- Build: `cmake --build kernels/manual/a2a3/moe_dispatch_combine_a8w8/build --target moe_dispatch_combine_a8w8 -j16`
  exited 0.
- Required small case: `ffn-v3-small`, `-pes 2 -M 16 -K 128 -N 128 -topK 2 -expertPerPe 2 --max-output-size 32`,
  `--first-device 4 --ndevices 7`, exited 0. Both ranks reported `pass=true` and `final_output.err_count=0`.
  Log: `/tmp/m3o5-small-final-20260601-134558.log`.
- Required large case: `ffn-v3-4097`, `-pes 2 -M 4097 -K 128 -N 128 -topK 2 -expertPerPe 2 --max-output-size 8194`,
  `--first-device 4 --ndevices 7`, exited 0. Both ranks reported `pass=true` and `final_output.err_count=0`.
  Log: `/tmp/m3o5-large-final-20260601-134614.log`.
- Small rank evidence includes `combine_active_aiv_workers=8`, `combine_owner_segment_workers=8`,
  `combine_worker_segment_counts=0:1,1:1,2:1,3:1,4:0,5:0,6:0,7:0`, `combine_mapped_segment_count=4`, and
  `combine_actual_write_count=4`.
- Large rank evidence includes rank0 `combine_worker_segment_counts=0:9,1:9,2:9,3:9,4:8,5:8,6:8,7:8`,
  `combine_mapped_segment_count=68`, `combine_actual_write_count=68`, and rank1
  `combine_worker_segment_counts=0:9,1:8,2:8,3:8,4:8,5:8,6:8,7:8`, `combine_mapped_segment_count=65`,
  `combine_actual_write_count=65`.
- Benchmark policy remains `warmup_iters=0` and `measure_iters=1` in both required runs.

## Delta And Handoff

- Design/task issue found: no; issue ID: none.
- User decision needed: no.
- Downstream notes: M3O.5 does not reopen fine-grained GMM2->combine overlap. M3O.6 may use this combine-local worker
  policy, but must still respect the M3O.4 activation shard blocker before reopening activation-related overlap.
- Next task: M3O.6.
