# moe_dispatch_combine_a8w8 Bugfix Notes

## 2026-06-01 M3 runtime on-mode hang / correctness pass

### Scope

- Target: runtime M3 path for `moe_dispatch_combine_a8w8`.
- M3R/M3S static work was discarded for now and is not part of this fix record.
- Priority: make real NPU launch pass with `OVERLAP_MODE=on`.
- Build validation must use a clean build because stale build artifacts previously caused misleading results.

### Observed symptoms

- `OVERLAP_MODE=on` cases could appear to hang after dispatch/GMM stages.
- `on/skewed` and `on/zero-token` could have long periods without new log output.
- A false hang can happen because `scripts/run_a3.sh` fixes `warmup_iters=3` and `measure_iters=5`; each fused iteration and verify step can take seconds, and logs only flush at stage or verify boundaries.
- A true hang risk is usually a rank waiting forever in a GM poll loop or at an MPI/HCCL barrier after another rank fails to advance.

### Key root-cause areas

1. M3N10 dispatch-to-GMM1 scoreboard async sync

   Fine-grained scoreboard async adds a producer/consumer dependency between dispatch and GMM1. If a producer status or domain min status is not advanced for an empty/skewed expert, GMM1 workers can poll forever. For the current runtime target this is not needed. Expert-level synchronization, matching the original FFN-style behavior, is sufficient.

2. Cross-rank GM signal visibility

   Count ready signals, combine done signals, return segment counters, and return payload metadata are synchronization data. Async copy paths or multi-writer paths can expose stale or partially visible values to remote ranks, causing polling loops to wait indefinitely.

3. Dispatch scale/cacheline sharing

   Multiple dispatch workers writing adjacent scale/signal locations can create false sharing or visibility ambiguity. The observed mitigation is to avoid concurrent writers for the dispatch scale path and use scalar bit-preserving stores/loads for scale values.

4. Empty expert and zero-token boundaries

   `skewed` and `zero-token` are the highest-risk cases because some experts have no rows while other ranks still participate in collective and polling protocols. Any missing skip/ready signal for an empty segment can deadlock the consumer side.

5. Rank-level barrier masking

   If one rank throws, times out, or fails an ACL/HCCL call, the other ranks can look stuck in `MpiBarrier`, `mpirun`, or a poll loop. Always inspect all ranks' logs, not just the last visible line.

6. Build artifact confusion

   Concurrent clean builds in the same build directory previously caused transient build failures such as missing `.o.d` files. Do not start another build while an M3 suite clean build is running.

### Implemented direction

- Disable M3N10 scoreboard async for the runtime path and fall back to expert-level M3N5 dispatch-to-GMM1 synchronization.
- Keep dispatch/GMM1 synchronization at expert granularity for now.
- Use scalar GM store/load/poll for cross-rank synchronization signals and counters.
- Use safer scalar copies for dispatch ledger/scale data.
- Avoid multi-worker false sharing on dispatch scale.
- Keep AIC-side fallback/coherent scalar paths where needed to avoid shared-state visibility problems.

### Validation command

Run from this directory:

```bash
bash scripts/run_a3.sh --m3-suite 1 --backend int8 --clean-build 1 --dry-run 0 --skip-kernel-launch 0 --first-device 0 --ndevices 8 --timeline 0
```

Validation log:

```text
out/m3_suite_clean_after_fix.log
```

### Passing evidence

- Clean M3 suite completed after the fixes.
- `OVERLAP_MODE=on` passed for:
  - `small`
  - `balanced`
  - `skewed`
  - `zero-token`
- The log contains `final_output.err_count=0` for all expected rank summaries.
- The log contains `pass=true` for all expected correctness/perf summaries.
- `scoreboard_async_enabled=false` appears in the reports, confirming the runtime path is using the expert-level sync fallback.
- Failure scan was clean for:
  - `runtime_error`
  - `FAILED`
  - `pass=false`
  - non-zero `mismatches`
  - non-zero `final_output.err_count`

### Useful diagnosis checklist

When execution looks stuck:

1. Check whether it is a real hang or just a long fused iteration with no flush:

   ```bash
   tail -n 120 out/m3_suite_clean_after_fix.log
   ps -eo pid,ppid,stat,etime,cmd | rg 'moe_dispatch_combine_a8w8|mpirun|run_a3|timeout'
   ```

2. Search for true failure signatures:

   ```bash
   rg -n 'runtime_error|FAILED|\bFAIL\b|mismatch_count=[1-9]|mismatches=[1-9]|final_output\.err_count=[1-9]|pass=false|Traceback|Segmentation|Killed|\[ERROR\]' out/m3_suite_clean_after_fix.log
   ```

3. For suspected sync deadlock, inspect:

   - `scoreboard_async_enabled`
   - `overlap_timeout_count`
   - `m3n9_timeout_dump_present`
   - `timeout_dump_stage_name`
   - `producer_status_checksum`
   - `scoreboard_min_status_checksum`
   - `combine_done_signal_checksum`
   - `return_segment_counters_checksum`

4. Do not diagnose with stale binaries. Re-run with `--clean-build 1` when validating a fix.
