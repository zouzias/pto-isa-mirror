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

   Fine-grained scoreboard async adds a producer/consumer dependency between dispatch and GMM1. If a producer status or domain min status is not advanced for an empty/skewed expert, GMM1 workers can poll forever. For the current runtime target this is not needed. Expert-level synchronization, matching the original FFN-style behavior, is sufficient. The current implementation has removed the scoreboard task map, producer status, min-status, worker wait counter, timeout counter layout, kernel paths, and host report fields.

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

- Delete M3N10 scoreboard async runtime code rather than keeping a disabled runtime path.
- Keep dispatch/GMM1 synchronization at expert granularity for now.
- Keep timeout dump focused on ready-signal diagnostics: dispatch ready, GMM1 ready, activation ready, GMM2 ready, and ready expert.
- Use scalar GM store/load/poll for cross-rank synchronization signals and counters.
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
- Source scan over `include/`, `kernel/`, `host/`, and active `scripts/` has no `scoreboard`/`M3N10` runtime references.
- Reports no longer emit `scoreboard_async_enabled` or producer/min-status checksum fields; dispatch-to-GMM1 reports `dispatch_gmm1_sync=expert_ready`.
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

   - `overlap_timeout_count`
   - `m3n9_timeout_dump_present`
   - `timeout_dump_stage_name`
   - `combine_done_signal_checksum`
   - `return_segment_counters_checksum`

4. Do not diagnose with stale binaries. Re-run with `--clean-build 1` when validating a fix.

## 2026-06-01 large case apparent hang after scoreboard deletion

### Symptom

- Command:

  ```bash
  bash scripts/run_a3.sh --backend int8 --skip-build 1 --clean-build 0 --dry-run 0 --skip-kernel-launch 0 --overlap-mode on --case-name ffn-v3-4097 -pes 2 -M 4097 -K 128 -N 128 -topK 2 -expertPerPe 2 --max-output-size 8194
  ```

- The process can look stuck for several minutes because `run_a3.sh` fixes `warmup_iters=3` and `measure_iters=5`, while each large fused iteration takes about 51 seconds on the current scalar/block0 GMM path.
- During the apparent hang, `ps` showed two rank processes at high CPU, not a sleeping `mpirun`/barrier wait.

### Evidence

- One-iteration direct host run passed:
  - `--warmup 0 --iters 1`
  - `ffn-v3-4097`, `maxOutputSize=8194`, overlap on
  - both ranks `pass=true`
  - `e2e_us.avg` about `51,459,882 us`
- Full `run_a3.sh` large on run passed:
  - `warmup_iters=3`, `measure_iters=5`
  - both ranks `pass=true`
  - rank0 `e2e_us.avg=50,875,830.3`
  - rank1 `e2e_us.avg=50,875,833.0`
  - rank1 reported `drop_triggered=true`, `dropped_rows=1`, which is the expected capacity-boundary behavior for `maxOutputSize=8194`.

### Conclusion

- This large case was not a scoreboard deadlock after the deletion. It was a long fixed 8-iteration run with little progress output.
- The main runtime cost remains the current correctness-first scalar/block0 GMM path. It is tracked in DESIGN M3O as the AIC GMM tile multi-core optimization gap.
