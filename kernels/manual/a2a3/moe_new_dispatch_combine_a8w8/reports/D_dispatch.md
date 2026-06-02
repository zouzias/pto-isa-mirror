# Dispatch Stage Acceptance Report

Date: 2026-06-02

Scope: `moe_new_dispatch_combine_a8w8` dispatch stage, from D0 `PublishCounts` to D4 `DispatchGroupReady`.

## Summary

- Implemented FFN dispatch semantics in PTO: source rows use `preSumBeforeRank`, destination rows use
  `dispatchOffset + cumsumMM previous`.
- Remote payload and scale gather use PTO row-block `TGET`; local rows use PTO tile load/store.
- `dispatchGroupReady[localExpert]` is set after all owner rows for that expert are written.
- Dispatch e2e timing is recorded once per rank by the standalone M2 dispatch owner AIV using `SYS_CNT`.
- Host/device M2 peer-window layout now agrees on `debugCounters` size, so the peer timeline offset is consistent.
- No token/row-level debug dump was added.

## Verification Commands

Build:

```bash
bash scripts/run_a3.sh --backend int8 --case-name ffn-v3-small -pes 2 -M 16 -K 128 -N 128 -topK 2 -expertPerPe 2 --max-output-size 32 --dispatch-only 1 --dry-run 0 --skip-kernel-launch 0 --skip-run 1 --clean-build 1
```

Small dispatch-only:

```bash
bash scripts/run_a3.sh --backend int8 --case-name ffn-v3-small -pes 2 -M 16 -K 128 -N 128 -topK 2 -expertPerPe 2 --max-output-size 32 --dispatch-only 1 --dry-run 0 --skip-kernel-launch 0 --skip-build 1 --clean-build 0 --first-device 6
```

Large dispatch-only:

```bash
timeout 900 bash scripts/run_a3.sh --backend int8 --case-name ffn-v3-4097 -pes 2 -M 4097 -K 128 -N 128 -topK 2 -expertPerPe 2 --max-output-size 8194 --dispatch-only 1 --dry-run 0 --skip-kernel-launch 0 --skip-build 1 --clean-build 0 --first-device 6
```

Review checks:

```bash
rg -n "M2CopyRemoteDispatchRowsToGmm1Scalar|CopyRemoteDispatchRowsToGmm1Scalar|remote.*scalar|scalar remote" kernels/manual/a2a3/moe_new_dispatch_combine_a8w8/kernel kernels/manual/a2a3/moe_new_dispatch_combine_a8w8/host
git diff --check -- kernels/manual/a2a3/moe_new_dispatch_combine_a8w8
```

## Results

| Case | Devices | Result | e2e cycles/us |
| --- | --- | --- | --- |
| `ffn-v3-small` | 6, 7 | rank0/rank1 metadata, payload, scale, ready mismatch all 0 | rank0 1777 cycles / 1.0 us; rank1 1243 cycles / 0.7 us |
| `ffn-v3-4097` | 6, 7 | rank0/rank1 metadata, payload, scale, ready mismatch all 0 | rank0 205065 cycles / 110.8 us; rank1 207873 cycles / 112.4 us |

Device note: `npu-smi info` showed devices 4 and 5 in `Alarm`, so NPU correctness was run on healthy devices 6 and 7.

## Review

- `preSumBeforeRank[tokenOwner][localExpert]` is written from the raw global-expert source prefix.
- `cumsumMM[tokenOwner][localExpert]` is written from effective destination rows after capacity clipping.
- Dispatch gather computes `srcStart = preSumBeforeRank[...]` and `dstStart = dispatchOffset + previous`.
- Remote gather calls `M2CopyRemoteDispatchRowsToGmm1Pto`, which uses `M2TGetRowsInt8` and `M2TGetRowsFloat`.
- Old remote scalar dispatch copy is absent from source search.
- Ready publication remains after gather completion.
- `git diff --check` passed.

## Residual Work

- D10 hot expert rowBlock parallelism is still a performance follow-up and does not block dispatch correctness.
- Device e2e timing is currently for standalone M2 dispatch-only acceptance; fused overlap timing should be revisited when GMM1 overlap is enabled.
