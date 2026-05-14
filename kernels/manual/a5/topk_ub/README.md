# A5 TopK — local UB radix variant (`topk_ub`)

This directory is **not** the upstream `kernels/manual/a5/topk` scaffold from `cann/pto-isa`. It holds a **local implementation** (single GM load, full-width `THISTOGRAM` / `TGATHER`, `TCONCAT`, etc.) so you can `git pull` upstream without overwriting your experiment.

- Upstream reference layout: [`../topk/README.md`](../topk/README.md)
- Build/run: same as `topk`, but the executable is **`topk_ub`**:

```bash
cd kernels/manual/a5/topk_ub
bash run.sh -r sim -v Ascend910_9599
```

- Data: `scripts/gen_data.py` writes `input/` and `output/` under **this** directory.

## Low-range ST (keys in 0~10)

To reproduce tie-heavy TopK behavior quickly:

```bash
cd kernels/manual/a5/topk_ub
bash scripts/st_range_0_10.sh            # default seed: 20260511
bash scripts/st_range_0_10.sh 123456789  # custom seed
```

This ST does:

- `python3 scripts/gen_data.py --min-key 0 --max-key 10 --seed <seed>`
- `python3 scripts/radix_topk_golden_stats.py`

`scripts/gen_data.py` now supports:

- `--min-key` (inclusive, uint16 range)
- `--max-key` (inclusive, uint16 range)

## RemainK corner case (`winner == 0`)

For MSB winner selection, `remain_k` must match radix golden:

- `remain_k = (N - TopK) - C[winner-1]`
- define `C[-1] = 0`

In kernel code, `winner-1` is clamped before gather for memory safety, so when `winner == 0` we must explicitly force `cw = 0` before `TSUB`; otherwise it incorrectly uses `C[0]` and may underflow in low-range/tie-heavy inputs.

## `include/` A5 header changes

Local edits under `include/pto/npu/a5/*.hpp` are exported to **`include_hpp_changes.patch`** (from repo root: `git diff include/`). To re-apply after pulling upstream:

```bash
cd /path/to/pto-isa
git apply kernels/manual/a5/topk_ub/include_hpp_changes.patch
```

Regenerate the patch after you change headers: `git diff --no-color include/ > kernels/manual/a5/topk_ub/include_hpp_changes.patch`.

## Performance Analysis (A5 sim, 2026-05-14)

Dataset: `build/OPPROF_20260514154240_NYBGOWGNYQBFCTNM` on `core0.veccore0`.

Notes:

- IPC for `VF01~VF17` is cycle-accurate (from retire/issue queue logs).
- IPC for `VF18~VF26` is trace-window estimated (retire records are truncated in `ccu.vec_issque` tail).

### Phase-level summary

| Phase | VF range | Duration (us) | IPC | Main PTO ISA | Comment |
|---|---:|---:|---:|---|---|
| Phase1 | VF01-VF02 | 1.358 | 1.503 | `TASSIGN/TLOAD/THISTOGRAM(BYTE_1)/TMOV` | MSB histogram main compute |
| Phase2 | VF03-VF13 | 0.838 | 0.447 | `TCMPS/TCI/TSELS/TROWMIN/TGATHER/TSUB` | Winner MSB + remainK (control-heavy) |
| Phase3 | VF14 | 1.464 | 1.481 | `TCVT/THISTOGRAM(BYTE_0)/TMOV` | LSB histogram main compute |
| Phase4 | VF15-VF17 | 0.359 | 0.148 | `TCMPS/TSELS/TROWMIN/TGATHER/TSHLS/TOR` | Winner LSB + packed threshold |
| Phase5 | VF18-VF26 | 6.803 | 0.864* | `TGATHER<GT>/TGATHER<EQ>/TCONCAT_IMPL/TSTORE` | Full-width gather + concat + store |

\* Phase5 IPC includes estimated segments (VF18~VF26).

### Key VF hotspots (for regression tracking)

| VF | PC | Duration (us) | IPC | Dominant RV ISA cluster | Code intent |
|---|---|---:|---:|---|---|
| VF02 | `0x10d0d16c` | 1.327 | 1.527 | `RV_VCVT_I2I/RV_VADD/RV_VMOV` | MSB histogram body in `Phase1` |
| VF14 | `0x10d0d764` | 1.464 | 1.481 | `RV_VCVT_I2I/RV_VADD/RV_VMOV` | LSB histogram body in `Phase3` |
| VF24 | `0x10d0dcc8` | 2.891 | 0.976* | `RV_VLD/RV_VADD/RV_VCMP_GT` | `TGATHER<GT>` heavy segment in `Phase5` |
| VF25 | `0x10d0dd14` | 2.959 | 0.966* | `RV_VSTUR/RV_VSQZ/RV_VLD` | `TGATHER<EQ>` heavy segment in `Phase5` |
| VF16 | `0x10d0d85c` | 0.247 | 0.080 | `RV_VLD/RV_PLT/RV_VLOOPv2` | Small control/predicate-heavy segment in `Phase4` |

\* VF24/VF25 IPC is trace-window estimated.
