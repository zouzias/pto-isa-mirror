# A5 TopK — local UB radix variant (`topk_ub`)

This directory is **not** the upstream `kernels/manual/a5/topk` scaffold from `cann/pto-isa`. It holds a **local implementation** (single GM load, full-width `THISTOGRAM` / `TGATHER`, `TCONCAT`, etc.) so you can `git pull` upstream without overwriting your experiment.

- Upstream reference layout: [`../topk/README.md`](../topk/README.md)
- Build/run: same as `topk`, but the executable is **`topk_ub`**:

```bash
cd kernels/manual/a5/topk_ub
bash run.sh -r sim -v Ascend950PR_9599
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

## TGATHER EQ mode and buffer sizing

- `PTO_A5_TGATHER_B16_EQ_USE_VSCATTER` defaults to `1` in `TGather.hpp`.
- Without vscatter, reserve **`N × 4` bytes** for the EQ list (worst case: all keys equal threshold).
- With vscatter, the EQ tile can be **TopK**-sized; still cover tie-heavy cases in golden data.

## Performance analysis (A5 sim, 2026-05-19)

Dataset: `build/OPPROF_20260519153609_JHDTPDFLGZWDKNLC` (`core0.veccore0`).

- **Duration**: VF `dur` (µs) from `simulator/trace.json`.
- **Cycles**: `vf_real_execute_time` from `PUSHQ … VF` in `instr_log.dump` (cycle-accurate).
- **IPC**: `RV_*` events in the VF trace window ÷ `vf_real_execute_time`.
- **PTO-ISA**: aligned with the five `Phase*` functions in `draft.cpp`.

Kernel trace wall span ≈ **26.2 µs**. Dominant cost: **Phase5 `TGATHER<EQ>` (VF24)** plus histogram **VF02 / VF14**.

### Phase summary (VF ↔ Phase ↔ PTO-ISA)

| Phase | Function | VF range | trace (µs) | vf_real | IPC | Main PTO-ISA |
|---|---|---|---:|---:|---:|---|
| 1 | `Phase1_LoadAndHistogramMsb` | VF01–VF02 | 1.358 | 2417 | 1.50 | `TASSIGN` / `TLOAD` / `THISTOGRAM<BYTE_1>` / `TMOV` |
| 2 | `Phase2_WinnerMsbAndRemainK` | VF03–VF13 | 0.838 | 851 | 0.44 | `TCMPS` / `TCI` / `TSELS` / `TROWMIN` / `TGATHER` / `TSUB` |
| 3 | `Phase3_HistogramLsb` | VF14 | 1.464 | 2609 | 1.48 | `TCVT` / `THISTOGRAM<BYTE_0>` / `TMOV` |
| 4 | `Phase4_WinnerLsbRemainKAndPackedThresholdTor` | VF15–VF22 | 0.485 | 447 | 0.50 | `TCMPS` / `TSELS` / `TROWMIN` / `TGATHER` / `TCVT` / `TSHLS` / `TOR` |
| 5 | `Phase5_TgatherGtEqTconcatAndStore` | VF23–VF25 | 20.088 | 35680 | 0.52 | `TGATHER<GT>` / `TGATHER<EQ>` / `TCONCAT_IMPL` / `TSTORE` |

Per-VF IPC and PTO-ISA: [below](#per-vf-ipc-and-pto-isa-reference). Methodology: [`draft_topk_radix_ub.md`](draft_topk_radix_ub.md) §4.

## Per-VF IPC and PTO-ISA reference

Same dataset as above (`OPPROF_20260519153609`). **IPC** = `RV_*` events in the VF trace window ÷ `vf_real_execute_time` from `instr_log`.

| VF | Ph | PC | trace (µs) | vf_real | **IPC** | **PTO-ISA** |
|:---:|---:|---|---:|---:|---:|---|
| VF01 | 1 | `0x10d0d0dc` | 0.031 | 56 | **0.38** | `TASSIGN`, MTE2↔V sync |
| VF02 | 1 | `0x10d0d158` | 1.327 | 2361 | **1.53** | `TLOAD`, `TEXPANDS`, `THISTOGRAM<BYTE_1>`, `TMOV` |
| VF03 | 2 | `0x10d0d200` | 0.074 | 68 | **0.60** | `TCMPS`(GE), `TCI`, `TSELS` |
| VF04 | 2 | `0x10d0d24c` | 0.062 | 63 | **0.68** | `TCI`, `TSELS` |
| VF05 | 2 | `0x10d0d328` | 0.080 | 119 | **0.64** | `TROWMIN`, `TGATHER` → `msbWinnerSaved` |
| VF06 | 2 | `0x10d0d3c4` | 0.068 | 55 | **0.36** | `TROWMIN`, `TSUB`, `TCMPS`, `TSEL` |
| VF07 | 2 | `0x10d0d454` | 0.142 | 214 | **0.36** | `TCMPS`, `TSEL`, `TGATHER` → `msbWinnerBin` |
| VF08 | 2 | `0x10d0d494` | 0.117 | 55 | **0.29** | `TGATHER` → `msbWinnerBin` |
| VF09 | 2 | `0x10d0d4d8` | 0.042 | 33 | **0.36** | `TEXPANDS`, `TGATHER`(C[w]), `TCMPS`, `TSEL` |
| VF10 | 2 | `0x10d0d528` | 0.059 | 55 | **0.29** | `TSEL`, `TSUB` → `remainKTile` |
| VF11 | 2 | `0x10d0d5a4` | 0.079 | 101 | **0.37** | `TSUB`, sync |
| VF12 | 2 | `0x10d0d628` | 0.070 | 52 | **0.31** | `TEXPANDS`, `TCMPS`(EQ), `TSEL` (`cwFix`) |
| VF13 | 2 | `0x10d0d6d4` | 0.045 | 36 | **0.53** | `TSUB` (`remainK`), epilogue |
| VF14 | 3 | `0x10d0d754` | 1.464 | 2609 | **1.48** | `TCVT`, `THISTOGRAM<BYTE_0>`, `TMOV` |
| VF15 | 4 | `0x10d0d800` | 0.073 | 68 | **0.62** | `TCMPS`(GT), `TCI`, `TSELS` |
| VF16 | 4 | `0x10d0d84c` | 0.057 | 63 | **0.70** | `TCI`, `TSELS` |
| VF17 | 4 | `0x10d0d91c` | 0.080 | 119 | **0.65** | `TROWMIN`, `TGATHER` → `lsbWinnerBin` |
| VF18 | 4 | `0x10d0d98c` | 0.068 | 55 | **0.31** | `TROWMIN`, `TGATHER` |
| VF19 | 4 | `0x10d0da0c` | 0.042 | 34 | **0.29** | `TCVT`, `TSHLS`, `TCVT` |
| VF20 | 4 | `0x10d0dab8` | 0.052 | 36 | **0.36** | `TOR`, V↔S sync |
| VF21 | 4 | `0x10d0db38` | 0.056 | 34 | **0.29** | `TASSIGN` (`packedThrU`) |
| VF22 | 4 | `0x10d0dbe8` | 0.057 | 38 | **0.29** | `TASSIGN`, sync |
| VF23 | 5 | `0x10d0dc70` | 2.910 | 5174 | **0.99** | **`TGATHER<GT>`** (full N) |
| VF24 | 5 | `0x10d0dd04` | 16.882 | 30236 | **0.44** | **`TGATHER<EQ>`** (vscatter) |
| VF25 | 5 | `0x10d0dd4c` | 0.296 | 270 | **0.58** | **`TCONCAT_IMPL`**, V↔MTE3 sync, **`TSTORE`** |
