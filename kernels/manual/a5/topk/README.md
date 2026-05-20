# A5 TopK operator (scaffold)

Device-side code lives in **`draft.cpp`**: a **radix-select TopK** for **16-bit sortable keys**, using **`THISTOGRAM`** (`HistByte::BYTE_1` then filtered `BYTE_0`) and **compare `TGATHER`** (`CmpMode::GT` / `EQ`) to collect indices. This differs from the sort/merge style in `kernels/manual/a2a3/topk`.

The structure matches **`kernels/manual/a5/topk_ub`**: **five `Phase*`** functions; all PTO instructions (`TASSIGN`, `TLOAD`, histogram, `TGATHER`, etc.) live in those phases. This directory uses **tiled `TLOAD`** (N = 2048, 256 columns per tile); `topk_ub` keeps full keys in UB and does full-width gathers.

## Current case

- Input: **`[1, 2048]`** `uint16` keys (`input/keys.bin` from `scripts/gen_data.py`)
- Output: **512** indices for the **512 largest** keys in `uint16` order — **no ordering requirement** on the output
- Host check: multiset of `keys[out[i]]` must match the reference multiset (`output/golden_topk_multiset.bin` from `gen_data.py`)

## Pipeline (see `draft.cpp` header)

1. **Phase1** — `TASSIGN` UB, stream keys from GM in tiles, `THISTOGRAM<BYTE_1>`, cumulative `chistMSB` (`TEXPANDS` + per-tile `TADD`)
2. **Phase2** — `TCMPS` / `TCI` / `TSELS`, raw MSB bin + `WinnerBinU8` path; `winner==0` fixes `C[-1]=0` for `remain_k`; `TGATHER` + `TSUB` for `remainK`
3. **Phase3** — `TCVT` MSB into `idxFilter`, stream keys again, `THISTOGRAM<BYTE_0>`, cumulative `chistLSB`
4. **Phase4** — LSB winner (`TCMPS` `GT` vs `remainK` tile), `TROWMIN`, `TOR` packed **uint16** threshold at `kRemainUbOut`
5. **Phase5** — Per-tile `TLOAD` + compare `TGATHER` (GT/EQ); per-tile **six-arg `TCONCAT_IMPL`** (`NeetCntDstIdx`, byte counts from `TGATHER` concat tiles) accumulates into `gtSeg`/`eqSeg`; `TMOV` updates segment and `idx*Acc`; final five-arg merge + `TSTORE`

## Data generation

```bash
python3 scripts/gen_data.py --seed <int>      # random keys
python3 scripts/gen_data.py --const 0x1234  # all-equal keys (EQ-only stress)
```

## Layout

- `CMakeLists.txt` — builds `draft.cpp` into `libtopk_kernel.so` (`dav-c310-vec`); **`target_include_directories(... BEFORE ...)`** prefers this repo’s `include/pto` over `$ASCEND_HOME_PATH/include` (same idea as `topk_ub`)
- `draft.cpp` — `RunRadixTopKDraft` / `LaunchRadixTopKDraft`
- `main.cpp` — ACL host: read inputs, `LaunchRadixTopKDraft<512>`, multiset validation
- `scripts/gen_data.py` — random or constant keys + golden top-512 value multiset
- `scripts/radix_topk_golden_stats.py` — print theoretical MSB/LSB winner and GT|EQ counts
- `run.sh` — `gen_data`, then configure, build, run `topk`

## Related in this repo

- **UB variant (full-width keys, same five-phase idea):** `kernels/manual/a5/topk_ub/`
- A5 `TMRGSORT` / value-only top-k style tests: `tests/npu/a5/src/st/testcase/tmrgsort/`
- A5 `TSORT32` / `TGATHER`: `tests/npu/a5/src/st/testcase/tsort32/`, `tgather/`
- A2/A3 full TopK example (algorithm reference only): `kernels/manual/a2a3/topk/`

## Build

Source your CANN environment (`set_env.sh`), ensure **`bisheng`** is on `PATH`, then:

**Simulator** (same SOC string as other `kernels/manual/a5/*` examples, e.g. `topk_ub` / `flash_atten` — use **`Ascend950PR_9599`**, **not** `Ascend310P*`):

```bash
cd kernels/manual/a5/topk
bash run.sh -r sim -v Ascend950PR_9599
```

**On-device**:

```bash
bash run.sh -r npu -v <SOC string for your A5 board>
```

`run.sh` prepends `$ASCEND_HOME_PATH/tools/simulator/$SOC_VERSION/lib` to `LD_LIBRARY_PATH` for sim; if the host executable fails to load Ascend libraries, source `set_env.sh` first so `lib64` paths are set.

## Regression (2K)

After code changes, from `kernels/manual/a5/topk`:

```bash
source $ASCEND_HOME_PATH/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$ASCEND_HOME_PATH/aarch64-linux/lib64:$LD_LIBRARY_PATH
cd build && make -j16 && cd ..
for s in 1241200609 2203936584 191132090; do
  python3 scripts/gen_data.py --seed $s && cd build && ./topk | grep RESULT; cd ..
done
python3 scripts/gen_data.py --const 0x1234 && cd build && ./topk | grep RESULT
```

Expect **`RESULT: PASS`** for all runs (~90–120 s per sim run on 2K).
