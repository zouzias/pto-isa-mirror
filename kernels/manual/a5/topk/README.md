# A5 TopK operator (scaffold)

Device-side code lives in **`draft.cpp`**: a **radix-select TopK** for **16-bit sortable keys**, using **`THISTOGRAM`** (MSB then filtered LSB) and **`TGATHER`** compare (`GT` / `EQ`) to collect indices. This differs from the sort/merge style in `kernels/manual/a2a3/topk`.

## Current case

- Input: **`[1, 2048]`** `uint16` keys (`input/keys.bin` from `scripts/gen_data.py`)
- Output: **512** indices for the **512 largest** keys in `uint16` order — **no ordering requirement** on the output
- Host check: multiset of `keys[out[i]]` must match the reference multiset (`output/golden_topk_multiset.bin` from `gen_data.py`)

## Layout

- `CMakeLists.txt` — builds `draft.cpp` into `libtopk_kernel.so` (`dav-c310-vec`)
- `draft.cpp` — `RunRadixTopKDraft` / `LaunchRadixTopKDraft` (see file header for pipeline)
- `main.cpp` — ACL host: read inputs, `LaunchRadixTopKDraft<512>`, multiset validation
- `scripts/gen_data.py` — random keys + golden top-512 value multiset
- `run.sh` — `gen_data`, then configure, build, run `topk`

## References inside this repo

- A5 `TMRGSORT` / value-only top-k style tests: `tests/npu/a5/src/st/testcase/tmrgsort/`
- A5 `TSORT32` / `TGATHER`: `tests/npu/a5/src/st/testcase/tsort32/`, `tgather/`
- A2/A3 full TopK example (algorithm reference only): `kernels/manual/a2a3/topk/`

## Build

Source your CANN environment (`set_env.sh`), ensure **`bisheng`** is on `PATH`, then:

**Simulator** (same SOC string as other `kernels/manual/a5/*` examples, e.g. `flash_atten` / `engram_simt` — **not** `Ascend310P*`):

```bash
cd kernels/manual/a5/topk
bash run.sh -r sim -v Ascend910_9599
```

**On-device**:

```bash
bash run.sh -r npu -v <SOC string for your A5 board>
```

`run.sh` prepends `$ASCEND_HOME_PATH/tools/simulator/$SOC_VERSION/lib` to `LD_LIBRARY_PATH` for sim; if the host executable fails to load Ascend libraries, source `set_env.sh` first so `lib64` paths are set.
