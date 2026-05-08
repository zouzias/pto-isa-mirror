# topk — auto-mode A3 prototype (v1)

Values-only top-K via merge sort, single AICORE, single row. Project layout
mirrors [kernels/automode/a2a3/add_tile_array/](../add_tile_array/), so the
same one-liner works:

```bash
bash run.sh -r npu -v Ascend910B1
```

## Supported AI Processors

- A3 only.

## Directory Layout

```
kernels/automode/a2a3/topk/
├── scripts/
│   └── gen_data.py             # Generates pre-sorted input + golden top-K
├── CMakeLists.txt              # Build configuration (mirrors add_tile_array)
├── topk_kernel.cpp             # Kernel implementation (auto mode)
├── main.cpp                    # Host-side entry point
└── run.sh                      # Convenience script
```

After running `bash run.sh`, the project also contains:

```
├── input/                      # input_src.bin                  (gen_data.py)
├── output/                     # golden_val.bin, output_val.bin (gen_data.py + main.cpp)
└── build/                      # CMake / make build tree
```

## Operator Description

### Function

Top-K of a single 1-D array of float32. Returns the top-512 values
(descending) from a 1×1280 input row.

### Specification

| Item              | Value |
|-------------------|-------|
| OpType            | `topk` |
| Input             | `src`: float32 of shape `(1, 1280)`, **pre-sorted in 64-element blocks descending** (see v1 limitation below) |
| Output            | `out`: float32 of shape `(1, 512)`, fully descending |
| Kernel name       | `topk_kernel` |

### Algorithm

The kernel adapts
[tests/npu/a2a3/src/st/testcase/tmrgsort/tmrgsort_kernel.cpp](../../../../tests/npu/a2a3/src/st/testcase/tmrgsort/tmrgsort_kernel.cpp)'s
`RunTMrgsortTopk` (specifically the `LanchTMrgsortTopK<float, 1, 1280, 1, 1280, 512>`
instantiation, which is in `ALL_TESTCASES` and known to build under auto
mode). The flow:

1. `TLOAD srcTile`
2. Loop: `TMRGSORT(blockLen)` followed by `TMOV` for `blockLen ∈ {64, 256}`
   (4-way merge each step, until `blockLen * 4 > kTCols_`).
3. Tail block: when the final `blockLen < kTCols_`, `SortTailBlock` merges
   the leftover prefix using `FillMrgArray` to plan the merge schedule.
4. `TMOV` final result into the top-K view, `TSTORE` to GM.

`TRESHAPE` and `TSUBVIEW` are used as auto-mode aliasing hints. `TASSIGN`
calls are no-ops in auto mode (preserved for cross-mode compilation).

## Auto-mode constraints honored

- User-facing PTO instructions only: `TLOAD`, `TMRGSORT`, `TMOV`,
  `TRESHAPE`, `TSUBVIEW`, `TSTORE`. No raw CCE intrinsics in the kernel
  body (see `docs_for_ai/auto_mode_bad_patterns.md §3`).
- `set_flag` / `wait_flag` / `pipe_barrier` are wrapped in
  `#ifndef __PTO_AUTO__` guards (preserved from the source for cross-mode
  builds; auto-sync handles ordering in auto mode). See
  `docs_for_ai/auto_mode_bad_patterns.md §2.1, §2.2` and
  `compile_error_logbook.md §E7`.
- No double / multi-buffering, no `TPipe` / `TPUSH` / `TPOP`
  (see `§2.4, §2.5`).
- No `Tile::data()` from kernel code; no `*_IMPL` calls
  (see `§3.3, §3.4`, `compile_error_logbook.md §E5`).
- No multi-core `block_idx` partitioning — single AICORE.
- Auto mode is enabled by adding `--cce-enable-pto-passes` to the kernel
  target's compile options (see `CMakeLists.txt` function
  `pto_example_vec_auto`).

## I/O shapes and formats

All `.bin` files are raw little-endian float32, contiguous, no header.

| File                    | Shape         | Bytes  | Source              | Consumer                |
|-------------------------|---------------|--------|---------------------|-------------------------|
| `input/input_src.bin`   | `(1, 1280)`   | 5 120  | `scripts/gen_data.py` (with 64-block pre-sort) | `main.cpp` (host)       |
| `output/golden_val.bin` | `(1, 512)`    | 2 048  | `scripts/gen_data.py`  | `main.cpp` (`ResultCmp`) |
| `output/output_val.bin` | `(1, 512)`    | 2 048  | `main.cpp`             | `main.cpp` (`ResultCmp`) |

## Numerical tolerance

`scripts/gen_data.py` draws inputs as random `float32` in `[-1000, 1000]`,
then pre-sorts each 64-element block in descending order. The golden output
is the descending sort of the original (random) row, prefixed to 512.

The kernel's merge-sort produces the same descending sort on the same data.
Where ties occur (duplicate values), order is implementation-defined; with
continuous-distribution inputs the probability of ties is effectively 0.

`main.cpp` calls `ResultCmp(golden, devFinal, 0.001f)`. Since both the
kernel and the Python golden operate on identical input bytes, the
expected element-wise difference is `0.0`.

If you change the dtype to `half` or change the tolerance, update both
`gen_data.py` and `main.cpp`.

## Build and Run

1. Configure your Ascend CANN environment:

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
# or:  source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. Run the example:

```bash
cd ${git_clone_path}/kernels/automode/a2a3/topk
bash run.sh -r npu -v Ascend910B1
```

For the simulator:

```bash
bash run.sh -r sim -v Ascend910B4
```

If the run succeeds, the output prints:

```text
test data success
test success
```

The build is **not** verified locally (this environment has no compiler).
Likely first-build failures, in order:

1. `ASCEND_HOME_PATH` unset → CMake stops at the env-var check.
2. SoC mismatch — pick the value matching your A3 hardware.
3. `MrgSortExecutedNumList` not found — fix: ensure
   `<pto/pto-inst.hpp>` and `<pto/common/pto_tile.hpp>` are included
   (already done in `topk_kernel.cpp`). If still missing, copy the
   include block from
   [tests/npu/a2a3/src/st/testcase/tmrgsort/tmrgsort_kernel.cpp:11-13](../../../../tests/npu/a2a3/src/st/testcase/tmrgsort/tmrgsort_kernel.cpp#L11-L13).
4. UB-budget overflow — the `(1, 1280)` shape is the same as the in-tree
   `tmrgsort` ST instantiation that is known to build, so this is unlikely
   to fire.

## v1 limitations and v2 roadmap

This is **the simplest correct top-K in auto mode**. Several pieces are
deliberately simplified vs. [kernels/manual/a2a3/topk/](../../../manual/a2a3/topk/):

| Feature | v1 (this) | Manual mode | v2 / future |
|---|---|---|---|
| Index tracking | none (values only) | `TSORT32` produces `(val, idx)` pairs; `TGATHER P0101 / P1010` extracts values / indices | add `TSORT32` + `TGATHER` |
| Random-input handling | requires Python pre-sort in 64-blocks | `TSORT32` provides 32-block pre-sort in-kernel | add `TSORT32` |
| Multi-core | single AICORE | 48-core via `block_idx * validRow * Cols` partitioning | add `block_idx` work split |
| Multi-row | single row | `SINGLE_LOOP_ROW = 2` rows per iter | add row loop (or row-tiling) |
| Pipelining | none | double-buffered with `set_flag`/`wait_flag` between MTE2/V/MTE3 | wait for sanctioned auto-mode pipeline abstraction |

Each follow-up iteration should:
- Capture the new pattern as an `A12 / A13 / …` entry in
  `docs_for_ai/known_good_kernel_examples.md` after a confirmed build.
- Log new compile errors as `E10 / E11 / …` in
  `docs_for_ai/compile_error_logbook.md`.
- Move resolved questions into `docs_for_ai/assumptions_to_verify.md §11`.

## References

- Pattern source: [tests/npu/a2a3/src/st/testcase/tmrgsort/tmrgsort_kernel.cpp](../../../../tests/npu/a2a3/src/st/testcase/tmrgsort/tmrgsort_kernel.cpp) — in `ALL_TESTCASES`, builds under auto mode.
- Project layout source: [kernels/automode/a2a3/add_tile_array/](../add_tile_array/) — confirmed-built baseline.
- Manual top-K reference (do **not** copy verbatim — uses double-buffer + manual sync): [kernels/manual/a2a3/topk/](../../../manual/a2a3/topk/).
- `docs_for_ai/known_good_kernel_examples.md §A11` — add_tile_array baseline reference.
- `docs_for_ai/auto_mode_bad_patterns.md` — what to avoid in auto-mode kernel code.
- `docs_for_ai/compile_error_logbook.md §E5, §E7, §E8, §E9` — gotchas observed during the add_tile_array build.
