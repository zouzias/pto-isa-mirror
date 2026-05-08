# add_tile_array — auto-mode A3 prototype

Element-wise `C = A + B` on a 2-D float32 array, processed as a serial
loop of 64×64 tiles by a single AICORE. Correctness-first; no performance
optimization. Project layout and build/run flow mirror
[kernels/manual/a2a3/topk/](../../../manual/a2a3/topk/) so the same
one-liner works:

```bash
bash run.sh -r npu -v Ascend910B1
```

## Supported AI Processors

- A3 only.

## Directory Layout

```
kernels/automode/a2a3/add_tile_array/
├── scripts/
│   └── gen_data.py             # Generates input and golden output
├── CMakeLists.txt              # Build configuration (mirrors topk)
├── add_tile_array_kernel.cpp   # Kernel implementation (auto mode)
├── main.cpp                    # Host-side entry point
└── run.sh                      # Convenience script
```

After running `bash run.sh`, the project also contains:

```
├── input/                      # input_a.bin, input_b.bin       (gen_data.py)
├── output/                     # golden_c.bin, output_c.bin     (gen_data.py + main.cpp)
└── build/                      # CMake / make build tree
```

## Operator Description

### Function

Element-wise sum on a 2-D float32 array of shape
`(NUM_TILES * TILE_ROWS, TILE_COLS) = (4 * 64, 64) = (256, 64)`. The
kernel iterates `NUM_TILES = 4` times; each iteration loads a
`(TILE_ROWS, TILE_COLS) = (64, 64)` tile from A and B, adds them with
`TADD`, and stores the result to C. Consecutive tiles cover the array
contiguously along the row axis.

### Specification

| Item        | Value |
| ----------- | ----- |
| OpType      | `add_tile_array` |
| Inputs      | `A`, `B`: float32 tensors of shape `(256, 64)` (16 384 elements each) |
| Output      | `C`: float32 tensor, same shape |
| Kernel name | `add_tile_array_kernel` |

### Tiling

Single AICORE for v1; the in-kernel `for` loop iterates `NUM_TILES = 4`
times. Per-iteration shape is `(64, 64)` float32. Multi-core
`block_idx`-based partitioning is a follow-up.

## Auto-mode constraints honored

- User-facing PTO instructions only: `TLOAD`, `TADD`, `TSTORE`. No raw
  CCE intrinsics in the kernel body
  (see `docs_for_ai/auto_mode_bad_patterns.md §3`).
- No `set_flag` / `wait_flag` / `pipe_barrier`; no `Event<>`
  (see `§2`, `compile_error_logbook.md §E7`).
- No `TASSIGN` aliasing tricks; tiles are placed by the auto allocator
  (see `§1.1, §1.2`).
- No double / multi-buffering, no `TPipe` / `TPUSH` / `TPOP`
  (see `§2.4, §2.5`).
- No `Tile::data()` in kernel code; no `*_IMPL` calls
  (see `§3.3, §3.4`, `compile_error_logbook.md §E5`).
- Auto mode is enabled by adding `--cce-enable-pto-passes` to the kernel
  target's compile options (see `CMakeLists.txt`,
  function `pto_example_vec_auto`).

## I/O shapes and formats

All `.bin` files are raw little-endian float32, contiguous, no header.
Length per file: `256 × 64 = 16 384` elements = **65 536 bytes**.

| File                  | Source              | Consumer            |
|-----------------------|---------------------|---------------------|
| `input/input_a.bin`   | `scripts/gen_data.py` | `main.cpp` (host) |
| `input/input_b.bin`   | `scripts/gen_data.py` | `main.cpp` (host) |
| `output/golden_c.bin` | `scripts/gen_data.py` | `main.cpp` (`ResultCmp`) |
| `output/output_c.bin` | `main.cpp`          | `main.cpp` (`ResultCmp`) |

## Numerical tolerance

`scripts/gen_data.py` draws inputs as random integers in `[1, 10]` cast
to float32. Element-wise addition of small integers is exact in
IEEE-754 float32. `main.cpp` calls `ResultCmp(golden, devFinal, 0.001f)`
which is comfortably wider than the expected `0.0` error.

If you change the input range or move to `half` (FP16) /
`bfloat16_t`, raise the tolerance and update both `gen_data.py` and
`main.cpp`.

## Build and Run

1. Configure your Ascend CANN environment:

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
# or:  source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. Run the example:

```bash
cd ${git_clone_path}/kernels/automode/a2a3/add_tile_array
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

The build is **not** verified in this environment (no compiler access).
First-build failures, in order of likelihood:

1. `ASCEND_HOME_PATH` unset → CMake stops at the env-var check.
2. SoC mismatch — pick the value that matches the compiler-server hardware.
3. The auto-mode flag `--cce-enable-pto-passes` may produce errors on the
   tile declarations or `TLOAD`/`TADD`/`TSTORE` calls; if so, capture the
   first compiler error and consult
   `docs_for_ai/compile_error_logbook.md` (E5/E6/E7 are likely starting
   points).
4. Static valid-region `Tile<…, TILE_ROWS, TILE_COLS>` may need to be
   swapped to the demo-style dynamic form (`-1, -1` plus constructor
   args) if the build harness expects that.

## Known limitations (v1)

- **Single AICORE.** No `block_idx` work distribution. v2 will partition
  the loop across `BLOCK_DIM` cores (mirroring
  [demos/auto_mode/baseline/add/csrc/kernel/add_custom.cpp](../../../../demos/auto_mode/baseline/add/csrc/kernel/add_custom.cpp)).
- **No tail handling.** Total length must be a multiple of
  `TILE_ROWS * TILE_COLS`. The current shape `(256, 64)` is exactly
  `4 × (64, 64)`. Non-aligned shapes need a partial-tile epilogue —
  follow-up.
- **No double / multi-buffering.** Auto mode does not support
  kernel-managed ping-pong today
  (per `docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md §1.4`).
  Revisit when a sanctioned auto-mode pipeline abstraction lands.
- **A3 only.** A5 differs in the `TileLeft` `BLayout` split (irrelevant
  for this Vec-only kernel) and in the SoC compile flag (`dav-c310-vec`
  vs `dav-c220-vec`). An A5 sibling project would change just the
  CMakeLists arch flag.
- **`float` only.** Other dtypes (`half`, `int16`, `int32`,
  `bfloat16_t`) require additional `launchAddTileArray<T>` instantiations
  in `add_tile_array_kernel.cpp` and matching dtype changes in
  `gen_data.py` / `main.cpp`.

## References

- Pattern source: [kernels/manual/a2a3/topk/](../../../manual/a2a3/topk/) — project layout and `run.sh` / CMakeLists shape.
- Auto-mode kernel pattern: [demos/auto_mode/baseline/add/](../../../../demos/auto_mode/baseline/add/).
- Closest dual-mode test: [tests/npu/a2a3/src/st/testcase/tadd/](../../../../tests/npu/a2a3/src/st/testcase/tadd/).
- `docs_for_ai/known_good_kernel_examples.md §A1, §A2`.
- `docs_for_ai/auto_mode_bad_patterns.md` — what to avoid in auto-mode kernel code.
- `docs_for_ai/tile_type_reference.md §1, §6, §8.1` — `Tile<TileType::Vec, …>` shape.
- `docs_for_ai/compile_error_logbook.md` — file new compile errors here when you build.
