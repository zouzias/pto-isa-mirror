# topk — auto-mode A3 prototype (v1, full TopK with values + indices)

Full top-K (values + matching original indices) for a single 1-D float32 array,
single AICORE, single row, no buffering. Project layout mirrors
[kernels/automode/a2a3/add_tile_array/](../add_tile_array/), so the same
one-liner works:

```bash
bash run.sh -r npu -v Ascend910B1
```

**Note:** the previous values-only shortcut (which assumed pre-sorted
64-element blocks and skipped TSORT32 / index tracking) has been
**removed**. The kernel now does the full TopK pipeline — TSORT32 → merge →
TGATHER values + TGATHER indices — closer to the manual TopK in
[kernels/manual/a2a3/topk/](../../../manual/a2a3/topk/) but stripped of
double-buffering / multi-core / TPipe / manual sync.

## Supported AI Processors

- A3 only.

## Directory Layout

```
kernels/automode/a2a3/topk/
├── scripts/
│   └── gen_data.py             # Generates inputs + golden values + golden indices
├── CMakeLists.txt              # Build configuration (mirrors add_tile_array)
├── topk_kernel.cpp             # Kernel implementation (auto mode, full TopK)
├── main.cpp                    # Host-side entry point (4 GM tensors)
└── run.sh                      # Convenience script
```

After running `bash run.sh`, the project also contains:

```
├── input/                      # input_src.bin, input_idx.bin           (gen_data.py)
├── output/                     # golden_val.bin, golden_idx.bin         (gen_data.py)
│                               # output_val.bin, output_idx.bin         (main.cpp)
└── build/                      # CMake / make build tree
```

## Operator Description

### Function

Returns the top-K values and their **original (unsorted)** indices from a
random 1-D float32 row. Output is descending by value.

| Item              | Value                                      |
|-------------------|--------------------------------------------|
| OpType            | `topk`                                     |
| Input `src`       | float32, shape `(1, 1280)` — random unsorted |
| Input `idx`       | uint32, shape `(1, 1280)` — identity `[0..1279]` |
| Output `out_val`  | float32, shape `(1, 512)` — top-K values descending |
| Output `out_idx`  | uint32, shape `(1, 512)` — matching original indices |
| Kernel name       | `topk_kernel`                              |

### Algorithm

The kernel mirrors the manual `runTOPK` pipeline structurally, simplified
for correctness-first auto-mode v1:

```
TLOAD srcTile              ← input_src.bin
TLOAD idxTile              ← input_idx.bin (identity [0..kCols-1])

TSORT32(packed, src, idx, scratch)          # per-32-block sort, emits (val, idx) packed

main TMRGSORT loop:                         # 4-way self-merge, blockLen *= 4
    blockLen = 64 * TYPE_COEF
    for each iter:
        srcSortedView = TRESHAPE(sort32DstTile)         # source prefix view
        tmpSortedView = TRESHAPE(mrgScratchTile)        # destination prefix view (independent buffer)
        TMRGSORT(tmpSortedView, srcSortedView, blockLen)
        TMOV(srcSortedView, tmpSortedView)              # promote merged result back to source

if (blockLen < kCols):                      # tail block — non-power-of-4 residual
    FillMrgArray(...) plans the 2-list merges
    for each planned merge:
        src0View   = TRESHAPE(sort32DstTile)            # already-merged prefix
        src1View   = TSUBVIEW(sort32DstTile, mrgSortedLen)  # next tail run (offset view)
        curDstView = TRESHAPE(mrgScratchTile)           # INDEPENDENT destination
        TMRGSORT(curDstView, executedNumList, tmp, src0View, src1View)
        TMOV(TRESHAPE(sort32DstTile), curDstView)       # copy back so next iter sees it

TGATHER<P0101>(outValTile, sortedTopKView)              # extract values
TGATHER<P1010>(outIdxTile, sortedTopKIdxView)           # extract indices via TRESHAPE float→uint32 type-pun

TSTORE outValTile, outIdxTile               → output_val.bin, output_idx.bin
```

Key auto-mode-safety choices:
- TASSIGN (numeric-address placement) is **never** used; the auto allocator
  owns tile placement.
- `TRESHAPE` / `TSUBVIEW` are used **only** for semantic aliasing
  (`(prefix view) | (offset view) | (different-layout view)`); each call
  carries an inline comment per the project rule
  ([tile_type_reference.md §11 item 14](../../../../docs_for_ai/tile_type_reference.md)).
- `SortTailBlock` writes to an **independent** destination tile
  (`mrgScratchTile`) and `TMOV`s the result back to `sort32DstTile` prefix.
  The manual's in-place merge (`curDstTile` aliased onto `srcTile.data()`)
  is **not** used; that pattern produced interleaved output in the
  previous values-only port.

## Auto-mode constraints honored

- User-facing PTO instructions only: `TLOAD`, `TSORT32`, `TMRGSORT`, `TMOV`,
  `TRESHAPE`, `TSUBVIEW`, `TGATHER`, `TSTORE`. No raw CCE intrinsics.
- No `TASSIGN(tile, numeric_addr)` (auto allocator owns placement).
- No `set_flag` / `wait_flag` / `pipe_barrier` at kernel scope (auto-sync
  handles ordering).
- No double / multi-buffering, no `TPipe` / `TPUSH` / `TPOP`.
- No `Tile::data()` from kernel code; no `*_IMPL` calls.
- Single AICORE; single row.

## I/O shapes and formats

All `.bin` files are raw little-endian, contiguous, no header.

| File                        | Shape         | Bytes  | Source                | Consumer                  |
|-----------------------------|---------------|--------|-----------------------|---------------------------|
| `input/input_src.bin`       | `(1, 1280)` f32 | 5 120 | `scripts/gen_data.py` | `main.cpp`                |
| `input/input_idx.bin`       | `(1, 1280)` u32 | 5 120 | `scripts/gen_data.py` | `main.cpp`                |
| `output/golden_val.bin`     | `(1, 512)`  f32 | 2 048 | `scripts/gen_data.py` | `main.cpp` (`ResultCmp`)  |
| `output/golden_idx.bin`     | `(1, 512)`  u32 | 2 048 | `scripts/gen_data.py` | `main.cpp` (`ResultCmp`)  |
| `output/output_val.bin`     | `(1, 512)`  f32 | 2 048 | `main.cpp`            | `main.cpp` (`ResultCmp`)  |
| `output/output_idx.bin`     | `(1, 512)`  u32 | 2 048 | `main.cpp`            | `main.cpp` (`ResultCmp`)  |

## Numerical tolerance

- **Values**: tolerance `0.001f` (`ResultCmp`). With unique random inputs the
  expected element-wise diff is `0.0` — values are produced by gathering
  the same float bytes from the same source elements.
- **Indices**: exact uint32 match expected (no tolerance applies meaningfully
  to integers; a 1-bit difference is a real bug).

If duplicate values appear by chance and the kernel's tie-breaking differs
from NumPy's stable argsort, the indices may swap among tied positions —
this is a real possibility worth knowing about, though with continuous
random inputs the probability of ties is effectively 0.

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

If the run succeeds, the output prints:

```text
test value success
test index success
test success
```

## v1 limitations and v2 roadmap

| Feature | v1 (this) | Manual mode | v2 / future |
|---|---|---|---|
| Random-input handling | yes (TSORT32 in-kernel) | yes | — |
| Index tracking | yes (TGATHER P1010 + type-pun) | yes | — |
| dtype | float32 only | float32 + half | add half (P0001 mask + half index extraction) |
| Multi-core | single AICORE | 48-core via `block_idx` | add `block_idx` work split |
| Multi-row | single row | `SINGLE_LOOP_ROW = 2` | add row loop |
| Pipelining | none | double-buffered manual sync | wait for sanctioned auto-mode pipeline abstraction |
| Tail-block merge | independent dst + TMOV-back | in-place via TASSIGN aliasing | revisit if in-place is auto-safe with proper hints |

## Likely first failure points (in order)

1. **`TGATHER<DstTile, SrcTile, MaskPattern>(dst, src)` template form** —
   if the bisheng-CCE wrapper expects a different signature, capture the
   first error and we'll match against `include/pto/npu/a2a3/TGather.hpp`.
2. **`TRESHAPE` between `Tile<Vec, float, ...>` and `Tile<Vec, uint32, ...>`**
   at kernel level — if the type-pun is rejected, the index extraction has
   to be reworked (e.g., TSTORE the packed buffer to a scratch GM tensor
   and have main.cpp do the host-side reinterpret + index extraction).
3. **TSORT32 wrong output** if `tmp` content matters — currently we declare
   `sort32TmpTile` as independent uninitialized UB, on the (Inferred)
   assumption that it's pure scratch; can be debugged by adding a
   zero-fill via a host-prepared GM tensor + `TLOAD`.
4. **Mask-pattern direction** (`P0101` vs `P1010` flipped) — values would
   land in `output_idx.bin` and indices in `output_val.bin`; trivial to
   detect.
5. **`SortTailBlock` `TMOV`-back invisible to next iteration's `src0View`**
   — for `kCols=1280, kTopK=512` the tail loop runs only once, so this
   wouldn't manifest at the current shape. If we change dims and it shows
   up, the auto-sync between `TMOV` and the next `TRESHAPE`-based read
   needs investigation.

## References

- Pattern source (full TopK): [kernels/manual/a2a3/topk/topk_kernel.cpp](../../../manual/a2a3/topk/topk_kernel.cpp).
- Pattern source (auto-clean TMRGSORT/TSORT32): [tests/npu/a2a3/src/st/testcase/tmrgsort/](../../../../tests/npu/a2a3/src/st/testcase/tmrgsort/), [tsort32/](../../../../tests/npu/a2a3/src/st/testcase/tsort32/).
- Project layout source: [kernels/automode/a2a3/add_tile_array/](../add_tile_array/) — confirmed-built baseline.
- Cross-element-type TRESHAPE pattern: [include/pto/npu/a2a3/TQuant.hpp:108-114](../../../../include/pto/npu/a2a3/TQuant.hpp#L108-L114) (auto branch).
- `docs_for_ai/known_good_kernel_examples.md §A4, §A11` — TRESHAPE-as-aliasing-hint, baseline confirmed-built reference.
- `docs_for_ai/auto_mode_bad_patterns.md` — what to avoid in auto-mode kernel code.
- `docs_for_ai/tile_type_reference.md §11 item 14` — TRESHAPE intent rule.
- `docs_for_ai/compile_error_logbook.md §E5, §E7, §E8, §E9, §E11, §E12` — gotchas observed during this project's evolution.
