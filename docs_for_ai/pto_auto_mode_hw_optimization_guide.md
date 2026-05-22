# Hardware-Aware Optimization Guide for PTO Auto-Mode Kernels

**Scope:** A3/A5-oriented PTO auto-mode kernels. The guide is motivated by a working optimized Flash Attention (FA) auto-mode prototype, a correct single-buffer A3 auto-mode TopK prototype, and the paper *Parallel Scan on Ascend AI Accelerators*. The FA source is used as a source of optimization ideas, not as a clean template to copy line-by-line.

**Main message:** High-performance auto mode is not just “make it compile without manual `TASSIGN`.” The goal is to expose the right hardware structure to the compiler: cube/vector separation, staged pipelines, multi-buffering, cross-stage FIFO handoff, careful tile geometry, and minimized synchronization.

---

## 1. Hardware mental model

### 1.1 AI core structure

A practical mental model for Ascend-style kernels:

```text
AI Core
├── Cube side
│   ├── matrix engine
│   ├── L1 / L0A / L0B / L0C / BT / FP style buffers
│   └── matmul / accumulation / format movement
├── Vector side
│   ├── vector engine(s), usually two vector subblocks on 910B-like parts
│   ├── UB
│   └── reductions, row-wise ops, gather/scatter, elementwise math
├── MTE / data-movement engines
│   └── move data between GM/L2 and local buffers
└── scalar/control side
    └── index arithmetic, branching, pipe and sync control
```

The paper reinforces three important architectural ideas:

1. Cube and vector units are separate computing resources.
2. MTE/data movement and compute engines can run through separate queues.
3. The programmer or compiler must expose synchronization and data dependencies clearly.

### 1.2 Consequence for auto-mode code

For optimized kernels, write code as if cube and vector are separate compiler/execution views. In practice, code often uses build macros like:

```cpp
#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif

#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif
```

and separates sections:

```cpp
if constexpr (DAV_CUBE) {
    // cube-only work
}

if constexpr (DAV_VEC) {
    // vector-only work
}
```

This separation matters because memory allocation, auto-sync, and available instructions may be seen differently by the cube and vector compilation paths.

**Rule:** Do not write one mixed blob and hope auto mode finds the hardware structure. Explicitly separate cube stages and vector stages.

### 1.3 A3 / Ascend 910B1 hardware facts and data flow

Source: user-provided architecture briefing (2026-05). A5 (`dav-c310`) capacities are **Unknown** — see [a3_a5_differences.md §10](a3_a5_differences.md). GM total size is **Unknown** — see [assumptions_to_verify.md §6.1](assumptions_to_verify.md).

**User-provided hardware briefing (A3 / Ascend 910B1):**

```text
AI cores: 25
Vector cores: 50 total, 2 vector subblocks per AI core
AICPUs: 4
Cube frequency: 1850 MHz

Per-AI-core local memory:
- L1:  512 KB
- L0A: 64 KB
- L0B: 64 KB
- L0C: 128 KB
- UB:  192 KB
```

These are physical capacities. Kernel generation should use custom usable
budgets below these physical limits.

Cube is matmul/GEMM-only; everything non-matmul (TMAXS / ReLU, casts, element-wise math, reductions, gather/scatter) runs on vector. (Known: user briefing.)

**Per-AI-core buffer capacities (A3):**

| Buffer | Size | Side | Role |
|---|---|---|---|
| L1  | 512 KB | cube  | staging from GM; feeds L0A / L0B; also receives L0C via FixPipe |
| L0A |  64 KB | cube  | left operand of `TMATMUL` (`TileLeft`, `__ca__`) |
| L0B |  64 KB | cube  | right operand of `TMATMUL` (`TileRight`, `__cb__`) |
| L0C | 128 KB | cube  | accumulator out of `TMATMUL` (`TileAcc`, `__cc__`); fp32 |
| UB  | 192 KB | vec   | vector staging from GM; feeds the vector unit |
| GM  | (Unknown) | shared | the only buffer cube and vector both see |

The element-count-vs-byte distinction for `tile_size(N)` is documented in [tile_type_reference.md §1](tile_type_reference.md) — these capacity numbers are **bytes**, so a fp16 (`half`) L0B tile holds at most 32K elements, a fp32 L0C tile holds at most 32K elements, etc. (Known.)

**Data-flow shape (A3):**

```text
                    ┌─────────────────── GM ───────────────────┐
                    │           (shared, both sides)           │
                    └────────┬────────────────────┬────────────┘
                             │                    │
                  ┌──────────┴─────────┐ ┌────────┴─────────┐
                  │      cube side      │ │    vector side   │
                  │                     │ │                  │
                  │   GM ↔ L1 (512 KB)  │ │   GM ↔ UB (192K) │
                  │           │         │ │           │      │
                  │   L1 → L0A (64 KB)  │ │   UB ↔ Vector    │
                  │   L1 → L0B (64 KB)  │ │   unit (TMAXS,   │
                  │           │         │ │   element ops,   │
                  │      [ CUBE ]       │ │   reductions,    │
                  │           │         │ │   gather/scatter)│
                  │   L0C (128 KB, fp32 │ │                  │
                  │   accumulator)      │ │   Scalar ↔ UB    │
                  │           │         │ │   Scalar ↔ GM    │
                  │   FixPipe → L1 or   │ │                  │
                  │             GM      │ │                  │
                  └─────────────────────┘ └──────────────────┘
```

**The cube/vec handoff round-trips through GM.** Cube cannot read UB; vector cannot read L1/L0A/L0B/L0C. Any kernel that mixes matmul with non-matmul work (e.g., `GEMM → ReLU → GEMM`) must `TSTORE` cube output to GM, then a vector TU `TLOAD`s from GM, does the op, `TSTORE`s back, then a second cube TU `TLOAD`s from GM and does the next matmul. (Known: user briefing; consistent with the cube/vec build-target split in [tests/npu/a2a3/src/st/testcase/CMakeLists.txt:14](../tests/npu/a2a3/src/st/testcase/CMakeLists.txt#L14) and [:43](../tests/npu/a2a3/src/st/testcase/CMakeLists.txt#L43).)

**Practical tile-budget implications (fp16 inputs, fp32 accumulator):**

- L0A holds at most `64 KB / 2 = 32K` fp16 elements → e.g., M·K ≤ 32768 for the left operand.
- L0B holds at most `32K` fp16 elements → K·N ≤ 32768 for the right operand. A full `256×256` fp16 weight tile (`131072` bytes) **does not fit**; must Split-K or Split-N.
- L0C holds at most `128 KB / 4 = 32K` fp32 elements → M·N ≤ 32768 for the accumulator.
- UB holds at most `192 KB` for vector-side staging — divide by element size and by the number of simultaneously-live tiles.

For Split-K/Split-N inside one cube TU, use the [§A6 Split-K pattern](known_good_kernel_examples.md#L81) (`if (i == 0) TMATMUL else TMATMUL_ACC`). When the accumulator is consumed by FixPipe (`TSTORE` / Acc→GM, or a UF-aware FIFO path), also reason about the unit-flag phase; see [§22.3](#223-unit-flag-phases-for-tmatmul--tmatmul_acc-and-l0c-readers).

## Memory-budget-first kernel generation

Before generating any nontrivial auto-mode kernel, define an explicit memory
budget for each local memory level used by the kernel.

| Memory level | Physical capacity on A3 / Ascend 910B1 | Custom usable budget |
|---|---:|---:|
| L1  | 512 KB | kernel-specific |
| L0A | 64 KB  | kernel-specific |
| L0B | 64 KB  | kernel-specific |
| L0C | 128 KB | kernel-specific |
| UB  | 192 KB | kernel-specific |

Do not plan tile shapes using the full physical capacity by default. Use a
custom usable limit per level to leave room for compiler allocation, temporary
tiles, alignment/padding, FIFOs, and future scheduling changes.

For each kernel, list all tiles that are live together at each memory level.
The budget must include input tiles, output tiles, accumulator tiles, temporary
tiles, scratch tiles, and FIFO staging tiles that can overlap in lifetime.

Example checklist:

```text
L1 live set:
- A L1 tile: ...
- B L1 tile: ...
- optional scratch / staging: ...
- total <= custom L1 budget

L0A live set:
- A operand tile(s): ...
- total <= custom L0A budget

L0B live set:
- B operand tile(s): ...
- total <= custom L0B budget

L0C live set:
- accumulator tile(s): ...
- total <= custom L0C budget

UB live set:
- vector input/output/temp tiles: ...
- reduction scratch: ...
- gather/scatter scratch: ...
- total <= custom UB budget
```

The budget should be based on bytes, not only element counts:

```text
bytes = rows x cols x sizeof(dtype) x number_of_live_tiles
```

For cube GEMM, also account for 16x16 fractal blocks in L0A/L0B/L0C. For the
common FP16 x FP16 -> FP32 case:

```text
A/B fractal: 16 x 16 x sizeof(half)  = 512 bytes
C fractal:   16 x 16 x sizeof(float) = 1024 bytes
```

A larger matrix tile is a grid of these 16x16 fractal blocks. Cube tile sizes
should be chosen as multiples of the cube-friendly block structure unless there
is explicit tail handling. See [tile_type_reference.md §4.1](tile_type_reference.md)
for the tile-type view of `BLayout`, `SLayout`, and `SFractalSize`.

## Operation granularity and loop tiling

For each kernel, define the smallest compute operation that is fed to the
hardware engine.

Examples:

```text
Cube:
- smallest TMATMUL / TMATMUL_ACC operation:
  M_tile x N_tile x K_tile
- accumulator tile:
  M_tile x N_tile
- Split-K / Split-N / Split-M strategy:
  ...

Vector:
- smallest vector tile operation:
  rows x cols
- row-slice or subblock split:
  ...
- reduction/gather/scatter granularity:
  ...
```

Every loop in the kernel should be explainable as tiling one logical dimension:

```text
for m_tile in M:
  for n_tile in N:
    for k_tile in K:
      feed one cube operation M_tile x N_tile x K_tile
```

or, for vector work:

```text
for token_tile in T:
  for hidden_tile in H:
    feed one vector operation token_tile x hidden_tile
```

When generating a kernel, write the loop nest first as a logical tiling plan.
For large/general kernels, prefer to tile-and-loop over most logical dimensions
rather than making tile sizes equal to the full input sizes. Input dimensions
such as `M`, `N`, `K`, `T`, `H`, `E`, `TopK`, `S0`, or `S1` should usually drive
loop bounds and GM offset progression. Tile sizes should be inferred from memory
budgets, operation granularity, cube fractal constraints, alignment, and the
tail strategy.

This distinction is important for generalizability:

```text
input shape:  large/model-specific
tile shape:   hardware-budget-specific
loop nest:    maps input shape onto repeated tile operations
```

Then map each loop level to:

1. GM offset progression,
2. tile shape,
3. memory level used,
4. live-tile budget impact,
5. whether the loop can be unrolled, peeled, or split,
6. tail handling strategy.

Use compile-time unrolling/peeling only when it helps the compiler see a stable
dependency pattern or when the loop bound is naturally small/fixed. Do not
confuse this with the broader rule above: the default is tile-and-loop over
dimensions, not hard-code one full-shape tile.

Default shape assumption for planning: most major dense dimensions can be
treated as multiples of 32 unless the kernel family says otherwise. Do not spend
the first design pass optimizing for unusual non-divisible dense shapes. The
main exceptions to plan for early are data-dependent MoE token counts per expert
and `TopK` / `kTopK`-style axes, where tails or non-divisible logical counts are
natural.

---

## 2. Optimization mindset

### 2.1 First question: which hardware resource should do the work?

Before coding, classify every stage:

| Work type | Preferred engine |
|---|---|
| Matrix multiply / matmul accumulation | Cube |
| Row reductions / row max / row sum | Vector |
| Gather/scatter / index extraction | Vector |
| Elementwise math / exp / multiply / divide | Vector |
| Large structured scan/reduction that can be matrix-reformulated | Possibly cube + vector |
| Data movement / layout transfer | MTE + layout-aware tile ops |

The paper’s scan algorithms are important because they show that cube units can accelerate operations that look vector/reduction-like by reformulating local scans as matrix operations. This generalizes beyond FA: do not assume cube is only useful for GEMM.

### 2.2 Second question: what should be overlapped?

A high-performance kernel usually tries to overlap:

```text
load next tile      with compute current tile
compute upstream    with consume downstream
cube stage          with vector stage
producer stage      with consumer stage
```

But overlap adds complexity. Use an optimization ladder:

```text
Level 0: correctness baseline
Level 1: local tile reuse
Level 2: local ping-pong
Level 3: one producer/consumer FIFO
Level 4: preload + warmup/main/drain
Level 5: periodic sync reduction
Level 6: multi-core / CV communication
```

Do not jump directly from Level 0 to Level 6.

---

## 3. Stage graph: required before high-performance code

For optimized kernels, always draw a stage graph before writing or modifying code.

Example from Flash Attention:

```text
compute_qk  (cube)   produces QK tile
    ↓ QK FIFO
compute_p   (vector) consumes QK, produces P/x_exp and softmax state
    ↓ P FIFO
compute_pv  (cube)   consumes P and V, produces PV partial
    ↓ PV FIFO
compute_gu  (vector) consumes PV and softmax state, updates O
```

For each edge, write an edge table:

| Field | Meaning |
|---|---|
| Producer stage | Which stage writes the data |
| Consumer stage | Which stage reads it |
| Producer engine | Cube/vector |
| Consumer engine | Cube/vector |
| Data type | half, float, uint32, packed format, etc. |
| Logical tile shape | e.g., `CUBE_S0 x CUBE_S1` |
| Physical storage | UB, L1, L0, GM FIFO, etc. |
| Split pieces | sub-tiles, row slices, vector subblocks |
| Record boundary | when the producer marks it ready |
| Wait boundary | when the consumer waits |
| Free/reuse boundary | when the slot can be reused |
| Tail/drain behavior | how final pending tiles/events are completed |

**Claude rule:** If the task involves buffering or pipelining, first output the stage graph and edge table. Do not patch code until the stage graph is coherent.

---

## 4. Cube/vector split pattern

### 4.1 Why it matters

A single algorithm can have stages that belong on different engines. In FA:

```text
compute_qk: Q x K^T      → cube
compute_p:  softmax      → vector
compute_pv: P x V        → cube
compute_gu: O update     → vector
```

The dependencies are sequential at the math level, but the implementation can pipeline them across S1 tiles.

### 4.2 General pattern

```cpp
template <...>
__global__ AICORE void Kernel(...) {
    // Shared setup: pointers, block ids, static constants.

    if constexpr (DAV_CUBE) {
        // Cube-side producer/consumer stages.
    }

    if constexpr (DAV_VEC) {
        // Vector-side producer/consumer stages.
    }
}
```

### 4.3 Generalizable examples

| Kernel family | Cube candidates | Vector candidates |
|---|---|---|
| Flash Attention | QK, PV | softmax, GU/output normalization |
| GEMM + epilogue | main matmul | activation, quant/dequant, bias, row stats |
| Conv-like kernels | im2col/matmul | padding, activation, normalization |
| TopK/sort | possible radix/split/scan variants | TSORT/TMRGSORT/TGATHER/index handling |
| Scan/reduction | local scan via matrix identity | block correction, row/vector propagation |

---

## 5. Multi-buffering: not one concept

Developers often say “double buffering,” but optimized kernels contain several different buffering concepts.

### 5.1 Local ping-pong buffering

Alternating local tiles inside one stage.

Example concept:

```text
buffer 0: being consumed/used
buffer 1: being loaded/produced
```

Use when load/compute/store within a stage can overlap.

### 5.2 FIFO depth

Number of outstanding logical tiles between a producer and a consumer.

```text
QK FIFO depth = how many QK tiles can exist between compute_qk and compute_p
```

Use to decouple engines that run at different speeds.

### 5.3 Preload depth

How far an upstream stage is allowed to run ahead.

```text
QK_PRELOAD = number of QK tiles produced before steady-state overlap starts
```

Use to fill a pipeline so downstream work does not starve.

### 5.4 Vector subblock slicing

Splitting vector work across vector subblocks is not “buffering,” but it interacts with buffering. If a tile is split by vector subblock or row-slice, the FIFO entry offset and wait/free boundaries must match that split.

### 5.5 Rule

Whenever proposing “buffering,” name which one:

```text
local ping-pong
producer/consumer FIFO depth
preload depth
vector row-slice/subblock split
accumulator ping-pong
```

Do not mix these up.

---

## 6. MultiBuffered pattern

### 6.1 What it does

`MultiBuffered<N>` is a compile-time loop scheduling helper. It does not magically allocate correct buffers; it maps iterations onto buffer lanes and helps the compiler form a multi-buffered schedule.

Representative pattern:

```cpp
MultiBuffered<kMatTNBuffers> mb;
using InnerMultiBuffered =
    MultiBuffered<kMatTNBuffers>::NestedLoopInvoker<Range<kTileFactor>>;

mb.loop<Range<qkPreloadNum, kTileFactor>>([&](auto ctxOuter,
                                               InnerMultiBuffered inner) {
    int tile_id = ctxOuter.iter;
    inner.loop([&](auto ctxInner) {
        int sub_tile = ctxInner.iter;
        TileMatKData kMatTile;
        qkPipe.prod.setTileId(tile_id, sub_tile);
        compute_qk(...);
    });
});
```

### 6.2 When to use it

Use it when a stage has repeated independent work over tiles/sub-tiles and you want the compiler to schedule the repeated operations with multiple buffer lanes.

Good candidates:

```text
preload QK tiles
preload P tiles
load/compute/store over repeated K panels
vector row-slice operations
```

### 6.3 Important caveat

More buffering can hurt performance. In the FA prototype, the main cube loop deliberately avoids additional local double buffering because the staged QK/PV pipeline already saturates the useful resource pattern. Use measurements.

**Rule:** Add `MultiBuffered` only when it is clear which resource bottleneck it improves.

---

## 7. MultiStaged pattern

### 7.1 What it does

`MultiStaged<N>` schedules different computations on the same resource in a staged/pipelined way.

FA example concept:

```cpp
MultiStaged<2> qk_pv_stages;

for (int tile_id = 0; tile_id < num_tiles_s1 - qkPreloadNum; ++tile_id) {
    for (int sub_tile = 0; sub_tile < kTileFactor; ++sub_tile) {
        qk_pv_stages.run(
            [&]() {
                // produce future QK
                qkPipe.prod.setTileId(tile_id + qkPreloadNum, sub_tile);
                compute_qk(... tile_id + qkPreloadNum ...);
            },
            [&]() {
                // consume current P and produce current PV
                pPipe.cons.setTileId(tile_id, sub_tile);
                pvPipe.prod.setTileId(tile_id, sub_tile);
                compute_pv(... tile_id ...);
            });
    }
}
```

### 7.2 Why it helps

It overlaps two same-engine stages that are offset by a dependency distance. In FA, `compute_pv(tile_id)` depends on `compute_p(tile_id)`, while `compute_qk(tile_id + preload)` is independent once Q is available. So cube can do:

```text
future QK work + current PV work
```

instead of doing all QK then all PV.

### 7.3 Precondition

This only works if the producer/consumer dependency is offset correctly:

```text
compute_qk(t + preload) must not depend on compute_pv(t)
compute_pv(t) must have P(t) ready
```

### 7.4 Rule

When using `MultiStaged`, state:
- stage count,
- resource being shared,
- dependency offset,
- warmup requirement,
- drain requirement,
- why local double buffering is or is not still useful.

---

## 8. Warmup / main / drain scheduling

### 8.1 Why these phases exist

If stage B consumes what stage A produces, a pipelined schedule cannot start with both stages immediately unless A has already produced some data. This creates phases:

```text
Warmup/prologue:
    run producer only for first preload tiles

Main:
    run future producer and current consumer together

Drain/epilogue:
    run consumer only for last preload tiles
```

### 8.2 FA cube side

```text
Warmup:
    compute_qk[0 .. QK_PRELOAD-1]

Main:
    compute_qk[t + QK_PRELOAD]
    compute_pv[t]

Drain:
    compute_pv[num_tiles - QK_PRELOAD .. num_tiles-1]
```

### 8.3 FA vector side

```text
Warmup:
    compute_p[0 .. QK_PRELOAD-1]

Main:
    compute_p[t + QK_PRELOAD]
    compute_gu[t]

Drain:
    compute_gu[num_tiles - QK_PRELOAD .. num_tiles-1]
```

### 8.4 Compiler phases

Some code uses phase enums such as:

```text
Phase::Prologue
Phase::Main
Phase::Epilogue
```

These are useful when the first/last few iterations need different synchronization, final store behavior, or accumulator handling.

Example concept:

```cpp
if constexpr (GU_Phase == Phase::Epilogue) {
    TSTORE(outGlobal, runningOTile);
}
```

### 8.5 Rule

Any pipelined kernel must explicitly handle:
- `num_tiles > preload`,
- `num_tiles == preload`,
- `num_tiles < preload`,
- tail store,
- pending notification drain.

---

## 9. Producer/consumer FIFO protocol

### 9.1 FIFO objects

The FA prototype uses logical pipes such as:

```text
QKPipe: qk -> p
PPipe:  p  -> pv
PVPipe: pv -> gu
```

A pipe usually has:
- producer side,
- consumer side,
- FIFO depth,
- sync period,
- source/consumer tile types,
- GM FIFO base pointer.

### 9.2 Producer protocol

General producer pattern:

```cpp
pipe.prod.setTileId(tile_id, sub_id);

// Optional: wait/allocate before overwriting FIFO slot.
pipe.prod.setAllocateStatus(should_wait_or_allocate);

// Compute or prepare tile.
compute_stage(...);

// Offset inside logical FIFO entry.
pipe.prod.setEntryOffset(offset_bytes);

// Record only when a full logical tile is ready.
pipe.prod.setRecordStatus(is_last_piece);

TPUSH(tile, pipe);
```

### 9.3 Consumer protocol

General consumer pattern:

```cpp
pipe.cons.setTileId(tile_id, sub_id);

// Wait before reading first required piece.
pipe.cons.setWaitStatus(is_first_piece_or_required_wait);

// Offset inside logical FIFO entry.
pipe.cons.setEntryOffset(offset_bytes);

TPOP(tile, pipe);

// Free only after the full logical tile is consumed.
pipe.cons.setFreeStatus(is_last_piece_or_periodic_free);
```

### 9.4 Boundary rule

If a logical tile is split into pieces:

```text
producer record: last piece
consumer wait: first piece
consumer free: last piece
```

This reduces synchronization overhead while preserving the true logical dependency.

### 9.5 Entry offset rule

If using `setEntryOffset`, always document:

```text
offset unit: bytes or elements
logical tile shape
piece shape
sub-tile ID
row-slice ID
vector subblock ID
formula
```

Wrong offset is a silent wrong-answer bug.

---

## 10. Periodic consumption notification

### 10.1 Motivation

Freeing/acknowledging every tile can add synchronization overhead. The FA code uses helper logic like:

```text
should_wait_consumption(fifo_size, sync_period, sync_iter)
should_notify_consumption(fifo_size, sync_period, sync_iter)
pending_consumption_events(...)
```

### 10.2 Concept

```text
Producer:
    wait only after the FIFO could be full

Consumer:
    notify/free every SyncPeriod tiles instead of every tile

Tail:
    drain pending consumption events
```

### 10.3 When to use

Only after the simple version works with every-tile synchronization.

### 10.4 Common deadlock

```text
Consumer skips final notify because total tile count is not a multiple of SyncPeriod.
Producer waits forever for a free slot.
```

### 10.5 Rule

Periodic notification requires a tail-drain function and tests for tile counts that are not multiples of the period.

---

## 11. Multi-core and CV communication

### 11.1 Work partitioning

For kernels like FA, a typical split is along output rows:

```text
block_idx -> block of S0 rows
```

For BNSD tensors, split candidates include:

```text
B
N/head
S0/CUBE_S0
```

FA-like kernels usually avoid splitting the S1 reduce axis unless there is a separate reduction/GU combination step.

### 11.2 CV communication slots

When logical blocks exceed physical communication resources, the code may map block IDs to communication slots. This is advanced and may use special synchronization helpers such as CV ID sync and FFTS-level cross-core sync.

### 11.3 Auto-mode limitation

Do not assume PTO auto mode automatically handles all multi-core communication. Current practice may still require explicit FFTS/CV sync instructions or pragmas for cross-core behavior.

### 11.4 Rule

For multi-core optimization:
1. first get one-core correctness,
2. split work without cross-core reduction if possible,
3. add cross-core synchronization only when the data dependency requires it,
4. document whether the communication is auto-managed or explicit.

---

## 12. Pragmas and barriers

### 12.1 Existing pattern

The FA code uses:

```cpp
#pragma pto v_loop_barrier
```

The code comments indicate this is a workaround for current auto-sync/memory-allocation limitations.

Repo sources also use loop-unroll pragmas:

```cpp
#pragma unroll
#pragma unroll(4)
```

Known examples include [include/pto/npu/a2a3/TCI.hpp](../include/pto/npu/a2a3/TCI.hpp) and A5 gather/scatter helpers under [include/pto/npu/a5/](../include/pto/npu/a5/). User-provided guidance: in auto-mode kernel work, loop-unroll pragmas may sometimes be used as a temporary hack when auto-sync does not recognize a complex pattern, especially nested loops. This is expected to be fixed later in the compiler.

### 12.2 How to treat it

Treat these pragmas as implementation-specific hints, not as clean general rules. They should explain a concrete compiler / auto-sync limitation, not merely force an optimization by habit.

### 12.3 Rule

Claude should not introduce `#pragma pto v_loop_barrier`, `#pragma unroll`, or `#pragma unroll(N)` unless:
- a known pattern requires it,
- the exact pipeline stage boundary is understood,
- for unroll pragmas, the loop pattern and the expected unrolled dependency shape are understood,
- the user explicitly allows it,
- the code comment marks it as a workaround.

---

## 13. Online softmax as an optimization pattern

### 13.1 Recurrence

For tiled softmax over S1:

```text
local_max = rowmax(X_tile)
new_max   = max(old_max, local_max)
exp_max   = exp(scale * (old_max - new_max))

P_tile    = exp(scale * (X_tile - new_max))
local_sum = rowsum(P_tile)
new_sum   = old_sum * exp_max + local_sum
```

For output accumulation:

```text
O = O * exp_max + P_tile @ V_tile

last tile:
    O = O / new_sum
```

### 13.2 Why it matters

It avoids materializing the full attention matrix and gives numerical stability.

### 13.3 General uses

This is useful beyond FA:
- streaming softmax,
- tiled normalization,
- top-p probability processing,
- any reduce-axis exponential normalization.

### 13.4 Implementation hints

Use row-reduction and row-broadcast operations:

```text
TROWMAX
TROWSUM
TROWEXPANDSUB
TROWEXPANDMUL
TROWEXPANDDIV
TEXP
TCVT
```

Avoid materializing a full 2D broadcast tile when a row-expand op can do it.

---

## 14. Stationary operand reuse

### 14.1 Pattern

In FA, Q can be stationary across multiple K/V tiles:

```text
Q block stays fixed for an S0 block.
K and V stream over S1 tiles.
```

### 14.2 General rule

Identify the reuse axis:

```text
stationary operand: reused across many loop iterations
streaming operand: changes every tile
```

### 14.3 Examples

| Kernel | Stationary candidate | Streaming candidate |
|---|---|---|
| FA QK | Q block | K tiles |
| FA PV | sometimes P tile / V tile depending loop order | V/P counterpart |
| GEMM | A or B depending tiling | other operand |
| Conv | filter or activation tile | other operand |
| Scan with constant matrix | triangular/all-ones matrix | input tiles |

### 14.4 Rule

Load stationary operands once when possible. Do not reload them in every sub-tile unless required by layout or capacity.

---

## 15. Hardware-aware algorithm redesign

### 15.1 Lesson from scan paper

The scan paper shows that some scan/reduction primitives can be reformulated as matrix operations and accelerated using cube units. This is not just an implementation trick; it is an algorithm-design technique.

Examples from the paper:
- ScanU uses multiplication by an upper-triangular all-ones matrix for local scans.
- ScanUL1 uses multiple matrix operations and cube accumulation to compute local scans.
- Multi-core scan combines cube and vector work.
- Split, compress, radix sort, top-p, and top-k-related workloads can be built around scan-like primitives.

### 15.2 General design question

For any performance-critical kernel, ask:

```text
Can a vector-looking operation be expressed as:
- matrix multiply with a structured constant,
- scan,
- split/partition,
- gather/scatter,
- radix-like loop,
- local reduction + propagation?
```

### 15.3 Tradeoff checklist

Cube reformulation is worth considering only if:
- extra computation is cheaper than memory traffic,
- structured constants can be reused,
- local memory capacity is sufficient,
- vector-side correction is small,
- synchronization does not dominate.

---

## 16. Recompute vs communication

### 16.1 Principle

Sometimes recomputing a small value is cheaper than storing/loading/synchronizing it.

The scan paper uses recomputation ideas in multi-core scan. FA-like kernels can also benefit from this mindset:

```text
Do not store every intermediate just because it exists.
Ask whether it can be recomputed or kept local.
```

### 16.2 Examples

Potential candidates:
- small row-wise scale factors,
- local reductions,
- masks,
- block-level offsets,
- some tile metadata.

### 16.3 Rule

For every intermediate:
1. estimate storage cost,
2. estimate reload cost,
3. estimate recompute cost,
4. check whether recomputation can overlap with idle hardware.

---

## 17. Shape and width correctness

### 17.1 Lesson from TopK

After `TSORT32`, data changes from raw values to packed `(value, index)` pairs; every subsequent merge / `TGATHER` step must use the **packed** width, not the source width. Full anti-pattern catalog + fix at [auto_mode_bad_patterns.md §5.7](auto_mode_bad_patterns.md); confirmed recipe at [known_good_kernel_examples.md §A12](known_good_kernel_examples.md).

### 17.2 General rule

After every transformation, update the shape contract:

```text
raw shape
packed shape
fractal shape
row-sliced shape
subblock shape
FIFO logical shape
physical byte stride
```

### 17.3 Claude rule

Any code that uses packed or layout-transformed data must define:
- logical element count,
- physical element count,
- byte stride,
- dtype reinterpretation,
- gather/scatter mask interpretation.

---

## 18. Static constraints and supported shapes

Optimized kernels often rely on fixed tile divisibility.

Use static assertions for constraints like:

```text
S1 % TILE_S1 == 0
TILE_S1 % CUBE_S1 == 0
CUBE_S0 % VEC_CORES == 0
QK_PRELOAD >= 1
CV_FIFO_SIZE >= 1
CV_FIFO_CONS_SYNC_PERIOD >= 1
```

First support a narrow shape set. Add dynamic tails only after the fixed-shape kernel is correct.

## Model-shape-driven test selection

Kernel tests should not only use arbitrary toy sizes. For each kernel family,
include at least one test shape inspired by real model configurations.

For MoE kernels, inspect representative SOTA model configurations such as
DeepSeek-style MoE shapes before choosing tests. The test plan should include:

1. a tiny debug shape,
2. a medium shape that exercises tiling,
3. a realistic model-inspired shape,
4. a tail/non-divisible shape if the kernel claims tail support.

For the first design pass, it is acceptable to choose major dense dimensions as
multiples of 32. Do not treat every odd/non-divisible dense shape as a primary
requirement unless the kernel explicitly claims broad tail support. Prioritize
natural irregular axes first: MoE per-expert token counts, routing capacity,
`TopK` / `kTopK`, packed widths after sort/gather transforms, and other
data-dependent loop bounds.

For each test, document:

```text
T / token count:
H / hidden size:
E / number of experts:
TopK:
intermediate FFN size:
dtype:
expected routing shape:
expected input/output GM buffers:
```

Do not claim support for a model-scale shape unless the kernel has a memory
budget and loop tiling plan for that shape. Do not hard-code exact model
configuration numbers unless they are already present in repo docs or verified
from source material.

---

## 19. Measurement-guided optimization

Do not assume an optimization helps.

The FA notes mention that main-loop local double buffering was not used because it hurt performance. This is important:

```text
More buffers can increase pressure or reduce scheduling quality.
```

Always define:
- expected bottleneck,
- expected improvement,
- timing/profiling point,
- rollback path.

Potential profile categories:
```text
cube QK time
vector P time
cube PV time
vector GU time
MTE load/store time
FIFO wait time
TSTORE bottleneck
```

---

## 20. Development workflow for optimized auto-mode kernels

### Step 1: correctness baseline

- One row/block if possible.
- One buffer.
- No cross-stage FIFO.
- Python reference.
- Fixed shapes.

### Step 2: shape and representation audit

- Confirm dtype.
- Confirm layout.
- Confirm packed/fractal widths.
- Confirm output mapping.

### Step 3: local reuse

- Keep stationary operands.
- Pick the view operator (`TSUBVIEW` for slices, `TRESHAPE` for reinterpret) by intent — full convention table at [tile_type_reference.md §11.1](tile_type_reference.md).
- Avoid in-place aliasing unless proven.

### Step 4: local multi-buffer

- Add `MultiBuffered` inside one stage.
- Test.

### Step 5: one cross-stage FIFO

- Add one producer/consumer edge.
- Start with simple wait/free every logical tile.
- Test.

### Step 6: preload

- Add warmup/main/drain.
- Start with preload 1 or 2.
- Test.

### Step 7: staged same-resource pipeline

- Use `MultiStaged` for offset independent stages.
- Test.

### Step 8: periodic sync reduction

- Add sync period.
- Add pending drain.
- Test non-multiple tile counts.

### Step 9: multi-core

- Add block partition.
- Add cross-core sync only if required.
- Test varied block counts and tails.

---

## 21. Deadlock checklist

Before running a pipelined kernel, verify:

### Producer side
- Does producer wait/allocate before overwriting a FIFO slot?
- Does producer record exactly when the logical tile is complete?
- If tile is split, is record on the last piece?
- If computation is skipped, is a needed record still emitted?

### Consumer side
- Does consumer wait before reading?
- Does consumer free exactly when the logical tile is fully consumed?
- If free is periodic, are pending frees drained?
- Does consumer use the same tile/subtile ID convention as producer?

### Schedule
- Does warmup produce enough tiles?
- Does main loop have correct bounds?
- Does drain consume all remaining tiles?
- What happens if tile count is less than preload?
- Are tail masks/causal skips still signalled?

### Shape/offset
- Are offsets bytes or elements?
- Does subblock row offset match store/load offset?
- Does FIFO modulo indexing match FIFO depth?
- Are packed widths used after packing?

---

## 22. Concrete API names confirmed in the auto-mode FA kernel

This section maps guide concepts to the specific identifiers observed in [kernels/automode/a2a3/flash_atten/fa_performance_kernel.cpp](../kernels/automode/a2a3/flash_atten/fa_performance_kernel.cpp). All items are **Inferred** auto-mode-safe unless marked Known (hardware-confirmed per README performance tables).

### 22.1 Cross-stage FIFO: `TMPipe` (not `TPipe`)

The auto-mode-safe FIFO class is `TMPipe`, from `<pto/npu/a2a3/custom/TSync_Custom.hpp>`. The old `TPipe` uses `TASSIGN` internally and is NOT auto-mode-safe. (Known — see [auto_mode_bad_patterns.md §2.4](auto_mode_bad_patterns.md).)

```cpp
using QKPipe = TMPipe<BUF0_QK_READY, FIFOType::GM_FIFO,
                      CV_FIFO_SIZE, CV_FIFO_CONS_SYNC_PERIOD,
                      TileQKData, TileDataF_T,
                      UF_ENABLE ? true : false, 0>;
QKPipe qkPipe(qk_tile_fifo_block_ptr);
```

Use `TPUSH(tile, pipe)` / `TPOP(tile, pipe)` via the setter API (§9.2 / §9.3 of this guide).

### 22.2 L1→L0 panel extraction: `TEXTRACT`

```cpp
TEXTRACT(al0Tiles, aMatTile, 0, k * Cube_K);   // row 0, col k*Cube_K
TEXTRACT(bl0Tiles, bMatTile, k * Cube_K, 0);   // row k*Cube_K, col 0
```

Extracts a `Cube_K`-wide sub-panel from an L1 `TileType::Mat` tile into an `L0A` `TileLeft` or `L0B` `TileRight`. Used inside `pto_macro_matmul` for K-split; the macro auto-selects `Cube_K` via `calculateFittingCubeK(M, N)`. (Inferred — part of the hardware-confirmed FA build.)

### 22.3 Unit-flag phases for TMATMUL / TMATMUL_ACC and L0C readers

```cpp
// First K slice: initialize the L0C accumulator.
TMATMUL<AccPhase::Partial>(cAccTile, al0Tiles, bl0Tiles);

// Middle K slices: accumulate, but do not publish the L0C blocks as complete.
TMATMUL_ACC<AccPhase::Partial>(cAccTile, al0Tiles, bl0Tiles);

// Last K slice: finish accumulation and mark each completed L0C block ready.
TMATMUL_ACC<AccPhase::Final>(cAccTile, al0Tiles, bl0Tiles);

// Reader side: wait on ready L0C blocks and clear/free them after read.
TSTORE<STPhase::Final, AccTile, GlobalDataC, AtomicType::AtomicNone>(cGlobal, cAccTile);
```

`AccPhase` is the public PTO hook for the MMAD-side unit-flag control. The enum is defined at [include/pto/common/type.hpp:233-239](../include/pto/common/type.hpp#L233-L239): `Unspecified = 0x0`, `Partial = 0x2`, `Final = 0x3`. The public overloads are in [include/pto/common/pto_instr.hpp:641-675](../include/pto/common/pto_instr.hpp#L641-L675), and A2/A3 lowers the template value into the `mad(...)` phase argument at [include/pto/npu/a2a3/TMatmul.hpp:37-52](../include/pto/npu/a2a3/TMatmul.hpp#L37-L52). (Known.)

`STPhase` is the matching L0C-reader side. It has the same bit encoding at [include/pto/common/type.hpp:224-230](../include/pto/common/type.hpp#L224-L230), the public `TSTORE<STPhase::...>` overloads are at [include/pto/common/pto_instr.hpp:351-374](../include/pto/common/pto_instr.hpp#L351-L374), and A2/A3 writes the phase into `copy_matrix_cc_to_gm`'s `Xt[33:32]` unit-flag field at [include/pto/npu/a2a3/TStore.hpp:318-335](../include/pto/npu/a2a3/TStore.hpp#L318-L335) and [TStore.hpp:374-394](../include/pto/npu/a2a3/TStore.hpp#L374-L394). (Known.)

Interpretation for kernel planning (Known for enum values and source plumbing; hardware semantics from user-provided internal docs, not in this repo):
- `0x0` / `Unspecified` disables the unit-flag mechanism.
- `0x2` / `Partial` enables unit-flag checks without post-update. For MMAD writes this means "wait until the L0C block is writable, write partial data, but do not mark it ready for the reader." Use this for every non-final K/F accumulation slice.
- `0x3` / `Final` enables check-and-update. For MMAD writes this means "wait until writable, write the final data, then mark the L0C block ready." For FixPipe/MTE reads this means "wait until ready, read, then clear/free it."

Rule of thumb: pair `AccPhase::Partial` on all non-final accumulation slices with exactly one `AccPhase::Final` on the final slice that completes the accumulator, then consume the accumulator with a matching reader phase such as `TSTORE<STPhase::Final, ...>`. If a GEMM has only one K slice, use `TMATMUL<AccPhase::Final>`. If a GEMM has multiple K slices, use `TMATMUL<AccPhase::Partial>` for the first slice, `TMATMUL_ACC<AccPhase::Partial>` for middle slices, and `TMATMUL_ACC<AccPhase::Final>` for the last slice.

The `pto_macro_matmul<M, K, N>()` wrapper in [kernels/automode/a2a3/flash_atten/pto_macro_matmul.hpp:101-123](../kernels/automode/a2a3/flash_atten/pto_macro_matmul.hpp#L101-L123) resolves this through its `AccMode` enum and emits the phase-specific calls at [pto_macro_matmul.hpp:154-171](../kernels/automode/a2a3/flash_atten/pto_macro_matmul.hpp#L154-L171). (Known; the surrounding Flash Attention kernel is hardware-confirmed per [known_good_kernel_examples.md §A19](known_good_kernel_examples.md#A19-flash_atten-auto-mode-a3-full-flash-attention-20-with-cubevector-spmd-pipeline--hardware-confirmed-optimized-prototype).)

### 22.4 Vector subblock SPMD: `get_subblockid()`

Returns `0` or `1` for the 2 vector subblocks within one AI core. Use to partition row work:

```cpp
const size_t subblock_base_rows = (Cube_S0 / VEC_CORES) * get_subblockid();
const size_t row_offset = subblock_base_rows + row_slice * Vec_S0;
```

Each subblock operates on its own row range with no inter-subblock sync required. `VEC_CORES = 2` is a build-time constant. (Inferred — part of the hardware-confirmed FA build.)

### 22.5 Triangular causal mask: `TTRI`

```cpp
TTRI<TileDataS1, 1>(triu, 1 + (s0_index % TileDataS1::Cols));
TMULS(triu, triu, negInf);
TADD(input_x, input_x, triu);  // apply -inf mask to above-diagonal positions
```

`TTRI<TileType, kDiag>(dst, diagOffset)` fills a triangular mask tile. Used for causal attention masking when `CAUSAL_MASK=true`. (Inferred — compiles under the `CAUSAL_MASK` template path; causal correctness not independently validated on hardware.)

### 22.6 L2 prefetch: `PTO_PREFETCH`

Host-side only (not in kernel code). Launches an SDMA prefetch of a GM tensor into L2 before the main kernel:

```cpp
#include <pto/npu/kernels/Pto_prefetch.hpp>
PTO_PREFETCH((__gm__ void *)q, tensor_bytes, stream);     // SDMA variant
PTO_PREFETCH<false, kPrefetchAivCores>(..., stream);       // AIV variant
```

Call before the main kernel launch on the same stream. Requires warmup kernel (`warmup_kernel<<<24, nullptr, stream>>>`) to be launched first so all AIV cores are ready. (Inferred — part of the hardware-confirmed FA build.)

### 22.7 `pto_example_mixed` CMake function

Mixed cube+vector kernels use `pto_example_mixed(NAME)` instead of `pto_example_vec_auto` / `pto_example_cube_auto`. Key difference: uses `--cce-aicore-arch=dav-c220` (no `-cube`/`-vec` suffix), which covers both cube and vector code paths in one CMake target with `--cce-fatobj-link`. (Inferred — from [CMakeLists.txt](../kernels/automode/a2a3/flash_atten/CMakeLists.txt).)

---

## 23. Guidance for Claude

When using this document (including the new §22 API names):

1. Do not copy FA performance code blindly.
2. Extract the stage graph first.
3. Start with correctness, then add one optimization at a time.
4. Avoid advanced FIFO/FFTS/CV sync unless requested.
5. Always state assumptions and unsupported shapes.
6. Always include a Python reference for numerical kernels.
7. When optimizing, explain which hardware resource becomes better utilized.

### Prompt skeleton

```text
Use CLAUDE.md and docs_for_ai.

Goal: optimize or generate a high-performance auto-mode kernel.

Before writing code:
1. Identify cube stages and vector stages.
2. Draw the stage graph.
3. Define every producer/consumer edge.
4. Define tile shapes and transformed widths.
5. Identify stationary operands.
6. Choose the optimization level:
   - correctness baseline
   - local reuse
   - local ping-pong
   - FIFO
   - preload
   - staged pipeline
   - periodic sync
   - multi-core
7. List risks and tests.

Do not introduce TPUSH/TPOP/TMPipe/v_loop_barrier/FFTS sync unless explicitly requested and justified.
Do not claim correctness without a test.
```

---

## 23. Compact rules

1. Separate cube and vector code.
2. Build a stage graph.
3. Distinguish FIFO depth, preload depth, and local ping-pong.
4. Use warmup/main/drain for pipelined dependencies.
5. Record/free at logical tile boundaries.
6. Use entry offsets carefully and document byte formulas.
7. Use packed/transformed widths after layout-changing ops.
8. Keep stationary operands resident.
9. Prefer row-expand ops over materialized broadcast buffers.
10. Treat pragmas and FFTS/CV sync as advanced, not default.
11. Consider cube reformulations for scans/splits/reductions.
12. Optimize by measurement, not by intuition.
