# PTO Knowledge

## Why this file exists

This file is a working engineering note for future work inside `pto-isa-zy`.

It is intentionally biased toward:

- how this repository is organized
- where the real source of truth lives
- how PTO models compute, communication, tiles, layouts, and synchronization
- which kernels and tests are the best entry points for future design work
- what the two key overlap kernels in this repo teach about "PTO-ized" kernel structure

This note is based on the current repository state. If this file ever disagrees with current code, trust the code first.

## Scope and source priority

This note is grounded in repo-local sources:

1. Public API and type system:
   - [`../include/pto/pto-inst.hpp`](../include/pto/pto-inst.hpp)
   - [`../include/pto/common/pto_instr.hpp`](../include/pto/common/pto_instr.hpp)
   - [`../include/pto/common/pto_tile.hpp`](../include/pto/common/pto_tile.hpp)
   - [`../include/pto/common/memory.hpp`](../include/pto/common/memory.hpp)
   - [`../include/pto/common/type.hpp`](../include/pto/common/type.hpp)
   - [`../include/pto/comm/pto_comm_inst.hpp`](../include/pto/comm/pto_comm_inst.hpp)
   - [`../include/pto/comm/comm_types.hpp`](../include/pto/comm/comm_types.hpp)
2. Repo docs:
   - [`../include/README.md`](../include/README.md)
   - [`coding/ProgrammingModel.md`](coding/ProgrammingModel.md)
   - [`coding/Tile.md`](coding/Tile.md)
   - [`coding/GlobalTensor.md`](coding/GlobalTensor.md)
   - [`coding/Event.md`](coding/Event.md)
   - [`isa/comm/README.md`](isa/comm/README.md)
   - [`isa/TLOAD.md`](isa/TLOAD.md)
   - [`isa/TMATMUL.md`](isa/TMATMUL.md)
   - [`isa/TSTORE_FP.md`](isa/TSTORE_FP.md)
3. Repo-local knowledge routing:
   - [`../.ai-knowledge/router/ROUTER.md`](../.ai-knowledge/router/ROUTER.md)
   - [`../.ai-knowledge/cards/tables/pto-primitives-map.md`](../.ai-knowledge/cards/tables/pto-primitives-map.md)
   - [`../.ai-knowledge/cards/tables/pto-kernel-pattern-map.md`](../.ai-knowledge/cards/tables/pto-kernel-pattern-map.md)
   - [`../.ai-knowledge/cards/tables/pto-test-entry-map.md`](../.ai-knowledge/cards/tables/pto-test-entry-map.md)
4. Example kernels and tests:
   - [`../kernels/README.md`](../kernels/README.md)
   - [`../kernels/manual/README.md`](../kernels/manual/README.md)
   - [`../tests/README_zh.md`](../tests/README_zh.md)
   - selected kernels and testcase families listed later in this file

For future work, use this priority:

1. Current code in `include/pto/**`
2. Current code in representative kernels/tests
3. Instruction reference in `docs/isa/**`
4. `.ai-knowledge` routing pages and maps
5. Summary documents like this file

## Fast reading order

When entering this repo from scratch, the most reliable reading order is:

1. If the question is broad, start with:
   - [`../.ai-knowledge/router/ROUTER.md`](../.ai-knowledge/router/ROUTER.md)
   - [`../.ai-knowledge/cards/tables/pto-primitives-map.md`](../.ai-knowledge/cards/tables/pto-primitives-map.md)
   - [`../.ai-knowledge/cards/tables/pto-kernel-pattern-map.md`](../.ai-knowledge/cards/tables/pto-kernel-pattern-map.md)
   - [`../.ai-knowledge/cards/tables/pto-test-entry-map.md`](../.ai-knowledge/cards/tables/pto-test-entry-map.md)
2. Unified PTO entry:
   - [`../include/pto/pto-inst.hpp`](../include/pto/pto-inst.hpp)
3. Common API surface:
   - [`../include/pto/common/pto_instr.hpp`](../include/pto/common/pto_instr.hpp)
4. Core data model:
   - [`../include/pto/common/pto_tile.hpp`](../include/pto/common/pto_tile.hpp)
   - [`../include/pto/common/memory.hpp`](../include/pto/common/memory.hpp)
   - [`../include/pto/common/type.hpp`](../include/pto/common/type.hpp)
5. Communication API:
   - [`../include/pto/comm/pto_comm_inst.hpp`](../include/pto/comm/pto_comm_inst.hpp)
   - [`../include/pto/comm/comm_types.hpp`](../include/pto/comm/comm_types.hpp)
6. Backend-specific implementations:
   - [`../include/pto/npu/a2a3/`](../include/pto/npu/a2a3/)
   - [`../include/pto/npu/a5/`](../include/pto/npu/a5/)
   - [`../include/pto/cpu/`](../include/pto/cpu/)
7. Instruction docs:
   - [`isa/`](isa/)
   - [`isa/comm/`](isa/comm/)
8. Real code:
   - [`../kernels/manual/`](../kernels/manual/)
   - [`../tests/`](../tests/)

The main idea is:

- `pto-inst.hpp` is the unified include
- `common/` is the portable programming model
- `comm/` is the communication ISA layer
- `npu/*` and `cpu/*` are implementation backends
- `docs/isa/*` explains instruction contracts
- `kernels/` and `tests/` show how the model is actually used

## Programming model

PTO exposes two orthogonal axes.

### 1. Auto vs Manual

From [`coding/ProgrammingModel.md`](coding/ProgrammingModel.md):

- `PTO-Auto`
  - compiler/runtime manages placement and required synchronization
  - better entry point for correctness and portability
- `PTO-Manual`
  - developer controls placement, address binding, ordering, and scheduling
  - uses explicit `TASSIGN`, events, and pipeline reasoning
  - this is the dominant style in `kernels/manual/**`

In practice, most performance-oriented work in this repo is closer to PTO-Manual.

### 2. SPMD vs MPMD

Also from [`coding/ProgrammingModel.md`](coding/ProgrammingModel.md):

- `SPMD`: all cores run the same tile program but on different data slices
- `MPMD`: different cores or groups of cores can run different stages/programs

This matters a lot for fused and communication-heavy kernels. Some kernels are regular SPMD math loops; others are staged protocols with distinct compute and communication roles.

## Backend and platform picture

From [`../include/README.md`](../include/README.md) and [`../include/pto/README.md`](../include/pto/README.md):

- CPU simulator backend exists and is the recommended first stop for beginners
- A2 and A3 currently share the `include/pto/npu/a2a3/` backend
- A5 uses `include/pto/npu/a5/`
- public APIs are meant to stay stable while backend details differ

Practical rule:

- use CPU simulation first when learning an instruction or debugging basic semantics
- use the matching `a2a3` or `a5` backend headers and testcase families when doing hardware-specific work

## Core data abstractions

### Tile

The Tile programming model is described in [`coding/Tile.md`](coding/Tile.md), while the concrete enums live in [`../include/pto/common/memory.hpp`](../include/pto/common/memory.hpp).

Important axes:

- `TileType`
  - `Vec`
  - `Mat`
  - `Left`
  - `Right`
  - `Acc`
  - `Bias`
  - `Scaling`
  - `ScaleLeft`
  - `ScaleRight`
- `BLayout`
  - `RowMajor`
  - `ColMajor`
- `SLayout`
  - `NoneBox`
  - `RowMajor`
  - `ColMajor`

What these mean in practice:

- `Vec` tiles are the normal vector or UB-side working tiles
- `Mat` tiles are GM-to-matrix or L1-style tiles
- `Left` / `Right` / `Acc` are matmul-specialized tiles
- `BLayout` is the outer 2D layout
- `SLayout` is the inner boxed or fractal layout

Valid region is first-class:

- tile capacity is compile-time (`Rows x Cols`)
- valid rows/cols can be static or dynamic
- dynamic valid region is how PTO expresses tail tiles cleanly

This matters a lot in real kernels:

- `TLOAD` and `TSTORE` use the valid region as the transfer domain
- tail chunks often require `pto::DYNAMIC`
- communication wrappers often use dynamic tile sizes for the last chunk

### Common tile aliases

[`../include/pto/common/pto_tile.hpp`](../include/pto/common/pto_tile.hpp) provides aliases such as:

- `TileLeft`
- `TileLeftCompact`
- `TileRight`
- `TileRightCompact`
- `TileAcc`
- `TileAccCompact`

These aliases are important enough to memorize:

- they encode legal matmul-oriented layout combinations
- they are easier to reason about than writing the full `Tile<...>` template
- `Compact` variants are worth special attention for layout-sensitive stores

One non-obvious backend detail:

- `TileLeft` is backend-dependent
- on `PTO_NPU_ARCH_A2A3`, `TileLeft` uses outer `BLayout::RowMajor`
- on non-`A2A3` or CPU simulation, `TileLeft` uses outer `BLayout::ColMajor`

So for cross-backend reasoning, do not treat the alias name alone as the whole layout fact. Always check the alias definition in `pto_tile.hpp`.

### Compact vs non-compact accumulator tiles

`TileAcc` and `TileAccCompact` are both valid PTO accumulator abstractions, but they are not interchangeable in layout-sensitive store paths.

Use this as a practical engineering rule:

- if a path writes accumulator data back to GM and stride/layout details matter, verify whether `TileAccCompact` is the correct representation
- do not assume `TileAcc` and `TileAccCompact` produce the same effective store behavior

This repo already has real code that relies on this distinction:

- [`../kernels/manual/a2a3/dispatch_combine_moe/include/pto_block_mmad_helper.h`](../kernels/manual/a2a3/dispatch_combine_moe/include/pto_block_mmad_helper.h)

### GlobalTensor

`GlobalTensor` is the GM-side typed view object. The model is documented in [`coding/GlobalTensor.md`](coding/GlobalTensor.md), and the implementation lives in [`../include/pto/common/pto_tile.hpp`](../include/pto/common/pto_tile.hpp).

Core contract:

- template shape: `pto::GlobalTensor<Element_, Shape_, Stride_, Layout_>`
- it wraps a `__gm__` pointer plus 5-D shape and stride metadata
- it is a view, not a GM allocator and not a lifetime owner
- shape and stride are expressed in elements, not bytes
- most 2D code sets the first three dimensions to `1` and uses DIM_3/DIM_4 as rows/cols
- static dimensions come from template `Shape` / `Stride`; dynamic dimensions must be supplied consistently at construction
- `TASSIGN(globalTensor, ptr)` only binds or rebinds the underlying GM pointer; the pointer type must match `GlobalTensor::DType`
- `globalTensor.data()` returns the underlying `__gm__` pointer

Important `Layout` values from [`../include/pto/common/type.hpp`](../include/pto/common/type.hpp):

- `ND`
- `DN`
- `NZ`
- `SCALE`
- `MX_A_*`
- `MX_B_*`
- convolution-related formats such as `NC1HWC0`, `NCHW`, `NHWC`

Mental model:

- tile layout answers: "how is the on-chip tile organized?"
- global layout answers: "what GM storage pattern is this view hinting at?"
- `GlobalTensor` describes the GM side for `TLOAD`, `TSTORE`, `MGATHER`, `MSCATTER`, and communication wrappers

Practical rule:

- keep raw `__gm__ *` pointers at broad kernel/helper boundaries when that is simpler
- instantiate or bind `GlobalTensor` close to the PTO primitive that needs shape/stride/layout semantics
- do not replace local UB lifetime management with `GlobalTensor`; it is GM-side metadata, not a UB queue or tile allocator
- for 2D helper code, prefer explicit shape/stride aliases such as `TileShape2D` and `BaseShape2D` when they match the primitive contract

Do not mix GM-side `Layout` with tile-side `BLayout` / `SLayout`.

### Address binding

`TASSIGN` is the manual binding primitive. From [`../include/pto/common/pto_instr.hpp`](../include/pto/common/pto_instr.hpp):

- `TASSIGN(tile, addr)` binds a tile to an address or resource in manual mode
- `TASSIGN<Addr>(tile)` also exists as a compile-time address form

In manual kernels, `TASSIGN` is usually the first signal that the author is controlling exact placement instead of relying on PTO-Auto.

### Events and synchronization

From [`coding/Event.md`](coding/Event.md), [`../include/pto/common/pto_instr.hpp`](../include/pto/common/pto_instr.hpp), [`../include/pto/common/event.hpp`](../include/pto/common/event.hpp), and backend sync headers such as [`../include/pto/npu/a2a3/TSync.hpp`](../include/pto/npu/a2a3/TSync.hpp), [`../include/pto/npu/a2a3/SyncAll.hpp`](../include/pto/npu/a2a3/SyncAll.hpp), [`../include/pto/npu/a5/SyncAll.hpp`](../include/pto/npu/a5/SyncAll.hpp):

- many intrinsics return `RecordEvent`
- explicit `Event<SrcOp, DstOp>` objects encode pipeline dependencies
- most intrinsics accept trailing `WaitEvents&...`
- `TSYNC(events...)` waits on those events before issuing the next op
- `TSYNC<OpCode>()` is a single-pipeline barrier form
- `SYNCALL<CoreType>()` is the hard all-core barrier form
- `SYNCALL<Mode, CoreType>(gmWorkspace, ubOrL1Workspace, usedCores)` is the explicit-workspace software barrier form

PTO does have synchronization and event APIs. The important boundary is that they are not a blanket drop-in replacement for every AscendC `SetFlag` / `WaitFlag` / `PipeBarrier` / `SyncAll` / cross-core sequence.

Layered model:

1. Public instruction wait flow: PTO primitives may produce events, later primitives can accept `WaitEvents&...`, and `TSYNC(events...)` waits for a set of event objects.
2. Typed pipeline event flow: `Event<SrcOp, DstOp, AutoToken, EventID>` maps PTO `Op` values to backend pipes, allocates or uses event IDs, and lowers to backend `set_flag` / `wait_flag` or cross-core FFTS sync.
3. Single-op barrier flow: `TSYNC<OpCode>()` lowers through `TSYNC_IMPL<OpCode>()`; in the A2/A3 backend this form statically requires the mapped pipe to be `PIPE_V`, so it is a vector-pipeline barrier, not a generic pipe barrier.
4. All-core barrier flow: hard `SYNCALL` uses backend pipe barriers and cross-core sync; software `SYNCALL` uses explicit GM workspace plus UB/L1 workspace and has backend-specific A2/A3 vs A5 behavior.
5. Custom C/V flow: [`../include/pto/npu/a2a3/custom/TSync_Custom.hpp`](../include/pto/npu/a2a3/custom/TSync_Custom.hpp) provides a specialized producer/consumer helper for `TSTORE_C2GM` or `TSTORE_V2GM` feeding `TLOAD`, including forward `record/wait` and backward `allocate/free` dependencies.
6. C/V slot discovery flow: `TSYNC_CVID` in [`../include/pto/npu/a2a3/custom/TSyncCVID.hpp`](../include/pto/npu/a2a3/custom/TSyncCVID.hpp) uses reserved CV communication flags for cube/vector slot handoff; treat it as a C/V coordination helper, not a general event abstraction.
7. Internal substrate flow: low-level helpers such as `PtoSetWaitFlag<SrcPipe, DstPipe>` still directly wrap backend flag primitives. Seeing `set_flag`, `wait_flag`, `pipe_barrier`, `dsb`, or `dcci` inside PTO backend code does not mean the user kernel should bypass PTO; it marks the substrate boundary.

Migration rule for manual kernels:

- first classify the existing sync as local pipeline ordering, hard pipe barrier, all-core barrier, cross-core C/V handoff, or cache-coherence sequence
- simple local PTO-producer to PTO-consumer dependencies are the best candidates for `Event<SrcOp, DstOp>` or trailing wait events
- vector-only barrier points may map to `TSYNC<OpCode>()` if the backend static constraints match
- C/V producer-consumer points may need `TSync_Custom`, but only when the producer/consumer pair matches its supported operation types
- `SyncAll`, cache maintenance, cross-core flags, queue lifetime, and mixed AIC/AIV coordination should stay on the existing substrate until their exact backend semantics are proven
- do not mechanically replace AscendC sync calls by name; replace only after the PTO op-to-pipe mapping, event ID ownership, cross-core ID, and memory visibility semantics are all matched

Practical rule:

- when reading a manual kernel, do not only track data flow
- also track event flow, hard barriers, software barriers, cross-core handoffs, and backend-level `set_flag` / `wait_flag` / `pipe_barrier` / `dsb` / `dcci` sequences
- if a kernel still uses AscendC-style wrappers for some sync points, that can be a deliberate substrate boundary rather than an incomplete PTO conversion

## Compute primitive families

The common API surface is declared in [`../include/pto/common/pto_instr.hpp`](../include/pto/common/pto_instr.hpp), while per-instruction references live in [`isa/`](isa/).

### 1. Load / store / movement

Representative primitives:

- `TLOAD`
- `TSTORE`
- `TSTORE_FP`
- `TPREFETCH`
- `TMOV`
- `TTRANS`
- `TRESHAPE`
- `TEXTRACT`
- `TINSERT`
- `TIMG2COL`

What they are for:

- move data between GM and tiles
- move data between tile forms
- convert between memory-friendly and compute-friendly views

Good repo entry points:

- [`../tests/npu/a2a3/src/st/testcase/tload_gm2mat/`](../tests/npu/a2a3/src/st/testcase/tload_gm2mat/)
- [`../tests/npu/a2a3/src/st/testcase/tstore_mat2gm/`](../tests/npu/a2a3/src/st/testcase/tstore_mat2gm/)
- [`../tests/npu/a5/src/st/testcase/tload_mx_NZ/`](../tests/npu/a5/src/st/testcase/tload_mx_NZ/)
- [`../tests/npu/a5/src/st/testcase/textract/`](../tests/npu/a5/src/st/testcase/textract/)
- [`../tests/npu/a5/src/st/testcase/textract_compact/`](../tests/npu/a5/src/st/testcase/textract_compact/)
- [`../tests/npu/a5/src/st/testcase/ttrans/`](../tests/npu/a5/src/st/testcase/ttrans/)

### 2. Vector elementwise / scalar ops

Representative primitives:

- `TADD`, `TSUB`, `TMUL`, `TDIV`
- `TMAX`, `TMIN`
- `TEXP`, `TLOG`, `TSQRT`, `TRSQRT`
- scalar forms such as `TADDS`, `TMULS`, `TDIVS`

These are the bread-and-butter ops for:

- vector post-processing
- activation logic
- normalization and scaling
- row or column pipeline stages that stay on the vector side

### 3. Row/column reduce and expand

Representative primitives:

- `TROWMAX`
- `TROWMIN`
- `TROWSUM`
- `TROWEXPAND*`
- `TCOLMAX`
- `TCOLMIN`
- `TCOLSUM`
- `TCOLEXPAND*`

These are key for:

- softmax-style pipelines
- row or column statistics
- reduction followed by broadcast or expand patterns

Good repo entry points:

- [`../tests/npu/a2a3/src/st/testcase/trowsum/`](../tests/npu/a2a3/src/st/testcase/trowsum/)
- [`../tests/npu/a2a3/src/st/testcase/trowmax/`](../tests/npu/a2a3/src/st/testcase/trowmax/)
- [`../tests/npu/a5/src/st/testcase/trowexpand_trowsum/`](../tests/npu/a5/src/st/testcase/trowexpand_trowsum/)
- [`../tests/npu/a5/src/st/testcase/trowsum_trowexpand/`](../tests/npu/a5/src/st/testcase/trowsum_trowexpand/)

### 4. Matmul family

Representative primitives:

- `TMATMUL`
- `TMATMUL_ACC`
- `TMATMUL_BIAS`
- `TMATMUL_MX`
- `TGEMV`
- `TGEMV_ACC`

What to remember:

- `TMATMUL` is the main cube or matmul entry
- `TMATMUL_ACC` is the accumulate-on-existing-accumulator form
- legal tile roles matter: left tile, right tile, accumulator tile
- valid row/col determine the effective `M/K/N`

The most educational testcase family is:

- [`../tests/npu/a5/src/st/testcase/tmatmul/`](../tests/npu/a5/src/st/testcase/tmatmul/)

It is useful because it shows the normal PTO cube pattern explicitly:

- `TLOAD`
- `TMOV` or `TEXTRACT`
- `TMATMUL` / `TMATMUL_BIAS`
- `TMATMUL_ACC` for split-K accumulation
- `TSTORE`

Also useful:

- [`../tests/npu/a5/src/st/testcase/tmatmul_mx/`](../tests/npu/a5/src/st/testcase/tmatmul_mx/)
- [`../tests/npu/a5/src/st/testcase/tstore_acc2gm/`](../tests/npu/a5/src/st/testcase/tstore_acc2gm/)

### 5. Quant / dequant / format-sensitive paths

Representative primitives:

- `TSTORE_FP`
- `TQUANT`
- `TDEQUANT`
- MX-related load/store and format conversions

This family deserves special care because it often combines:

- accumulator tiles
- special scale tiles
- layout-sensitive GM views
- backend-specific legality checks

Important repo-specific caution:

- [`../include/README.md`](../include/README.md) is the first support matrix to consult
- but it is not always the final engineering truth
- for example, `TSTORE_FP` is still marked `TODO` in that matrix, while:
  - the API is declared in [`../include/pto/common/pto_instr.hpp`](../include/pto/common/pto_instr.hpp)
  - constraints and examples exist in [`isa/TSTORE_FP.md`](isa/TSTORE_FP.md)
  - current repo code already uses it in real kernel work

Two useful testcase families here are:

- [`../tests/npu/a5/src/st/testcase/tquant/`](../tests/npu/a5/src/st/testcase/tquant/)
- [`../tests/npu/a5/src/st/testcase/tdequant/`](../tests/npu/a5/src/st/testcase/tdequant/)

And for format-sensitive A5 layout reasoning:

- [`../tests/npu/a5/src/st/testcase/tload_mx_NZ/`](../tests/npu/a5/src/st/testcase/tload_mx_NZ/)
- [`../tests/npu/a5/src/st/testcase/tload_mx_ND_DN/`](../tests/npu/a5/src/st/testcase/tload_mx_ND_DN/)

## Communication primitives

Communication has its own public entry:

- [`../include/pto/comm/pto_comm_inst.hpp`](../include/pto/comm/pto_comm_inst.hpp)
- [`../include/pto/comm/comm_types.hpp`](../include/pto/comm/comm_types.hpp)
- overview doc: [`isa/comm/README.md`](isa/comm/README.md)

This layer is one of the most important parts of the repo for future fused-kernel and distributed-kernel work.

### Communication families

#### Point-to-point

- `TPUT`
- `TGET`

What to remember:

- `TPUT`
  - local GM -> UB tile -> remote GM
  - supports `AtomicNone` and `AtomicAdd`
  - supports ping-pong double buffering
  - supports auto chunking, irregular tails, and 2D sliding when the view is larger than the staging tile
- `TGET`
  - remote GM -> UB tile -> local GM
  - supports ping-pong double buffering
  - is the symmetric "pull" side of the same model

Best entry tests:

- [`../tests/npu/a2a3/comm/st/testcase/tput/`](../tests/npu/a2a3/comm/st/testcase/tput/)
- [`../tests/npu/a5/comm/st/testcase/tput/`](../tests/npu/a5/comm/st/testcase/tput/)
- [`../tests/npu/a2a3/comm/st/testcase/tget/`](../tests/npu/a2a3/comm/st/testcase/tget/)
- [`../tests/npu/a5/comm/st/testcase/tget/`](../tests/npu/a5/comm/st/testcase/tget/)

The `tput` family is especially valuable because it covers:

- 1D vector movement
- 2D shapes
- large-shape auto chunking
- multi-dimensional outer-loop traversal
- irregular tail chunks through dynamic valid region
- 2D sliding
- ping-pong double buffering
- `AtomicAdd`

#### Signal-based synchronization

- `TNOTIFY`
- `TWAIT`
- `TTEST`

Important facts from the comm docs and `comm_types.hpp`:

- signal dtype is `int32_t`
- `NotifyOp`
  - `Set`
  - `AtomicAdd`
- `WaitCmp`
  - `EQ`, `NE`, `GT`, `GE`, `LT`, `LE`
- `TWAIT` is blocking
- `TTEST` is non-blocking
- for signal tensors, `TWAIT` and `TTEST` use ALL-satisfied semantics, not ANY-satisfied semantics

Useful types:

- `comm::Signal`
- `comm::Signal2D<Rows, Cols>`
- `comm::GlobalSignal<...>`

Best entry tests:

- [`../tests/npu/a2a3/comm/st/testcase/tnotify/`](../tests/npu/a2a3/comm/st/testcase/tnotify/)
- [`../tests/npu/a2a3/comm/st/testcase/twait/`](../tests/npu/a2a3/comm/st/testcase/twait/)
- [`../tests/npu/a2a3/comm/st/testcase/ttest/`](../tests/npu/a2a3/comm/st/testcase/ttest/)
- [`../tests/npu/a5/comm/st/testcase/tnotify/`](../tests/npu/a5/comm/st/testcase/tnotify/)
- [`../tests/npu/a5/comm/st/testcase/twait/`](../tests/npu/a5/comm/st/testcase/twait/)
- [`../tests/npu/a5/comm/st/testcase/ttest/`](../tests/npu/a5/comm/st/testcase/ttest/)

The `twait` family is especially good for learning:

- single-signal waits
- comparison-mode waits
- multi-rank atomic-add counters
- `Signal2D` matrix waits
- sub-region waits with custom stride

#### Collective communication

- `TGATHER`
- `TSCATTER`
- `TBROADCAST`
- `TREDUCE`

These work through `ParallelGroup<GlobalData>` from [`../include/pto/comm/comm_types.hpp`](../include/pto/comm/comm_types.hpp).

Best entry tests:

- [`../tests/npu/a2a3/comm/st/testcase/tgather/`](../tests/npu/a2a3/comm/st/testcase/tgather/)
- [`../tests/npu/a2a3/comm/st/testcase/tscatter/`](../tests/npu/a2a3/comm/st/testcase/tscatter/)
- [`../tests/npu/a2a3/comm/st/testcase/tbroadcast/`](../tests/npu/a2a3/comm/st/testcase/tbroadcast/)
- [`../tests/npu/a2a3/comm/st/testcase/treduce/`](../tests/npu/a2a3/comm/st/testcase/treduce/)
- [`../tests/npu/a5/comm/st/testcase/tgather/`](../tests/npu/a5/comm/st/testcase/tgather/)
- [`../tests/npu/a5/comm/st/testcase/tscatter/`](../tests/npu/a5/comm/st/testcase/tscatter/)
- [`../tests/npu/a5/comm/st/testcase/tbroadcast/`](../tests/npu/a5/comm/st/testcase/tbroadcast/)
- [`../tests/npu/a5/comm/st/testcase/treduce/`](../tests/npu/a5/comm/st/testcase/treduce/)

Use them when:

- the data movement is regular and group-structured
- there is a natural root or collective semantics

Do not force collectives onto clearly irregular peer-to-peer exchange problems.

### Communication design rules worth remembering

1. Treat payload movement and readiness signaling as separate protocol steps.
2. Prefer `TPUT` / `TGET` for irregular or explicitly peer-to-peer movement.
3. Prefer collectives when the communication really is collective.
4. Keep signal views small and explicit.
5. Use summary counters when many waits would otherwise touch a large sparse flag matrix.
6. Publish doorbells only after the producer-side data path is globally visible; in this repo that often appears as `pipe_barrier(PIPE_ALL) + dsb(DSB_DDR)` before `TNOTIFY`.
7. Use ping-pong staging when bandwidth or overlap matters and the staging tiles can remain non-overlapping.

## PTO-Manual pipeline and tiling

This is the third main axis of this note after compute primitives and communication primitives. PTO-Manual is not just a list of instructions; it is a way to make placement, movement, compute, synchronization, and reuse explicit.

### What PTO-Manual owns

PTO-Manual gives the kernel author control over:

- local-memory placement through `TASSIGN(tile, addr)` or `TASSIGN<addr>(tile)`;
- GM-side view description through `GlobalTensor` shape, stride, layout, and bound pointer;
- tile-level movement and compute through `TLOAD`, `TMOV`, `TMATMUL`, `TSTORE`, sort/gather, vector/reduce, and communication primitives;
- pipeline ordering through `Event<SrcOp, DstOp>`, `TSYNC`, and backend-level flag/barrier helpers;
- ping-pong or double-buffer schedules by explicitly choosing non-overlapping addresses and alternating them by loop phase.

Do not confuse these with a high-level runtime allocator. `TASSIGN` binds a tile to an already chosen on-chip address. It does not allocate a queue slot, prove live-range non-overlap, or free a buffer for reuse.

### Buffer ownership boundary

In current manual kernels, two styles coexist:

1. **Hybrid AscendC allocator + PTO instructions**
   - `TQue` / `TBuf` / `TPipe::InitBuffer` allocate or reserve local buffers and manage queue slot lifetime.
   - PTO helpers take the resulting physical address, commonly through `PtoUbBaseAddr(...)`, and issue tile/vector instructions.
   - `LocalTensor` handles must remain when they are still passed to `AllocTensor`, `EnQue`, `DeQue`, or `FreeTensor`.

2. **Pure PTO-Manual buffer plan**
   - the kernel manually partitions UB/L1/L0/FixBuf with byte offsets;
   - tiles are bound to those addresses with `TASSIGN`;
   - ordering is expressed with PTO events or explicit backend barriers;
   - the author must prove address non-overlap, tail bounds, alignment, and reuse timing.

Do not mechanically delete `TQue` or `LocalTensor` just because a PTO tile can bind an address. First decide whether the stage is staying hybrid or becoming pure PTO-Manual.

### Tiling decisions to record before coding

For each stage, write down:

1. **Iteration domain**
   - per-core rows / tokens / experts / blocks;
   - per-loop rows, columns, or K-slices;
   - whether the stage is SPMD or an MPMD role.

2. **Tile validity**
   - static tile capacity;
   - dynamic valid rows/cols for tails;
   - whether GM shape/stride and tile valid region describe the same logical slice.

3. **Local-memory budget**
   - UB vector scratch, input staging, output staging, metadata, quant scale;
   - L1 / L0A / L0B / L0C / FixBuf for cube paths;
   - ping/pong slots if load/compute/store or comm/compute overlap is required;
   - 32B alignment and any backend-specific larger alignment.

4. **Live-range table**
   - each buffer's producer, consumers, last use, and reuse point;
   - which buffers can alias after an event/barrier;
   - which buffers must remain distinct because MTE/V/Cube/comm operations overlap.

### Pipeline primitive selection

Use this decision table:

- use `Event` / `TSYNC` when dependencies are between first-class PTO instructions;
- use backend flag/barrier helpers when wrapping existing AscendC-style helper paths;
- use `TQue` only when keeping AscendC queue-managed buffer lifetime;
- use `TNOTIFY` / `TWAIT` / `TTEST` when readiness must be visible across ranks, blocks, or protocol roles;
- publish readiness only after the data path is visible to the consumer side.

The important split is payload vs readiness:

- payload movement can be `TLOAD/TSTORE`, `TPUT/TGET`, `TSTORE_IMPL<AtomicAdd>`, or collectives;
- readiness is a protocol object, usually a signal, counter, ready matrix, or queue entry;
- a good overlap pipeline keeps those two concepts separate.

### Conversion checklist

When PTO-izing an existing manual kernel:

1. move helper APIs to raw GM pointers, `GlobalTensor` views where needed, and `uint64_t` local addresses;
2. keep queue-owned `LocalTensor` handles until the whole stage has a pure PTO buffer plan;
3. replace implementation internals with tile/global/event primitives one seam at a time;
4. verify the seam by grep, build, and representative runtime cases;
5. only then consider deleting the AscendC allocator layer for that stage.

### Practical rule for `LocalTensor` removal

A `LocalTensor` can usually be removed from helper interfaces when it is only a view used to obtain an address. A `LocalTensor` should remain, or be replaced only by a larger pure-PTO buffer plan, when it is the object passed to `AllocTensor`, `EnQue`, `DeQue`, or `FreeTensor`.

### PTO knowledge points that are easy to miss

- `GlobalTensor` is GM-side metadata; it does not manage UB/L1/L0 lifetime.
- Tile capacity and tile valid region are different facts; tails should normally be expressed through valid rows/cols, not by inventing a new tile type.
- Shape/stride are element-based, while manual address partitioning is byte-based; do not mix the units.
- Tile-side layout (`BLayout` / `SLayout`) and GM-side layout (`Layout`) must be checked independently at every load/store boundary.
- Communication kernels often use PTO at the protocol level: the most important structure may be the ready queue, signal matrix, or summary counter, not the payload primitive alone.
- A3/A5 backend differences are real; validate support against current headers, instruction docs, and the nearest testcase before assuming a primitive is portable.

## Tile layout and memory-layout mental model

This repo uses two different layout vocabularies, and confusing them causes real bugs.

### Tile-side layout

Used by `Tile<...>`:

- `BLayout`
- `SLayout`
- tile alias families such as `TileLeft`, `TileRight`, `TileAcc`

This is about the on-chip representation of a tile.

### GM-side layout

Used by `GlobalTensor<..., Layout::...>`:

- `Layout::ND`
- `Layout::DN`
- `Layout::NZ`
- `Layout::SCALE`
- `Layout::MX_*`

This is about the global-memory view and lowering hint.

### Practical consequence

For future work, always ask two separate questions:

1. What tile type and layout does the compute primitive require?
2. What `GlobalTensor` shape, stride, and layout does the load/store or comm primitive require?

If a path crosses the compute/GM boundary, verify both sides independently.

## PTO-ized kernel patterns

The most useful mental shift for reading this repo is:

- PTO-ization does not mean "replace every line with one PTO instruction family"
- it means "re-express data movement, compute, synchronization, and communication protocol using PTO's tile/global/event model"

In practice, repo kernels tend to fall into three PTO-ization styles:

1. **Compute-centric PTO**
   - the kernel is mostly about `TLOAD/TEXTRACT/TMATMUL/TSTORE`
   - examples: GEMM-like compute kernels
2. **Comm-centric PTO**
   - the kernel is mostly about remote GM movement and signaling
   - examples: `TPUT/TGET/TNOTIFY/TWAIT/TTEST`-driven kernels
3. **Protocol-centric PTO**
   - PTO is used to build a larger producer/consumer protocol
   - the core idea is not just a single primitive, but a chunk queue, ready matrix, summary counter, or owner-local handoff

The last category is especially important for future fused-kernel work.

## Two key kernel case studies

The two most useful project-level case studies for PTO programming style in this repo are:

- [`../kernels/manual/a2a3/allgather_gemm/`](../kernels/manual/a2a3/allgather_gemm/)
- [`../kernels/manual/a2a3/gemm_ar/`](../kernels/manual/a2a3/gemm_ar/)

They are not just examples of "using a few PTO APIs". They show how PTO is used to structure the whole kernel design.

### `allgather_gemm`: chunk streaming, comm drives compute

This project is the clearest example of a streaming communication-first pipeline.

Key files:

- [`../kernels/manual/a2a3/allgather_gemm/allgather_gemm_comm_kernel.cpp`](../kernels/manual/a2a3/allgather_gemm/allgather_gemm_comm_kernel.cpp)
- [`../kernels/manual/a2a3/allgather_gemm/allgather_gemm_compute_kernel.cpp`](../kernels/manual/a2a3/allgather_gemm/allgather_gemm_compute_kernel.cpp)
- [`../kernels/manual/a2a3/allgather_gemm/ready_queue.hpp`](../kernels/manual/a2a3/allgather_gemm/ready_queue.hpp)
- [`../kernels/manual/a2a3/allgather_gemm/README.md`](../kernels/manual/a2a3/allgather_gemm/README.md)

What it teaches:

- communication kernel and compute kernel are launched separately
- communication works at chunk granularity, not whole-tensor granularity
- compute begins as soon as a chunk is known to be ready
- PTO communication and PTO compute are glued together by a device-side ready protocol, not by a host-side global barrier

Important design ideas:

- **comm side**
  - remote data movement is expressed with `pto::comm::TPUT`
  - local and remote buffers are wrapped as dynamic `GlobalTensor<..., Layout::ND>`
  - UB staging uses explicit `Tile<Vec, ...>` plus ping/pong buffering
  - chunk completion is published with `TNOTIFY(AtomicAdd)`
- **compute side**
  - the normal PTO compute pipeline is still `TLOAD -> TEXTRACT -> TMATMUL / TMATMUL_ACC -> TSTORE`
  - compute does not busy-spin over all chunks blindly
  - it first checks a summary counter or ready count, then blocks with `TWAIT` only when necessary
- **protocol side**
  - `ChunkFlagMatrix` is the real heart of the design
  - a chunk-ready flag matrix plus a summary counter turns communication completion into a PTO-visible device-side protocol
  - local data is processed zero-wait first, then remote chunks are consumed in arrival order
  - `ComputeOptimalChunkSize()` shows that chunk size itself is part of the overlap design, not just a build constant

What to learn from it:

- PTO communication is strongest when it is paired with an explicit chunk protocol
- `TWAIT` is more useful as a device-side protocol primitive than as a generic polling replacement
- the best PTO-ized overlap design usually separates:
  - payload transfer
  - ready publication
  - compute consumption

### `gemm_ar`: ready queue, dual-stream overlap, protocol split by role

This project is the clearest example of a dual-stream compute/comm decouple design.

Key files:

- [`../kernels/manual/a2a3/gemm_ar/gemm_compute_kernel.cpp`](../kernels/manual/a2a3/gemm_ar/gemm_compute_kernel.cpp)
- [`../kernels/manual/a2a3/gemm_ar/comm_kernel.cpp`](../kernels/manual/a2a3/gemm_ar/comm_kernel.cpp)
- [`../kernels/manual/a2a3/gemm_ar/ready_queue.hpp`](../kernels/manual/a2a3/gemm_ar/ready_queue.hpp)
- [`../kernels/manual/a2a3/gemm_ar/README.md`](../kernels/manual/a2a3/gemm_ar/README.md)

What it teaches:

- compute and communication are not just different instructions, they are different roles
- compute blocks and comm blocks communicate through a queue protocol
- the comm kernel itself is internally split into RS and AG behavior, but still uses PTO concepts end-to-end

Important design ideas:

- **compute kernel**
  - uses a classic PTO manual compute stack:
    - `TLOAD`
    - `TEXTRACT`
    - `TMATMUL / TMATMUL_ACC`
    - `TSTORE`
  - L1 batching (`stepK`) plus L0 ping/pong lets data movement and cube compute overlap
  - block swizzle is part of the PTO kernel design because it changes L1 reuse, not just host scheduling
  - after a tile is finished, the compute block publishes work through a per-block ready queue
- **queue protocol**
  - queue is single-producer single-consumer
  - the comm block first probes queue progress with `TTEST`
  - it only falls back to `TWAIT` when a queue is not yet ready
  - this is a practical example of using PTO comm primitives as hardware-assisted protocol operations
- **comm kernel**
  - synchronization protocol is PTO-native:
    - `TNOTIFY`
    - `TWAIT`
    - `TTEST`
  - data path is not limited to the `pto::comm::TPUT/TGET` family only
  - it also uses PTO tile/global/store building blocks such as `TLOAD` and `TSTORE_IMPL<AtomicAdd>` for subtile RS execution and `TSTORE_IMPL<AtomicNone>` for AG distribution
  - owner-local `subtile-ready` and `ag-summary` counters turn RS/AG overlap into an explicit device protocol
  - `pipe_barrier(PIPE_ALL) + dsb(DSB_DDR)` appears before readiness publication because visibility and readiness are intentionally separated

That last point is important:

- the repo's "PTO communication kernel" design is not always a pure `TPUT/TGET` wrapper
- sometimes the best PTO-ized data path is:
  - protocol and readiness by `pto::comm::*`
  - tile data movement and reduction by `TLOAD/TSTORE_IMPL/AtomicAdd`

What to learn from it:

- PTO-ization can happen at the protocol level, not only at the instruction-family level
- a good comm kernel often combines:
  - PTO comm primitives for signaling and queue readiness
  - PTO compute/store primitives for the actual subtile data path
- when reading or designing such kernels, separate:
  - who owns readiness
  - who owns data movement
  - who owns accumulation or broadcast semantics

### `allgather_gemm` vs `gemm_ar`

These two kernels are worth studying together because they solve different overlap directions:

- `allgather_gemm`
  - communication arrival drives later compute
  - protocol center = chunk-ready matrix plus summary counter
  - main remote payload primitive is visibly `TPUT`
- `gemm_ar`
  - compute completion drives later communication
  - protocol center = per-block ready queue plus owner-local subtile counters
  - main data path mixes comm signaling with `TLOAD/TSTORE_IMPL` subtile execution

This is one of the most important repo-specific conclusions:

- there is no single "PTO communication kernel template"
- the right PTO-ized protocol depends on whether compute waits on communication, communication waits on compute, or both are interleaved through a shared protocol

## Tests as the real primitive cookbook

The `tests/` tree is not just for regression. It is also the best primitive cookbook in the repo.

Use this rule:

- if you want the smallest working example of a PTO primitive, start in `tests/`
- if you want to see how multiple PTO primitives compose into a performance kernel, then move to `kernels/`

### Communication primitive examples in `tests`

The clearest communication examples live here:

- [`../tests/npu/a2a3/comm/st/testcase/`](../tests/npu/a2a3/comm/st/testcase/)
- [`../tests/npu/a5/comm/st/testcase/`](../tests/npu/a5/comm/st/testcase/)

Main primitive-to-example mappings:

- `TPUT`
  - [`../tests/npu/a2a3/comm/st/testcase/tput/`](../tests/npu/a2a3/comm/st/testcase/tput/)
  - [`../tests/npu/a5/comm/st/testcase/tput/`](../tests/npu/a5/comm/st/testcase/tput/)
- `TGET`
  - [`../tests/npu/a2a3/comm/st/testcase/tget/`](../tests/npu/a2a3/comm/st/testcase/tget/)
  - [`../tests/npu/a5/comm/st/testcase/tget/`](../tests/npu/a5/comm/st/testcase/tget/)
- `TNOTIFY`
  - [`../tests/npu/a2a3/comm/st/testcase/tnotify/`](../tests/npu/a2a3/comm/st/testcase/tnotify/)
  - [`../tests/npu/a5/comm/st/testcase/tnotify/`](../tests/npu/a5/comm/st/testcase/tnotify/)
- `TWAIT`
  - [`../tests/npu/a2a3/comm/st/testcase/twait/`](../tests/npu/a2a3/comm/st/testcase/twait/)
  - [`../tests/npu/a5/comm/st/testcase/twait/`](../tests/npu/a5/comm/st/testcase/twait/)
- `TTEST`
  - [`../tests/npu/a2a3/comm/st/testcase/ttest/`](../tests/npu/a2a3/comm/st/testcase/ttest/)
  - [`../tests/npu/a5/comm/st/testcase/ttest/`](../tests/npu/a5/comm/st/testcase/ttest/)
- collectives
  - `TGATHER`: [`../tests/npu/a2a3/comm/st/testcase/tgather/`](../tests/npu/a2a3/comm/st/testcase/tgather/), [`../tests/npu/a5/comm/st/testcase/tgather/`](../tests/npu/a5/comm/st/testcase/tgather/)
  - `TSCATTER`: [`../tests/npu/a2a3/comm/st/testcase/tscatter/`](../tests/npu/a2a3/comm/st/testcase/tscatter/), [`../tests/npu/a5/comm/st/testcase/tscatter/`](../tests/npu/a5/comm/st/testcase/tscatter/)
  - `TBROADCAST`: [`../tests/npu/a2a3/comm/st/testcase/tbroadcast/`](../tests/npu/a2a3/comm/st/testcase/tbroadcast/), [`../tests/npu/a5/comm/st/testcase/tbroadcast/`](../tests/npu/a5/comm/st/testcase/tbroadcast/)
  - `TREDUCE`: [`../tests/npu/a2a3/comm/st/testcase/treduce/`](../tests/npu/a2a3/comm/st/testcase/treduce/), [`../tests/npu/a5/comm/st/testcase/treduce/`](../tests/npu/a5/comm/st/testcase/treduce/)

What these tests are especially good for:

- verifying rank topology assumptions
- learning what a legal `GlobalTensor` comm view looks like
- seeing when ping-pong double buffering is needed
- seeing how `AtomicAdd` and `WaitCmp::*` are actually used

### Compute primitive examples in `tests`

The main compute examples live here:

- [`../tests/npu/a2a3/src/st/testcase/`](../tests/npu/a2a3/src/st/testcase/)
- [`../tests/npu/a5/src/st/testcase/`](../tests/npu/a5/src/st/testcase/)

The most useful mapping is by question type:

- "I want the smallest read-compute-write example"
  - `tadd`
- "I want load/store and layout behavior"
  - `tload`
  - `tstore`
  - `tload_gm2mat`
  - `tstore_mat2gm`
  - `tload_mx_NZ`
  - `tload_mx_ND_DN`
  - `tload_shape2d`
- "I want matmul and accumulator behavior"
  - `tmatmul`
  - `tmatmul_mx`
  - `tstore_acc2gm`
- "I want layout conversion and movement between tile roles"
  - `tmov`
  - `tmov_acc2mat`
  - `tmov_acc2vec`
  - `tmov_mx`
  - `textract`
  - `textract_compact`
  - `ttrans`
- "I want quant/dequant examples"
  - `tquant`
  - `tdequant`
- "I want row/column reduction and expand examples"
  - `trowmax`
  - `trowsum`
  - `trowexpand*`
  - `tcolmax`
  - `tcolsum`
  - `tcolexpand*`
- "I want fused or composed vector examples"
  - `tmuls_trowsum`
  - `trowsum_trowexpand`
  - `trowexpand_trowsum`
  - `tmul_tadds`
  - `tadd_tdiv`
  - `tsub_texp`
- "I want cube/vec FIFO cooperation examples"
  - `tpushpop_cv`
  - `tpushpop_vc`

### Which test families are most educational

If I had to recommend a short reading ladder for future PTO design work, it would be:

1. `tadd`
   - minimal PTO dataflow
2. `tload` / `tstore`
   - GM/tile contract
3. `tload_gm2mat` and `tload_mx_NZ`
   - layout-sensitive loads
4. `tmatmul` and `tstore_acc2gm`
   - cube path and accumulator writeback
5. `tquant` / `tdequant`
   - format-sensitive path
6. `trow*` / `tcol*`
   - reduction-expansion vocabulary
7. `tput` / `twait`
   - communication payload and signal programming

### Practical repo rule for future work

When you need to use a PTO primitive in a new kernel:

1. first locate the public API in `include/pto/**`
2. then read the matching `docs/isa/**`
3. then find the smallest matching testcase in `tests/**`
4. only after that copy the pattern into a manual kernel or fused project

This is usually faster and safer than jumping directly into a complex kernel.

## Where to study real code by problem type

### Minimal read-compute-write

Best entry tests:

- [`../tests/npu/a2a3/src/st/testcase/tadd/`](../tests/npu/a2a3/src/st/testcase/tadd/)
- [`../tests/npu/a5/src/st/testcase/tadd/`](../tests/npu/a5/src/st/testcase/tadd/)

### Layout and load/store behavior

Best entry tests:

- [`../tests/npu/a2a3/src/st/testcase/tload_gm2mat/`](../tests/npu/a2a3/src/st/testcase/tload_gm2mat/)
- [`../tests/npu/a2a3/src/st/testcase/tstore_mat2gm/`](../tests/npu/a2a3/src/st/testcase/tstore_mat2gm/)
- [`../tests/npu/a5/src/st/testcase/tload_mx_NZ/`](../tests/npu/a5/src/st/testcase/tload_mx_NZ/)

### Matmul pipeline

Best entry points:

- [`../kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp`](../kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp)
- [`../tests/npu/a5/src/st/testcase/tmatmul/`](../tests/npu/a5/src/st/testcase/tmatmul/)

### Vector/reduce pipeline

Best entry point:

- [`../kernels/manual/a2a3/topk/topk_kernel.cpp`](../kernels/manual/a2a3/topk/topk_kernel.cpp)

### Mixed cube + vector stages

Best entry points:

- [`../kernels/manual/common/flash_atten/fa_performance_kernel.cpp`](../kernels/manual/common/flash_atten/fa_performance_kernel.cpp)
- [`../kernels/manual/common/flash_atten/pto_macro_matmul.hpp`](../kernels/manual/common/flash_atten/pto_macro_matmul.hpp)
- [`../kernels/manual/common/flash_atten/pto_macro_fa_softmax.hpp`](../kernels/manual/common/flash_atten/pto_macro_fa_softmax.hpp)

### Communication and compute/comm decoupling

Best entry points:

- [`../kernels/manual/a2a3/gemm_ar/`](../kernels/manual/a2a3/gemm_ar/)
- [`../kernels/manual/a2a3/allgather_gemm/`](../kernels/manual/a2a3/allgather_gemm/)
- [`../tests/run_comm_test.sh`](../tests/run_comm_test.sh)

### Complex fused workload that already mixes PTO with higher-level kernel logic

Useful reference:

- [`../kernels/manual/a2a3/dispatch_combine_moe/`](../kernels/manual/a2a3/dispatch_combine_moe/)

Why it matters:

- it is a realistic example of PTO being integrated into a larger fused kernel project
- it includes PTO communication wrappers, PTO tile/global layout bridging, and partial compute migration
- it is a good place to study how PTO is consumed by a real workload, not just isolated toy examples

## Where to start for future tasks

If the next task is mostly about one of these areas, start here:

- communication primitive design
  - `include/pto/comm/pto_comm_inst.hpp`
  - `include/pto/comm/comm_types.hpp`
  - `docs/isa/comm/*`
  - `tests/npu/a2a3/comm/st/testcase/*`
  - `tests/npu/a5/comm/st/testcase/*`
  - `kernels/manual/a2a3/gemm_ar/*`
  - `kernels/manual/a2a3/allgather_gemm/*`
- tile/layout reasoning
  - `include/pto/common/pto_tile.hpp`
  - `include/pto/common/memory.hpp`
  - `include/pto/common/type.hpp`
  - `docs/coding/Tile.md`
  - `docs/coding/GlobalTensor.md`
  - `tests/npu/a2a3/src/st/testcase/tload_gm2mat/*`
  - `tests/npu/a5/src/st/testcase/tload_mx_NZ/*`
  - `tests/npu/a5/src/st/testcase/tstore_acc2gm/*`
- compute primitive replacement or kernel design
  - `include/pto/common/pto_instr.hpp`
  - `docs/isa/TLOAD.md`
  - `docs/isa/TSTORE.md`
  - `docs/isa/TSTORE_FP.md`
  - `docs/isa/TMATMUL*.md`
  - `tests/npu/a2a3/src/st/testcase/tmatmul/*`
  - `tests/npu/a5/src/st/testcase/tmatmul/*`
  - `tests/npu/a5/src/st/testcase/tquant/*`
  - `kernels/manual/a2a3/gemm_performance/*`
  - `kernels/manual/common/flash_atten/*`
  - `kernels/manual/a2a3/dispatch_combine_moe/*`
- broad PTO routing
  - `.ai-knowledge/router/ROUTER.md`
  - `.ai-knowledge/cards/tables/pto-primitives-map.md`
  - `.ai-knowledge/cards/tables/pto-kernel-pattern-map.md`
  - `.ai-knowledge/cards/tables/pto-test-entry-map.md`

## Validation and execution entry points

For future work, keep these entry points in mind:

- CPU simulator
  - [`../tests/run_cpu.py`](../tests/run_cpu.py)
  - getting started doc: [`getting-started.md`](getting-started.md)
- NPU ST
  - [`../tests/script/run_st.py`](../tests/script/run_st.py)
  - overview: [`../tests/README_zh.md`](../tests/README_zh.md)
- communication ST
  - [`../tests/run_comm_test.sh`](../tests/run_comm_test.sh)

Recommended workflow:

1. learn or prototype on CPU when possible
2. validate semantics with the closest NPU testcase
3. study the nearest manual kernel before changing a performance-sensitive path
4. for overlap or protocol work, study the nearest ready-queue or ready-counter kernel before inventing a new synchronization shape

## Short takeaways

- PTO is not just an instruction list; it is a tile/global-tensor/event programming model.
- `include/pto/common/` is the center of gravity for compute-side understanding.
- `include/pto/comm/` is the center of gravity for communication-side understanding.
- tile layout and global layout are different abstractions and must be reasoned about separately.
- `Compact` tile aliases, dynamic valid region, and explicit shape/stride are recurring sharp edges.
- kernel PTO-ization in this repo often happens at the protocol level, not only at the instruction-family level.
- `allgather_gemm` and `gemm_ar` are the two best overlap case studies in the current repo.
- the best way to understand PTO in this repo is to move back and forth between:
  - public API headers
  - instruction docs
  - manual kernels
  - focused tests
  - `.ai-knowledge` route maps
