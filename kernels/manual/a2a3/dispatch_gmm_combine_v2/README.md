# dispatch_combine_moe_v2

> Last refreshed: 2026-05-24

## Scope

`dispatch_combine_moe_v2` is the current A2/A3 manual kernel workspace for turning the fused dispatch → FFN → combine flow into a PTO-oriented programming style.

The practical target is not to remove every `__gm__` pointer. `__gm__` is allowed at ABI, protocol, workspace-layout, and `pto::GlobalTensor` view-construction boundaries. The main algorithmic chain should move away from naked GM scalar loops and toward explicit PTO stages.

## Current implementation status

### Host pipeline

Batch 13 has switched the active host path to the方案B two-kernel ready-queue pipeline:

```text
dispatch_stream: launchCommVecQueuePipeline(...)      // dav-c220-vec, dispatch/dequant/SwiGLU/combine/restore
compute_stream:  launchComputeCubeQueuePipeline(...)  // dav-c220-cube, GMM1/GMM2
```

The two kernels communicate through four GM queue sets:

```text
dispatchToGmm1Queue -> gmm1ToSwiGluQueue -> swiGluToGmm2Queue -> gmm2ToCombineQueue
```

`main.cpp` allocates and resets the queue workspaces per iteration, launches the two pipeline kernels, and synchronizes both streams. The old split-stage launches remain declared and built, but are no longer the active host main chain.

### CMake targets

Current build has five device shared libraries plus one host executable:

```text
dispatch_combine_moe_v2_dispatch_combine_kernel   -> op_kernel/dispatch_ffn_combine.cpp, dav-c220-vec
dispatch_combine_moe_v2_compute_vec_kernel        -> op_kernel/compute_vec/compute_vec_kernel.cpp, dav-c220-vec
dispatch_combine_moe_v2_comm_vec_queue_kernel     -> op_kernel/compute_vec/comm_vec_queue_kernel.cpp, dav-c220-vec
dispatch_combine_moe_v2_compute_cube_kernel       -> op_kernel/compute/gmm_cube_kernel.cpp, dav-c220-cube
dispatch_combine_moe_v2_compute_cube_queue_kernel -> op_kernel/compute/compute_cube_queue_kernel.cpp, dav-c220-cube
dispatch_combine_moe_v2                           -> host executable
```

`op_kernel/compute_kernel.cpp` is no longer part of the active CMake build and is retained only as an inactive pointer to the split stage sources.

### Dispatch stage

Dispatch uses PTO communication for remote pulls and Batch 8C publishes range completion from one ordered producer block:

```text
remote path: TGET
local self-copy: TLOAD/TSTORE with Batch 9 full-chunk ping-pong
range completion: one dispatch block loops ranges and publishes per-range done signals for dequant tile wait
```

The local self-copy path in `op_kernel/dispatch/dispatch_pull.hpp` now uses PTO `TLOAD/TSTORE`.

### Compute stages

Current compute is split into four launch boundaries:

```text
dispatch_combine_moe_v2_dequant
  WaitExpertSafeRows on Batch 8A host-preseeded dequant sentinel when requiredSafeRows != 0
  WaitDispatchTileReady on Batch 8C per tile source segments / dispatch done signals
  PTO dequant TLOAD/TCVT/TROWEXPANDMUL -> row-wise TSTORE dequantOut[tile, 8, 2]

 dispatch_combine_moe_v2_gmm1
  dcci dequantOut tile
  RunGmm1Tile8

 dispatch_combine_moe_v2_swiglu
  dcci gmm1Out rows
  PTO SwiGLU TLOAD/TMUL/TMULS -> row-wise TSTORE swigluOut[tile, 8, 2]

 dispatch_combine_moe_v2_gmm2
  dcci swigluOut rows
  RunGmm2TileRowsN
  direct TSTORE to final gmm2Out rows
  PublishGmm2Ready
```

Current PTO coverage:

- GMM1/GMM2 are routed through `op_kernel/substrate/pto_mmad_shell.hpp` and use `TLOAD / TMOV / TMATMUL / TSTORE`.
- Dequant is now a Vec-target PTO stage using padded Vec tiles, `TLOAD/TCVT/TROWEXPAND/TROWEXPANDMUL`, and row-wise PTO `TSTORE` into compact `[8,2]` GM layout.
- SwiGLU is now a Vec-target PTO stage using padded Vec tiles, `TLOAD/TMUL/TMULS`, and row-wise PTO `TSTORE` into compact `[8,2]` GM layout.
- GMM2 now stores each N tile directly into final `gmm2Out`; `localPartialDev` no longer aliases packed GMM2 scratch in the active path.

Important constraint: do not put vector PTO primitives such as `TCVT`, `TROWEXPANDMUL`, `TMUL`, or `TMULS` into cube-target code. Batch 2 and Batch 4 replaced the Vec-stage scalar bodies with Vec PTO primitives in the Vec target.

### Combine and restore stages

Combine uses PTO communication for remote pushes:

```text
remote path: TPUT
local self-copy: TLOAD/TSTORE with Batch 9 full-chunk ping-pong
```

Restore/reduction already uses vector PTO-style operations:

```text
TLOAD -> TMULS -> TADD/TMOV -> TSTORE
```

The local self-copy path in `op_kernel/combine/combine_push.hpp` now uses PTO `TLOAD/TSTORE`.

## Target PTO architecture

The active architecture is now the two-kernel queue pipeline:

```text
comm_vec_queue_kernel dav-c220-vec:
  dispatch remote/local copy -> dequant -> poll GMM1 ready -> SwiGLU -> poll GMM2 ready -> combine/restore

compute_cube_queue_kernel dav-c220-cube:
  poll dispatch/dequant ready -> GMM1 -> poll SwiGLU ready -> GMM2
```

Batches 1-9 replaced the stage bodies, GMM2 scratch path, and local self-copy paths with PTO primitives. Batches 10-13 connected those bodies with GM ready queues and switched the active host main chain to two queue pipeline launches. Batches 14-19 then tightened the overlap granularity and routing semantics: dispatch/dequant no longer waits for a whole dispatch pre-stage, combine consumes GMM2 ready windows, cube GMM producers can split by producer queue, queue consumers poll round-robin, expert/group metadata is carried through queue entries, and runtime routing/capacity/dropPad counters are visible in profile metadata.

## Key files

```text
kernels/manual/a2a3/dispatch_combine_moe_v2/
  README.md                         // this status entry
  design.md                         // current design and staged plan
  task.md                           // live progress checklist
  kernel_launch.hpp                 // stage launch ABI declarations
  main.cpp                          // host orchestration and black-box matrix driver
  CMakeLists.txt                    // split device targets and host executable
  run.sh                            // build + black-box run wrapper
  op_kernel/dispatch_ffn_combine.cpp
  op_kernel/compute_vec/compute_vec_kernel.cpp
  op_kernel/compute/gmm_cube_kernel.cpp
  op_kernel/compute_kernel.cpp      // inactive Batch 1 split pointer, not built
  op_kernel/compute/compute_chain.hpp
  op_kernel/substrate/pto_mmad_shell.hpp
  op_kernel/dispatch/dispatch_pull.hpp
  op_kernel/combine/combine_push.hpp
  op_kernel/output/unpermute_reduce.hpp
```

`desing.md` is a compatibility pointer only. Use `design.md` as the canonical design document.

## Development workflow

1. Read `README.md`, `design.md`, and `task.md` before editing.
2. Keep `design.md` aligned with any stage-design changes.
3. Keep `task.md` aligned with actual progress.
4. Follow coding → review. Do not write white-box tests just for convenience.
5. Prefer black-box validation through `run.sh --matrix`.
6. Watch for hangs; do not keep waiting indefinitely on stuck NPU/MPI runs.

## Build and black-box validation

Use the project environment convention first:

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
export PATH=/home/ntlab/miniconda3/envs/ltr_pto/bin:$PATH
export LD_LIBRARY_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib:$LD_LIBRARY_PATH
export MPI_LIB_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib/libmpi.so
```

Then run:

```bash
kernels/manual/a2a3/dispatch_combine_moe_v2/run.sh --matrix
```

`run.sh --matrix` currently builds the five device kernel libraries and host executable, then runs three `mpirun -n 2` cases:

```text
default shape
--m 16 --max-output-size 32
--m 4097 --max-output-size 32
```

Latest Batch 20 verification on A3 completed with rank 0/1 `perf-main full-pipeline accuracy PASS` for all three matrix cases after the Batch 19 routing/capacity fixes.

Because the current machine is A3, A3 cases can run directly. A5-specific cases should be treated as compile-only unless explicitly moved to an A5 environment.

## Current gaps

- The public A3 matrix passes with the two-kernel queue pipeline active.
- Queue entry payload visibility is part of correctness: producer writes entry fields, flushes the entry cacheline, then publishes count.
- Per-iteration reset must clear dispatch done, combine done, summary, compute, combine, and queue workspaces; otherwise later warmup/measure iterations can consume stale readiness.
- The current Batch 17 cube multi-producer assignment is static by tile id. Further performance work should measure whether Vec-side dispatch/dequant/SwiGLU remains the bottleneck before adding new producer classes.

## Next recommended batch

Batch 20 is the current closure batch: keep the final code review focused on active two-kernel queue pipeline integrity, target separation, per-producer queue ownership, owner completion, docs alignment, and A3 matrix/performance evidence. The non-PTO reference projects remain algorithm references only; do not copy their AscendC/Catlass programming style.
