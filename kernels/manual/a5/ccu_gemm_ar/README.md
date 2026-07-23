# CCU GEMM + AllReduce Fusion Example

## Overview

This example shows how to implement a multi-rank GEMM + AllReduce fused operator on A5 with PTO and **CCU**: the Compute Stream runs AIC GEMM, AIV only handles readiness and CKE scheduling, and the data path uses a **persistent CCU** kernel for owner-scoped Pull Reduce + Push Broadcast (logically equivalent to RS + AG). The host registers and launches microcode via `HcommCcuKernelRegister*` / `HcommCcuKernelLaunch`.

Compared with the sibling demo `gemm_ar` (AIV `TPUT` / AtomicAdd data path), the communication volume is the same order of magnitude, about `2*(P-1)/P * D`, but the payload movement runs on CCU.

## Supported AI Processors

- Ascend950PR

## Directory Layout

```text
kernels/manual/a5/ccu_gemm_ar/
├── CMakeLists.txt                      # Build config (3 targets: cube / vec / host)
├── run.sh                              # One-click build + run (auto HCCL_BUFFSIZE, MPI discovery)
├── config.h                            # Global config (shape, tiles, groups, signal layout)
├── main.cpp                            # Entry: MPI/HCCL, CCU register, timing, verify
├── compute_kernel.cpp                  # GEMM compute kernel (Cube, `dav-c310-cube`)
├── scheduler_kernel.cpp                # AIV progress / gate / unpack (`dav-c310-vec`)
├── ccu_reduce_broadcast_kernel.hpp     # Host-side fused RS+AG CCU synth + register
├── kernel_launchers.h                  # Host kernel launcher declarations
├── comm_context.h                      # CommDeviceContext (window addresses, etc.)
├── ready_queue.hpp                     # Tile queue for USE_PROGRESS=0 escape path
└── comm_mpi.h                          # MPI dynamic loading wrapper
```

Shared headers: `include/pto/comm/async/ccu/ccu_gate_registry.hpp`, `ccu_loopgroup_utils.hpp`.

## Operator Description

### Functionality

This example implements multi-rank GEMM + AllReduce:

$$
C_{final} = \sum_{i=0}^{nranks-1} A_i \times B
$$

Where:

- `A_i` is `M x K` and is private to each rank
- `B` is `K x N` and shared by all ranks
- `C_i` is the local GEMM result of shape `M x N`
- `C_final` is the final `M x N` output after AllReduce

The default reference configuration in `config.h` is `M=5416, K=6144, N=1408`, aligned with `gemm_ar`.

### Specification

| Item | Value |
| --- | --- |
| OpType | `GEMM + AllReduce` (CCU data path) |
| Input | `A_i`: `M x K`, `float16`, `ND` (per-rank); `B`: `K x N`, `float16`, `DN` (shared) |
| Output | `C_final`: `M x N`, `float16`, `ND` (AllReduce result) |
| Compute kernel | `CcuGemmArComputeKernel` (Cube, `dav-c310-cube`) |
| Scheduler kernel | `ccu_gemm_ar_progress_kernel` (default) / `ccu_gemm_ar_scheduler_kernel` (`USE_PROGRESS=0`) |
| CCU data path | fused Pull Reduce + Push Broadcast (`ccu_reduce_broadcast_kernel.hpp`) |

## Optimization Notes

This example uses Ascend950PR as the validation platform. Cube (AIC) and Vector (AIV) are physically separate, which enables dual-stream compute/comm overlap; the communication payload is moved by CCU, and AIV does not perform RS/AG data copies.

> Prefer CANN `platform_config` for core counts. For example on `950PR_958b`:
>
> - `cube_core_cnt=32` (Cube / AIC parallelism)
> - `vector_core_cnt=64` (Vector / AIV parallelism)

- **Dual-stream overlap**: Compute Stream runs AIC GEMM; AIV Stream runs progress/gate; CCU Stream runs fused RS+AG. After each tile/group becomes ready, a CKE fires CCU work that overlaps with later compute.
- **Logical RS + AG, fused execution**: one CCU mission per owner group — Pull Reduce to the owner, then Push Broadcast; volume about `2*(P-1)/P * D`.
- **Owner-scoped packed layout**: `owner = tile % nranks`, packed owner-contiguously so CCU can issue fixed-length group WQEs; residual groups pad each owner shard to a multiple of `comm-group-tiles`.
- **Progress sync (default)**: AIC `AtomicAdd`s into local-window `groupDone[flat]`; peer AIV polls then `TNOTIFY(owner groupReady)`; owner waits `>= P-1` then `TriggerProgressCke` (**depth-1** to avoid lost edges).
- **CKE lifetime**: `CompletedEvent` is published only after `RegisterEnd`/Translate so physical `DieId`/`Id` are valid.
- **AG stagger**: CCU Broadcast peer write order rotates by `(rankId + group) % (P-1)`.
- **Block Swizzle + L1/L0 double buffering**: same compute-side pattern as `gemm_ar` (zigzag tiles, `stepK=4`, L0 ping/pong).
- **Per-tile store fence**: after `TSTORE`, `pipe_barrier(PIPE_FIX) + dsb`, then signal `groupDone`.
- **Stable constraint**: `CCU_MISSION_PARALLEL=1` (one gate descriptor per rank).

## Tiling Parameters

| Parameter | Value |
| --- | --- |
| `M` (raw) | 5416 |
| `K` | 6144 |
| `N` (raw) | 1408 |
| `M` (padded) | 5504 |
| `N` (padded) | 1536 |
| `baseM` | 128 |
| `baseK` | 64 |
| `baseN` | 256 |
| `stepKa` / `stepKb` | 4 |
| `commSubM` | 128 (`== baseM`; current path requires subtile=1) |
| `commGroupTiles` | default 16 (prefer 8 on 4 ranks) |
| Number of tiles | 258 (`43 x 6`) |
| `COMPUTE_BLOCK_NUM` | 24 (override with `--compute-blocks`) |
| `COMM_BLOCK_NUM` | 24 |
| `CCU_MISSION_PARALLEL` | 1 |

## Overall Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│  Compute Stream (AIC)     AIV Stream (progress)        CCU Stream            │
│                                                                              │
│  CcuGemmArComputeKernel:  ccu_gemm_ar_progress_kernel:  fused RS+AG:         │
│  ┌────────────────────┐   ┌─────────────────────────┐  ┌──────────────────┐  │
│  │ for each tile:     │   │ poll groupDone          │  │ WaitEvent(gate)  │  │
│  │   K-loop → TSTORE  │──►│ TNOTIFY owner groupReady│  │ per group:       │  │
│  │   PIPE_FIX + dsb   │   │ owner: wait >= P-1      │──►│   Pull Reduce    │  │
│  │   AtomicAdd        │   │ TriggerProgressCke      │  │   Push Broadcast │  │
│  │     groupDone[flat]│   │ (depth-1 backpressure)  │  │ itemsDone++      │  │
│  └────────────────────┘   └─────────────────────────┘  └──────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Compute Kernel Details

Each AIC owns a subset of tiles assigned by `block_idx`. For each tile:

1. **Block Swizzle**: zigzag traversal (odd rows reversed) to improve L1 reuse of `B`.
2. **K-loop**: batched `TLOAD` into L1 with `stepKa=4`, `TEXTRACT` into L0, then `TMATMUL` / `TMATMUL_ACC`.
3. **TSTORE**: L0C FP32 is cast to FP16 by FixPipe and written to **owner-packed** `gemm_output`.
4. **`pipe_barrier(PIPE_FIX) + dsb(DSB_DDR)`**: ensure the store is visible before signaling.
5. **Progress path**: `AtomicAdd(+1)` into this rank's window `groupDone[flat(owner,g)]`.  
   **Escape (`USE_PROGRESS=0`)**: `PerBlockQueueEnqueueFast` into the readyQueue.

## Communication Path Details

### AIV Progress (default)

1. Each rank's AIV polls local `groupDone[flat]`; when it reaches `CcuOwnerGroupTilesInGroup`, it `TNOTIFY(+1)`s the **owner** `groupReady[flat]`.
2. Owner AIV, per flat group: after `TTEST(groupReady >= P-1)` and depth-1 backpressure allow it, `TriggerProgressCke` (ping-pong slots A/B).
3. Host `PrepareCcu`: launch persistent CCU → poke gate CKE; CCU waits on `WaitEvent(gate)`, then enters `CCU_WHILE` and consumes groups on progress CKEs.

### CCU fused RS + AG

For each ready group (contiguous owner-local tiles):

1. **Pull Reduce**: `Read` each peer's packed `gemm_output` into MS, `LocalReduce`, write into the owner's `reduced_output` shard.
2. **Push Broadcast**: `Write` to each rank's `reduced_output` at the same offset, with staggered peer order.
3. Increment `itemsDone` so AIV can apply backpressure before firing the next group.

### Sequential baseline

Compute finishes → `hostBarrier` → **one** one-shot fused RS→AG over the real owner footprint (not group-strided), without AIV progress.

Note: the reported Sequential `comm` wall includes `HcclHostBarrier` plus CCU; newer builds also print a barrier / `ccu_event` breakdown.

### Escape: `USE_PROGRESS=0`

Uses `readyQueue` + multi-block scheduler. Heavier; for comparison only.

## Memory Layout and HCCL Window

CCU payloads (`gemm_output` / `reduced_output`) and final `row_output` live in a high-bandwidth HBM block (with token for CCU access). Cross-rank sync counters live in the HCCL window `signal_matrix`.

| Buffer | Size | Location | Why |
| --- | --- | --- | --- |
| `gemm_output` (packed) | owner-padded × tile bytes | **HBM + token** | CCU Pull source |
| `reduced_output` (packed) | same | **HBM + token** | CCU Reduce dst / Broadcast src+dst |
| `row_output` | `M x N x 2B` | **HBM block** | unpacked output for verify |
| `signal_matrix` | `G_SIGNAL_MATRIX_SLOTS x 4B` | **HCCL window** | `groupDone` / `groupReady` (and escape `subtile_ready`) |
| `src0_dev`, `src1_dev` | input matrices | **aclrtMalloc** | local-only |

`run.sh` estimates and raises `HCCL_BUFFSIZE` from the padded footprint (about `3 x packed + 128MB` margin).

## Measured Performance (reference)

Measured on 2-rank Ascend950PR with `M=5416, K=6144, N=1408` (padded `5504x1536`), `258 tiles (43x6)`, `compute_blocks=32`, `ccu_group_tiles=16`. Each rank computes full GEMM `C_i = A_i x B`; AllReduce sums the two `C_i`. `comm_data=0.016 GB/rank`.

| Metric | Value |
| --- | --- |
| Compute-only | `316.1 us` (`296480 GFLOPS`) |
| Sequential | `815.0 us` (compute `315.7 us` + one-shot fused RS→AG `499.3 us @ 31.5 GB/s`) |
| Pipelined | **`567.1 us`** (compute done `303.4 us`, comm done `566.6 us @ 27.8 GB/s`) |
| Speedup | `1.437x` |
| Time saved | `247.8 us` (`30.4%`) |
| Overlap eff | `75.6%` |
| Throughput | `330443 GFLOPS` (total) |

### What these numbers mean

- **Compute-only**: pure GEMM with no communication. Here `316.1 us` → `296480 GFLOPS`.
- **Sequential**: compute then one CCU one-shot, no overlap. Here `815.0 us` with compute `315.7 us` and comm `499.3 us`.
- **Pipelined**: AIC / AIV / CCU overlapped end-to-end. Here `567.1 us`, `1.437x` vs Sequential; `compute done = 303.4 us`.
- **Speedup**: Sequential / Pipelined.
- **Time saved**: wall time saved vs the serial path. Here `247.8 us` (~`30.4%`).
- **Overlap eff**: time saved by overlap as a percentage of the shorter serial phase. Pipelined `comm done` is wall-to-CCU-finish (includes overlap), so it is not pure CCU latency.

## Build and Run

1. Set up the Ascend CANN environment (**cann-9.2** required):

```bash
export ASCEND_CANN_PATH=/usr/local/Ascend/cann-<version>/set_env.sh
source "${ASCEND_CANN_PATH}"
```

2. Run the 2-rank example:

```bash
cd ${git_clone_path}/kernels/manual/a5/ccu_gemm_ar
./run.sh -r npu -v Ascend950PR_958b -n 2 -d 2 --compute-blocks 32
```

3. Choose the first device id:

```bash
FIRST_DEVICE=0 ./run.sh -r npu -v Ascend950PR_958b -n 2 -d 2 --compute-blocks 32
```

4. 4 ranks (prefer `comm-group-tiles=8`):

```bash
FIRST_DEVICE=0 ./run.sh -r npu -v Ascend950PR_958b -n 4 -d 4 \
  --compute-blocks 32 --comm-group-tiles 8
```

On success:

```text
CCU GEMM AllReduce demo completed successfully.
```

### Environment Variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `ASCEND_HOME_PATH` | CANN root (required by cmake) | set by `set_env.sh` |
| `ASCEND_CANN_PATH` | Full path to CANN `set_env.sh` | set for your install |
| `MPI_SEARCH_DIRS` | MPI `bin/` search paths | common mpich locations |
| `MPI_LIB_PATH` | `libmpi.so` | set by `run.sh` |
| `HCCL_BUFFSIZE` | HCCL window size (MB) | auto-raised by `run.sh` from M/N |
| `HCCL_CCU_CUSTOM_OP_MODE` | custom CCU kernel mode | set to `1` by `run.sh` |
| `FIRST_DEVICE` | first NPU device id | `0` |
| `USE_PROGRESS` | `1` progress path / `0` readyQueue | `1` |
| `CCU_MISSION_PARALLEL` | CCU mission parallelism | must be `1` |
| `HCOMM_PKG_INC` | optional internal hcomm pkg_inc | auto-probed by cmake |

## Changing Matrix Dimensions

Edit `CONFIG_G_M` / `CONFIG_G_K` / `CONFIG_G_N` in `config.h`, or pass env vars:

```bash
G_M=8192 G_K=8192 G_N=2048 ./run.sh -r npu -v Ascend950PR_958b -n 2 -d 2
```

Constraints:

- `K` must be divisible by `G_BASE_K x stepKa` (default `64 x 4 = 256`)
- `M` / `N` are padded to `baseM` / `baseN`
- `CONFIG_COMM_SUB_M == G_BASE_M`
- `CCU_MISSION_PARALLEL=1`

## Build System

- **Compiler**: bisheng (bundled with CANN)
- **Cube kernel**: `--cce-aicore-arch=dav-c310-cube`
- **Vec kernel**: `--cce-aicore-arch=dav-c310-vec`
- **Host**: `-xc++`, links `runtime`, `ascendcl`, `hccl`, `hcomm`, `tiling_api`, etc.
- **ABI**: host `-D_GLIBCXX_USE_CXX11_ABI=0` (match hcomm/hccl)
- pto-isa `include/` must come before CANN bundled headers
