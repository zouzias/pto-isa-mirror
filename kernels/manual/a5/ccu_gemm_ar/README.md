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
├── host_comm.hpp                       # Host communication stack (HCCL window + CCU channel/CKE/register/launch)
├── main.cpp                            # Demo orchestration: MPI, bench, verify, data I/O
├── compute_kernel.cpp                  # Device compute (AIC Cube, `dav-c310-cube`)
├── progress_kernel.cpp                 # Device communication control (AIV progress / seq peer-sync / gate / unpack)
├── ccu_reduce_broadcast_kernel.hpp     # CCU microcode synthesis (host-authored, CCU-executed)
├── kernel_launchers.h                  # Device kernel launcher declarations
├── comm_context.h                      # CommDeviceContext (window addresses, etc.)
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
| Progress kernel | `ccu_gemm_ar_progress_kernel` (Vector, `dav-c310-vec`) |
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
- **Progress sync**: AIC `AtomicAdd`s into local-window `groupDone[flat]`; peer AIV polls then `TNOTIFY(owner groupReady)`; owner waits `>= P-1` and the previous group to retire, then `TriggerProgressCke` (single progress CKE, one-at-a-time).
- **CKE lifetime**: seq one-shot and pipelined register in **one** `RegisterStart/End` batch; after Translate they Publish to `seqGate` / `gate` (+ progress) so physical CKEs are not reused across paths.
- **AG stagger**: CCU Broadcast peer write order rotates by `(rankId + group) % (P-1)`.
- **Block Swizzle + L1/L0 double buffering**: same compute-side pattern as `gemm_ar` (zigzag tiles, `stepK=4`, L0 ping/pong).
- **Per-tile store fence**: after `TSTORE`, `pipe_barrier(PIPE_ALL) + dsb`, then signal `groupDone`. `PIPE_FIX` only drains FixPipe and does **not** order the following scalar `st_atomic(groupDone)`; without `PIPE_ALL` the CCU can read a half-written tile (more visible with small groups / short payloads).
- **Optional Broadcast source**: Pipelined defaults to Reduce→HBM then whole-group Broadcast; `CCU_FUSED_RB=1` Writes from MS after each 4KB Reduce (saves the HBM round-trip, but small-message count scales with peer count). Sequential one-shot **always** Broadcasts from HBM.
- **Stable constraint**: `CCU_MISSION_PARALLEL=1`; each rank has a distinct pipelined `gate` and Sequential `seqGate`. Per group the CCU runs Reduce→Broadcast serially on one channel.

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
| `commGroupTiles` | default 16; measured prefer 2-rank `26`, 4-rank `13` |
| Number of tiles | 258 (`43 x 6`) |
| `COMPUTE_BLOCK_NUM` | default 24 (use `--compute-blocks 32` for reported numbers) |
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
│  │   PIPE_ALL + dsb   │   │ owner: wait >= P-1      │──►│   Pull Reduce    │  │
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
4. **`pipe_barrier(PIPE_ALL) + dsb(DSB_DDR)`**: drain FixPipe across pipes before signaling (there is no `set_flag(PIPE_FIX→S)` for the scalar `groupDone` path).
5. **Progress signal**: `AtomicAdd(+1)` into this rank's window `groupDone[flat(owner,g)]`.

## Communication Path Details

### AIV Progress

1. Each rank's AIV polls local `groupDone[flat]`; when it reaches `CcuOwnerGroupTilesInGroup`, it `TNOTIFY(+1)` the **owner** `groupReady[flat]`.
2. Owner AIV, per flat group: after `TTEST(groupReady >= P-1)` and the previous group has retired, `st_dev` `TriggerProgressCke` (single slot).
3. Host `PrepareCcu`: Launch persistent CCU → **single** `st_dev` gate poke (same as `treduce_ccu`, no Host poll); after `WaitEvent(gate)` CCU consumes groups on progress CKEs.

### CCU fused RS + AG

For each ready group (owner-contiguous tiles):

1. **Pull Reduce**: `Read` peer packed `gemm_output` into MS (4KB slices), `LocalReduce`.
2. **Push Broadcast** (Pipelined only; two options):
   - **Default (HBM)**: `LocalCopy` the reduce result into owner `reduced_output`, then one whole-group `Write` per peer.
   - **`CCU_FUSED_RB=1` (MS)**: after each 4KB Reduce, `Write` from MS to every peer (and still LocalCopy to local HBM). Each Write stays 4KB, so cost scales with peer count.
3. Increment `itemsDone` for AIV backpressure on the next group.

### Sequential baseline

Separate from Pipelined: data plane stays one-shot (**seqGate** CKE only; no progress CKE / per-group fire), while peer readiness reuses the same ready-counter `TNOTIFY` walk. `seqGate` is a different physical CKE from the pipelined `gate`, so `PrepareCcu`'s gate poke cannot release Sequential.

1. Host `Launch` one-shot (CCU parks on `WaitEvent(seqGate)`, **outside the timed window**; Launch may precede GEMM, same as Pipelined).
2. Full GEMM → sync (compute must finish before communication; CCU must still be in `WaitEvent`).
3. AIV `seq_peer_sync_gate`: shared peer-ready walk (**only after full GEMM**), then **one** Trigger of `seqGate` (no follow-up poke — that latches the next WaitEvent).
4. CCU runs **one** fused RS→AG over the full owner footprint.

Timing: `seq` is continuous wall-clock `GEMM → peer sync → CCU sync`; `seq_comm` covers peer sync + one-shot comm. Use `PTO_CCU_GEMM_AR_COMM_DIAG=1` for `aiv_wall` / `ccu_wall` (`ccu_wall≈0` means the gate did not hold).

## Memory Layout and HCCL Window

CCU payloads (`gemm_output` / `reduced_output`) and final `row_output` live in a high-bandwidth HBM block (with token for CCU access). Cross-rank sync counters live in the HCCL window `signal_matrix`.

| Buffer | Size | Location | Why |
| --- | --- | --- | --- |
| `gemm_output` (packed) | owner-padded × tile bytes | **HBM + token** | CCU Pull source |
| `reduced_output` (packed) | same | **HBM + token** | CCU Reduce dst / Broadcast src+dst |
| `row_output` | `M x N x 2B` | **HBM block** | unpacked output for verify |
| `signal_matrix` | `G_SIGNAL_MATRIX_SLOTS x 4B` | **HCCL window** | `groupDone` / `groupReady` |
| `src0_dev`, `src1_dev` | input matrices | **aclrtMalloc** | local-only |

`run.sh` estimates and raises `HCCL_BUFFSIZE` from the padded footprint (about `3 x packed + 128MB` margin).

## Measured Performance (reference)

Same shape: `M=5416, K=6144, N=1408` (padded `5504x1536`), `258 tiles (43x6)`, `compute_blocks=32`.  
Date: **2026-08-05** (`FIRST_DEVICE=2`, devices 2–5; includes the `PIPE_ALL` store fence).  
Format: **avg** `[med=, std=]`. Headline metric is **Pipelined**.  
Default Broadcast = HBM (`CCU_FUSED_RB` unset or `0`).

### 2 ranks (`--comm-group-tiles 26`, HBM Broadcast)

`comm_data=0.016 GB/rank`. VERIFY `err=0` PASS.

| Metric | Value |
| --- | --- |
| Compute-only | `308.0 us` (`304261 GFLOPS`) `[med=308.1, std=0.3]` |
| Sequential | `777.0 us` `[med=773.1, std=11.2]` (compute `308.4 us` + one-shot comm `468.6 us @ 33.6 GB/s`) |
| Pipelined | **`488.1 us`** `[med=484.5, std=18.8]` (compute done `299.1 us`, comm done `487.7 us @ 32.3 GB/s`) |
| Speedup | `1.592x` |
| Overlap eff | `94.1%` |
| Throughput | `383993 GFLOPS` (total) |

### 4 ranks (`--comm-group-tiles 13`, HBM Broadcast)

`comm_data=0.024 GB/rank`. VERIFY `err=0` PASS.

| Metric | Value |
| --- | --- |
| Compute-only | `306.5 us` (`305748 GFLOPS`) `[med=306.8, std=0.7]` |
| Sequential | `641.2 us` `[med=636.8, std=19.6]` (compute `308.0 us` + one-shot comm `333.2 us @ 71.1 GB/s`) |
| Pipelined | **`388.1 us`** `[med=380.7, std=23.2]` (compute done `299.9 us`, comm done `387.7 us @ 61.1 GB/s`) |
| Speedup | `1.652x` |
| Overlap eff | `83.1%` |
| Throughput | `965752 GFLOPS` (total, ×ranks) |

### HBM vs MS Broadcast (same-day, `CCU_FUSED_RB`)

| Config | Broadcast | Pipelined | Sequential | Speedup |
| --- | --- | --- | --- | --- |
| 2-rank g=26 | HBM (`0`) | 488.1 `[med=484.5]` | 777.0 | 1.592x |
| 2-rank g=26 | **MS (`1`)** | **465.0** `[med=462.1]` | 796.8 | **1.714x** |
| 4-rank g=13 | HBM (`0`) | 388.1 `[med=380.7]` | 641.2 | 1.652x |
| 4-rank g=13 | MS (`1`) | 383.0 `[med=378.7]` | 615.8 | 1.608x |

Takeaway: on 2 ranks MS is ~**23 µs** faster (HBM round-trip savings beat single-peer 4KB tax); on 4 ranks the two are tied (~5 µs, noise) because peer=3 roughly triples per-group small Writes. Sequential ignores `CCU_FUSED_RB` (always HBM one-shot); Sequential deltas above are run-to-run noise.

### What these numbers mean

- **Compute-only**: pure GEMM with no communication.
- **Sequential**: full GEMM, then peer-sync + one-shot CCU, no compute/comm overlap; Launch outside the timer.
- **Pipelined**: `PrepareCcu` outside the timer; AIC / AIV / CCU overlapped end-to-end (primary cross-config metric).
- **Speedup**: Sequential / Pipelined (treat carefully when Sequential variance is high).
- **Overlap eff**: time saved by overlap as a percentage of the shorter serial phase.

## Build and Run

1. Set up the Ascend CANN environment (**cann-9.2** required):

```bash
export ASCEND_CANN_PATH=/usr/local/Ascend/cann-<version>/set_env.sh
source "${ASCEND_CANN_PATH}"
```

2. Run the 2-rank example (use `--comm-group-tiles 26` to match the numbers above):

```bash
cd ${git_clone_path}/kernels/manual/a5/ccu_gemm_ar
./run.sh -r npu -v Ascend950PR_958b -n 2 -d 2 --compute-blocks 32 --comm-group-tiles 26
```

3. Choose the first device id:

```bash
FIRST_DEVICE=0 ./run.sh -r npu -v Ascend950PR_958b -n 2 -d 2 \
  --compute-blocks 32 --comm-group-tiles 26
```

4. 4 ranks (matches the numbers above with `--comm-group-tiles 13`):

```bash
FIRST_DEVICE=0 ./run.sh -r npu -v Ascend950PR_958b -n 4 -d 4 \
  --compute-blocks 32 --comm-group-tiles 13
```

On 4 ranks owner shards are 65/65/64/64; `13` divides 65 (less pad on the critical owner). Default `16` pads 65 to 80.

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
| `HCCL_CCU_CUSTOM_OP_MODE` | custom CCU kernel mode (required for `HcommCcuKernelRegister`) | set to `1` by `run.sh` |
| `CCU_FUSED_RB` | Pipelined: `1` = MS-direct Broadcast, else HBM Broadcast | HBM by default |
| `FIRST_DEVICE` | first NPU device id | `0` |
| `CCU_MISSION_PARALLEL` | CCU mission parallelism | must be `1` |
| `HCOMM_PKG_INC` | optional internal hcomm pkg_inc | auto-probed by cmake |
| `PTO_CCU_GEMM_AR_VERBOSE` | log seq/pipe gate VAs | off |
| `PTO_CCU_GEMM_AR_COMM_DIAG` | log seq/pipe `aiv_wall`/`ccu_wall` | off |

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
