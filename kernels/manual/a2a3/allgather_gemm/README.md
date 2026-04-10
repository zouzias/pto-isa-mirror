# AllGather + GEMM Communication-Compute Fusion Example

## Overview

This example demonstrates a fused AllGather + GEMM operator on Ascend AI Cores using **M-dimension splitting** and a **chunk streaming pipeline**. In multi-card LLM inference, each rank holds a local slice of matrix `A` along the M dimension. Instead of completing the AllGather before starting GEMM, this implementation overlaps communication and computation at chunk granularity — the compute kernel begins processing each chunk as soon as the communication kernel signals its arrival, effectively hiding communication latency behind compute.

## Supported AI Processors

- A2/A3

## Directory Layout

```
kernels/manual/a2a3/allgather_gemm/
├── main.cpp                           # Host entry: HCCL init, dual-stream dispatch, warmup, verification, perf stats
├── allgather_gemm_comm_kernel.cpp     # AIV communication kernel: AllGather via TPUT
├── allgather_gemm_compute_kernel.cpp  # AIC compute kernel: streaming GEMM with chunk-flag waiting
├── ready_queue.hpp                    # ChunkFlagMatrix / summary counter metadata
├── run.sh                             # Build & run script (multi-device, CSV batch, perf mode)
├── scripts/
│   ├── gen_data.py                    # Input data generation (FP16 A slices + B)
│   ├── test_shapes.csv                # Test shape configurations (M, K, N)
│   └── verify_result.py               # Golden comparison (FP32, rtol/atol=0.001)
└── CMakeLists.txt                     # Build configuration
```

## Operator Description

### Function

This example implements AllGather followed by GEMM:

$$
C = A \times B
$$

Where:

- Each of `n_ranks` ranks holds a local M-slice of `A`: rows `[rank * M/n_ranks, (rank+1) * M/n_ranks)`.
- `B` is replicated across all ranks (`K × N`, FP16).
- After AllGather collects the full `A` (`M × K`, FP16), each rank computes the full `C` (`M × N`, FP32).

The AllGather and GEMM are fused into a streaming pipeline so that computation begins before the full AllGather completes.

### Specification

| Item         | Value |
| ------------ | ----- |
| OpType       | `AllGather + GEMM` (communication-compute fusion) |
| Inputs       | `A`: `M × K`, `float16`, `ND` (M-sliced across ranks); `B`: `K × N`, `float16`, `ND` (replicated) |
| Output       | `C`: `M × N`, `float32`, `ND` |
| Comm kernel  | `RingCommStreamingKernel` (AIV) |
| Compute kernel | `AllGatherGemmComputeStreamingKernel` (AIC) |

## Architecture

### Dual-Stream Concurrency

The communication and compute kernels run on two independent AICPU streams, launched concurrently from the host:

- **Comm stream** → `RingCommStreamingKernel` runs on **AIV** (Vector) cores.
- **Compute stream** → `AllGatherGemmComputeStreamingKernel` runs on **AIC** (Cube) cores.

The host dispatches both kernels back-to-back and synchronizes after both complete.

### AI Core Resources

| Unit            | Hardware Engine | Role in This Example |
| --------------- | --------------- | -------------------- |
| **AIC (Cube)**  | Matrix engine   | Compute kernel: GEMM (`TMATMUL` / `TMATMUL_ACC`) |
| **AIV (Vector)**| Vector / DMA    | Comm kernel: RDMA data transfer (`TPUT`) + signaling (`TNOTIFY`) |

### Streaming Pipeline

```
Sequential execution:
  [ AllGather completes entirely ] ──► [ GEMM completes entirely ]

Streaming pipelined execution:
  Comm (AIV):   [chunk0 TPUT][TNOTIFY] [chunk1 TPUT][TNOTIFY] [chunk2 TPUT][TNOTIFY] ...
                      │                      │                      │
                      ▼                      ▼                      ▼
  Compute (AIC): [local GEMM]  [TWAIT chunk0][GEMM chunk0] [TWAIT chunk1][GEMM chunk1] ...
                  (zero-wait)
```

The compute kernel operates in two phases:

1. **Phase 1 (local)**: Processes the local rank's row-groups immediately (data already resident in shared memory, no waiting).
2. **Phase 2 (remote)**: For each remote rank's row-groups, uses `TWAIT` on the summary counter to block until chunks arrive, then computes as soon as each chunk is ready.

## Optimization Notes

- **Summary monotonic counter + TWAIT**: The comm kernel atomically increments a per-source summary counter (`TNOTIFY` AtomicAdd) after each chunk transfer. The compute kernel uses hardware `TWAIT` (compare-and-block) to wait for the counter to reach the expected value — zero polling overhead, no busy-spin.
- **Local data zero-wait priority**: The compute kernel processes the local rank's row-groups first (Phase 1) with no flag checks, overlapping with remote chunk transfers.
- **Send order aligned with consumption order**: The comm kernel transmits chunks in the same order the compute kernel consumes them, minimizing wait time.
- **Continuous K accumulation pipeline**: Within each row-group, K-tiles are processed with `TMATMUL` (first iteration) followed by `TMATMUL_ACC` (subsequent iterations), maintaining a continuous accumulation pipeline without intermediate store/reload.
- **L1/L0 two-level double buffering**: `aMatTile[2]` / `bMatTile[2]` in L1 and `aTile[2]` / `bTile[2]` in L0A/L0B enable overlapped DMA (`TLOAD`) ↔ extract (`TEXTRACT`) ↔ compute (`TMATMUL`).
- **Parallel AIV full-mesh communication**: In the full-mesh mode, each rank's AIV cores directly `TPUT` data to all other ranks simultaneously, with multiple AIV blocks assigned per destination for bandwidth utilization.
- **Dynamic chunk size**: `ComputeOptimalChunkSize()` automatically selects chunk granularity to keep the number of chunks per source in the 64–128 range, balancing pipeline depth against signaling overhead.
- **Flexible block allocation**: The comm kernel adapts to the available block count — when blocks outnumber destinations, blocks are evenly distributed per destination; otherwise, work items are round-robin scheduled across blocks.

## Measured Performance (Reference)

The following measurements were collected on Ascend A3 (910B1) with fp16 inputs → fp32 output, using `aclrtEvent` timing (3 warmup + 10 timed iterations, average reported). TFLOPS is computed as `2 × M × K × N / time`.

### 2-rank

| M | K | N | Execution time (ms) | TFLOPS |
| --- | --- | --- | --- | --- |
| 2048 | 2048 | 1024 | 0.297 | 28.96 |
| 4096 | 4096 | 2048 | 1.098 | 62.57 |
| 4096 | 4096 | 4096 | 1.231 | 111.62 |
| 8192 | 4096 | 4096 | 2.519 | 109.13 |

### 4-rank

| M | K | N | Execution time (ms) | TFLOPS |
| --- | --- | --- | --- | --- |
| 4096 | 4096 | 4096 | 0.986 | 139.42 |
| 8192 | 4096 | 4096 | 1.648 | 166.75 |

### 8-rank

| M | K | N | Execution time (ms) | TFLOPS |
| --- | --- | --- | --- | --- |
| 8192 | 4096 | 4096 | 1.439 | 191.03 |
| 16384 | 4096 | 4096 | 2.585 | 212.71 |

### What the numbers suggest

- **Multi-rank scaling**: For the same total GEMM shape (M=8192, K=4096, N=4096), throughput scales from 109 TFLOPS (2-rank) to 167 TFLOPS (4-rank) to 191 TFLOPS (8-rank). This reflects effective communication-compute overlap — as each rank computes a smaller local GEMM, the relative communication overhead increases, but the streaming pipeline successfully hides a significant portion of it.
- **Larger M improves throughput**: With 8 ranks, doubling M from 8192 to 16384 increases throughput from 191 to 213 TFLOPS, because the compute-to-communication ratio grows and the pipeline has more chunks to overlap.
- **Small shapes are communication-dominated**: The 2048×2048×1024 (2-rank) case achieves only 29 TFLOPS — the AllGather data volume is small but the fixed communication overhead (HCCL setup, signaling) is not amortized.

## Build and Run

1. Configure your Ascend CANN environment:

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

2. Run the example:

```bash
cd ${git_clone_path}/kernels/manual/a2a3/allgather_gemm
bash run.sh -r npu -v Ascend910B1 -n 2
```

If the run succeeds, the output prints:

```text
test success
```
