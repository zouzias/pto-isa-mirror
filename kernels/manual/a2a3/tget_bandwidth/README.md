# TGET / TGET_ASYNC Bandwidth Comparison Example

## Overview

This example compares the point-to-point communication bandwidth of **TGET** (synchronous remote read) and **TGET_ASYNC** (asynchronous SDMA remote read), sweeping transfer sizes from 4 KB to 4 MB, and measuring both host-side bandwidth (GB/s) and device-side average execution cycles.

- **TGET** performs remote reads via UB (Unified Buffer) staging: `Remote GM → UB → Local GM`. Bandwidth saturates at approximately 4 GB/s due to UB throughput limits.
- **TGET_ASYNC** performs remote reads via the SDMA engine: `Remote GM → SDMA → Local GM`. By bypassing the UB bottleneck, it reaches approximately 13–14 GB/s at 4 MB.

## Supported AI Processors

- A2/A3

## Directory Layout

```
kernels/manual/a2a3/tget_bandwidth/
├── scripts/
│   └── plot_bw_compare.py           # Generate bandwidth comparison plot
├── CMakeLists.txt                   # Build configuration
├── tget_bandwidth_kernel.cpp        # Kernel implementation (AICORE + host orchestration)
├── tget_bandwidth_kernel.h          # Kernel header
├── main.cpp                         # Host-side entry point (MPI initialization)
├── run.sh                           # Convenience script
├── README_zh.md                     # Chinese version
└── README.md                        # This file
```

## Operator Description

### Data Flow

**TGET (synchronous)**:
```
Peer NPU GM ──TGET──▶ Local UB ──TSTORE──▶ Local GM
```

**TGET_ASYNC (asynchronous)**:
```
Peer NPU GM ──SDMA──▶ Local GM   (direct transfer, no UB staging)
```

### Test Procedure

1. Each rank prepares send data in HCCL shared memory (`PrepareSendBufferKernel`)
2. The root rank runs TGET and TGET_ASYNC for each transfer size
3. Host-side timing measures bandwidth; device-side `SYS_CNT` measures cycles
4. Received data is verified for correctness

### Specification

| Item           | Value |
| -------------- | ----- |
| Data type      | `float` |
| NPU count      | 2 (point-to-point) |
| Transfer sizes | 4 KB, 16 KB, 64 KB, 256 KB, 1 MB, 4 MB |
| Metrics        | host bandwidth (GB/s), device average cycles |

## Measured Performance (Reference)

The following measurements were collected on Ascend A2/A3 (float type, 2-NPU point-to-point).

| Transfer Size | TGET BW (GB/s) | TGET_ASYNC BW (GB/s) | TGET Device Avg Cycles | TGET_ASYNC Device Avg Cycles |
| ------------- | --------------: | --------------------: | ---------------------: | ---------------------------: |
| 4 KB          | 0.21            | 0.19                  | 50.85                  | 118.18                       |
| 16 KB         | 0.72            | 0.75                  | 202.05                 | 166.42                       |
| 64 KB         | 1.75            | 2.55                  | 780.73                 | 338.10                       |
| 256 KB        | 3.01            | 6.08                  | 3347.12                | 1094.37                      |
| 1 MB          | 3.75            | 10.48                 | 12703.39               | 3791.18                      |
| 4 MB          | 3.99            | 12.95                 | 52878.12               | 14834.47                     |

### Analysis

- **TGET** bandwidth gradually increases with transfer size but saturates at approximately **4 GB/s** — the throughput ceiling of the UB staging path.
- **TGET_ASYNC** significantly outperforms TGET for large transfers (≥256 KB), reaching approximately **13 GB/s** at 4 MB, close to the theoretical SDMA engine bandwidth.
- For very small transfers (4 KB), TGET_ASYNC is slightly slower than TGET due to SDMA launch overhead.

### Bandwidth Comparison Plot

Run the plotting script to generate the comparison chart:

```bash
python3 scripts/plot_bw_compare.py
```

## Build and Run

### Prerequisites

- CANN Toolkit >= 8.5.0 (TGET synchronous instruction); >= 9.0.0 (TGET_ASYNC asynchronous instruction)
- MPI >= 3.2.1 (e.g. OpenMPI)
- 2 or more Ascend NPUs

### Steps

1. Configure your Ascend CANN environment:

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

2. Run the example (2-NPU by default):

```bash
cd ${git_clone_path}/kernels/manual/a2a3/tget_bandwidth
bash run.sh -r npu -v Ascend910B1
```

Use `-n` to specify the number of ranks (default is 2):

```bash
bash run.sh -r npu -v Ascend910B1 -n 2
```

On success, the output looks like:

```text
================ TGET/TGET_ASYNC Bandwidth Sweep ================
peer_rank=1 dtype=float tile_elems=1024
[BW] instr=TGET bytes=4096 iters=1000 ...
[BW] instr=TGET_ASYNC bytes=4096 iters=1000 ...
...
test success
```

## Changelog

| Date       | Change |
| ---------- | ------ |
| 2026-04-02 | Migrated from ST test to standalone performance example |
