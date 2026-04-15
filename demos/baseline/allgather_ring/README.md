# Ring Allgather Demo

Demonstrates the allgather collective operation using PTO's `TPUT_ASYNC` (async remote write) SDMA instruction with the **Ring algorithm**.

## Prerequisites

- CANN Toolkit (version 9.0.0 or above) installed (`ASCEND_HOME_PATH` set via `set_env.sh`)
- CANN Ops package (version 9.0.0 or above) installed
- MPICH installed
- At least 2 NPU devices on the machine

## Quick Start

```bash
source /path/to/set_env.sh
./run.sh                      # 8 ranks, default SoC
./run.sh 4                    # 4 ranks
./run.sh 2 Ascend910_9599     # 2 ranks, A5 devices
```

## What It Does

Each rank contributes 256 `int32_t` values. After allgather, every rank holds all ranks' data.

### Ring Algorithm

Ring allgather runs N-1 rounds for N ranks. Each round, every rank pushes one chunk to its ring neighbor (next rank):

- **Round 0**: Rank i copies its own sendBuf to recvBuf[i] (local copy), then pushes sendBuf to rank (i+1)'s recvBuf[i] via `TPUT_ASYNC`.
- **Round r (r >= 1)**: Rank i forwards the chunk received in the previous round to rank (i+1), i.e. pushes recvBuf[(i-r+N)%N] to rank (i+1)'s recvBuf[(i-r+N)%N].

Each round is launched as a separate kernel. Host-side `aclrtSynchronizeStream` + `HcclHostBarrier` between rounds ensures all SDMA writes complete before the next round begins.

### Known Limitation

For N > 2, rounds >= 1 read data that was written by a remote rank's SDMA in the previous round. On current hardware, SDMA remote writes do not invalidate the local AICORE's L2 cache, which may cause stale reads. This demo serves as a reference implementation with this limitation documented.

## Project Structure

```
allgather_ring/
  CMakeLists.txt                            -- Build configuration (bisheng + CCE)
  csrc/kernel/allgather_ring_kernel.cpp     -- AICORE kernel + host-side launcher
  csrc/kernel/allgather_ring_kernel.h       -- Host-side function declaration
  csrc/host/main.cpp                        -- Entry point (MPI init, run demo, report)
  run.sh                                    -- One-click build and run
```

## Manual Build

```bash
mkdir -p build && cd build
cmake .. -DSOC_VERSION=ascend910b1
make -j$(nproc)
cd ..
mpirun -n 2 ./build/bin/allgather_ring_demo
```
