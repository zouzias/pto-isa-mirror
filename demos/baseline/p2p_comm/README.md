# P2P Communication Demo

Minimal point-to-point communication demo using PTO's `TPUT` (remote write) and `TGET` (remote read) instructions between two NPU devices.

## Prerequisites

- CANN toolkit installed (`ASCEND_HOME_PATH` set via `set_env.sh`)
- MPI library available (e.g. mpich)
- At least 2 NPU devices on the machine

## Quick Start

```bash
source /path/to/set_env.sh   # set ASCEND_HOME_PATH
./run.sh                      # build + run (default: ascend910b1)
./run.sh Ascend910_9599       # for A5 devices
```

## What It Does

1. **TPUT Demo**: Rank 0 writes 256 `int32_t` values to Rank 1's shared memory via `pto::comm::TPUT`. Rank 1 reads and verifies the received data.

2. **TGET Demo**: Rank 0 prepares data in its shared memory. Rank 1 pulls the data via `pto::comm::TGET` and verifies it.

## Project Structure

```
p2p_comm/
  CMakeLists.txt                 -- Build configuration (bisheng + CCE)
  csrc/kernel/p2p_kernel.cpp     -- AICORE kernels + host-side launchers
  csrc/kernel/p2p_kernel.h       -- Host-side function declarations
  csrc/host/main.cpp             -- Entry point (MPI init, run demos, report)
  run.sh                         -- One-click build and run
```

## Manual Build

```bash
mkdir -p build && cd build
cmake .. -DSOC_VERSION=ascend910b1
make -j$(nproc)
cd ..
mpirun -n 2 ./build/bin/p2p_demo
```
