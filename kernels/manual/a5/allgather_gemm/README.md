# AllGather GEMM A5 v3

This directory is regenerated from the current `kernels/manual/a2a3/allgather_gemm` baseline and keeps only the A5-specific changes that are required.

## What stays aligned with A3

- same algorithm structure
- same tile split and streaming pipeline
- same A3-style `run.sh` flow: generate data, rebuild, run
- same overall `CMakeLists.txt` layout

## A5-only changes

- `dav-c310-cube` / `dav-c310-vec`
- A5 comm testcase headers under `tests/npu/a5/comm/st/testcase`
- kernel include root detection for `x86_64-linux` / `aarch64-linux`
- HCCL window guard and size check in `main.cpp`
- `pipe_barrier(PIPE_ALL)` plus `block_idx/block_num` rename in compute kernel
- local `test_common.h` retained for the compare-count fix

## Usage

Prepare the CANN and MPI environment first, then run:

```bash
bash run.sh -r npu -v Ascend910_950z -n 2
```

Compile-only validation has been completed on the current non-A5 machine. Runtime validation still needs an A5 machine.
