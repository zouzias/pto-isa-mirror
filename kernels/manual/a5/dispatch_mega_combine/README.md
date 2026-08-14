# PTO MegaMoE Dispatch + Combine Fusion Example

[中文文档](README_zh.md)

## Overview

This example implements an end-to-end MegaMoE fused operator with PTO Manual kernels. It fuses the traditional MoE reorder, AlltoAllV-style data exchange, grouped FFN compute, combine, and unpermute flow into one large kernel, and overlaps AIC and AIV work at local-expert granularity.

The main device pipeline is:

```text
FrontReorder -> Dispatch -> GMM1 -> SwiGLU -> GMM2 -> Combine -> Unpermute
```

## Supported AI Processors

- Ascend950PR (A5, arch35 / DAV_3510 family)
- Toolchain-compatible alias `Ascend910_9599`

This project always builds the mixed-core kernel with `dav-c310` and `PTO_NPU_ARCH_A5`. `Ascend910B` is an A3 target and is not a valid `--soc` value for this directory.

## Directory Layout

```text
kernels/manual/a5/dispatch_mega_combine/
├── CMakeLists.txt                  # Build configuration for host executable and device kernel shared library
├── run.sh                          # One-click data generation, build, and mpirun entry
├── main.cpp                        # Host entry: case loading, ACL/HCCL/MPI setup, launch, and verification
├── kernel_launch.cpp               # Device kernel launch wrapper
├── runtime_context.*               # Single-rank runtime, HCCL window, device/context management
├── tiling_builder.*                # Host tiling and workspace planning
├── data_utils.*                    # Case file IO and validation helpers
├── comm_mpi.h                      # MPI dynamic loading wrapper
├── scripts/
│   └── gen_data.py                 # Synthetic input, weight, and golden generation
├── op_kernel/
│   ├── dispatch_mega_combine.h     # MegaMoE device pipeline entry
│   ├── front_reorder.h             # Token-order quantization and route-mask publication
│   ├── dispatch.h                  # Mask decode, remote token pull, and GMM1 input build
│   ├── gmm_common.h                # Shared GMM tile scheduling and helpers
│   ├── gmm1.h                      # First grouped matmul
│   ├── swiglu.h                    # SwiGLU activation and dynamic quantization
│   ├── gmm2.h                      # Second grouped matmul
│   ├── combine.h                   # Route-metadata prefetch and route-slot remote writeback
│   ├── unpermute.h                 # Route-slot TopK reduction in original token order
│   └── utils/                      # PTO vector, sync, HCCL window, and GMM pipeline helpers
```

## Operator Description

### Functionality

The operator implements the multi-rank MoE FFN main path:

```text
x[rank, M, K] + expertId[rank, M, topK] + probs[rank, M, topK]
  -> route-mask pull into destination expert-major rows
  -> grouped GMM1
  -> SwiGLU activation + dynamic quantization
  -> grouped GMM2
  -> combine back to source ranks
  -> TopK weighted reduction
  -> out[rank, M, K]
```

Conceptually:

```text
for each rank, token:
  out[token] = sum_{topK route} probs[token, route] * FFN_expert(x[token])
```

`FFN_expert` consists of MXFP8 GMM1, SwiGLU, and MXFP8 GMM2. Cross-rank data exchange uses the HCCL RDMA window and PTO communication/synchronization helpers.

### Specification

| Item | Value |
| --- | --- |
| OpType | `MegaMoE Dispatch + FFN + Combine` |
| Input | `x`: `[M, K]`, `bfloat16`; `expertId`: `[M, topK]`, `int32`; `probs`: `[M, topK]`; `weight1/weight2`: per-local-expert E4M3 data; `scale1/scale2`: E8M0 scales |
| Output | `out`: `[M, K]`, `half/bfloat16` |
| Kernel name | `dispatch_mega_combine_kernel` |
| Host executable | `dispatch_mega_combine` |
| Default script case | `worldSize=2, M=2048, K=7168, N=4096, topK=8, expertPerRank=16, maxOutputSize=81940` |

## Optimization Notes

- **Expert-level overlap**: AIC-side GMM1/GMM2 and AIV-side Dispatch/SwiGLU/Combine progress by local expert group, connected by cache-line-isolated GM epochs.
- **Mask Pull Front**: tokens are quantized once in source-token order; PTO `TCMPS` builds per-expert route masks, and Dispatch pulls only matching token records.
- **PTO double buffering**: Front quant/mask, 8-row Dispatch gather, and 16-row Combine metadata/scale prefetch use independent ping-pong UB buffers and pipeline events.
- **PTO tile GMM optimization**: GMM1/GMM2 use output-tile swizzle, L1-to-L0 multi-level reuse, double buffering, and fixpipe quant/cast.
- **Per-expert SwiGLU overlap**: GMM1 publishes each completed expert to SwiGLU, which publishes that same expert to
  GMM2 after all AIV rows are written. No segment metadata workspace or segment-level scheduling remains.
- **Streaming Combine/Unpermute**: Combine writes source-rank route slots and publishes expert progress; Unpermute consumes ready routes without a final all-rank barrier.

The host queries the AICore count with `aclrtGetDeviceInfo(..., ACL_DEV_ATTR_AICORE_CORE_NUM, ...)` and selects a
validated mixed-core mapping. Every physical block contains one AIC and two AIV subblocks. The default mappings are:

| AIC | AIV | Default GMM1 / Dispatch / SwiGLU blocks | Default GMM2 / Combine blocks |
| ---: | ---: | --- | --- |
| 28 | 56 | `0..19` (20) | `20..27` (8) |
| 32 | 64 | `0..20` (21) | `21..31` (11) |
| 36 | 72 | `0..23` (24) | `24..35` (12) |

The first GMM1 wave uses all available AICs. The default split is used after that; once GMM1 finishes, its AICs
may join the GMM2 tail. RankStreaming first releases the two AIV subblocks paired with the GMM1 group. After active
Combine lanes finish and local metadata is reset, the AIVs paired with GMM2 join as helpers.

`run.sh --aicore-num 0|28|32|36` selects the effective launch count (`0` uses the runtime-reported count). A requested
count must not exceed the physical runtime count.

## Tiling Parameters

| Parameter | Default / Description |
| --- | --- |
| `M` | Set by `run.sh --m` or `case.json` |
| `K` | Hidden size for GMM input; must satisfy packed-row and GMM tile alignment |
| `N` | FFN intermediate size; GMM1 output and `N/2` after SwiGLU |
| `topK` | Number of routed experts per token |
| `expertPerRank` | Number of local experts per rank |
| `worldSize` | MPI/HCCL rank count |
| `maxOutputSize` | Per-rank routed-row workspace limit |
| `aicNum` | Effective AICore launch count; validated values are 28, 32, and 36 |
| `aivNum` | Derived as `2 * aicNum`: 56, 64, or 72 |
| `GMM baseM/baseN` | Main output tile shape is `128 x 256` |
| `Front Mask Pull` | The only Front implementation; host rejects clipping, invalid experts, inactive tokens, and insufficient receive capacity |
| `Fixed-role AIV UB` | Dispatch, SwiGLU, Combine, and Unpermute use a 216 KiB main region; the final 40 KiB is reserved for synchronization snapshots |
| `Dispatch / Unpermute tiles` | Dispatch gathers 8 packed rows per UB batch; Unpermute selects up to 8192 columns within the 216 KiB main region |
| `Combine metadata batch` | 16 contiguous GMM rows per metadata/scale load, two UB buffers |

## Supported Cases

Representative validation cases keep all major parameters fixed except `M`:

```text
worldSize=8
K=7168
N=4096
topK=8
expertPerRank=16
aicNum=runtime (28, 32, or 36), or `run.sh --aicore-num 28|32|36`
aivNum=2 * aicNum
```

| M | maxOutputSize | Command |
| --- | --- | --- |
| 16 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 16 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 32 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 32 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 64 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 64 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 128 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 128 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 512 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 512 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 1024 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 1024 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |
| 2048 | 81940 | `bash run.sh --world-size 8 --first-device 0 --m 2048 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data` |

## Overall Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ Front Mask Pull (AIV)                                                       │
│   token-order quant + per-expert route masks/counts                         │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │ independent front-ready epoch
┌──────────────────────────────▼───────────────────────────────────────────────┐
│ Expert-level overlapped pipeline                                             │
│                                                                              │
│ AIV: Dispatch(group i) -> SwiGLU(group i) -> Combine(group i)                 │
│ AIC:                    GMM1(group i)      -> GMM2(group i)                  │
│                                                                              │
│ Stages communicate through per-expert GM arrival/ready epochs                 │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │ final boundary
┌──────────────────────────────▼───────────────────────────────────────────────┐
│ Unpermute (AIV)                                                              │
│   route-slot output + probs -> TopK weighted reduce -> out[M, K]              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## FrontReorder Stage

Front keeps one quantized record per source token and publishes one bitmask slot per global expert/source rank:

```text
x[M, K] + expertId[M, topK]
  -> sourceTokenRecords[M, K + 32]
  -> routeMaskSlots[localExpert, srcRank, mask + laneCapacity * 32B partial counts]
  -> cumsumMM[srcRank, localExpert] / expertTokenNums[localExpert]
```

Tokens are split across all AIVs without splitting K. Global experts receive balanced AIV lane intervals. Each lane
writes disjoint 32B mask blocks (256 route slots per block) plus an independent aligned partial-count record. If there
are fewer AIVs than experts, each AIV handles multiple experts with one lane per expert. Quant and mask batches use two
PTO UB buffers. After every source publishes its independent front-ready epoch, one coordinator loads and reduces the
lane count records in UB, builds source-major `cumsumMM`, and releases Dispatch.

## Dispatch Stage

Dispatch runs on the destination rank. Each source rank gets `dispatchGroupSize / worldSize` AIV0 lanes; every lane
scans the source mask but consumes only its match-ordinal shard:

```text
srcRank.sourceTokenRecords[routeSlot / topK]
  -> workspace.gmA[dstRowBase : dstRowBase + rows, 0:K]
  -> workspace.perTokenScale1[dstRowBase : dstRowBase + rows]
  -> workspace.routeMeta[dstRow] = {srcRank, routeSlot, 0...}
```

Matches are accumulated in 8-row batches. Two packed/meta UB buffers overlap remote MTE2 loads with contiguous MTE3
stores. After all active lanes finish one local expert, Dispatch releases the existing expert-level GMM1 ready epoch.

## GMM1 / SwiGLU / GMM2 Stages

### GMM1

GMM1 runs on AIC and performs the first grouped matmul:

```text
gmA[int8] x weight1[int8]
  -> int32 accumulator
  -> fixpipe scale1
  -> gmC[half]
```

Each local expert is split into `128 x 256` output tiles. Linear tile ids are mapped to `(blockM, blockN)` with swizzle to improve L1 reuse of the B-side weights.

### SwiGLU

SwiGLU runs on AIV and consumes GMM1 `gmC` plus Dispatch-generated `perTokenScale1`:

```text
gmC * perTokenScale1
  -> silu(up) * gate
  -> dynamic quantization
  -> gmPermutedToken[int8] + perTokenScale2[float]
```

SwiGLU advances one expert at a time. It waits for the GMM1 ready epoch, splits the current expert rows across the
active SwiGLU AIVs, drains activation and scale stores, then publishes one ready slot per GMM2 AIC after all producer
arrivals are visible.

### GMM2

GMM2 runs on AIC and performs the second grouped matmul:

```text
gmPermutedToken[int8] x weight2[int8]
  -> int32 accumulator
  -> fixpipe scale2
  -> gmm2Output[half]
```

After GMM2 finishes a local expert group, either the GMM2 group or all available AICs publish arrivals. A single AIV
coordinator collects them and releases the active Combine group through one shared ready slot per GMM2 AIC.

## Combine / Unpermute Stages

Combine runs on AIV and writes GMM2 output directly to the source route slot after applying `perTokenScale2`:

```text
gmm2Output[srcRow, 0:K] half
  -> fp32
  -> * perTokenScale2[srcRow]
  -> OutputElement
  -> srcRank.combineOutputByRouteSlot[routeMeta[srcRow].routeSlot, 0:K]
```

Each source-rank lane prefetches 16 contiguous `routeMeta` records and scales into two UB buffers. This removes the
per-row `DCCI` and scalar GM scale load while preserving full-row K processing and expert-progress publication.

Unpermute restores the source-rank token order:

```text
combineOutputByRouteSlot[token * topK + topk] + probs
  -> TopK weighted accumulation
  -> out[M, K]
```

## Memory Layout and HCCL Window

The HCCL remote window carries cross-rank visible data:

| Buffer | Location | Purpose |
| --- | --- | --- |
| `sourceTokenRecords` | HCCL window | One packed int8 record per source token; Dispatch pulls by `routeSlot / topK` |
| `routeMaskSlots` | HCCL window | Per-local-expert/source-rank mask plus aligned count record |
| `combineOutputByRouteSlot` | HCCL window | Combine writes source route slots; Unpermute consumes directly |
| `gmA` | workspace GM | GMM1 input generated by Dispatch |
| `gmC` | workspace GM | GMM1 output and SwiGLU input |
| `gmPermutedToken` | workspace GM | SwiGLU dynamic-quant output and GMM2 input |
| `gmm2Output` | workspace GM | GMM2 output and Combine input |
| `routeMeta` | workspace GM | 32B mapping from GMM row to `{srcRank, routeSlot}` |
| `cumsumMM` | workspace GM | source-major cumulative rows for Dispatch destination offsets |

`run.sh` estimates the HCCL window from `M`, `topK`, `K`, expert topology, and mask-lane capacity, then raises
`HCCL_BUFFSIZE` when needed.

## Build and Run

Configure the Ascend CANN environment:

```bash
source ~/zy/set_evn.sh
```

Build the A5 mixed-core kernel and host without generating case data:

```bash
cd ${git_clone_path}/kernels/manual/a5/dispatch_mega_combine
bash run.sh --build-only
```

Run on contiguous available A5 devices. The environment must provide MPICH; OpenMPI is not supported by the MPI
compatibility wrapper.

```bash
bash run.sh --soc Ascend910_9599 --world-size 2 --first-device 2 --m 2048 --k 7168 --n 4096 \
  --topk 8 --experts 16 --max-output-size 81940 --reuse-data
```

This maps ranks 0 and 1 to physical devices 2 and 3. `Ascend910_9599` is the A5 toolchain alias here; it must not be
replaced with the A3 target `Ascend910B`.

For another `M` value, keep the initial two-rank setup, for example:

```bash
bash run.sh --soc Ascend910_9599 --world-size 2 --first-device 2 --m 512 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data
```

### Environment Variables

| Environment Variable | Purpose | Default Behavior |
| --- | --- | --- |
| `ASCEND_HOME_PATH` | CANN installation path | Must be set before running |
| `CMAKE_COMPILER` | Compiler used by CMake | `bisheng` |
| `FIRST_DEVICE` | First physical device in the contiguous rank mapping | `0`; overridden by `--first-device` |
| `MPI_LIB_PATH` | Optional absolute path to the MPICH `libmpi.so` | Otherwise resolved from `LD_LIBRARY_PATH` |
| `MPI_RUNNER` | MPICH launch command | `mpirun` from the sourced environment |
| `HCCL_BUFFSIZE` | HCCL RDMA window size | Raised automatically by `run.sh` when needed |
| `DISPATCH_MEGA_COMBINE_AICORE_NUM` | Effective AIC count | `0`, which uses the runtime-reported count |
| `DISPATCH_MEGA_COMBINE_WARMUP_ITERS` | Warmup launches before timing | `3` |
| `DISPATCH_MEGA_COMBINE_MEASURE_ITERS` | Timed launches used for the kernel summary | `5` |

### Kernel Performance

After the timed launches, rank 0 prints one `[KERNEL_PERF]` summary for the complete kernel. Each AIC/AIV records only
its overall start/end system-counter values; the summary reports the maximum rank duration per iteration together with
average, minimum, maximum, standard deviation, token throughput, equivalent compute TFLOPS, and equivalent
communication bandwidth.

## Changing Case Parameters

When changing `M`, keep the other major parameters consistent with a supported topology and update the receive
capacity as needed.

```bash
bash run.sh --world-size 8 --first-device 0 --m 512 --k 7168 --n 4096 --topk 8 --experts 16 --max-output-size 81940 --reuse-data
```

Common constraints:

- `K` must satisfy packed-row, GMM1/GMM2 tile, and quantization-path alignment requirements.
- `N` is the GMM1 output dimension; GMM2 consumes `N / 2` after SwiGLU.
- `expertPerRank` must be one of `4`, `8`, `16`, or `32`, matching the compiled kernel specializations. The underlying
  C2V/V2C flag layout has an upper bound of 45 experts.
- `maxOutputSize` must cover the per-rank routed-row workspace limit.
- Synthetic `expert_idx` uses global token round-robin so small-M cases still cover global experts.

## FAQ

| Problem | Cause and Fix |
| --- | --- |
| `ASCEND_HOME_PATH must be set` | Source the CANN environment and export `ASCEND_HOME_PATH` before running `run.sh` |
| HCCL window too small | The manually set `HCCL_BUFFSIZE` is below the case requirement; unset it or increase it |
| MPI launch fails | Source the project environment and verify `mpirun --version` reports MPICH/HYDRA; OpenMPI is unsupported |
| Golden generation is slow | Reuse the generated files with `--reuse-data` after the first run; the chunk size is fixed internally |
| Result diff is abnormal | Check whether old generated data was reused; do not reuse stale `out/` after changing expert distribution or key case parameters |

## Build System

- **Compiler**: `bisheng`
- **Device kernel flags**: `-xcce --cce-aicore-arch=${CCE_AICORE_ARCH}`
- **Host executable**: `-xc++ -std=c++17`
- **Targets**: `dispatch_mega_combine_kernel`, `dispatch_mega_combine`
- **Linked libraries**: `stdc++`, `ascendcl`, `hcomm`, `runtime`, `tiling_api`, `platform`, `nnopbase`, `pthread`, and others
- **PTO include**: repository root `include/` is added to the include path for PTO tile/communication helpers

## Changelog

| Date | Change |
| --- | --- |
| 2026-06-26 | Added `dispatch_mega_combine` README covering the MegaMoE operator, stage flow, build/run, and FAQ |
| 2026-07-27 | Ported the optimized A3 schedule to the A5 backend and completed compile-only validation |
| 2026-08-14 | Removed development instrumentation and prepared the A5 target for production use |
