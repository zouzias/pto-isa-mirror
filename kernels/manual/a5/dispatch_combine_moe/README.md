# Dispatch Combine MoE A5 Fused Operator Example

[中文文档](README_zh.md)

## Overview

This directory implements an A5 / Ascend950-oriented fused MoE kernel for the `dispatch -> FFN -> combine` path. The kernel fuses cross-rank token dispatch, two int8 grouped matmuls, the post-GMM1 SwiGLU path, the post-GMM2 probability-weighted combine path, and final token restore into one mixed AIC/AIV kernel. The implementation explicitly builds the compute-communication pipeline with PTO tile/vector/communication primitives and HCCL RDMA remote windows.

## Supported AI Processors

- A5 / Ascend950 family, with device compilation target `dav-c310`.

## Source Layout

```text
kernels/manual/a5/dispatch_combine_moe/
├── CMakeLists.txt                         # builds the kernel shared library and standalone host executable
├── run.sh                                 # generates the case, configures the build directory, and starts multi-rank smoke runs with MPI
├── main.cpp                               # standalone host runner: MPI/ACL/HCCL init, kernel launch, timing, and verification
├── kernel_launch.hpp                      # launch argument structure and launchDispatchCombineMoe host interface declaration
├── op_host/
│   ├── comm_mpi.h                         # loads MPI symbols at runtime instead of binding to one MPI implementation at link time
│   ├── data_utils.{hpp,cpp}               # reads/writes case and rank binary files and compares FP16 outputs
│   ├── runtime_context.{hpp,cpp}          # manages ACL/HCCL streams, communicator, and remote-window lifetime
│   └── tiling_builder.{hpp,cpp}           # builds tiling/workspace/blockDim and checks the remote-window layout
├── op_kernel/
│   ├── dispatch_combine_moe.cpp           # device kernel symbol, tiling registration, and host launch stub
│   ├── dispatch_combine_moe.h             # turns host tiling into A5 policies, layouts, and kernel params
│   ├── dispatch_combine_moe_kernel.hpp    # mixed AIC/AIV orchestrator for routing, dispatch, GMM, epilogue, and restore
│   ├── dispatch_combine_moe_tiling.h      # host/device shared tiling, runtime, and launch config structures
│   ├── token_reorder/
│   │   ├── routing/
│   │   │   ├── moe_init_routing_quant.cpp                 # routing entry, dispatching full-load/sort/gather-quant paths from tiling results
│   │   │   ├── moe_init_routing_quant_tiling.h            # routing quant tiling, branch selection, and workspace calculation
│   │   │   ├── moe_init_routing_tiling_common.h           # shared routing sub-stage tiling structures and tiling base
│   │   │   ├── moe_init_routing_sort.h                    # one-core/multi-core routing sort orchestration
│   │   │   ├── moe_packed_sort_merge.h                    # packed sort-record merge and output extraction
│   │   │   ├── moe_pto_sort.h                             # PTO UB sort, packed sort, vector helpers, and sync bridge
│   │   │   ├── moe_init_routing_expert_tokens.h           # expert token count/cumsum and expandedRowIdx/src-to-dst mapping
│   │   │   ├── moe_init_routing_fullload_dynamic_quant.h  # full-load path for sort, statistics, gather, and row-wise dynamic quant
│   │   │   ├── moe_init_routing_gather_dynamic_quant.h    # gather path that reads tokens by expandedRowIdx and writes int8 payload/scale
│   │   │   └── moe_common.h                               # routing constants, alignment helpers, and GM initialization utilities
│   │   └── unpermute/
│   │       ├── moe_token_unpermute.h                      # accumulates expanded outputs back to original token order using expandedRowIdx/probs
│   │       └── moe_token_unpermute_tiling.h               # token/core and hidden-chunk tiling for restore/unpermute
│   └── utils/
│       ├── block_mmad_preload_async_fixpipe_quant.hpp # AIC-side GM-L1-L0-MMAD-fixpipe staged pipeline
│       ├── block_epilogue_pertoken_swiglu.hpp         # dequantizes GMM1 output, applies SwiGLU, then per-token requantizes
│       ├── block_epilogue_pertoken_row.hpp            # CombineV1: row-wise dequant and local store or remote TPUT
│       ├── block_epilogue_pertoken_v2.hpp             # CombineV2: splits GMM tiles by destination rank and writes back to offsetD
│       ├── pto_mmad_ops.hpp               # AIC matmul PTO tile wrappers plus scale/fixpipe-store helpers
│       ├── pto_vector_ops.hpp             # UB/Vec tile movement, cast, arithmetic, and reduce PTO helpers
│       ├── moe_pto_utils.hpp              # shared shape/layout, arch resource, workspace, sync, and copy helpers
│       ├── hccl_context.hpp               # device-side HCCL remote-window context data structures
│       ├── hccl_window.hpp                # remote-window address mapping plus cross-rank TGET/TPUT notify/wait wrappers
│       ├── layout3d.hpp                   # tokenPerExpert[dst][src][expert] 3D index layout
│       ├── const_args.hpp                 # common constants for alignment, flags, and remote-window offsets
│       └── dispatch_policy_custom.hpp     # A5 MMAD policy and the three epilogue policy tags
├── scripts/gen_data.py                    # generates rank inputs, CPU golden outputs, case.json, and verification data
├── README_zh.md                           # Chinese README
└── README.md                              # English README
```

## Operator Semantics

Each rank owns its local input tokens, local expert weights, and scales. `expert_idx` stores global expert ids, and experts are sharded by rank:

```text
global_expert = dst_rank * expert_per_rank + local_expert
```

For each active token and each top-k expert, the kernel computes:

$$
Y_i = \sum_{j=0}^{topK-1} prob_{i,j} \cdot FFN_{expert_{i,j}}(X_i)
$$

Current FFN data path:

```text
BF16 X
  -> routing / expand / per-token quantize to int8
  -> GMM1: int8 X @ int8 W1
  -> per-token/per-channel dequant
  -> SwiGLU: split N into N/2 + N/2
  -> per-token quantize to int8
  -> GMM2: int8 hidden @ int8 W2
  -> per-token/per-channel dequant
  -> top-k probability weighted combine
  -> restore to original token order
```

## Specification and Constraints

| Item | Current value |
| ---- | ------------- |
| Project / host target | `dispatch_combine_moe` |
| Kernel shared target | `dispatch_combine_moe_kernel` |
| Device kernel symbol | `dispatch_combine_moe` |
| Kernel type | `KERNEL_TYPE_MIX_AIC_1_2` |
| Input `x` | `M×K`, BF16 bits represented as `uint16_t` files |
| Input `weight1` | `expert_per_rank×K×N`, `int8`, Zn packed |
| Input `weight2` | `expert_per_rank×(N/2)×K`, `int8`, Zn packed |
| Input `expert_idx` | `M×topK`, `int32`, global expert id |
| Input `scale1/scale2` | FP32 scales viewed through packed `int64` storage |
| Input `probs` | `M×topK`, `float32` |
| Input `x_active_mask` | `M`, `uint8` |
| Output `out` | `M×K`, `float16` |
| Output `expert_token_nums` | `expert_per_rank`, `int32` |

Constraints:

- `N` must be even because the GMM1 output is split into `N/2 + N/2` for SwiGLU.
- `expert_idx` must be in `[0, world_size * expert_per_rank)`. Inactive tokens are rewritten by `ApplyXActiveMask()` to a sentinel expert id.
- `max_output_size` caps the routed-token workspace per rank. Too small a value truncates routed tokens in both CPU golden generation and the device path.
- HCCL context parsing stores `windowIn/windowOut` up to `PTO_HCCL_MAX_RANKS`.

## Host Layer

### Host execution loop

```text
run.sh
  -> scripts/gen_data.py generates case.json and rank*.bin
  -> cmake -S dispatch_combine_moe -B build_dir
  -> cmake --build build_dir --target dispatch_combine_moe
  -> mpirun -n world_size build_dir/dispatch_combine_moe
```

`main.cpp` implements the standalone run loop:

1. Initialize MPI through `comm_mpi.h` and bind each rank to the NPU device with the same id.
2. Rank 0 creates `HcclRootInfo` and broadcasts it to all ranks.
3. `InitStandaloneRankRuntime()` creates ACL/HCCL streams, HCCL communicator, and the HCCL remote window.
4. `LoadCaseConfig()` reads `case.json`; `BuildRankFileSet()` locates per-rank input and golden files.
5. `BuildDispatchCombineMoeTiling()` builds `DispatchCombineMoeTilingData`, workspace bytes, and block dim.
6. Before each warmup/measure iteration, clear HCCL windows, workspace, `out`, and `expert_token_nums`.
7. Launch the device kernel through `launchDispatchCombineMoe()`.
8. Measure with ACL events; rank 0 reports max-rank latency as `[PROFILE] dispatch_combine_moe`.
9. In verification, copy `out` back to host, write `output_rank*.bin`, and compare against `expected_out` as FP16.

### Host tiling builder

`op_host/tiling_builder.cpp` exposes `BuildDispatchCombineMoeTiling()`. It fills four categories of data:

1. `DispatchCombineMoeInfo`: `M/K/N/topK/expertPerRank/worldSize/maxOutputSize/aivNum`.
2. `CoCTiling`: `m0=128`, `k0=256`, `n0=256`, `ubMoveNum=16KiB`, `commNpuSplit=world_size`, routing quant tiling, and related parameters.
3. `DispatchCombineMoeRuntimeInfo`: rank id, rank size, and device-side remote-window context address.
4. `DispatchCombineMoeLaunchConfig`: `blockDim`, kernel branch key, and `workspaceBytes`.

It also validates the remote-window capacity on host before kernel launch.

## Device Entry and Policy Assembly

`op_kernel/dispatch_combine_moe.cpp` defines the device symbol:

```text
dispatch_combine_moe(..., workspaceGM, tilingGM)
  -> REGISTER_TILING_DEFAULT(DispatchCombineMoeTilingData)
  -> match the fixed kernel branch key
  -> KERNEL_TYPE_MIX_AIC_1_2
  -> DispatchCombineMoe<int8_t, DTYPE_W1, DTYPE_OUT, false, true>
  -> op.Init(...)
  -> op.Process()
```

`op_kernel/dispatch_combine_moe.h` converts tiling into the A5 execution policy inside `DispatchCombineMoe::Process()`:

- `ArchTag = pto_ext::Arch::AtlasA5`
- `L1TileShape = GemmShape<128, 256, 512>`
- `L0TileShape = GemmShape<128, 256, 128>`
- `MmadAtlasA5PreloadAsyncFixpipe<preloadStages=1, l1Stages=2, l0AStages=2, l0BStages=2, l0CStages=1, enableShuffleK=true>`
- `BlockEpilogue1 = EpilogueAtlasA5PerTokenDequantSwigluQuant`
- `BlockEpilogue2 = EpilogueAtlasA5PerTokenDequant`
- `BlockEpilogue3 = EpilogueAtlasA5PerTokenDequantV2`
- `BlockScheduler = GemmIdentityBlockSwizzle<9, 1>`

It then assembles `DispatchCombineMoeKernel::Params` and lets `DispatchCombineMoeKernel` dispatch by `g_coreType`.

## Main Kernel Pipeline

`op_kernel/dispatch_combine_moe_kernel.hpp` is the main device-side orchestrator. It fuses AIV data organization, remote-window communication, AIC grouped matmul, epilogue processing, and restore into one pipeline.

### AIC/AIV split

```text
DispatchCombineMoeKernel::operator()
├── AIC: RunGmm1Impl()
│        -> RunGmmInterlockImpl()
│        -> RunGmm2Impl()
└── AIV: RunRoutingImpl()
         -> RunDispatchGatherImpl()
         -> RunSwigluImpl()
         -> RunCombineImpl()
         -> RunRestoreImpl()
```

### Main data flow

```text
x / expert_idx / probs / x_active_mask
  -> AIV RunRoutingImpl
       ApplyXActiveMask
       moe_init_routing_quant
       token-count all-gather + preSumBeforeRank + cumsumMM
  -> AIV RunDispatchGatherImpl
       peer window packed rows -> local gmA expert-major input
  -> AIC RunGmm1Impl
       grouped matmul by local expert
  -> AIV RunSwigluImpl
       GMM1 C -> dequant -> SwiGLU -> dynamic quant -> gmPermutedToken
  -> AIC RunGmm2Impl
       grouped matmul by local expert
  -> AIV RunCombineImpl
       GMM2 C2 -> dequant -> local/remote offsetD
  -> AIV RunRestoreImpl
       unpermute + top-k weighted accumulation -> out[M, K]
```

### Key code reading order

1. `op_kernel/dispatch_combine_moe.cpp`: device symbol, tiling key, mixed AIC/AIV task type, host launch stub.
2. `op_kernel/dispatch_combine_moe.h`: `Init()` for shape/rank/window state and `Process()` for policy and parameter assembly.
3. `op_kernel/dispatch_combine_moe_kernel.hpp`: `operator()<AIC/AIV>()`, five AIV `Run*Impl()` stages, and three AIC `Run*Impl()` stages.
4. `op_kernel/token_reorder/routing/moe_init_routing_quant.cpp`: routing branch selection by tiling key.
5. `op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp`: AIC GMM data movement and `Finalize()` synchronization points.
6. `op_kernel/utils/block_epilogue_pertoken_swiglu.hpp`: GMM1 output to GMM2 input.
7. `op_kernel/utils/block_epilogue_pertoken_row.hpp` and `op_kernel/utils/block_epilogue_pertoken_v2.hpp`: GMM2 output store/TPUT by row or tile.
8. `op_kernel/token_reorder/unpermute/moe_token_unpermute.h`: final accumulation back to `out[M, K]` using `expandedRowIdx` and `probs`.

### Core variable dictionary

| Variable / region | Meaning | Main producer | Main consumer |
| ----------------- | ------- | ------------- | ------------- |
| `problemShape = [M, N, K]` | `M` tokens, `N` GMM1/SwiGLU dimension, `K` input/output hidden size | host tiling | GMM1/GMM2/epilogue |
| `EP` | world size / expert-parallel rank count | host tiling | count exchange, dispatch, combine |
| `expertPerRank` | local expert count per rank | host tiling | `groupIdx` loops, tokenPerExpert layout |
| `topK` | experts per token | host tiling / input | routing, restore |
| `expandedRowIdx` | expanded/top-k row to original token/top-k slot mapping | routing | unpermute/restore |
| `tokenPerExpert[dst][src][expert]` | routed-token count for each target rank, source rank, and local expert | routing count exchange | dispatch-gather, combine |
| `preSumBeforeRank[dst][expert]` | current source-rank start offset inside a destination-rank expert segment | count exchange | dispatch-gather, CombineV2 |
| `cumsumMM` | expert-major GMM row boundary after source-rank prefix sum | `GetCumsumForMMAIV()` | AIC GMM, dispatch-gather, combine |
| `gmA` | local expert-major int8 GMM1 input | `RunDispatchGatherImpl()` | `GMM1()` |
| `gmPerTokenScale1` | per-token scale for GMM1 input | routing / dispatch-gather | SwiGLU epilogue |
| `gmC` | GMM1 output workspace | `GMM1()` | `RunSwigluImpl()` |
| `gmPermutedToken` | int8 GMM2 input after SwiGLU and requantization | `RunSwigluImpl()` | `GMM2()` |
| `gmPerTokenScale2` | per-token scale for GMM2 input | `RunSwigluImpl()` | combine epilogue |
| `gmC2` | GMM2 output workspace | `GMM2()` | `CombineV1()` / `CombineV2()` |
| remote `offsetD` | expanded output returned to source-rank windows | combine epilogue | unpermute/restore |

### Workspace layout

`WorkspaceInfo` slices the GM workspace passed to the kernel launch (`ptrWorkspace`) into these major regions:

| Region | Purpose |
| ------ | ------- |
| `expandedRowIdx` | expanded row index generated by routing; restore uses it to recover token order |
| `ptrcumsumMM` | GMM row boundaries from `tokenPerExpert` source-rank prefix sums |
| `ptrPerTokenScale` | GMM1 per-token scale from dispatch/routing quantization |
| `ptrPerTokenScale2` | GMM2 per-token scale generated after SwiGLU requantization |
| `ptrC` | GMM1 fixpipe output workspace |
| `ptrC2` | GMM2 fixpipe output workspace |
| `ptrA` | expert-major int8 GMM1 input after dispatch-gather |
| `ptrPermutedToken` | int8 GMM2 input after SwiGLU |
| `ptrSumBeforeRank` | peer-rank prefix sums needed by combine/restore |
| `ptrSoftFlagBase` | soft-progress helper region |

Cross-rank visible data lives in the HCCL remote window, not in ordinary workspace.

## Routing Subsystem: `token_reorder/routing`

The routing entry point is `moe_init_routing_quant()` in `token_reorder/routing/moe_init_routing_quant.cpp`. It only runs on AIV cores; AIC cores return immediately.

### Routing inputs and outputs

Inputs:

- `x`: original BF16 tokens.
- `expertIdx`: `M×topK` global expert ids.
- `scale` / `offset`: optional dynamic-quant smoothing inputs; the main path passes scale and keeps offset reserved.
- `workspace`: routing-internal sort and temporary-index workspace.
- routing tiling: generated by `MoeInitRoutingQuantTilingBase::DoTiling()`; selects the full-load, one-core sort, or multi-core sort path.

Outputs:

- `expandedX`: int8 payload after routing-order expansion and per-token quantization; the main path writes it to `remoteWindow + offsetA`.
- `expandedRowIdx`: original token/top-k mapping for each expanded row; the main path stores it in ordinary workspace.
- `expertTokensCountOrCumsum`: local token-per-expert statistics; the main path stores it in the remote-window token-count region.
- `dynamicQuantScale`: per-expanded-row scale; the main path stores it in `remoteWindow + offsetPeerPerTokenScale`.

### Routing branches

```text
moe_init_routing_quant()
├── full-load path
│   └── RunFullLoadDynamicQuant()
│       └── MoeFullLoadDynamicQuant
│           sort + count/cumsum + gather/quant in the full-load path
├── one-core sort path
│   ├── RunSortStage<MoeSortOneCore>()
│   ├── RunExpertTokenOut()
│   ├── RunSrcToDst()
│   └── RunGatherDynamicQuant()
└── multi-core sort path
    ├── RunSortStage<MoeSortMultiCore>()
    ├── RunExpertTokenOut()
    ├── RunSrcToDst()
    └── RunGatherDynamicQuant()
```

Static-check refactoring split the entry into small helpers, so each routing stage now has one clear call boundary:

| Helper | Wrapped implementation | Role |
| ------ | ---------------------- | ---- |
| `RunFullLoadDynamicQuant()` | `MoeFullLoadDynamicQuant` | one-pass full-load routing path; owns sort, expert statistics, source-to-destination mapping, gather, and per-row quantization |
| `RunSortStage<SortOp>()` | `MoeSortOneCore` / `MoeSortMultiCore` | runs the one-core or multi-core sort selected by host tiling and materializes ordered routing records |
| `RunExpertTokenOut()` | `MoeExpertTokenOut` | emits expert token count/cumsum output when `expertTokensCountOrCumsumFlag` requests it |
| `RunSrcToDst()` | `MoeSrcToDstOp` | converts sorted routing records into `expandedRowIdx` source-to-destination rows |
| `RunGatherDynamicQuant()` | `MoeGatherDynamicQuant` | gathers original token rows by `expandedRowIdx`, computes per-token abs-max scale, and writes int8 payload plus scale |

### Routing file responsibilities

| File | Responsibility |
| ---- | -------------- |
| `moe_init_routing_quant.cpp` | routing subsystem entry; selects full-load, one-core sort, or multi-core sort from host tiling |
| `moe_init_routing_quant_tiling.h` | quant routing tiling: workspace size and full-load/gather branch parameters |
| `moe_init_routing_tiling_common.h` | common routing tiling: `AiCoreParams`, `TilingBaseClass`, VBS/VMS/sort-out/src-to-dst/gather sub-tiling |
| `moe_init_routing_sort.h` | local routing-sort orchestration; defines `MoeSortBase`, `MoeSortOneCore`, and `MoeSortMultiCore` |
| `moe_packed_sort_merge.h` | packed sort-record merge/extract: `MoeMrgsort`, `MoeMrgsortOut`, `PtoMergePackedSortRecords` |
| `moe_init_routing_expert_tokens.h` | `MoeExpertTokenOut` and `MoeSrcToDstOp`: expert token counts/cumsum, `expandedRowIdx`, source-to-destination row mapping |
| `moe_init_routing_fullload_dynamic_quant.h` | `MoeFullLoadDynamicQuant` full-load path: sort, count/cumsum, row-wise quant, int8 payload and scale output |
| `moe_init_routing_gather_dynamic_quant.h` | `MoeGatherDynamicQuant` gather path: read original tokens by `expandedRowIdx`, row-wise dynamic quant, write `expandedX` and scale |
| `moe_pto_sort.h` | PTO UB int32 sort, packed sort, vector helpers, and AscendC sync bridge |
| `moe_common.h` | constants, alignment helpers, and global memory initialization |

### Routing data semantics

1. `ApplyXActiveMask()` rewrites inactive tokens to the sentinel expert id `expertNum` before routing.
2. Sorting groups expanded tokens by expert while preserving `expandedRowIdx` for final restore.
3. Dynamic quantization computes an abs-max per row, emits the per-token scale, and writes int8 payload.
4. `expertTokensCountOrCumsum` first describes local expert counts; the main kernel then all-gathers it into `tokenPerExpert[dst_rank][src_rank][local_expert]`.

## Dispatch-gather and Cross-rank Count Synchronization

After routing, `RunRoutingImpl()` does two things:

1. `CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2()`:
   - publish local token counts to peer windows;
   - issue `TPUT` to peers and use remote-window token-ready notify/wait;
   - read the full `tokenPerExpert[dst_rank][src_rank][local_expert]` view;
   - compute `preSumBeforeRank` for each destination rank.
2. `GetCumsumForMMAIV()`:
   - prefix-sum `tokenPerExpert` along source rank;
   - generate `cumsumMM`, which is shared by AIV dispatch-gather and AIC GMM.

`RunDispatchGatherImpl()` then iterates local experts by `groupIdx`:

```text
for each local expert groupIdx:
  currentM = cumsumMM[last_src_rank, groupIdx]
  for each dstEpIdx assigned to this AIV core:
    CopyDispatchRowsForPeer(dstEpIdx, groupIdx)
      rows = tokenPerExpert[dstEpIdx][rank][groupIdx]
      rowStart = cumsumMM prefix + previous expert-group sum
      rowSrc = preSumBeforeRank prefix carried across dst ranks
      TGET peer offsetA packed rows
      strip UB_ALIGN payload metadata
      write gmA[rowStart, :]
      write gmPerTokenScale1[rowStart]
  SyncAll
  CrossCoreSetFlag -> AIC GMM1 can consume this expert group
  UpdateDispatchDequantSums(groupIdx, currentM)
```

The dispatch-gather stage is split into three layers:

| Layer | Code | Responsibility |
| ----- | ---- | -------------- |
| Expert loop orchestration | `RunDispatchGatherImpl()` | walks local experts, publishes the per-group GMM1 readiness flag, and records the two SwiGLU segment sizes |
| Peer-row copy | `CopyDispatchRowsForPeer()` | maps `dstEpIdx/groupIdx` into local `gmA` row offsets, clamps by `maxOutputSize`, and calls the packed-row copy helper |
| Packed payload transfer | `CopyGMToGMPerToken()` | issues `pto::comm::TGET` from peer `offsetA`, then splits the packed `[int8 payload | UB_ALIGN metadata]` row into `gmA` and `gmPerTokenScale1` |
| SwiGLU segment accounting | `UpdateDispatchDequantSums()` | accumulates `stageDequantSum1` before `epilogueGranularity` and `stageDequantSum2` after it, with the same `maxOutputSize` cap as GMM |

## AIC GMM Subsystem

AIC-side GMM1 and GMM2 both use `BlockMmad<MmadAtlasA5PreloadAsyncFixpipe<...>>`. This layer moves A/B/scale from GM into L1/L0 buffers, runs PTO matmul, and fixpipe-stores results back to GM workspace.

### MMAD staged flow

```text
GM A/B/Scale
  -> TLOAD GM -> L1(Mat)
  -> TMOV L1(Mat) -> L0A(Left) / L0B(Right)
  -> TMATMUL / TMATMUL_ACC
  -> TSTORE_FP L0C/Scale -> GM C
```

File responsibilities:

| File | Responsibility |
| ---- | -------------- |
| `block_mmad_preload_async_fixpipe_quant.hpp` | manage L1/L0A/L0B/L0C staged buffers, k-loop preload, and `CrossCoreSetFlag` finalize |
| `pto_mmad_ops.hpp` | wrap PTO primitives: `PtoLoadNdGmToNzL1` / `PtoLoadNzGmToNzL1` for GM→L1, `PtoMoveL1ToL0A` / `PtoMoveL1ToL0B` for L1→L0A/L0B, `PtoTileMmad` for matmul, `StagePerChannelScale` for per-channel scale staging, and `StoreAccumulator` / `PtoStoreAccToGm` for fixpipe store |
| `dispatch_policy_custom.hpp` | define `MmadAtlasA5PreloadAsyncFixpipe` and epilogue policy tags |
| `moe_pto_utils.hpp` | shared arch resource, shape/layout helpers, tile copy traits, and sync wrappers |

`pto_mmad_ops.hpp` now separates primitive wrappers from type-policy selection:

| Helper | Role |
| ------ | ---- |
| `LaunchPtoMatmul()` | chooses `TMATMUL` versus `TMATMUL_ACC` and maps `unitFlag` to normal / partial / final accumulation phases |
| `PtoTileMmad()` | binds L0A/L0B/L0C tile offsets and launches one matmul tile |
| `PtoMoveL1ToL0A()` / `PtoMoveL1ToL0B()` | perform `TMOV(Mat -> Left/Right)` after GM data has been staged into L1 Mat tiles |
| `StagePerChannelScale()` | loads per-channel scale from GM into an L1 Mat tile and moves it into a FIXPIPE Scaling tile |
| `StoreAccumulator()` | dispatches fixpipe store by input type: int8 uses per-channel scaling, half uses the non-scale accumulator store path |
| `MatmulShell` | centralizes derived element/layout/trait aliases used by `BlockMmad`, so the staged kernel code does not repeat type plumbing |

GMM differences:

- `GMM1` consumes `gmA`, `weight1`, and `scale1`, and writes `gmC`.
- `GMM2` consumes `gmPermutedToken`, `weight2`, and `scale2`, and writes `gmC2`.
- Both iterate local experts by `groupIdx`; `BlockScheduler` maps each expert's `[M, N, K]` problem to AIC block/tile work.

## Epilogue Subsystem

The three `block_epilogue_*` files serve different stages and granularities.

### `block_epilogue_pertoken_swiglu.hpp`: post-GMM1 path

Used by `RunSwigluImpl()`.

```text
gmC: GMM1 half output [routed_rows, N]
gmPerTokenScale1
  -> dequant to fp32
  -> split N into N/2 + N/2
  -> x0 * sigmoid(x0) * gate
  -> dynamic quant to int8
gmPermutedToken: GMM2 int8 input [routed_rows, N/2]
gmPerTokenScale2
```

Key points:

- Processes GMM1 output row by row; each row splits `N` into two `N/2` chunks.
- Casts half C to FP32 and multiplies by `gmPerTokenScale1[row]` for per-token dequant.
- Builds SwiGLU from PTO vector primitives: `TMULS`, `TEXP`, `TADDS`, `TDIV`, `TMUL`.
- Computes abs-max/reduce-max for the SwiGLU output, emits `gmPerTokenScale2[row]`, and writes quantized int8 data to `gmPermutedToken`.

### `block_epilogue_pertoken_row.hpp`: CombineV1 row-level path

Used when `RunCombineImpl()` selects `CombineV1`.

```text
gmC2: GMM2 half output [routed_rows, K]
gmPerTokenScale2
  -> dequant to fp32
  -> cast to output dtype
  -> local store or remote TPUT to peer offsetD
```

Key points:

- Processes one full row of `K` at a time.
- Loads GM to UB, casts half to float, multiplies by per-token scale, then casts to output dtype.
- Stores directly for local destination rank.
- For remote destination rank, writes a local remote-window scratch row and issues `pto::comm::TPUT` to the peer `offsetD` region.

### `block_epilogue_pertoken_v2.hpp`: CombineV2 tile-level path

Used when `RunCombineImpl()` selects `CombineV2`. Current `initBuffer()` logic is:

```text
isCombineV1 = true
if M * topK <= 4096:
  isCombineV1 = false   # small cases default to CombineV2
```

Key points:

- Processes GMM tile/block coordinates instead of scanning complete rows in order.
- Splits each tile into `m0 = 16` row blocks and distributes them across two AIV sub-cores.
- Uses `tokenPerExpert[dst_rank][src_rank][groupIdx]` and `preSumBeforeRank[dst_rank][groupIdx]` to decide which rows in the tile belong to which destination rank.
- `LoadTileC()` loads a non-contiguous `gmC2` tile into UB and casts half output to FP32 row by row.
- `ScaleAndCastTile()` caches per-token scales in UB, multiplies each FP32 row by its scale, then casts to the output dtype.
- `StoreTileD()` walks destination ranks, intersects the current tile row interval with each rank-owned expert interval, and advances a tile-local row offset.
- Local rows are stored with `PtoStoreMatrixRows()`; remote rows go through `StoreRemoteRows()`, which stores one row into a per-sub-core scratch row and issues `pto::comm::TPUT` to the peer `offsetD` slice.

## Combine and Restore

`RunCombineImpl()` builds two epilogues:

- `BlockEpilogue2 = EpilogueAtlasA5PerTokenDequant`, used by `CombineV1()`.
- `BlockEpilogue3 = EpilogueAtlasA5PerTokenDequantV2`, used by `CombineV2()`.

Both paths write expert-major GMM2 output back to each source rank's remote-window `offsetD`; they differ in granularity.

| Path | Granularity | Main file | Trait |
| ---- | ----------- | --------- | ----- |
| CombineV1 | expert group + row | `block_epilogue_pertoken_row.hpp` | simple row-level writeback after AIC group flags |
| CombineV2 | GMM block/tile + sub-core rows | `block_epilogue_pertoken_v2.hpp` | uses BlockScheduler tile coordinates and finer AIV sub-core parallelism |

`RunRestoreImpl()` is the final AIV stage:

```text
SyncAll
ResetTokenPerExpert
remoteWindow.CrossRankSync()
MoeTokenUnpermuteTiling(M * topK, K, topK, ...)
KernelMoeTokenUnpermute<ElementD2, int32_t, float, true>
  input: remoteWindow + offsetD
  index: expandedRowIdx
  prob: probs
  output: out
```

`token_reorder/unpermute/moe_token_unpermute.h` restores expanded/top-k ordered outputs back to the original token order:

1. Each AIV core owns a slice of output tokens.
2. For each output token, iterate its `topK` expanded rows.
3. Read original-token mapping from `expandedRowIdx`.
4. Read expert output slices from `remoteWindow + offsetD`.
5. If `PROBS=true`, read `probs` and apply probability weighting.
6. Process the hidden dimension in chunks to fit UB.
7. Write the accumulated result to final `out[M, K]`.

## Remote Window Layout

Only cross-rank visible data and signals live in the HCCL RDMA remote window. Ordinary inputs, outputs, workspace, and tiling remain allocated by `aclrtMalloc`.

```text
offsetA                  = 0
offsetPeerPerTokenScale  = AlignUp(windowBytes / 3, 512)
offsetD                  = offsetPeerPerTokenScale + 1 MiB
offsetPeerTokenPerExpert = windowBytes - 2 MiB
signalBase               = windowBytes - 1 MiB
```

| Region | Producer | Consumer | Purpose |
| ------ | -------- | -------- | ------- |
| `offsetA` | routing / dynamic quant | dispatch-gather | expanded int8 token payload |
| `offsetPeerPerTokenScale` | routing / dispatch-gather / combine scratch | GMM1 epilogue / TGET/TPUT scratch | per-token scales and temporary scratch |
| `offsetD` | combine | restore/unpermute | expanded output after GMM2 dequant, returned by source rank |
| `offsetPeerTokenPerExpert` | routing count exchange | dispatch-gather / combine | `tokenPerExpert[dst][src][expert]` |
| signal region | `PtoRemoteWindow` | `PtoRemoteWindow` | barriers, token-ready signals, notify/wait signals |

## Compute-Communication Overlap

The steady-state goal is expert-level skewed execution:

```text
Dispatch phase:  Comm(i + 1)  ||  GMM(i)
Combine phase:   GMM(i)       ||  Comm(i - 1)
```

Here `Comm` is not a standalone op-level AlltoAllV call. It is assembled from remote-window `TGET/TPUT`, token-ready signals, and rank-local cross-core flags.

Three dependency chains matter:

1. GMM1 only depends on the current expert group's input rows being dispatch-gathered into `gmA`; it does not need all experts to be gathered.
2. SwiGLU only depends on the previous segment of GMM1 output being finalized; it does not need all GMM1 groups to finish.
3. Combine only depends on the corresponding GMM2 group/tile being visible; restore waits until combine is synchronized.

```text
Time  ───────────────────────────────────────────────────────────────>

AIV : routing/count/quant -> token-count exchange -> cumsumMM
AIV : | dispatch gather expert0 | dispatch gather expert1 | dispatch gather expert2 | ...
AIC :                         | GMM1 expert0            | GMM1 expert1            | ...
AIV :                                      | dequant + SwiGLU + quant segment0 |
AIC :                                                         | GMM2 expert0 | GMM2 expert1 | ...
AIV :                                                  | combine tile/row expert-1 | combine expert0 | ...
AIV : restore / unpermute / probability accumulation
```

### Producer / consumer / signal map

| Producer | Consumer | Data | Synchronization | Code |
| -------- | -------- | ---- | --------------- | ---- |
| routing count exchange | dispatch-gather / GMM planning | `tokenPerExpert`, `preSumBeforeRank`, `cumsumMM` | `remoteWindow.NotifyRemoteTokenReady()` / `WaitTokenReady()` + `PtoSyncAll` | `CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2()`, `GetCumsumForMMAIV()` |
| AIV dispatch-gather | AIC GMM1 | `gmA`, `gmPerTokenScale1` | `CrossCoreSetFlag<0x2, PIPE_MTE3>` and AIC `CrossCoreWaitFlag` | `RunDispatchGatherImpl()`, `GMM1()` |
| AIC GMM1 | AIV SwiGLU | `gmC` | `blockMmad.Finalize(..., SYNCFLAGC2V)` and AIV `CrossCoreWaitFlag(SYNCFLAGC2V)` | `GMM1()`, `RunSwigluImpl()` |
| AIV SwiGLU | AIC GMM2 | `gmPermutedToken`, `gmPerTokenScale2` | `CrossCoreSetFlag(SYNCFLAGV2C)` and AIC wait | `RunSwigluImpl()`, `RunGmmInterlockImpl()`, `GMM2()` |
| AIC GMM2 | AIV combine | `gmC2` | AIC finalize flag; CombineV1/CombineV2 wait by group/tile | `GMM2()`, `CombineV1()`, `CombineV2()` |
| AIV combine | AIV restore | remote `offsetD` | `ResetTokenPerExpert()` + `remoteWindow.CrossRankSync()` | `RunCombineImpl()`, `RunRestoreImpl()` |

### How `epilogueGranularity` creates two-stage SwiGLU/GMM2 overlap

`epilogueGranularity` splits local experts into front and back segments:

```text
front groups: [0, epilogueGranularity)
back groups : [epilogueGranularity, expertPerRank)
```

- `RunDispatchGatherImpl()` accumulates `stageDequantSum1` and `stageDequantSum2`, the routed-row counts for the two SwiGLU segments.
- `GMM1()` emits `Finalize(..., SYNCFLAGC2V)` when `groupIdx + 1 == epilogueGranularity`, allowing AIV to process the first segment of `gmC`.
- `RunSwigluImpl()` emits `SYNCFLAGV2C` after finishing that first segment, so AIC `GMM2()` can consume the first segment of `gmPermutedToken` early.
- `GMM1()` emits a second `SYNCFLAGC2V` after the remaining groups; AIV then processes the second segment and emits another `SYNCFLAGV2C`.

This is why the `GMM1 -> SwiGLU -> GMM2` path is not a single global barrier chain.

## Defaults and Tiling

`run.sh` generates a small smoke case by default:

| Parameter | Default |
| --------- | ------- |
| `world_size` | 2 |
| `M` | 16 |
| `K` | 128 |
| `N` | 128 |
| `topK` | 2 |
| `expert_per_rank` | 2 |
| `max_output_size` | 32 |
| `seed` | 20260515 |
| `warmup_iters` | 3 |
| `measure_iters` | 5 |

Important host-tiling constants and derived values:

| Parameter | Value / source |
| --------- | -------------- |
| `m0` | 128 |
| `k0` | 256 |
| `n0` | 256 |
| `swizzleDirect` | 1 |
| `swizzleOffset` | 7 |
| `ubMoveNum` | 16 KiB |
| `commNpuSplit` | `world_size` |
| `commDataSplit` | 1 |
| `lenPerLoop` | `m0 * n0 / 2` |
| `totalUbSize` | 196352 |
| system workspace | 16 MiB |
| `aivNum` | `PlatformAscendCManager::GetCoreNumAiv()` |
| `block_dim` | `CalcTschBlockDim(aivNum, aicNum, aivNum)` |

## Build and Run

### Smoke run

Set the CANN and MPI environment before running on an A5-capable environment:

```bash
export ASCEND_CANN_PATH=/home/XXX/Ascend/cann-9.0.0/set_env.sh
export ASCEND_HOME_PATH=/home/XXX/Ascend/cann-9.0.0
source /home/XXX/Ascend/cann-9.0.0/set_env.sh
export PATH=/usr/local/mpich/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/mpich/lib:$LD_LIBRARY_PATH
export MPI_LIB_PATH=/usr/local/mpich/lib/libmpi.so
```

Then run the smoke case:

```bash
bash kernels/manual/a5/dispatch_combine_moe/run.sh \
  --soc Ascend950PR_958b \
  --world-size 2 \
  --m 16 \
  --k 128 \
  --n 128 \
  --topk 2 \
  --experts 2 \
  --max-output-size 32
```

`ASCEND_CANN_PATH` lets `run.sh` locate and source the CANN environment. `ASCEND_HOME_PATH` is the CANN include/lib/platform root used by CMake; `set_env.sh` normally sets it, but export it explicitly if the target environment does not.

If the platform config uses a different SoC name, replace `--soc` / `--soc-version` accordingly. If omitted, host tiling uses the default `PlatformAscendCManager::GetInstance()` platform.

### `run.sh` arguments

| Argument | Meaning | Default |
| -------- | ------- | ------- |
| `--soc`, `--soc-version` | SoC version passed to host tiling | empty |
| `--world-size` | MPI rank count | 2 |
| `--m` | token count | 16 |
| `--k` | input/output hidden size | 128 |
| `--n` | GMM1 output dimension before SwiGLU | 128 |
| `--topk` | experts per token | 2 |
| `--experts` | experts per rank | 2 |
| `--max-output-size` | routed-token workspace cap per rank | 32 |
| `--seed` | random seed for data generation | 20260515 |
| `--atol` | FP16 comparison absolute tolerance | 1e-3 |
| `--rtol` | FP16 comparison relative tolerance | 1e-3 |
| `--warmup-iters` | warmup iterations | 3 |
| `--measure-iters` | measured iterations | 5 |

### Environment variables

| Environment variable | Purpose | Default behavior |
| -------------------- | ------- | ---------------- |
| `ASCEND_CANN_PATH` | CANN install directory or `set_env.sh` path | auto-search latest `/usr/local/Ascend/cann-*/set_env.sh` |
| `ASCEND_HOME_PATH` | CANN include/lib/platform root for CMake | normally set by `set_env.sh`; CMake errors if absent |
| `ASCEND_DRIVER_PATH` | driver kernel include root | `/usr/local/Ascend/driver` |
| `MPI_ENV_BIN` | directory containing `mpirun` | prepended to `PATH` if valid |
| `MPI_ENV_LIB` | MPI library directory | prepended to `LD_LIBRARY_PATH`; defaults `MPI_LIB_PATH` |
| `MPI_SEARCH_DIRS` | MPI search directories | conda `ltr_pto` and common MPICH paths |
| `MPI_LIB_PATH` | absolute path to `libmpi.so` | inferred by `run.sh` when possible |
| `MPI_RUNNER` | optional MPI runner override | `mpirun` |
| `DISPATCH_COMBINE_MOE_BUILD_DIR` | CMake build directory | `/tmp/dispatch_combine_moe_a5_run_build` |
| `DISPATCH_COMBINE_MOE_CASE_DIR` | host case directory | `run.sh` exports this directory's `out/`; direct host runs default to `../out` |
| `DISPATCH_COMBINE_MOE_SOC_VERSION` | SoC version for host tiling | set by `--soc` / `--soc-version` |
| `DISPATCH_COMBINE_MOE_WARMUP_ITERS` | host warmup iterations | 3 |
| `DISPATCH_COMBINE_MOE_MEASURE_ITERS` | host measured iterations | 5 |

## Data Files

`gen_data.py` generates:

| File | Content |
| ---- | ------- |
| `case.json` | shapes, topK, expert counts, comparison tolerances, logical workload statistics |
| `rank{r}_x.bin` | `M×K` BF16 bits |
| `rank{r}_weight1.bin` | local W1 experts, int8 Zn packed |
| `rank{r}_weight2.bin` | local W2 experts, int8 Zn packed |
| `rank{r}_expert_idx.bin` | `M×topK` global expert ids |
| `rank{r}_scale1.bin` | W1 dequant scales, FP32 packed into int64 view |
| `rank{r}_scale2.bin` | W2 dequant scales, FP32 packed into int64 view |
| `rank{r}_probs.bin` | `M×topK` probabilities |
| `rank{r}_x_active_mask.bin` | active token mask |
| `rank{r}_expected_out.bin` | CPU golden FP16 output |
| `output_rank{r}.bin` | device output written during host verification |

The CPU golden follows the device semantics: iterate routed tokens by destination rank, local expert, source rank, token, and top-k slot. Routed tokens beyond `max_output_size` are truncated.

## Common Issues

| Issue | Cause / fix |
| ----- | ----------- |
| `Cannot find CANN set_env.sh` | Set `ASCEND_CANN_PATH` to the CANN install directory or `set_env.sh`. |
| `Cannot find ASCEND_HOME_PATH` | Source CANN `set_env.sh`, or export `ASCEND_HOME_PATH=<cann-install>`. |
| `Unsupported A5 SOC_VERSION` | `run.sh` expects `--soc` / `--soc-version` to start with `Ascend`; check the platform SoC name. |
| `per-token-scale region is too small` | `max_output_size` is too large or the window is too small; the scale region reaches `offsetD`. |
| `dispatch output region is too small` | `max_output_size * K * sizeof(int16_t)` exceeds the dispatch-output region. |
| `token-count region overlaps signal region` | token-count storage reaches the last 1 MiB signal region; use a larger window or smaller `world_size/expert_per_rank`. |
| hang at HCCL/MPI barrier | Check that all ranks start, rank/device ids match, and stale HCCL resources are cleaned. `run.sh` attempts `/dev/shm/sem.hccl*` and `ipcrm -a` cleanup. |
| `FAIL rank=<id>` or mismatch | Check shape constraints, even `N`, sufficient `max_output_size`, then inspect the first bad index in `output_rank*.bin` vs `expected_out`. |

## Build System

- Compiler: `bisheng`
- C++ standard: C++17
- A5 macro: `PTO_NPU_ARCH_A5`
- Kernel target: `dispatch_combine_moe_kernel`
- Host target: `dispatch_combine_moe`
- Kernel arch: `--cce-aicore-arch=dav-c310`
- Kernel dtype macros: `DTYPE_W1=int8_t`, `DTYPE_OUT=half`
- Kernel source: `op_kernel/dispatch_combine_moe.cpp`
- Host sources: `main.cpp`, `op_host/runtime_context.cpp`, `op_host/tiling_builder.cpp`, `op_host/data_utils.cpp`
- Main linked libraries: `runtime`, `ascendcl`, `hcomm`, `tiling_api`, `nnopbase`
- PTO include directories are placed before CANN include directories so the in-repo PTO headers are used.
