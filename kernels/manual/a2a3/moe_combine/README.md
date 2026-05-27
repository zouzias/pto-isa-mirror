# moe_combine - A2/A3 PTO MoE Combine Kernel

## Overview

This example implements the MoE combine stage with PTO on Ascend A2/A3-class chips. It is the return half of a
dispatch-compute-combine MoE pipeline: after local experts finish computing routed rows, the combine kernel returns those
rows to the original token owner rank and restores each token output with the gate weights.

The current kernel is a standalone combine kernel with an explicit low-level routing contract. It does not accept
MC2-style high-level inputs such as `expert_ids`, `assist_info_for_combine`, or `ep_send_counts` directly. Those fields
must already be lowered into `routeMeta`.

```text
expertOutput[local expert rows, K]
  -> variable-length return through HCCL peerWindow.ptrD
  -> cross-rank completion with TNOTIFY/TWAIT
  -> weighted restore: outputC[token, :] = sum(topK probs * returned rows)
```

## Supported AI Processors

- A2/A3, validated on Atlas 910B1

## Directory Layout

```text
kernels/manual/a2a3/moe_combine/
├── CMakeLists.txt           # Bisheng CCE + host build configuration
├── run.sh                   # One-click build/run wrapper, MPI discovery, HCCL_BUFFSIZE estimate
├── common.h                 # Shared ABI: shape, routeMeta layout, peerWindow layout, HCCL context
├── layout.h                 # Host-side layout calculators and HCCL_BUFFSIZE estimator
├── kernel_launchers.h       # Host-side kernel launcher declaration
├── moe_combine_kernel.cpp   # PTO AIV kernel: return + wait + weighted restore
├── main.cpp                 # Host orchestration: MPI, ACL, HCCL window, fixture, verify, profiling
├── golden.h                 # CPU golden route construction and output verification
├── hccl_context.h           # HCCL window bootstrap and peer-window address exchange
├── comm_mpi.h               # MPI dynamic loading wrapper
├── DESIGN.md                # Split-from-dispatch design notes
├── PHASE2_DESIGN.md         # PTO style refactor plan
├── IMPLEMENTATION_PLAN.md   # Historical implementation tracker
├── README.md                # English README
└── README_zh.md             # Chinese README
```

## Operator Description

### Functionality

For each rank, the operator consumes expert outputs that are already laid out by local expert and source rank. It then:

1. Reads `routeMeta` to know how many rows each source rank sent to each expert and where those rows are located in
   `expertOutput`.
2. Uses PTO `TPUT` to return each expert row to the token owner rank's HCCL peer window.
3. Uses `TNOTIFY` / `TWAIT` to wait until every peer has completed its return writes.
4. Reads `routeMeta.expandedRowIdx` and `probs` to restore `outputC[M, K]`.

For token `t`:

```text
outputC[t, :] = sum_{slot=0..topK-1} probs[t, slot] * ptrD[expandedRowIdx[t, slot], :]
```

### Scope

| Included | Not included |
| --- | --- |
| EP-domain combine return through HCCL window | Dispatch pack/gather kernel |
| Variable-length all-to-all-like return with `TPUT` | HCCL collective `AllToAllV` API |
| Weighted restore with `probs` | Expert FFN/GMM compute |
| Explicit low-level `routeMeta` contract | Quantization, TP ReduceScatterV, shared/copy/const experts |
| A2/A3 HCCL peer-window path | MC2 public ABI adapter |

## Entry Contract

### Kernel Launcher ABI

```cpp
void LaunchMoeCombineKernel(MoeCombineShape shape, uint32_t myRank,
                            uint8_t *expertOutput,
                            uint8_t *probs,
                            uint8_t *outputC,
                            uint8_t *routeMeta,
                            uint8_t *peerWindow,
                            uint8_t *hcclCtx,
                            uint8_t *workspace,
                            void *stream,
                            uint32_t launchBlockCount);
```

### Runtime Inputs

| Argument | Direction | Storage | Meaning |
| --- | --- | --- | --- |
| `shape` | input | value | Static shape and tuning fields (`ep`, `m`, `k`, `topK`, `expertPerRank`, `aivBlocks`, etc.) |
| `myRank` | input | value | Rank id in the EP domain |
| `expertOutput` | input | `aclrtMalloc` GM | Local expert result rows, shape `[maxOutputSize, K]`, fp16 |
| `probs` | input | `aclrtMalloc` GM | Gate weights, shape `[M, topK]`, fp32 |
| `outputC` | output | `aclrtMalloc` GM | Restored token output, shape `[M, K]`, fp16 |
| `routeMeta` | input | `aclrtMalloc` GM | Explicit combine routing ledger |
| `peerWindow` | input/output | HCCL RDMA window | Remote-visible `ptrD` return buffer and signal counters |
| `hcclCtx` | input | `aclrtMalloc` GM | Device-side HCCL window addresses for all ranks |
| `workspace` | scratch | `aclrtMalloc` GM | Local sync and scratch buffers |
| `stream` | input | ACL stream | Kernel launch stream |
| `launchBlockCount` | input | value | AIV block count for the kernel launch |

### `MoeCombineShape`

| Field | Meaning |
| --- | --- |
| `ep` | EP rank count |
| `m` | Tokens per rank |
| `k` | Hidden size |
| `topK` | Expert routes per token |
| `expertPerRank` | Local expert count per rank |
| `expertNum` | Global expert count, normally `ep * expertPerRank` |
| `maxOutputSize` | Per-rank expert-output row capacity |
| `aivBlocks` | Logical AIV block count; A3 default is `24`, can be overridden |

### `routeMeta` Layout

`routeMeta` is the explicit low-level combine ledger. It is local GM, not part of the HCCL window.

| Field | Shape | Meaning |
| --- | --- | --- |
| `peerTokenPerExpert` | `[ep, expertNumPadded]` int32 | Number of rows owned by each source rank for each global expert |
| `expandedRowIdx` | `[M * topK]` int32 | Token route to `peerWindow.ptrD` row mapping; `-1` means invalid route |
| `cumsumPerExpert` | `[ep, expertNumPadded]` int32 | Inclusive prefix by global expert for each source rank |
| `dispatchOffset` | `[expertPerRank]` int32 | Base row in `expertOutput` for each local expert |
| `prevSumBeforeRank` | `[ep, expertPerRank]` int32 | Per-source offset inside a local expert's rows |

This layout makes `moe_combine` independently callable without requiring callers to know the internal `workspace` or
`peerWindow` offsets.

## Optimization Notes

This kernel is an AIV-only combine kernel. It is bandwidth-bound for large hidden sizes such as `K=7168`, where one fp16
row is 14 KiB. The main optimization goal is to keep GM/HCCL-window movement streaming while minimizing control-path
overhead.

- **Explicit routeMeta**: routing metadata is passed as a separate GM buffer. `peerWindow` is reserved for remote-visible
  return data and signals, and `workspace` is reserved for local scratch.
- **Chunked return sharding**: the return phase iterates `src_rank x local_expert` segments and shards row chunks across
  AIV blocks with `chunkBase % blockNum`.
- **PTO `TPUT` ping/pong path**: remote return uses `TPUT(remoteDst, localSrc, ping, pong)`, allowing MTE2 load and MTE3
  store movement to pipeline through UB.
- **Route cache for restore**: when `topK <= 16`, route rows and probabilities are cached in scalar arrays per token so
  the inner restore loop does not reload route metadata.
- **DCCI batched acquire before restore**: each token refreshes the returned `ptrD` rows before consuming them, then uses
  one `dsb(DSB_DDR)` for the cached route batch.
- **TLOAD to TAXPY event chain**: restore accumulates each returned row with a PTO `TLOAD -> TAXPY` event dependency.
- **Soft AIV sync**: `SoftSyncAiv` separates return, wait, and restore stages within the same kernel launch.

## Tiling and Default Parameters

| Parameter | Default | Notes |
| --- | --- | --- |
| `PES` / `ep` | `2` | EP rank count |
| `M` | `64` | Tokens per rank |
| `K` | `7168` | Hidden size |
| `topK` | `8` | Expert routes per token |
| `expertPerPe` | `2` | Local experts per rank |
| `expertNum` | `4` | `PES * expertPerPe` |
| `maxOutputSize` | `PES * M * topK` | Default capacity, `1024` for the default shape |
| `aivBlocks` | `24` | A3 default logical AIV block count |
| Internal vector tile columns | `1024` | Fixed by the sample implementation |
| Internal return chunk | `8 rows` | Fixed return-stage row chunk |
| Internal metadata pad | `16` | Pads expert metadata rows |

For `PES=2, M=64, K=7168, topK=8, expertPerPe=2, aivBlocks=24`, the layouts are:

| Layout | Bytes |
| --- | --- |
| `workspace` | `22120704` |
| `routeMeta` | `2432` |
| `peerWindow` | `7340160` |

## Overall Architecture

```text
Host:
  ParseArgs -> ComputeWorkspaceLayout / ComputeCombineRouteMetaLayout / ComputePeerWindowLayout
    -> PrepareHostData and CPU golden
    -> Init HCCL peer window
    -> AllocateLocalBuffers(routeMeta/workspace/expertOutput/probs/outputC)
    -> loop(warmup + measured):
         ClearDeviceState
         PrepareCombineFixture -> writes routeMeta + expertOutput
         LaunchMoeCombineKernel
         Verify outputC

Device:
  ReturnExpertRowsToOwners -> WaitCombinePhase -> RestoreOutputRows
```

```text
Return phase:
  routeMeta(peerToken/cumsum/offset) + expertOutput
    -> local or remote peerWindow.ptrD
    -> TNOTIFY peer combineDoneSignal[myRank]

Restore phase:
  routeMeta.expandedRowIdx + probs + peerWindow.ptrD
    -> outputC
```

## Kernel Details

### Stage 1: ReturnExpertRowsToOwners

The kernel walks all local expert segments:

```text
segment = src_rank * expertPerRank + localExpert
globalExpert = myRank * expertPerRank + localExpert
rows = routeMeta.peerTokenPerExpert[src_rank, globalExpert]
```

For each non-empty segment:

1. `srcStart` is computed from `dispatchOffset[localExpert] + prevSumBeforeRank[src_rank, localExpert]`.
2. `dstStart` is computed from the previous expert prefix in `cumsumPerExpert`.
3. If `src_rank == myRank`, the row is copied locally into local `peerWindow.ptrD`.
4. Otherwise, PTO `TPUT` writes the row chunk into the source rank's remote peer window.

### Stage 2: WaitCombinePhase

After return writes finish, each rank notifies every token-owner rank:

```text
TNOTIFY(remotePeer.combineDoneSignal[myRank], AtomicAdd)
TWAIT(localPeer.combineDoneSignal[peer] >= 1)
```

The host clears `combineDoneSignal` before each iteration, so the kernel waits for one notify from every peer.

### Stage 3: RestoreOutputRows

Each AIV block owns a contiguous token shard. For each token and each column tile:

1. Initialize the output tile to zero with `TEXPANDS`.
2. For every valid route, load `ptrD[expandedRowIdx]`.
3. Accumulate with `TAXPY(outTile, ptrTile, prob)`.
4. Store the fp16 tile to `outputC`.

## Memory Layout and HCCL Window

Only buffers that remote ranks write must live in the HCCL window. Routing metadata and local scratch are ordinary GM
buffers.

| Buffer | Location | Contents |
| --- | --- | --- |
| `routeMeta` | `aclrtMalloc` | `peerTokenPerExpert`, `expandedRowIdx`, `cumsumPerExpert`, `dispatchOffset`, `prevSumBeforeRank` |
| `workspace` | `aclrtMalloc` | `localSync`, `floatScratch`, `dispatchedA`, `ptrDLocal` |
| `expertOutput` | `aclrtMalloc` | Local expert output rows |
| `probs` | `aclrtMalloc` | Gate weights |
| `outputC` | `aclrtMalloc` | Final token output |
| `peerWindow.ptrD` | HCCL window | Return destination rows, remotely written by `TPUT` |
| `peerWindow.combineDoneSignal` | HCCL window | Completion counters, remotely written by `TNOTIFY` |

On A2/A3, the live peer-window payload starts at the HCCL window base. There is no extra A5-style head guard in this
layout.

## Measured Performance

Latest validation in this workspace used 2 ranks on Atlas 910B1 with
`M=64, K=7168, topK=8, expertPerPe=2, aivBlocks=24`, `warmup=3`, and `iters=5`.

| Metric | Value |
| --- | --- |
| `workspace` | `22120704 bytes` |
| `routeMeta` | `2432 bytes` |
| `peerWindow` | `7340160 bytes` |
| `prepare_fixture` | `avg=70128.5 us`, `max=76100.4 us` |
| `combine_e2e` | `avg=631.3 us`, `max=1866.4 us` |
| Verification | `verify=PASS` |

`prepare_fixture` is host-side fixture copy time and is not part of the device kernel datapath. `combine_e2e` includes
kernel launch, return, wait, restore, stream sync, and MPI rank max.

## Build and Run

### Environment

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
export PATH=/home/ntlab/miniconda3/envs/ltr_pto/bin:$PATH
export LD_LIBRARY_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib:$LD_LIBRARY_PATH
export MPI_LIB_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib/libmpi.so
```

### Build Only

```bash
cd kernels/manual/a2a3/moe_combine
bash run.sh --skip-build 0 --clean-build 1
```

### Quick Verification

```bash
cd kernels/manual/a2a3/moe_combine
bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --aiv-blocks 2
```

### Default Shape

```bash
cd kernels/manual/a2a3/moe_combine
bash run.sh -pes 2 -M 64 -K 7168 -topK 8 -expertPerPe 2 --aiv-blocks 24
```

### Key CLI Options

| Option | Default | Meaning |
| --- | --- | --- |
| `-pes` | `2` | Rank count |
| `-M` | `64` | Tokens per rank |
| `-K` | `7168` | Hidden size |
| `-topK` | `8` | Routes per token |
| `-expertPerPe` | `2` | Experts per rank |
| `--max-output-size` | `PES * M * topK` | Expert output row capacity |
| `--aiv-blocks` | `0 -> 24` | Logical AIV block count, override when matching a hardware resource plan |
| `--device-base` | `0` | First device id used by rank-to-device mapping |
| `--ndevices` | `PES` | Visible device count used by the sample launcher |

## Verification

The host builds a deterministic CPU golden route ledger, writes it to `routeMeta`, copies `expertOutput`, launches the
kernel, and compares `outputC` with the CPU golden output. Verification is enabled by default.

Expected successful output:

```text
verify=PASS
```
