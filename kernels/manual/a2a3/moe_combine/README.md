# moe_combine — A2/A3 PTO MoE Combine Kernel

A hand-written PTO kernel implementing the **MoE combine** stage on Ascend A2/A3 (Atlas 910B).
It covers the full combine datapath after expert computation is finished:

```
expertOutput[localExperts × sourcesPerExpert, K]
  ─── variable-length return (TPUT) ──→  peerWindow.ptrD on token-owner rank
  ─── cross-rank done signal (TNOTIFY/TWAIT) ──→  synchronization barrier
  ─── weighted restore (TAXPY) ──→  outputC[M, K]
```

## Architecture Overview

```
┌─────────────────────────────── Device Kernel ────────────────────────────────┐
│                                                                              │
│  MoeCombineKernel(shape, myRank, expertOutput, probs, outputC,     │
│                              peerWindow, hcclCtx, workspace)                │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │ Stage 1: ReturnExpertRowsToOwners                                    │    │
│  │   • Iterate segments: src_rank × expertPerRank                       │    │
│  │   • Compute variable row count from peerTokenPerExpert               │    │
│  │   • Local return: TPUT (MTE2→MTE3 via UB ping/pong)                  │    │
│  │   • Remote return: TPUT via UB ping/pong to peer window              │    │
│  │   • Chunk-based sharding across AIV blocks                           │    │
│  │   • Post-return: TNOTIFY combineDoneSignal[myRank] on each peer      │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │ Stage 2: WaitCombinePhase                                            │    │
│  │   • TWAIT combineDoneSignal[peer] >= signalValue  ∀ peer             │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │ Stage 3: RestoreOutputRows                                           │    │
│  │   • Token-parallel across AIV blocks                                 │    │
│  │   • For each token: zero outputC, then                               │    │
│  │     for slot in topK:                                                │    │
│  │       outputC[token,:] += probs[token,slot] * ptrD[expandedRowIdx]   │    │
│  │   • Uses TEXPANDS (zero) / TLOAD / TAXPY / TSTORE                   │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Scope

| Included | Not Included |
|----------|--------------|
| Variable-length return via HCCL window | Dispatch pack/gather |
| Cross-rank signal synchronization | Expert FFN/GMM computation |
| Weighted restore with probs | HCCL collective AllToAllV |
| Multi-AIV-block parallelism | Dynamic routing / gating |

## Key Parameters

Defined in `common.h` → `MoeCombineShape`:

| Field | Description |
|-------|-------------|
| `ep` | Number of ranks (endpoint count) |
| `m` | Token count per rank |
| `k` | Hidden dimension |
| `topK` | Expert routes per token |
| `expertPerRank` | Local experts per rank |
| `expertNum` | Global expert count (`ep × expertPerRank`) |
| `maxOutputSize` | Max expert output rows per rank |
| `aivBlocks` | AIV block count for parallelism (0→24) |
| `tileCols` | Column tile width for vector operations (internal, 1024) |
| `rowChunk` | Return stage row chunk size (0→8) |
| `metadataPad` | Expert metadata padding granularity |
| `signalValue` | Current iteration signal epoch |

## Memory Layout

### Workspace (per-rank local)

| Field | Type | Description |
|-------|------|-------------|
| `localTokenPerExpert` | int32 | Token count per expert |
| `blockTokenPerExpert` | int32 | Per-block token assignments |
| `blockPrefixPerExpert` | int32 | Per-block prefix sums |
| `cumsumPerExpert` | int32 | Cross-rank cumulative sums |
| `dispatchOffset` | int32 | Local expert dispatch base offset |
| `prevSumBeforeRank` | int32 | Cumulative tokens before this rank |
| `localSync` | int32 | Soft sync workspace |
| `floatScratch` | float | Scratch buffer |
| `dispatchedA` | half | Dispatched input (combine fixture) |
| `ptrDLocal` | half | Local ptrD mirror |

### Peer Window (HCCL shared, per-rank)

| Field | Type | Description |
|-------|------|-------------|
| `peerTokenPerExpert` | int32 | Routing metadata (ep × expertNum) |
| `expandedRowIdx` | int32 | Token→ptrD row mapping |
| `packedA` | half | Packed dispatch data |
| `ptrD` | half | Return destination buffer |
| `countReadySignal` | int32 | Dispatch ready signals |
| `combineDoneSignal` | int32 | Combine done signals |

All fields are 64-byte aligned. Layout computed identically on host (`layout.h`) and device.

## UB Storage Plan

| Region | Offset | Size | Purpose |
|--------|--------|------|---------|
| Ping | 0x0000 | 4 KiB | Double-buffer tile A |
| Pong | 0x1000 | 4 KiB | Double-buffer tile B |
| Meta | 0x2000 | 4 KiB | Metadata TLOAD tile |
| Sync | 0x3000 | 256 B | SoftSyncAiv workspace |
| **Total** | | **~12.25 KiB** | ≤ 192 KiB budget |

## File Structure

```
moe_combine/
├── moe_combine_kernel.cpp  — PTO device kernel (return + wait + restore)
├── main.cpp                 — Host orchestration (MPI, HCCL, verify)
├── common.h                 — Shared ABI structs (Shape, Layout, Context)
├── kernel_launchers.h       — Kernel launch declaration
├── layout.h                 — Host-side layout calculators
├── args.h                   — CLI parsing and validation
├── golden.h                 — CPU golden reference and binary I/O
├── hccl_context.h           — HCCL window bootstrap (MESH/RING)
├── comm_mpi.h               — dlopen MPI wrapper
├── CMakeLists.txt           — Bisheng CCE + host build
├── run.sh                   — Build/run wrapper (mpirun)
├── DESIGN.md                — Split-from-dispatch design notes
├── PHASE2_DESIGN.md         — PTO style refactor plan
└── IMPLEMENTATION_PLAN.md   — Original task tracker
```

## Host Flow

```
ParseArgs → ValidateArgs → ComputeLayouts
  → InitRankInfo (MPI)
  → PrepareHostData (deterministic generate + CPU golden)
  → BindDeviceContinuous → CreateStreams → InitHccl
  → AllocateLocalBuffers → CopyInputsToDevice
  → loop(warmup + timed iterations):
       ClearDeviceState
       PrepareCombineFixture    ← writes metadata + expertOutput to device
       RunCombine               ← kernel launch + stream sync
       VerifyAndDump            ← compare outputC vs CPU golden
  → PrintProfileSummary → Cleanup
```

## Build & Run

### Prerequisites

- Ascend CANN 8.5+ with Bisheng compiler
- MPI library (`MPI_LIB_PATH`)
- PTO header library at `../../../../include`
- Hardware: Atlas 910B1 (A3) or compatible

### Build Only

```bash
bash run.sh --skip-build 0 --clean-build 1
```

### Quick Verification (small shape)

```bash
bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --debug 2
```

### Production-Scale Run

```bash
bash run.sh -pes 2 -M 64 -K 7168 -topK 8 -expertPerPe 2 --aiv-blocks 24
```

### Key CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `-pes` | 2 | Rank count |
| `-M` | 64 | Tokens per rank |
| `-K` | 7168 | Hidden dimension |
| `-topK` | 8 | Expert routes per token |
| `-expertPerPe` | 2 | Experts per rank |
| `--aiv-blocks` | 0 (→24) | AIV block parallelism |
| `--row-chunk` | 0 (→8) | Return chunk size |
| `--debug` | 0 | 0=quiet, 1=summary, 2=verbose |
| `--iters` | 1 | Measured iterations |
| `--warmup` | 1 | Warmup iterations |

Verification is always enabled; the kernel output is compared against a CPU golden reference with `rtol=1e-2, atol=1e-2`.

### Check Device Availability

```bash
npu-smi info
```

## Verification

The test framework uses an **identity fixture** (`expertOutput = dispatchedA`) to isolate combine correctness:

1. **CPU Golden**: `golden.h` computes the full dispatch→combine pipeline on host
2. **Full Verify**: compares device `outputC` against golden with `rtol=1e-2, atol=1e-2`
3. **Binary Dumps**: saved to `--data-dir` for offline inspection

Expected pass output:

```
verify=PASS
```

## Implementation Details

### Cross-Rank Communication

- **Local return** (`src == myRank`): synchronous `TPUT` through UB ping/pong tiles
- **Remote return** (`src != myRank`): synchronous `TPUT` through UB ping/pong tiles to peer window
- **Completion**: `TNOTIFY` with `AtomicAdd` on peer's `combineDoneSignal`
- **Barrier**: `TWAIT` with `GE` comparison against epoch `signalValue`

### Metadata Access

Uses `LoadMetadataScalar`: TLOAD a 256-element int32 tile from GM (bypasses scalar D-cache via MTE2), then extracts the target index from UB.

### Block Sharding

- Return stage: chunk-interleaved across blocks (`chunkBase % blockNum == blockId`)
- Restore stage: token-range partitioned (`TokenShardBegin/End`)
- Sync: `SoftSyncAiv` (SYNCALL Soft mode) between stages

## Performance Optimization & Overlap Analysis

### Current Pipeline Structure

```
Time ──────────────────────────────────────────────────────────────────────────→

Block 0: ┃ Return chunk0 ┃ Return chunk2 ┃ ... ┃ Wait ┃ Restore tok0..N/B ┃
Block 1: ┃ Return chunk1 ┃ Return chunk3 ┃ ... ┃ Wait ┃ Restore tokN/B..  ┃
          ├─── Stage 1: Return ───────────────┤sync├─ Stage 3: Restore ──┤
                                               ↑
                                          SoftSyncAiv
```

### Overlap Mechanisms Already Present

| Mechanism | Location | Overlap Effect |
|-----------|----------|----------------|
| **Ping/Pong double buffer** | Return stage: `TPUT(dst, src, ping, pong)` | MTE2 load of next chunk overlaps with MTE3 store of current chunk |
| **Event-driven TAXPY chain** | Restore stage: `TLOAD → TAXPY → TSTORE` with `Event` | MTE2 load, Vector compute, MTE3 store form a 3-stage pipeline within each tile iteration |
| **Multi-block parallelism** | All stages | Multiple AIV blocks execute in parallel, sharing work via chunk/token sharding |

### Key Performance Tuning Parameters

| Parameter | Impact | Recommendation |
|-----------|--------|----------------|
| `aivBlocks` | Core-level parallelism; more blocks = more parallel work | 24 (default) utilizes all available AIV blocks on 910B |
| `rowChunk` | Return granularity; controls overlap granularity and load balance | Larger = fewer metadata reads, less overhead; smaller = better block load balance |
| `K` (hidden size) | Determines data volume per row; dominates bandwidth cost | At K=7168 fp16 = 14 KiB/row → bandwidth-bound |

### Bottleneck Analysis

#### Stage 1: Return (Communication-Bound)

```
┌───── AIV Scalar Path ─────┐    ┌───── MTE2/MTE3 (ping/pong) ────┐
│ LoadMetadata(peerToken)    │    │                                  │
│ LoadMetadata(dispatchOff)  │    │  ┌─ TPUT chunk N (via UB) ──┐   │
│ LoadMetadata(prevSum)      │    │  │  MTE2 load → MTE3 store   │   │
│ LoadMetadata(cumsum)       │    │  └────────────────────────────┘  │
│ ... next segment ...       │    │  ┌─ TPUT chunk N+1 ─────────┐   │
│                            │    │  │  MTE2 load → MTE3 store   │   │
└────────────────────────────┘    │  └────────────────────────────┘  │
                                  └──────────────────────────────────┘
```

- **Dominant cost**: MTE transfer latency for remote writes (`rowChunk × K × 2B` per chunk)
- **Secondary cost**: Metadata scalar reads (4 × `LoadMetadataScalar` per segment → 4 × MTE2 TLOAD)
- **Optimization lever**: Increase `rowChunk` to amortize metadata overhead; at `rowChunk=8, K=7168` each transfer is 112 KiB

#### Stage 2: Wait (Latency-Bound)

- Pure synchronization; cost = max(peer return latency)
- Signal polling via `TWAIT` — hardware wait, near-zero AIV cycles
- **Optimization lever**: Overlap with Stage 3 (see Future Optimizations below)

#### Stage 3: Restore (Compute + Bandwidth-Bound)

```
Per token, per slot, per tile:
  TLOAD(ptrTile, ptrD[row])     ← MTE2: read ptrD row tile
  TLOAD(outTile, outputC[tok])  ← MTE2: read current accumulator
  TAXPY(outTile, ptrTile, prob) ← VEC:  fused multiply-add
  TSTORE(outputGlobal, outTile) ← MTE3: write back accumulator

Pipeline depth per tile = 3 stages (MTE2 → VEC → MTE3)
Total iterations per token = topK × ceil(K / tileCols)
```

- **Dominant cost**: MTE2 bandwidth for `2 × topK` loads per token-tile (ptrD + outputC read-back)
- **Secondary cost**: MTE3 store-back of accumulator after each slot
- **Optimization lever**: Increase `tileCols` to reduce loop overhead; fuse multiple slots before store-back

### Future Optimization Opportunities

| Optimization | Expected Gain | Complexity |
|--------------|---------------|------------|
| **Return ↔ Restore partial overlap** | Overlap local-return restore with remote-return wait; reduce Stage 2 idle | High — requires per-peer readiness tracking and row-level restore gating |
| **Multi-slot accumulation** | Accumulate multiple topK slots in UB before final TSTORE, reducing MTE3 writes by topK× | Medium — needs additional UB buffer for accumulator; UB budget allows ~4 extra tiles |
| **Metadata prefetch** | Batch-load all segment metadata at return start, eliminating per-segment TLOAD stalls | Low — allocate dedicated UB region for full metadata array |
| **Adaptive rowChunk** | Auto-tune chunk size per segment based on actual row count vs. block count | Low — add runtime heuristic in kernel prologue |
| **TPUT_ASYNC\<SDMA\> for remote return** | Decouple remote writes from MTE pipeline; multiple transfers in-flight | Medium — requires CANN 9.0+ with `aclnnShmemSdmaStarsQuery` support |
| **Restore row prefetch** | Prefetch next ptrD row into ping while computing on pong | Medium — extend double-buffering from return stage into restore stage |
| **K-dimension cube tiling** | For large K, use Cube (MTE1→Cube→MTE3) path for TAXPY equivalent | High — requires Cube tile reshape and different UB budget split |

### Overlap Diagram: Ideal Pipeline (Future)

```
Time ──────────────────────────────────────────────────────────────────────────→

MTE2 (load):  ║ meta_seg0    ║ meta_seg1    ║ ... ║ ptrD_tok0 ║ out_tok0 ║ ...║
VEC (compute):║              ║              ║     ║           ║ TAXPY_0  ║ ...║
MTE3 (store): ║ chunk0→peer  ║ chunk1→peer  ║ ... ║           ║          ║ ST ║
AIV (scalar): ║ addr_calc    ║ addr_calc    ║ ... ║ idx_read  ║ ...      ║   ║

Goal: Keep MTE2, VEC, MTE3 all busy simultaneously
      across the return→restore boundary when possible.
```

## Performance Reference

```
[PROFILE] CombineTile
  M=64 K=7168 ranks=2 topK=8 expertPerPe=2 warmup=1 measured=1 samples=1
  logical work: input tokens(all ranks)=128 routed tokens(all ranks)=1024
  combine_e2e: avg=963.7 us max=963.7 us
  verify=PASS
```

Configuration: Atlas 910B1, 2 ranks, AIV blocks=24.

## Current Limitations

- Combine-only scope — does not cover dispatch or expert computation
- Uses HCCL window + PTO TPUT/TWAIT protocol, not HCCL collective API
- Identity fixture (`expertOutput = dispatchedA`) for correctness isolation
- Remote return uses synchronous TPUT (SDMA async requires CANN 9.0+)
- Half precision (fp16) only for data tensors
