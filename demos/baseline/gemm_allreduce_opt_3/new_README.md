# GEMM AllReduce Optimized — ReduceScatter + AllGather

This project implements an optimized GEMM + AllReduce fused operator. Compared to the original scheme (TPUT\<AtomicAdd\> broadcast), this approach adopts a **ReduceScatter + AllGather** communication algorithm, combined with an increased K dimension and reduced communication block count, improving **Time Saved from 3.6% to 20.2%**.

## Optimization Summary

| Item | Original | Optimized |
|------|----------|-----------|
| Comm Algorithm | TPUT\<AtomicAdd\> broadcast to all ranks | ReduceScatter + AllGather (plain TPUT) |
| K Dimension | 4096 | **16384** |
| COMM_BLOCK_NUM | 40 | **20** |
| TPUT Count / rank | 14,336 | **3,584** (4× reduction) |
| Comm Data / rank | 1.75 GB | **448 MB** (~4× reduction) |
| Comm Time | ~31,330 us | **~9,945 us** |
| Time Saved | 3.6% | **20.2%** |
| Speedup | 1.034× | **1.253×** |

## Data Parallel Mode

### Matrix Partitioning Strategy (K-split)

Global matrix multiplication: C\[M×N\] = A\[M×K\] × B\[K×N\]

```
Default parameters: M=16384, K=16384, N=4096, 8 ranks

K dimension evenly split across ranks. Each rank holds:
  A_part [M × (K/r)] = [16384 × 2048]
  B_part [(K/r) × N] = [2048 × 4096]

Each rank computes independently:
  C_rank [M × N] = A_part × B_part = [16384 × 4096]

AllReduce target:
  C_final = C_rank_0 + C_rank_1 + ... + C_rank_7
```

### Mathematical Correctness

```
A × B = [A_0 | A_1 | ... | A_r-1] × [B_0; B_1; ...; B_r-1]
      = A_0 × B_0 + A_1 × B_1 + ... + A_r-1 × B_r-1
```

The sum of partial products from each rank after K-split equals the full matrix multiplication result.

## Overall Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  Compute Stream (Cube, 32 blocks)          Comm Stream (Vec, 20 blocks)          │
│                                                                                   │
│  ┌──────────────────────┐                                                         │
│  │ Compute Tile → TSTORE│──┐                                                      │
│  │ pipe_barrier(PIPE_ALL)│  │                                                     │
│  │ Enqueue tile_idx      │  │ Ready Queue                                         │
│  └──────────────────────┘  │                                                      │
│  ┌──────────────────────┐  │         ┌───────────────────────────────────────┐    │
│  │ Compute Tile → TSTORE│──┼────────→│ Phase 1: ReduceScatter TPUT           │    │
│  └──────────────────────┘  │         │   Send each tile to its owner rank    │    │
│  ┌──────────────────────┐  │         │ Phase 2: Barrier                      │    │
│  │ Compute Tile → TSTORE│──┘         │ Phase 3: Local TREDUCE (owned tiles)  │    │
│  └──────────────────────┘            │ Phase 4: AllGather TPUT               │    │
│         ...                          │   Broadcast reduced tiles to all ranks │    │
│                                      │ Phase 5: Barrier                      │    │
│                                      └───────────────────────────────────────┘    │
│                                                                                   │
│  Key Features:                                                                    │
│  - Dual-stream: Cube compute stream + Vector comm stream, overlapped execution    │
│  - Per-tile signaling: comm can start as soon as each tile is ready               │
│  - 4× less comm: ReduceScatter sends only to owner, AllGather only from owner     │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## Communication Algorithm: ReduceScatter + AllGather

### Output Matrix Tile Layout

```
C_rank / C_final [M × N] = 16384 × 4096

Tile size: 128 × 256 (128KB per tile)
  M direction: 16384/128 = 128 blocks
  N direction: 4096/256  = 16 blocks
  Total tiles: 128 × 16 = 2048

     ni →  0    1    2   ...   15
  mi ↓
    0    │  0  │  1  │  2  │ ... │ 15  │     tile_idx = mi × 16 + ni
    1    │ 16  │ 17  │ 18  │ ... │ 31  │
   ...   │     │     │     │     │     │
   127   │2032 │2033 │2034 │ ... │2047 │

Owner assignment (round-robin):
  owner(tile_idx) = tile_idx % 8
  → Rank 0 owns: tile 0, 8, 16, ..., 2040 (256 tiles)
  → Rank 1 owns: tile 1, 9, 17, ..., 2041 (256 tiles)
  → ...
```

### Phase 1: ReduceScatter — Send Each Tile Only to Its Owner

Comm blocks poll the Ready Queue; once a tile is ready, they check its owner:

```
Example: tile_idx = 42, owner = 42 % 8 = 2 (Rank 2)

Original (Broadcast):                  Optimized (ReduceScatter):
  Rank 0 → Rank 1,2,3,4,5,6,7           Rank 0 → Rank 2 (owner only)
  (7 TPUT calls)                         (1 TPUT call)

All ranks' operations for tile 42:
  Rank 0 → Rank 2  (1 TPUT)
  Rank 1 → Rank 2  (1 TPUT)
  Rank 2   local data, no send (0 TPUT)
  Rank 3 → Rank 2  (1 TPUT)
  ...
  Rank 7 → Rank 2  (1 TPUT)

  Global TPUT count for tile 42: 7 (vs. original 56)
```

Per-rank ReduceScatter send volume:

```
Out of 2048 tiles:
  256 tiles owned by self → no TPUT (0 sends)
  1792 tiles → 1 TPUT each to the respective owner

  Per-rank send: 1792 × 128KB = 224 MB (vs. original 1.75 GB)
```

### Phase 2: Barrier

`ShmemDeviceQuiet()` + `ShmemDeviceBarrierAll()` ensures all ReduceScatter writes are visible across all ranks.

### Phase 3: Local Reduction (Owned Tiles Only)

Each rank performs 8-way summation on **only its 256 owned tiles**:

```
Example: Rank 0 reduces tile 0:

  Inputs (8 sources):
    shmem_output[tile 0]     ← own partial result
    recv_buffers[1][tile 0]  ← from Rank 1
    recv_buffers[2][tile 0]  ← from Rank 2
    ...
    recv_buffers[7][tile 0]  ← from Rank 7
  ──────────────────────────
  Output:
    reduced_output[tile 0]   ← complete AllReduce result

  Method: TREDUCE_PINGPONG (half-height 64×256 sub-tiles)
  Tiles reduced per rank: 256 (vs. original 2048, 8× reduction)
```

**TREDUCE_PINGPONG** alternates EVENT_ID1/EVENT_ID2 to avoid V-pipeline event counter overflow — a critical issue discovered in previous optimization attempts.

### Phase 4: AllGather — Owner Broadcasts Reduced Results

Each rank TPUTs its 256 reduced tiles to all other ranks' `reduced_output`:

```
Rank 0 broadcasts tile 0's final result:

  reduced_output[tile 0] ─→ Rank 1 reduced_output[tile 0]
                          ─→ Rank 2 reduced_output[tile 0]
                          ─→ ...
                          ─→ Rank 7 reduced_output[tile 0]

  7 TPUT calls per owned tile
  Per-rank send: 256 × 7 × 128KB = 224 MB
```

### Phase 5: Barrier

`ShmemDeviceQuiet()` + `ShmemDeviceBarrierAll()` + `pipe_barrier(PIPE_ALL)`.

AllReduce is complete. Every rank's `reduced_output` now contains the full C\_final\[M×N\].

### Communication Volume Comparison

```
               ┌────────────────────────┬──────────────────────────────┐
               │    Original             │    Optimized                  │
               │  Broadcast Scatter      │  ReduceScatter + AllGather   │
┌──────────────┼────────────────────────┼──────────────────────────────┤
│ Send target  │ All 7 remote ranks     │ RS: 1 owner rank only        │
│ per tile     │                        │ AG: owned tiles → 7 ranks    │
├──────────────┼────────────────────────┼──────────────────────────────┤
│ TPUT count   │ 2048×7 = 14,336       │ 1,792 + 1,792 = 3,584       │
│ / rank       │                        │ (4× reduction)               │
├──────────────┼────────────────────────┼──────────────────────────────┤
│ Comm data    │ 1.75 GB               │ 224 + 224 = 448 MB           │
│ / rank       │                        │ (~4× reduction)              │
├──────────────┼────────────────────────┼──────────────────────────────┤
│ Reduce tiles │ 2048 (all)            │ 256 (owned only, 8×↓)        │
│ / rank       │                        │                              │
├──────────────┼────────────────────────┼──────────────────────────────┤
│ Measured     │ ~31,330 us            │ ~9,945 us (~3.2× faster)     │
│ Comm Time    │                        │                              │
└──────────────┴────────────────────────┴──────────────────────────────┘
```

## End-to-End Data Flow

```
                    Per Rank (local)
  A_part[M,K/r] × B_part[K/r,N] = C_rank[M,N]
          │                    │
          ▼                    ▼
  ┌──────────────┐      ┌──────────────┐
  │ GEMM Compute │      │ shmem_output │  2048 tiles, 128×256 each
  │ 32 blocks    │ ──▶  │ [M×N]        │
  └──────────────┘      └──────┬───────┘
          │                    │
          │ Enqueue tile_idx    │ ReduceScatter: each tile → owner only
          ▼                    ▼
  ┌──────────────┐      ┌──────────────┐
  │ Ready Queue  │      │ recv_buffers │  Per-slot receive from other ranks
  │ (tile signal)│      │[nranks × MN] │
  └──────────────┘      └──────┬───────┘
                               │ Barrier
                               ▼
                        ┌──────────────┐
                        │ TREDUCE      │  Owned tiles only: 8-way Sum
                        │ (PINGPONG)   │
                        └──────┬───────┘
                               │ AllGather: owner broadcasts to all ranks
                               ▼
                        ┌──────────────┐
                        │reduced_output│  Every rank has full C_final [M×N]
                        │ [M×N]        │
                        └──────────────┘
```

## Symmetric Heap Memory Layout

Each rank allocates three buffers on the symmetric heap:

| Buffer | Size | Purpose |
|--------|------|---------|
| `shmem_output` | M×N×4B = 256 MB | Local GEMM result C\_rank |
| `recv_buffers` | nranks × M×N×4B = 2048 MB | Per-rank slots for ReduceScatter data |
| `reduced_output` | M×N×4B = 256 MB | Final AllReduce result C\_final |

```
recv_buffers logical layout (Rank 0's perspective):
┌────────────────┬────────────────┬─────┬────────────────┐
│ slot 0 (rank0) │ slot 1 (rank1) │ ... │ slot 7 (rank7) │
│ [M×N] local    │ [M×N] from R1  │     │ [M×N] from R7  │
└────────────────┴────────────────┴─────┴────────────────┘
Note: slot 0 = own data, read directly from shmem_output (not via recv_buffers)
```

## Compute Kernel

### Core Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| G_M | 16384 | Matrix M dimension |
| G_K | 16384 | Matrix K dimension (global) |
| G_N | 4096 | Matrix N dimension |
| G_BASE_M | 128 | Tile M dimension |
| G_BASE_K | 64 | Tile K dimension |
| G_BASE_N | 256 | Tile N dimension |
| G_STEP_KA/KB | 4 | L1-cached K-slices (4× DMA reduction) |
| COMPUTE_BLOCK_NUM | 32 | Compute block count |
| k_per_rank | K/nranks = 2048 | Per-rank K dimension |

### Compute Flow

Each compute block handles a set of tiles (2048 / 32 = 64 tiles). For each tile:

1. **K-loop**: 2048/64 = 32 iterations
   - Every 4 iterations: TLOAD to L1 (4× DMA reduction)
   - Each iteration: TEXTRACT + TMATMUL\_ACC
2. **TSTORE**: Write result to `shmem_output`
3. **pipe\_barrier(PIPE\_ALL)**: Ensure GM write completion
4. **MultiBlockEnqueueFast**: Enqueue `tile_idx` to signal comm side

### Two-Level Double-Buffered Pipeline

```
Time →
L1 (MTE2):  [TLOAD A0,B0]              [TLOAD A1,B1]              ...
L0 (MTE1):       [EXT k0] [EXT k1] [EXT k2] [EXT k3] [EXT k0'] ...
Cube (M):             [MUL k0] [ACC k1] [ACC k2] [ACC k3] [MUL k0'] ...
                      ↑ Full three-stage pipeline overlap ↑
```

## Three Optimization Techniques and Their Impact

### 1. ReduceScatter + AllGather Communication Algorithm

- **Original**: Each tile broadcast to all 7 remote ranks (14,336 TPUTs/rank, 1.75 GB/rank)
- **Optimized**: ReduceScatter sends to owner only (1,792) + AllGather broadcasts from owner (1,792), total 3,584 TPUTs/rank, 448 MB/rank
- **Impact**: Comm time reduced from ~31ms to ~10ms

### 2. K Dimension Increased from 4096 to 16384

- Per-rank compute: 2 × M × (K/r) × N
  - Original: 2 × 16384 × 512 × 4096 = 137 GFLOP → ~0.7 ms
  - Optimized: 2 × 16384 × 2048 × 4096 = 550 GFLOP → ~1.7 ms
- **Impact**: Compute becomes a larger fraction of total time, making pipeline overlap more effective

### 3. COMM_BLOCK_NUM Reduced from 40 to 20

- Fewer comm blocks consume fewer AI Core resources
- **Impact**: Reduced Cube/Vector core resource contention; pipelined compute efficiency improved from 12.3% to 59.9%

## Performance Results

```
================================================================
  Matrix dimensions: M=16384, K=16384, N=4096
  Ranks: 8
  Compute blocks: 32, Comm blocks: 20
  Algorithm: ReduceScatter + AllGather
================================================================

  Compute Kernel (pure):     1707.5 us, 160981 GFLOPS

  Sequential (no overlap):   11747.1 us total
    Compute:  1802.0 us (152542 GFLOPS)
    Comm:     9945.1 us (201.1 GB/s)

  Pipelined (with overlap):  9374.4 us total
    Compute:  2852.3 us (96370 GFLOPS, 59.9% of pure)
    Comm:     9374.4 us (213.3 GB/s)

  Performance Comparison:
    Speedup:     1.253×
    Time Saved:  2372.7 us (20.198%)
```

## Build and Run

```bash
# One command: generate data + build + run
./run_performance_test.sh

# Specify parameters
./run_performance_test.sh --nranks 8 --first-device 0

# Run only (skip rebuild)
./run_performance_test.sh --skip-build
```

## File Structure

```
gemm_allreduce_opt_3/
├── CMakeLists.txt              # Build configuration
├── new_README.md               # This document (English)
├── new_README_zh.md            # This document (Chinese)
├── README_zh.md                # Legacy README (Chinese)
├── run_performance_test.sh     # Performance test entry script
├── main.cpp                    # Main program (dual-stream launch)
├── gemm_compute_kernel.cpp     # Compute kernel (Cube, 32 blocks)
├── comm_kernel.cpp             # Comm kernel (Vec, 20 blocks)
│                                 Contains ReduceScatter + AllGather logic,
│                                 host-side launch, verification, perf stats
├── ready_queue.hpp             # Multi-block lock-free queue
├── input/                      # Input data (generated by gen_data.py)
├── output/                     # Output data / golden reference
└── scripts/
    └── gen_data.py             # Data generation script (K=16384)
```

## Key Technical Points

### 1. Plain TPUT Replaces TPUT\<AtomicAdd\>

The new scheme uses plain TPUT (no atomic operations), avoiding write conflicts through the owner assignment in ReduceScatter:

- Each tile is sent to exactly one owner rank during ReduceScatter
- Different ranks write to different slots in the owner's `recv_buffers` (indexed by sender rank), eliminating contention
- During AllGather, only the owner writes each tile; other ranks only read

### 2. TREDUCE_PINGPONG Prevents Event Counter Overflow

- The V-pipeline event counters have only 2-bit capacity
- Regular TREDUCE accumulates EVENT_ID1 across loop iterations, causing overflow
- TREDUCE_PINGPONG alternates EVENT_ID1 and EVENT_ID2, keeping each counter at max 1

### 3. Lock-Free Ready Queue Signaling

- One independent queue per compute block (single-producer, single-consumer)
- Comm side uses TTEST/TWAIT hardware instructions for polling
- Enables per-tile compute-communication pipelining

## References

- [PTO Communication ISA](../../../include/pto/comm/)
- [TPUT Instruction](../../../include/pto/comm/TPut.hpp)
- [TREDUCE_PINGPONG Instruction](../../../include/pto/comm/TReduce.hpp)
