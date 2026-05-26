# MoE Dispatch — PTO-ISA Standalone Communication Operator

## Overview

Standalone implementation of the MegaMoE Dispatch communication operator using PTO-ISA instructions.
This operator pulls quantized tokens from remote ranks' shared memory, splitting interleaved
`[int8 token | float scale]` rows into compact separate outputs (`gmA` and `gmPerTokenScale`).

Two independent kernel paths are provided:

- **Direct (2-step)**: `TLOAD remote GM → UB → TSTORE split` — fast path with adaptive UB tiling
- **ViaGM (4-step)**: `TGET remote GM → local GM → TLOAD → UB → TSTORE split` — MegaMoE-compatible path

## Supported AI Processors

- Ascend A2
- Ascend A3

## Data Flow

```
Direct path (default):
  Remote GM ──TLOAD──▶ UB (ping/pong) ──TSTORE──▶ gmA (token)
                                        ──TSTORE──▶ gmPerTokenScale (scale)

ViaGM path:
  Remote GM ──TGET──▶ Local temp GM ──TLOAD──▶ UB (ping/pong) ──TSTORE──▶ gmA
                                                                ──TSTORE──▶ gmPerTokenScale
```

## Algorithm

```
for each local expert (groupIdx):
    for each remote rank (dstEpIdx, strided by coreIdx):
        1. Compute remote source address in peer shmem
        2. Compute local destination offset in gmA/gmPerTokenScale
        3. [Direct] TLOAD interleaved rows into UB, TSTORE split token and scale
           [ViaGM]  TGET rows to local GM, then TLOAD→UB→TSTORE split
        4. Event-driven ping-pong: overlap TLOAD(N+1) with TSTORE(N)
    // Cross-rank continuous pipeline: no bubble between ranks
```

## Key Features

- **Dual-path design**: Direct (2-step) for low latency; ViaGM (4-step) for MegaMoE compatibility
- **Adaptive MOVE_NUM**: Compile-time `DispatchTraits<TILE_COLS>` auto-shrinks rows/batch for large hiddenSize
- **Event-driven ping-pong**: `set_flag`/`wait_flag` overlaps MTE2 (TLOAD) and MTE3 (TSTORE) pipelines
- **Cross-rank continuous pipeline**: Ping-pong state persists across remote ranks — no flush between ranks
- **Multi-core parallel**: Each AIV core handles one or more remote ranks (strided assignment)
- **Token/scale separation**: Remote rows `[int8×K][float scale padded to 32B]` → compact token + scale outputs

## Specification

| Item | Value |
|------|-------|
| Data type (token) | `int8_t` |
| Data type (scale) | `float` (stored in 32B-aligned rows) |
| Remote row format | `hiddenSize` bytes token + `UB_ALIGN` (32) bytes padding (scale at offset 0) |
| Output token | `gmA[maxOutputSize, hiddenSize]` — compact, no padding |
| Output scale | `gmPerTokenScale[maxOutputSize]` — 32 bytes/row (float at offset 0) |
| Default hiddenSize | 128 |
| Execution model | AIV-only (vector cores), multi-rank via mpirun |

## Directory Layout

```
kernels/manual/a2a3/moe_dispatch/
├── moe_dispatch_kernel.cpp     # Device kernel: dual-path dispatch + ping-pong
├── main.cpp                    # Host driver: MPI init, data gen, launch, verify
├── moe_dispatch_config.h       # Shape constants, DispatchTraits, parameters
├── hccl_context.h              # Device-side HCCL context struct
├── CMakeLists.txt              # Build configuration (bisheng + dav-c220-vec)
├── run.sh                      # Build & run convenience script
└── README.md                   # This file
```

## Build & Run

```bash
# Set environment
source /mnt/data/ntlab/liulei/set_env_new.sh
export HCCL_WHITELIST_DISABLE=1

# Build & run Direct path (default), 2 ranks
bash run.sh all --ep 2 --mode direct

# Build & run ViaGM path, 4 ranks
bash run.sh all --ep 4 --mode viagm

# Use specific devices (start from device 4)
bash run.sh all --ep 4 --first-device 4 --mode direct

# Build only
bash run.sh build --ep 2 --hidden 128 --debug

# Run only (after build)
bash run.sh run --ep 2 --mode direct

# Clean build
bash run.sh all --ep 4 --mode viagm --clean
```

### run.sh Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ep N` | 2 | Number of ranks (EP count) |
| `--mode direct\|viagm` | direct | Kernel path selection |
| `--first-device N` | 0 | First NPU device ID |
| `--hidden N` | 128 | Hidden size (K) |
| `--tokens N` | 64 | Max tokens per rank |
| `--max-output N` | 512 | Max output rows |
| `--experts N` | 1 | Experts per rank |
| `--clean` | — | Force clean rebuild |
| `--debug` | — | Enable debug mode |

## Relation to MegaMoE

This operator validates the Dispatch phase of MegaMoE. It can serve as a direct building block
for the full MegaMoE fused operator:

```
MegaMoE full pipeline:
  InitRouting → [Dispatch] → GEMM (FFN) → Combine
                 ^^^^^^^^
                 This operator
```

- **Interface compatibility**: Parameters (`cumsumMM`, `tokenPerExpert`, `preSumBeforeRank`, `shmemBase`) match MegaMoE exactly
- **ViaGM path**: Functionally equivalent to MegaMoE's `DispatchCopyPerToken`
- **Direct path**: PTO-ISA optimization that bypasses intermediate GM buffer

## Reference

- MegaMoE source: `vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp`
- Design doc: `/mnt/data/ntlab/liulei/docs/megamoe/dispatch_pto_isa_design.md`
- PTO-ISA TGET API: `include/pto/comm/pto_comm_inst.hpp`
