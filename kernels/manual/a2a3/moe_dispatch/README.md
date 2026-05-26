# MoE Dispatch — PTO-ISA Standalone Communication Operator

## Overview

Standalone implementation of the MegaMoE Dispatch communication operator using PTO-ISA instructions.
This operator validates that PTO-ISA's `TGET` (remote read) can correctly implement the MegaMoE
dispatch pattern: multi-core parallel pull of quantized tokens from remote ranks' shared memory.

## Algorithm

```
for each local expert (groupIdx):
    for each remote rank (dstEpIdx, strided by coreIdx):
        1. Compute remote source address in peer shmem
        2. Compute local destination offset in gmA
        3. TGET: pull (token + scale) rows via ping-pong UB staging
        4. Separate token data and per-token scale into gmA and gmPerTokenScale
    SyncAll()  // All cores finished this expert group
```

## Key Features

- **Multi-core parallel TGET**: Each AIV core handles one or more remote ranks (strided assignment)
- **Ping-pong double buffering**: Overlaps remote read with local store via dual UB tiles
- **Token/scale separation**: Remote rows have interleaved layout `[int8 x K][float scale]`;
  on arrival, token data is stored compactly in `gmA`, scales in `gmPerTokenScale`
- **Per-expert synchronization**: `SyncAll()` after each expert group, enabling future
  pipeline overlap with GEMM computation

## Build & Run

```bash
# Set environment
source /mnt/data/ntlab/liulei/set_env_new.sh

# Build & run (2 ranks, K=128)
./run.sh all --ep 2 --hidden 128

# Larger configuration
./run.sh all --ep 8 --experts 2 --hidden 4096 --tokens 256 --max-output 2048

# Build only
./run.sh build --ep 4 --hidden 512 --debug

# Run only (after build)
./run.sh run --ep 4
```

## File Structure

| File | Description |
|------|-------------|
| `moe_dispatch_kernel.cpp` | Device kernel: TGET dispatch loop + ping-pong |
| `main.cpp` | Host driver: MPI init, data gen, launch, verify |
| `moe_dispatch_config.h` | Shape constants and parameters |
| `hccl_context.h` | Device-side HCCL context struct |
| `CMakeLists.txt` | Build configuration (bisheng + dav-c220-vec) |
| `run.sh` | Convenience build & run script |

## Reference

- MegaMoE source: `vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp` (L842-888)
- Design doc: `/mnt/data/ntlab/liulei/docs/megamoe/dispatch_pto_isa_design.md`
- Module split plan: `/mnt/data/ntlab/liulei/docs/megamoe/module_split_plan.md`
