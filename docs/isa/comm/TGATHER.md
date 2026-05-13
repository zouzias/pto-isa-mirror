# pto.tgather

## Summary

`pto.tgather` gathers per-rank buffers described by a parallel group into a destination GlobalTensor, using one or two explicit UB staging tiles.

## Semantics

The verified public wrapper accepts:

- `parallelGroup`,
- destination GlobalTensor,
- one staging tile or a ping/pong tile pair,
- optional event tokens.

The wrapper delegates to the backend gather implementation after waiting on all incoming events.

## Assembly Syntax

```text
pto.tgather %group, %dst : (!pto.group<...>, !pto.memref<...>)
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`.

```cpp
template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                             TileData &stagingTileData, WaitEvents &... events);

template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                             TileData &pingTile, TileData &pongTile, WaitEvents &... events);
```

## Constraints

!!! warning "Constraints"
    - Parallel-group buffer descriptors, destination tensor, and staging-tile element types must be compatible.
    - Staging tiles must be UB-resident.
    - Ping-pong staging tiles should be non-overlapping.
    - The wrapper itself does not encode root-only validation; users must follow the semantic contract expected by the backend collective implementation.

## Examples

```cpp
#include <pto/comm/pto_comm_inst.hpp>
using namespace pto;

void example_gather(auto &group, auto &dstG, auto &stagingTile) {
    comm::TGATHER(group, dstG, stagingTile);
}
```
