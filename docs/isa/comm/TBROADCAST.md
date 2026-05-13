# pto.tbroadcast

## Summary

`pto.tbroadcast` broadcasts a source GlobalTensor to the per-rank buffers described by a parallel group, using one or two explicit UB staging tiles.

## Semantics

The verified public wrapper accepts:

- `parallelGroup`,
- source GlobalTensor,
- one staging tile or a ping/pong tile pair,
- optional event tokens.

The wrapper delegates to the backend broadcast implementation after waiting on all incoming events.

## Assembly Syntax

```text
pto.tbroadcast %group, %src : (!pto.group<...>, !pto.memref<...>)
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`.

```cpp
template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TBROADCAST(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                                TileData &stagingTileData, WaitEvents &... events);

template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TBROADCAST(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                                TileData &pingTile, TileData &pongTile, WaitEvents &... events);
```

## Constraints

!!! warning "Constraints"
    - Parallel-group buffer descriptors, source tensor, and staging-tile element types must be compatible.
    - Staging tiles must be UB-resident.
    - Ping-pong staging tiles should be non-overlapping.
    - The wrapper itself does not encode root-only validation; users must follow the semantic contract expected by the backend collective implementation.

## Examples

```cpp
#include <pto/comm/pto_comm_inst.hpp>
using namespace pto;

void example_broadcast(auto &group, auto &srcG, auto &stagingTile) {
    comm::TBROADCAST(group, srcG, stagingTile);
}
```
