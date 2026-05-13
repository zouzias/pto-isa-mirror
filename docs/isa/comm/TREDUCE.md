# pto.treduce

## Summary

`pto.treduce` reduces per-rank buffers described by a parallel group into a destination GlobalTensor, using explicit accumulator/receive staging tiles.

## Semantics

The verified public wrapper accepts:

- `parallelGroup`,
- destination GlobalTensor,
- accumulator tile plus either one receive tile or ping/pong receive tiles,
- `ReduceOp`, and
- optional event tokens.

The wrapper delegates to the backend reduce implementation after waiting on all incoming events.

## Assembly Syntax

```text
pto.treduce %group, %dst {op = #pto.reduce_op<Sum>} : (!pto.group<...>, !pto.memref<...>)
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`.

```cpp
template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TREDUCE(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                             TileData &accTileData, TileData &recvTileData, ReduceOp op, WaitEvents &... events);

template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TREDUCE(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                             TileData &accTileData, TileData &pingTileData, TileData &pongTileData,
                             ReduceOp op, WaitEvents &... events);
```

## Constraints

!!! warning "Constraints"
    - Parallel-group buffer descriptors, destination tensor, and staging-tile element types must be compatible.
    - Accumulator/receive tiles must be UB-resident.
    - Ping-pong receive tiles should be non-overlapping.
    - The wrapper itself does not encode root-only validation; users must follow the semantic contract expected by the backend collective implementation.

## Examples

```cpp
#include <pto/comm/pto_comm_inst.hpp>
using namespace pto;

void example_reduce(auto &group, auto &dstG, auto &accTile, auto &recvTile) {
    comm::TREDUCE(group, dstG, accTile, recvTile, comm::ReduceOp::Sum);
}
```
