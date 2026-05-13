# pto.tget

## Summary

`pto.tget` performs a remote read from a remote GlobalTensor into a local GlobalTensor through one or two explicit UB staging tiles.

## Semantics

Conceptually, `TGET` copies data from the remote source tensor into the local destination tensor:

$$ \mathrm{dst}^{\mathrm{local}}_{i,j} = \mathrm{src}^{\mathrm{remote}}_{i,j} $$

The public API supports:

- single staging-tile form, and
- ping-pong double-buffered form.

## Assembly Syntax

```text
pto.tget %dst_local, %src_remote : (!pto.memref<...>, !pto.memref<...>)
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`.

```cpp
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGET(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData, TileData &stagingTileData,
                          WaitEvents &... events);

template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGET(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData, TileData &pingTile,
                          TileData &pongTile, WaitEvents &... events);
```

## Constraints

!!! warning "Constraints"
    - `dstGlobalData`, `srcGlobalData`, and staging tile element types must be compatible.
    - Source and destination layouts must be compatible.
    - Staging tiles must be UB-resident and sized for the selected chunking strategy.
    - Ping-pong staging tiles should be non-overlapping.

## Target-Visible Notes

- The wrapper always waits on incoming event tokens before issuing the implementation call.
- Double-buffering is an explicit API choice, not an implicit optimization of the single-tile overload.

## Examples

```cpp
#include <pto/comm/pto_comm_inst.hpp>
using namespace pto;

void example_get(auto &dstG, auto &srcG, auto &stagingTile) {
    comm::TGET(dstG, srcG, stagingTile);
}
```
