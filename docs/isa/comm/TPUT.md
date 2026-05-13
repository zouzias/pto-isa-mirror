# pto.tput

## Summary

`pto.tput` performs a remote write from a local GlobalTensor to a remote GlobalTensor through one or two explicit UB staging tiles.

## Semantics

Conceptually, `TPUT` copies data from the local source tensor into the remote destination tensor:

$$ \mathrm{dst}^{\mathrm{remote}}_{i,j} = \mathrm{src}^{\mathrm{local}}_{i,j} $$

The public API supports:

- single staging-tile form,
- ping-pong double-buffered form,
- compile-time atomic mode selection,
- runtime atomic mode selection for the single-tile form.

## Assembly Syntax

```text
pto.tput %dst_remote, %src_local : (!pto.memref<...>, !pto.memref<...>)
```

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`.

```cpp
template <AtomicType atomicType = AtomicType::AtomicNone, typename GlobalDstData, typename GlobalSrcData,
          typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TPUT(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData, TileData &stagingTileData,
                          WaitEvents &... events);

template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TPUT(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData, TileData &stagingTileData,
                          AtomicType atomicType, WaitEvents &... events);

template <AtomicType atomicType = AtomicType::AtomicNone, typename GlobalDstData, typename GlobalSrcData,
          typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TPUT(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData, TileData &pingTile,
                          TileData &pongTile, WaitEvents &... events);
```

## Constraints

!!! warning "Constraints"
    - `dstGlobalData`, `srcGlobalData`, and staging tile element types must be compatible.
    - Source and destination layouts must be compatible.
    - Staging tiles must be UB-resident and sized for the selected chunking strategy.
    - Runtime `AtomicType` dispatch is only exposed on the verified single-staging-tile overload.
    - Ping-pong staging tiles should be non-overlapping.

## Target-Visible Notes

- The exposed atomic modes are `AtomicType::AtomicNone` and `AtomicType::AtomicAdd`.
- The wrapper always waits on incoming event tokens before issuing the implementation call.
- Double-buffering is an explicit API choice, not an implicit optimization of the single-tile overload.

## Examples

```cpp
#include <pto/comm/pto_comm_inst.hpp>
using namespace pto;

void example_put(auto &dstG, auto &srcG, auto &stagingTile) {
    comm::TPUT(dstG, srcG, stagingTile);
    comm::TPUT<comm::AtomicType::AtomicAdd>(dstG, srcG, stagingTile);
}
```
