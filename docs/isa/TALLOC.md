# TALLOC

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-26T03:25:36.294Z pushedAt=2026-08-29T09:05:18.414Z -->

## Introduction

Allocates a producer FIFO slot from `TPipe` and exposes it as a `GlobalTensor` view.

`TALLOC` is used for the `GlobalData` split API. It allows the producer to obtain the current FIFO slot address, write data to the slot using ordinary memory instructions such as `TSTORE`, and then commit the slot through `TPUSH(Pipe&, GlobalData&)`.

## Operation Semantics

For the GlobalData flow:

1. `TALLOC(Pipe&, GlobalData&)` allocates a producer FIFO slot from `TPipe` and exposes it as a `GlobalTensor` view. The producer can write data to the slot through instructions such as `TSTORE`.
2. `TPUSH(Pipe&, GlobalData&)` records the data-ready synchronization for the slot already allocated by `TALLOC` and submits the FIFO slot to the consumer. It does not store tile data itself.
3. `TPOP(Pipe&, GlobalData&)` waits for data readiness, assigns `gmTensor` to the current FIFO slot address, and increments the consumer tile index. It does not load data into the local tile, nor does it release the slot. The consumer can read data from the slot through instructions such as `TLOAD`.
4. `TFREE(Pipe&, GlobalData&)` releases the FIFO slot view returned by `TPOP(Pipe&, GlobalData&)`, notifying the producer that the slot space is free.

`TALLOC` performs three steps:

1. When both `pipe.prod.getAllocateStatus()` and `Pipe::shouldWaitFree(pipe.prod.tileIndex)` are true, wait for idle FIFO space.
2. Compute the current FIFO slot address based on `pipe.prod.tileIndex`.
3. Assign `gmTensor` to the FIFO slot address and increment the producer tile index.

`TALLOC` does not write any data, nor does it notify the consumer. The producer must first write the slot content, and then call `TPUSH(Pipe&, GlobalData&)`.

## C++ Intrinsic

Declaration location: `include/pto/common/pto_instr.hpp`:

```cpp
template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TALLOC(Pipe &pipe, GlobalData &gmTensor, WaitEvents &... events);
```

`Pipe` is usually the `TPipe` declared in `include/pto/npu/a2a3/TPush.hpp` or `include/pto/npu/a5/TPush.hpp`:

```cpp
template <uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum,
          uint32_t LocalSlotNum = 2, bool IsNoSplit = false, bool EN_UNIT_FLAG = false>
struct TPipe;
```

## Constraints

- **GlobalData producer for Atlas A2/A3 training products/Atlas A2/A3 inference products**:
    - `GlobalData` must satisfy `is_global_data_v<GlobalData>`.
    - `Direction::DIR_C2V`: The producer can see the entire FIFO slot.
    - `Direction::DIR_V2C`: For vector sub-blocks, a split offset may be applied based on `Split`.
- **GlobalData producer for Ascend 950PR/Ascend 950DT**:
    - `GlobalData` must satisfy `is_global_data_v<GlobalData>`.
    - `Direction::DIR_C2V_GM`: The producer can see the entire FIFO slot.
    - `Direction::DIR_V2C_GM`: For vector sub-blocks, a split offset may be applied based on `Split`.
- **FIFO slot**:
    - `SlotSize` must be large enough to hold one logical FIFO entry.
    - `SlotNum >= 1`.
    - `Pipe::SyncPeriod` is derived from `SlotNum`: `(SlotNum <= 2) ? SlotNum : SlotNum / 2`.
- **Splitting behavior**:
    - `TileSplitAxis::TILE_NO_SPLIT`: No sub-vector offset is applied.
    - `TileSplitAxis::TILE_UP_DOWN`: Vector sub-blocks are mapped to the upper and lower row halves.
    - `TileSplitAxis::TILE_LEFT_RIGHT`: Vector sub-blocks are mapped to the left and right column halves.
- **Synchronization**:
    - Idle space waiting is sparse and is controlled by `Pipe::SyncPeriod`.
    - `TALLOC` does not record data readiness; after writing to the slot, use `TPUSH(Pipe&, GlobalData&)`.

## Examples

### Allocation, Storage, and Committing

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_talloc(__gm__ void *fifoMem)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint32_t FlagID = 0;
    constexpr uint32_t FifoDepth = 2;

    using Pipe = TPipe<FlagID, Direction::DIR_C2V, M * N * sizeof(T), FifoDepth>;
    using SlotGlobal = GlobalTensor<T, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using VecTile = Tile<TileType::Vec, T, M, N, BLayout::RowMajor, M, N>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    SlotGlobal slot;
    VecTile tile;
    TASSIGN(tile, 0x0);

    TALLOC<Pipe, SlotGlobal, TileSplitAxis::TILE_NO_SPLIT>(pipe, slot);
    TSTORE(slot, tile);
    TPUSH<Pipe, SlotGlobal, TileSplitAxis::TILE_NO_SPLIT>(pipe, slot);
}
```

### V2C Splitting Allocation

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_v2c_split(__gm__ void *fifoMem)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 128;
    constexpr uint32_t FullM = M * 2;
    constexpr uint32_t FlagID = 0;
    constexpr uint32_t FifoDepth = 2;

    using Pipe = TPipe<FlagID, Direction::DIR_V2C, FullM * N * sizeof(T), FifoDepth>;
    using SlotGlobal = GlobalTensor<T, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using VecTile = Tile<TileType::Vec, T, M, N, BLayout::RowMajor, M, N>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    SlotGlobal slot;
    VecTile tile;
    TASSIGN(tile, 0x0);

    TALLOC<Pipe, SlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pipe, slot);
    TSTORE(slot, tile);
    TPUSH<Pipe, SlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pipe, slot);
}
```

## ASM Examples

The currently published assembly reference has not yet defined a stable PTO-AS form for `TALLOC`. Use the C++ intrinsic form when hand-writing CV FIFO programs.

```text
