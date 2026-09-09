# TPUSH

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-26T04:43:50.491Z pushedAt=2026-08-29T09:05:18.454Z -->

## Introduction

Pushes a producer tile into the FIFO for data transfer between Cube and Vector and for inter-core synchronization.

This instruction supports pushing multiple types of data, including tile overload based on `TileSplitAxis`, simplified tile overload (with reversed parameter order and no split required), GlobalTensor overload, and TConfig-based overload.

## Operation Semantics

For the TileData flow:

1. `TPUSH(Pipe&, TileData&, Split)` stores the producer tile into the current FIFO slot and records the data-ready synchronization for the consumer. The producer tile index is incremented after the slot address is computed.
2. `TPOP(Pipe&, TileData&, Split)` waits for the producer's data-ready synchronization and loads the current FIFO slot into the consumer tile. The consumer tile index is incremented after the slot address is computed.
3. `TFREE(Pipe&, Split)` releases the slot space in the FIFO. On Atlas A2/A3 training products/Atlas A2/A3 inference products, this API is a null operation (`TPOP` already performs the free-space notification internally). On Ascend 950PR/Ascend 950DT, it releases the FIFO slot space used by `TPOP`.

For the GlobalData flow:

1. `TALLOC(Pipe&, GlobalData&)` allocates a producer FIFO slot from `TPipe` and exposes it as a `GlobalTensor` view. The producer can write data to this slot through instructions such as `TSTORE`.
2. `TPUSH(Pipe&, GlobalData&)` records the data-ready synchronization for the slot already allocated by `TALLOC` and submits the FIFO slot to the consumer. It does not store tile data itself.
3. `TPOP(Pipe&, GlobalData&)` waits for data readiness, assigns `gmTensor` to the current FIFO slot address, and increments the consumer tile index. It does not load data into a local tile, nor does it release the slot. The consumer can read data from the slot through instructions such as `TLOAD`.
4. `TFREE(Pipe&, GlobalData&)` releases the FIFO slot view returned by `TPOP(Pipe&, GlobalData&)`, notifying the producer that the slot space is free.

For the `TConfig` overload `TPUSH(Pipe&, TileProd&, TConfig)`, the `TConfig` template parameter is used to configure the fixpipe parameters from L0C to GM/UB.

## C++ Intrinsic

Declaration location: `include/pto/common/pto_instr.hpp`:

```cpp
template <typename Pipe, typename TileProd, TileSplitAxis Split,
          std::enable_if_t<is_tile_data_v<TileProd>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, TileProd &tile, WaitEvents &... events);

template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, GlobalData &gmTensor, WaitEvents &... events);

template <typename Pipe, typename TileProd, typename TConfig, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, TileProd &tile, WaitEvents &... events);
```

`Pipe` is usually the `TPipe` type declared in `TPush.hpp`:

```cpp
template <uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum,
          uint32_t LocalSlotNum = 2, bool IsNoSplit = false, bool EN_UNIT_FLAG = false>
struct TPipe;
```

## Constraints

- **TileData type producer**:
    - `TileProd::Loc` must be `TileType::Acc`, `TileType::Vec`, or `TileType::Ctrl`.
    - `Direction::DIR_C2V`: Cube produces an accumulator tile for vector consumption.
    - `Direction::DIR_V2C`: Vector produces a vector tile for cube consumption.
    - `Direction::DIR_BOTH`: The same pipe type supports both C2V and V2C producers.
- **FIFO slots**:
    - `SlotSize` must be large enough to hold one logical FIFO entry.
    - `SlotNum >= 1`.
- **Split behavior for Atlas A2/A3 training products/Atlas A2/A3 inference products**:
    - `TileSplitAxis::TILE_NO_SPLIT`: No split is performed. To enable this split mode on Atlas A2/A3 training products/Atlas A2/A3 inference products, AIV0 and AIV1 must perform accompanying synchronization operations.
    - `TileSplitAxis::TILE_UP_DOWN`: The vector sub-block is mapped to the upper and lower row halves.
    - `TileSplitAxis::TILE_LEFT_RIGHT`: The vector sub-block is mapped to the left and right column halves.
- **Ascend 950PR/Ascend 950DT split behavior**:
    - `TileSplitAxis::TILE_NO_SPLIT`: No split is performed.
    - `TileSplitAxis::TILE_UP_DOWN`: Splits the data into upper and lower halves. In the Cube->Vector direction over the L0C->UB path, this split mode supports only the b32 data type, and the validRows of srcTile must be an integer multiple of 2. In the Vector->Cube direction over the UB->L1 path, validRows must be an integer multiple of 32 bytes in this split mode.
    - `TileSplitAxis::TILE_LEFT_RIGHT`: Splits the data into left and right column halves. In the Cube->Vector direction over the L0C->UB path, this split mode supports only the b32 data type, and the validCols of srcTile must be an integer multiple of 32. In the Vector->Cube direction over the UB->L1 path, validCols must be an integer multiple of 32 bytes in this split mode.
- **Simplified TileData API**:
    - `TPUSH(TileData&, Pipe&)` internally uses `TileSplitAxis::TILE_NO_SPLIT` semantics.
    - `TileData::Loc` must be `TileType::Acc` or `TileType::Vec`.
- **TConfig API**:
    - `TConfig` is the configuration type that determines the push behavior (implementation-defined).
    - `TileProd::Loc` must be `TileType::Acc`, `TileType::Vec`, or `TileType::Ctrl`.
- **Synchronization**:
    - Free-space waiting is sparse and controlled by `Pipe::SyncPeriod`.
    - Each `TPUSH` issues a data-ready record.
- **GlobalData type producer**:
    - `gmTensor` must be a FIFO slot view returned by `TALLOC`.
    - Before calling `TPUSH(Pipe&, GlobalData&)`, data must have been written to `gmTensor`.
    - `TPUSH(Pipe&, GlobalData&)` ignores the tensor content and only submits the FIFO slot to the consumer.
- **Tile type support**:
    - **Tile types supported by TPUSH/TPOP**:
        - `TileType::Acc` (accumulator tile): used by the Cube core for C2V direction communication.
        - `TileType::Vec` (vector tile): used by the Vector core for V2C direction communication.
        - `TileType::Ctrl` (control tile): used by the Vector core for control signal communication in the V2C_CTRL direction.

## Defining TConfig

The `TConfig` template parameter in the `TPUSH(Pipe&, TileProd&, TConfig)` overload is a configuration structure used to control the fixpipe behavior during the push process. PTO provides the `FixpipeParams` structure to implement this functionality.

Declared in `include/pto/common/fixpipe.hpp`:

```cpp
template <LayoutMode_t layoutMode = LayoutMode_t::NZ2ND,
          QuantMode_t quantMode = QuantMode_t::NoQuant,
          ReluPreMode reluMode = ReluPreMode::NoRelu,
          STPhase phase = STPhase::Unspecified,
          uint8_t subBlockId = 0,
          AtomicType atomicT = AtomicType::AtomicNone,
          ClipReluMode_t clipReluMode = ClipReluMode_t::NOCLIP_RELU,
          bool isChannelSplit = false>
struct FixpipeParams {
    static constexpr LayoutMode_t LayoutMode = layoutMode;
    static constexpr QuantMode_t QuantPre = quantMode;
    static constexpr ReluPreMode ReluMode = reluMode;
    static constexpr STPhase Phase = phase;
    static constexpr uint8_t SubBlockId = subBlockId;
    static constexpr AtomicType AtomicT = atomicT;
    static constexpr ClipReluMode_t ClipReluMode = clipReluMode;
    static constexpr bool IsChannelSplit = isChannelSplit;
};
```

### TConfig Field Description

| Field | Type | Description |
|------|------|------|
| `LayoutMode` | `LayoutMode_t` | Output data layout: `NZ2NZ` (NZ→NZ), `NZ2ND` (NZ→Row-Major), `NZ2DN` (NZ→Column-Major). Default value: `NZ2ND`. |
| `QuantPre` | `QuantMode_t` | Quantization/dequantization mode (defined by CANN). Controls data type conversion during fixpipe push. Default value: `NoQuant`. |
| `ReluMode` | `ReluPreMode` | ReLU activation mode: `NoRelu` or `NormalRelu`. Default value: `NoRelu`. |
| `Phase` | `STPhase` | Storage phase (for the unit-flag path): `Unspecified`, `Partial`, or `Final`. Default value: `Unspecified`. |
| `SubBlockId` | `uint8_t` | Sub-block identifier, used for accumulator-to-vector move mode mapping (Ascend 950PR/Ascend 950DT only). Default value: `0`. |
| `AtomicT` | `AtomicType` | Atomic operation type for GM writes: `AtomicNone` or `AtomicAdd`. Default value: `AtomicNone`. |
| `ClipReluMode` | `ClipReluMode_t` | Clip ReLU mode: `NOCLIP_RELU` or `CLIP_RELU`. Default value: `NOCLIP_RELU`. |
| `IsChannelSplit` | `bool` | Whether to enable channel split. Default value: `false`. |

### TConfig Usage Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_tconfig_push(__gm__ void *fifoMem)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint32_t FlagID = 0;
    constexpr uint32_t FifoDepth = 2;

    using Pipe = TPipe<FlagID, Direction::DIR_C2V, M * N * sizeof(T), FifoDepth>;
    using AccTile = TileAcc<float, M, N, M, N>;

    // Define TConfig: NZ→row-major layout, dequantize to half, enable ReLU.
    using MyConfig = FixpipeParams<LayoutMode_t::NZ2ND, QuantMode_t::DEQF16, ReluPreMode::NormalRelu>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    AccTile acc;
    TASSIGN(acc, 0x0);

    TPUSH<Pipe, AccTile, MyConfig>(pipe, acc);
}
```

## Examples

### C2V Accumulator Push

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_c2v(__gm__ void *fifoMem)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint32_t FlagID = 0;
    constexpr uint32_t FifoDepth = 2;

    using Pipe = TPipe<FlagID, Direction::DIR_C2V, M * N * sizeof(float), FifoDepth>;
    using AccTile = TileAcc<float, M, N, M, N>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    AccTile acc;
    TASSIGN(acc, 0x0);

    // Fill acc with a cube computation before pushing.
    TPUSH<Pipe, AccTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, acc);
}
```

### V2C Vector Push

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_v2c(__gm__ void *fifoMem)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint32_t FlagID = 0;
    constexpr uint32_t FifoDepth = 2;

    using Pipe = TPipe<FlagID, Direction::DIR_V2C, M * N * sizeof(T), FifoDepth>;
    using VecTile = Tile<TileType::Vec, T, M, N, BLayout::RowMajor, M, N>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    VecTile tile;
    TASSIGN(tile, 0x0);

    TPUSH<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, tile);
}
```

### GlobalData Push Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_globaldata(__gm__ void *fifoMem)
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

## ASM Examples

The currently published assembly reference does not yet define a stable PTO-AS form for `TPUSH`. When hand-writing CV FIFO programs, use the C++ intrinsic form.

```text
