# Enhanced TPUSH/TPOP ISA Design for Intra-Cluster Function Group Data Communication

## Overview

This document specifies an enhanced ISA design for `TPUSH` and `TPOP` instructions to support **intra-cluster data communication across InCore kernels** within a function group.

### Cluster Architecture

Each cluster contains **1 Cube core** and **2 buddy Vector cores** that share a hardware flag-based synchronization mechanism:

```
┌─────────────────────── Cluster ───────────────────────┐
│                                                       │
│  ┌──────────┐    flags (8 per dir)    ┌──────────┐   │
│  │  Vector 0 │◄══════════════════════►│          │   │
│  └──────────┘   SET/WAIT V→C, C→V    │   Cube   │   │
│                                       │          │   │
│  ┌──────────┐    flags (8 per dir)    │          │   │
│  │  Vector 1 │◄══════════════════════►│          │   │
│  └──────────┘   SET/WAIT V→C, C→V    └──────────┘   │
│                                                       │
└───────────────────────────────────────────────────────┘
```

- A Vector core can **SET** a flag that the Cube core **WAITs** on, and vice versa.
- There are **8 flags per direction per peer** (Vector→Cube: 8 flags, Cube→Vector: 8 flags), for a total of 16 flags per Vector-Cube pair.
- With 2 buddy Vector cores, the cluster has **32 cross-core flags** in total (2 peers × 2 directions × 8 flags).

### AIV Communication Modes: 1:1 vs 1:2

The TPUSH/TPOP design supports two AIV communication modes:

| AIV_MODE | Description | Use Case |
|---|---|---|
| `AIV_SINGLE` (1:1) | Communication with **one** buddy Vector core only | Simple workloads, single Vector core active |
| `AIV_DUAL` (1:2) | Communication with **both** buddy Vector cores simultaneously | Full cluster utilization, tile split/combined across AIV0 and AIV1 |

**Assumption for 1:1 mode**: In the current design, 1:1 mode (`TILE_NO_SPLIT`) communicates with **AIV0 by default**. This simplifies the API and provides forward compatibility — when upgrading to 1:2 mode, AIV0's role remains the same (upper/left portion), only AIV1 is added.

### Tile Split/Combine Axis

When using 1:2 mode, the user **must specify** how the Cube tile is split (C2V) or combined (V2C):

```cpp
enum TileSplitAxis : uint8_t {
    TILE_NO_SPLIT   = 0,   // 1:1 mode, no split
    TILE_UP_DOWN    = 1,   // Split/combine along rows: AIV0=upper half, AIV1=lower half
    TILE_LEFT_RIGHT = 2,   // Split/combine along cols: AIV0=left half, AIV1=right half
};
```

**Why this must be specified**: The Vector kernel code cannot compute correctly without knowing which portion of the tile it receives or produces. With `TILE_UP_DOWN`, AIV0 knows it has the upper rows; with `TILE_LEFT_RIGHT`, AIV0 knows it has the left columns.

```
TILE_UP_DOWN:                          TILE_LEFT_RIGHT:

  Cube Tile [16, 128]:                   Cube Tile [16, 128]:
  ┌───────────────────┐                  ┌──────────┬──────────┐
  │ upper [8, 128]    │ ↔ AIV0           │ left     │ right    │
  ├───────────────────┤                  │[16, 64]  │[16, 64]  │
  │ lower [8, 128]    │ ↔ AIV1           │↔ AIV0    │↔ AIV1    │
  └───────────────────┘                  └──────────┴──────────┘
```

### Ring Buffer Data Channel

Data is moved between producer and consumer kernels through a **multi-slot ring buffer** with flow control. Each slot holds one fixed-size **Tile**. The ring buffer location depends on the platform:

| Platform | Ring Buffer Location | Description |
|---|---|---|
| **A3** | **Global Memory (GM)** | Ring buffer resides in off-chip DDR/HBM, accessible by all cores in the cluster |
| **A5** | **Consumer's on-chip SRAM** | Ring buffer resides in the consumer core's local memory: **Unified Buffer (UB)** if consumer is a Vector core, or **L1 Buffer** if consumer is a Cube core |

```
A3 Platform:                              A5 Platform:

Producer          GM            Consumer  Producer                    Consumer
┌──────┐   ┌────────────┐   ┌──────┐    ┌──────┐                 ┌──────────────┐
│      │──▶│ slot[0..N-1]│──▶│      │    │      │───────────────▶│ UB / L1      │
│ Cube │   │ (off-chip)  │   │ Vec  │    │ Cube │  DMA to local  │ slot[0..N-1] │
│ /Vec │   └────────────┘   │ /Cube│    │ /Vec │                 │ (on-chip)    │
└──────┘                     └──────┘    └──────┘                 └──────────────┘
```

The A5 placement in consumer-local SRAM eliminates the round-trip to GM, enabling lower-latency data handoff. The consumer can directly operate on tile data in its local buffer without an explicit TLOAD from GM.

The enhanced design extends `TPUSH`/`TPOP` to serve as the primary data communication mechanism between InCore kernels co-scheduled on these cores within the same cluster, enabling:

- **Producer kernel → Ring Buffer → Consumer kernel** tile-level data flow between Cube and buddy Vector cores
- Cross-core synchronization via the hardware flag mechanism (SET/WAIT, 8 flags per direction)
- Multi-slot ring buffer for pipelined execution (`SLOT_NUM` = 8 for unidirectional, 4 for bidirectional)
- Platform-adaptive buffer placement: GM (A3) or consumer-local SRAM (A5)

## Motivation: Intra-Cluster Function Group Communication

When the `ExpandMixedKernel` pass decomposes a mixed InCore function into multiple co-scheduled kernels (e.g., a data-movement kernel on Vector cores and a compute kernel on Cube cores), these kernels need an efficient, synchronized data communication channel within the same cluster.

`TPUSH`/`TPOP` with ring buffer flow control provides exactly this capability — the enhanced design formalizes how the compiler should emit `TPUSH`/`TPOP` pairs to connect the expanded kernel group.

## Enhanced Design: Tag-Based Dual-Channel FIFO Protocol

The enhanced TPUSH/TPOP design uses a **multi-slot ring buffer** with tag-based dual-channel flow control for moving fixed-size Tile data between producer and consumer kernels.

### Producer / Consumer Roles

The terms **producer** and **consumer** are conceptual roles, not bound to a specific core type:

- A **Cube core** can be a producer (e.g., matmul output → Vector for post-processing), or a consumer (e.g., receiving preprocessed data from Vector).
- A **Vector core** can be a producer (e.g., data loading / preprocessing → Cube), or a consumer (e.g., receiving matmul results from Cube).
- In some applications, **both cores are simultaneously producer and consumer** in opposite directions, forming a bidirectional data flow.

### Ring Buffer Structure

Each ring buffer is a **unidirectional** channel from one producer to one consumer. The number of slots is a compile-time constant parameter `SLOT_NUM`, specified during kernel initialization:

| Communication Pattern | SLOT_NUM | Flags Used | Description |
|---|---|---|---|
| **Unidirectional** (one direction only) | **8** | 8 flags for P2C + C2P | All 8 flags per direction dedicated to a single ring buffer |
| **Bidirectional** (both directions simultaneously) | **4** per direction | 4 flags for each of the 2 ring buffers | The 8 available flags are split equally between the two directions |

```
Unidirectional (SLOT_NUM=8):

    Cube (producer)  ──────▶  Vector (consumer)
    Ring Buffer: slot[0..7], using flags 0..7

Bidirectional (SLOT_NUM=4 per direction):

    Cube  ──── Ring Buffer A (slot[0..3], flags 0..3) ────▶  Vector
    Cube  ◀──── Ring Buffer B (slot[0..3], flags 4..7) ────  Vector
```

```
Ring Buffer  —  SLOT_NUM fixed-size Tile slots, indexed by tag

    ┌──────────────────────────────────────────────────────┐
    │  slot[0]    slot[1]    ...    slot[SLOT_NUM-1]       │   A3: Global Memory
    └──────────────────────────────────────────────────────┘   A5: Consumer's UB or L1

Signal Channels (mapped to hardware cross-core flags):
    P2C  —  Producer → Consumer  (data ready signal, indexed by tag)
    C2P  —  Consumer → Producer  (space free signal, indexed by tag)
```

Each ring buffer slot holds exactly one Tile and is identified by a **tag** (0 .. SLOT_NUM-1). The two signal channels `P2C` and `C2P` carry per-tag notifications:
- `SET P2C: tag` — producer signals "data in `slot[tag]` is ready"
- `SET C2P: tag` — consumer signals "`slot[tag]` is free for reuse"
- `WAIT P2C: tag` — consumer blocks until `slot[tag]` is ready
- `WAIT C2P: tag` — producer blocks until `slot[tag]` is free

### API Definition

#### Platform Constant

```cpp
enum PlatformID : uint8_t {
    PLATFORM_A2A3 = 0,   // A2/A3 platform: ring buffer in Global Memory
    PLATFORM_A5   = 1,   // A5 platform: ring buffer in consumer's on-chip SRAM
};
```

`PLATFORM_ID` is a **compile-time constant** generated by the compiler and embedded into the kernel binary. It is used by the initialization APIs and `tpush_*/tpop_*` instructions to select the appropriate behavior:

| PLATFORM_ID | Ring Buffer Location | `tpush_*` Behavior | `tpop_*` Behavior |
|---|---|---|---|
| `PLATFORM_A2A3` | GM (orchestration-allocated `GM_SLOT_BUFFER`) | DMA tile → GM slot | DMA GM slot → local tile |
| `PLATFORM_A5` | Consumer's on-chip SRAM (UB or L1) | DMA tile → consumer's local SRAM slot | Zero-copy: tile references local SRAM directly |

#### Direction Constants

```cpp
enum Direction : uint8_t {
    DIR_C2V = 0,   // Cube → Vector: Cube is producer, Vector is consumer
    DIR_V2C = 1,   // Vector → Cube: Vector is producer, Cube is consumer
};
```

A kernel uses `DIR_C2V` or `DIR_V2C` to specify the data flow direction. For bidirectional communication, both directions are active simultaneously (`DIR_C2V | DIR_V2C`).

#### Tile Split/Combine Axis Constants

```cpp
enum TileSplitAxis : uint8_t {
    TILE_NO_SPLIT   = 0,   // 1:1 mode, full tile to/from single AIV
    TILE_UP_DOWN    = 1,   // 1:2 mode, split/combine along rows (M axis)
    TILE_LEFT_RIGHT = 2,   // 1:2 mode, split/combine along cols (N axis)
};
```

#### `DIR_MASK`

A bitmask indicating which directions are active for this kernel:

| DIR_MASK | Value | Meaning | SLOT_NUM per direction |
|---|---|---|---|
| `DIR_C2V` | `0b01` | Unidirectional: Cube → Vector only | 8 |
| `DIR_V2C` | `0b10` | Unidirectional: Vector → Cube only | 8 |
| `DIR_C2V \| DIR_V2C` | `0b11` | Bidirectional: both directions | 4 |

#### `GM_SLOT_BUFFER` and `CONSUMER_BUFFER_BASE` / `CONSUMER_BUFFER_SIZE`

The ring buffer backing memory differs between A2A3 and A5:

| Platform | Ring Buffer Source | Mechanism |
|---|---|---|
| **A2A3** | `GM_SLOT_BUFFER` — orchestration-allocated GM buffer, passed as INOUT argument | Same as before |
| **A5** | `CONSUMER_BUFFER_BASE` / `CONSUMER_BUFFER_SIZE` — compiler-generated constant symbols per InCore function | See "Consumer SRAM Address Problem" section below |

**A2A3**: `GM_SLOT_BUFFER` is allocated in GM by the orchestration and passed to both InCore functions as INOUT.

**A5**: The ring buffer lives in the consumer's local SRAM. Its location is specified by `CONSUMER_BUFFER_BASE` and `CONSUMER_BUFFER_SIZE`, which are **constant symbols attached to each InCore function** (see the detailed design in the "Cross-Core Address Problem on A5" section below). The resolved `CONSUMER_BUFFER_BASE` values are passed as **explicit arguments** (`C2V_CONSUMER_BUF`, `V2C_CONSUMER_BUF`) to the initialization functions, avoiding special compiler requirements for implicit constant lookups.

```
Orchestration function (A2A3):
    gm_slot_buf = gm_alloc(2 * SLOT_NUM * SLOT_SIZE)    // bidirectional

    for ...:
        cube_kernel(  ..., GM_SLOT_BUFFER=gm_slot_buf, ...)   // INOUT
        vector_kernel(..., GM_SLOT_BUFFER=gm_slot_buf, ...)   // INOUT

Orchestration function (A5):
    // CONSUMER_BUFFER_BASE values are resolved by compiler and passed explicitly

    for ...:
        cube_kernel(  ..., GM_SLOT_BUFFER=nullptr, ...)
        vector_kernel(..., GM_SLOT_BUFFER=nullptr, ...)
```

#### `aic_initialize_pipe(DIR_MASK, SLOT_SIZE, GM_SLOT_BUFFER, C2V_CONSUMER_BUF, V2C_CONSUMER_BUF)`

**Called on the Cube (AIC) core at kernel startup.** Initializes the ring buffer pipe(s) for the specified direction(s).

| Parameter | Type | Description |
|---|---|---|
| `DIR_MASK` | `uint8_t` | Bitmask of active directions (`DIR_C2V`, `DIR_V2C`, or both) |
| `SLOT_SIZE` | `uint32_t` | Size of each ring buffer slot in bytes (= Tile size) |
| `GM_SLOT_BUFFER` | `__gm__ void*` | GM buffer allocated by orchestration (INOUT). Active on A2A3; `nullptr` on A5 |
| `C2V_CONSUMER_BUF` | `uint32_t` | Consumer's SRAM base address for C2V direction (Vector's UB). `0` on A2A3; explicit on A5 |
| `V2C_CONSUMER_BUF` | `uint32_t` | Consumer's SRAM base address for V2C direction (Cube's own L1). `0` on A2A3; explicit on A5 |

**Note**: The split axis (`TILE_UP_DOWN` / `TILE_LEFT_RIGHT`) is **not** specified during initialization. It is specified per-instruction on `tpush_to_aiv` and `tpop_from_aiv`.

**Description**: Binds the ring buffer pipe(s) to the appropriate backing memory based on `PLATFORM_ID`, computes `SLOT_NUM` from `DIR_MASK` (8 if unidirectional, 4 if bidirectional), and initializes internal state. On A5, the ring buffer base addresses are passed as **explicit arguments** (`C2V_CONSUMER_BUF`, `V2C_CONSUMER_BUF`) — no implicit constant symbol lookup is required. For each direction where the Cube is the **consumer** (`DIR_V2C`), it signals all slots as free to the Vector producer.

- On `PLATFORM_A2A3`: uses `GM_SLOT_BUFFER` in GM for all directions. `C2V_CONSUMER_BUF` and `V2C_CONSUMER_BUF` are ignored.
- On `PLATFORM_A5`:
  - **C2V (Cube is producer)**: uses `C2V_CONSUMER_BUF` — the Vector's UB address, passed explicitly.
  - **V2C (Cube is consumer)**: uses `V2C_CONSUMER_BUF` — Cube's own L1 address, passed explicitly.

**Pseudocode**:

```
function aic_initialize_pipe(DIR_MASK, SLOT_SIZE, GM_SLOT_BUFFER, C2V_CONSUMER_BUF, V2C_CONSUMER_BUF):
    if DIR_MASK == (DIR_C2V | DIR_V2C):
        SLOT_NUM = 4
    else:
        SLOT_NUM = 8

    if DIR_MASK & DIR_C2V:
        // Cube is PRODUCER in C2V direction
        if PLATFORM_ID == PLATFORM_A2A3:
            c2v_ring_buf = GM_SLOT_BUFFER                         // GM buffer
        else:  // PLATFORM_A5
            c2v_ring_buf = C2V_CONSUMER_BUF                       // Vector's UB (explicit argument)
        c2v_target_tag = 0

    if DIR_MASK & DIR_V2C:
        // Cube is CONSUMER in V2C direction
        if PLATFORM_ID == PLATFORM_A2A3:
            buf_offset = (DIR_MASK & DIR_C2V) ? SLOT_NUM * SLOT_SIZE : 0
            v2c_ring_buf = GM_SLOT_BUFFER + buf_offset            // GM buffer
        else:  // PLATFORM_A5
            v2c_ring_buf = V2C_CONSUMER_BUF                       // Cube's own L1 (explicit argument)
        v2c_target_tag = 0
        // Signal all slots as free to Vector producer(s)
        // Note: flags for both AIV0 and AIV1 are pre-signaled
        // The actual 1:1 vs 1:2 mode is determined per-instruction
        for (i = 0; i < SLOT_NUM; i++):
            SET flag_V2C_free[AIV0]: i
            SET flag_V2C_free[AIV1]: i
```

#### `aiv_initialize_pipe(DIR_MASK, SLOT_SIZE, GM_SLOT_BUFFER, C2V_CONSUMER_BUF, V2C_CONSUMER_BUF)`

**Called on a Vector (AIV) core at kernel startup.** Initializes the ring buffer pipe(s) for the specified direction(s).

| Parameter | Type | Description |
|---|---|---|
| `DIR_MASK` | `uint8_t` | Bitmask of active directions (`DIR_C2V`, `DIR_V2C`, or both) |
| `SLOT_SIZE` | `uint32_t` | Size of each ring buffer slot in bytes (= Tile size) |
| `GM_SLOT_BUFFER` | `__gm__ void*` | GM buffer allocated by orchestration (INOUT). Active on A2A3; `nullptr` on A5 |
| `C2V_CONSUMER_BUF` | `uint32_t` | Consumer's SRAM base address for C2V direction (Vector's own UB). `0` on A2A3; explicit on A5 |
| `V2C_CONSUMER_BUF` | `uint32_t` | Consumer's SRAM base address for V2C direction (Cube's L1). `0` on A2A3; explicit on A5 |

**Note**: The split axis (`TILE_UP_DOWN` / `TILE_LEFT_RIGHT`) is **not** specified during initialization. It is specified per-instruction on `tpush_to_aic` and `tpop_from_aic`.

**Description**: Binds the ring buffer pipe(s) to the appropriate backing memory based on `PLATFORM_ID`, computes `SLOT_NUM`, and initializes internal state. On A5, the ring buffer base addresses are passed as **explicit arguments** (`C2V_CONSUMER_BUF`, `V2C_CONSUMER_BUF`) — no implicit constant symbol lookup is required. For each direction where the Vector is the **consumer** (`DIR_C2V`), it signals all slots as free to the Cube producer.

**Pseudocode**:

```
function aiv_initialize_pipe(DIR_MASK, SLOT_SIZE, GM_SLOT_BUFFER, C2V_CONSUMER_BUF, V2C_CONSUMER_BUF):
    if DIR_MASK == (DIR_C2V | DIR_V2C):
        SLOT_NUM = 4
    else:
        SLOT_NUM = 8

    my_aiv_idx = get_my_aiv_idx()   // 0 or 1

    if DIR_MASK & DIR_C2V:
        // Vector is CONSUMER in C2V direction
        if PLATFORM_ID == PLATFORM_A2A3:
            c2v_ring_buf = GM_SLOT_BUFFER                         // GM buffer
        else:  // PLATFORM_A5
            c2v_ring_buf = C2V_CONSUMER_BUF                       // Vector's own UB (explicit argument)
        c2v_target_tag = 0
        // Signal all slots as free to Cube producer (only signal my own flag)
        for (i = 0; i < SLOT_NUM; i++):
            SET flag_C2V_free[my_aiv_idx]: i

    if DIR_MASK & DIR_V2C:
        // Vector is PRODUCER in V2C direction
        if PLATFORM_ID == PLATFORM_A2A3:
            buf_offset = (DIR_MASK & DIR_C2V) ? SLOT_NUM * SLOT_SIZE : 0
            v2c_ring_buf = GM_SLOT_BUFFER + buf_offset            // GM buffer
        else:  // PLATFORM_A5
            v2c_ring_buf = V2C_CONSUMER_BUF                       // Cube's L1 (explicit argument)
        v2c_target_tag = 0
```

#### Buffer Layout and Cross-Core Address Problem on A5

##### A2A3: Straightforward GM Layout

On A2A3, the ring buffer resides in GM. The orchestration allocates a single `GM_SLOT_BUFFER` and passes it to both InCore functions. Both cores access the same physical GM addresses.

```
GM_SLOT_BUFFER (total size = 2 * SLOT_NUM * SLOT_SIZE for bidirectional):

    ┌─────────────────────────────┬─────────────────────────────┐
    │  C2V ring buffer            │  V2C ring buffer            │
    │  slot[0] .. slot[SLOT_NUM-1]│  slot[0] .. slot[SLOT_NUM-1]│
    │  offset: 0                  │  offset: SLOT_NUM*SLOT_SIZE │
    └─────────────────────────────┴─────────────────────────────┘
```

##### A5: Consumer SRAM Address Problem

On A5, the ring buffer for each direction resides in the **consumer's on-chip SRAM** (UB or L1). This creates a fundamental problem:

1. The ring buffer is a **local memory region in the consumer's InCore function**, allocated by the compiler in the consumer's local address space (UB or L1).
2. The **producer** needs to know this address to DMA data into it — but it lives in another core's address space.
3. In standard C/C++ semantics, a local symbol's address from one function cannot be referenced by another function. This violates symbol locality.

```
A5 Problem: C2V direction (Cube produces → Vector consumes)

    Cube InCore function:                    Vector InCore function:
    ┌─────────────────────┐                 ┌─────────────────────┐
    │  tpush_to_aiv       │   ??? how to    │  consumer_buf =     │
    │  DMA to Vector's UB │ ──────────────▶ │  UB[BASE..BASE+SIZE]│
    │  at what address?   │   get address?  │  // local segment   │
    └─────────────────────┘                 └─────────────────────┘
```

##### Solution: `CONSUMER_BUFFER_BASE` / `CONSUMER_BUFFER_SIZE` Constant Symbols

The solution defines two **constant symbols** that are attached to **each InCore function** that participates in TPUSH/TPOP communication:

```cpp
const uint32_t CONSUMER_BUFFER_BASE;   // base address of the consumer's ring buffer in its local SRAM
const uint32_t CONSUMER_BUFFER_SIZE;   // total size in bytes (= SLOT_NUM * SLOT_SIZE)
```

These symbols represent a **reserved memory segment** in the consumer InCore function's local SRAM (UB for Vector, L1 for Cube). The key properties are:

1. **Per-function constants**: Each InCore function that acts as a **consumer** in any TPUSH/TPOP direction has its own `CONSUMER_BUFFER_BASE` and `CONSUMER_BUFFER_SIZE`. Each InCore function that acts as a **producer** also receives the peer consumer's `CONSUMER_BUFFER_BASE` and `CONSUMER_BUFFER_SIZE` so it knows the DMA target.

2. **Value origin**:
   - **Auto-generated kernels** (`auto_incore` / `ExpandMixedKernel`): The values are generated by the `ExpandMixedKernel` pass, which has visibility into both producer and consumer functions' memory layouts and can assign a non-overlapping SRAM region for the ring buffer.
   - **Manually written kernels**: The programmer specifies `CONSUMER_BUFFER_BASE` and `CONSUMER_BUFFER_SIZE` as explicit constant declarations in the InCore function. The values must be chosen to not conflict with other SRAM usage.

3. **Address allocator reservation**: The downstream **memory address allocator** (e.g., `AllocateMemoryAddr` pass) must treat the segment `[CONSUMER_BUFFER_BASE, CONSUMER_BUFFER_BASE + CONSUMER_BUFFER_SIZE)` as **occupied / in-use** in the consumer function's SRAM. It must **not** allocate any other symbols (tiles, temporaries, etc.) into this region. This ensures the ring buffer and the function's normal tile allocations do not overlap.

4. **Cross-function visibility**: The `CONSUMER_BUFFER_BASE` value of a consumer function is visible to its paired producer function as a compile-time constant. The compiler ensures this by:
   - Generating both functions in the same compilation unit (natural for `ExpandMixedKernel`).
   - Emitting the consumer's `CONSUMER_BUFFER_BASE` as a constant in the producer's initialization code.

##### Buffer Layout Summary

```
A2A3 (ring buffer in GM, GM_SLOT_BUFFER active):

    GM_SLOT_BUFFER:
    ┌─────────────────────────────┬─────────────────────────────┐
    │  C2V ring buffer            │  V2C ring buffer            │
    │  slot[0] .. slot[SLOT_NUM-1]│  slot[0] .. slot[SLOT_NUM-1]│
    └─────────────────────────────┴─────────────────────────────┘

A5 (ring buffer in consumer SRAM, CONSUMER_BUFFER_BASE/SIZE):

    Vector UB (for C2V, Vector is consumer):
    ┌──────────┬──────────────────────────────┬───────────┐
    │ normal   │  CONSUMER_BUFFER segment     │ normal    │
    │ tiles    │  [BASE .. BASE+SIZE)         │ tiles     │
    │          │  slot[0] .. slot[SLOT_NUM-1] │           │
    └──────────┴──────────────────────────────┴───────────┘
    ◄─── allocator avoids this region ───►

    Cube L1 (for V2C, Cube is consumer):
    ┌──────────┬──────────────────────────────┬───────────┐
    │ normal   │  CONSUMER_BUFFER segment     │ normal    │
    │ tiles    │  [BASE .. BASE+SIZE)         │ tiles     │
    │          │  slot[0] .. slot[SLOT_NUM-1] │           │
    └──────────┴──────────────────────────────┴───────────┘
```

#### Data Transfer Instructions

The ISA defines **four distinct instructions**, each executed on a specific core type with an implicit direction. Each instruction takes a `TileSplitAxis` parameter to specify 1:1 or 1:2 mode.

| Instruction | Executed On | Role | Direction | Description |
|---|---|---|---|---|
| `tpush_to_aiv(TILE, SPLIT)` | **Cube** | Producer | C2V | Push tile from Cube to buddy Vector(s) |
| `tpush_to_aic(TILE, SPLIT)` | **Vector** | Producer | V2C | Push tile from Vector to buddy Cube |
| `tpop_from_aic(TILE, SPLIT)` | **Vector** | Consumer | C2V | Pop tile that Cube pushed |
| `tpop_from_aiv(TILE, SPLIT)` | **Cube** | Consumer | V2C | Pop tile that Vector(s) pushed |

**`SPLIT` parameter values**:
- `TILE_NO_SPLIT`: 1:1 mode (full tile to/from single AIV)
- `TILE_UP_DOWN`: 1:2 mode (split/combine along rows)
- `TILE_LEFT_RIGHT`: 1:2 mode (split/combine along cols)

#### `tpush_to_aiv(TILE, SPLIT)` — C2V Direction

**Executed on Cube (AIC).** Pushes a tile into the C2V ring buffer destined for buddy Vector core(s).

**1:1 Mode** (`SPLIT == TILE_NO_SPLIT`):
```
function tpush_to_aiv(TILE, TILE_NO_SPLIT):
    // Wait for single AIV's free flag
    WAIT flag_free[C2V, target_aiv]: target_tag

    // DMA full tile to target AIV's slot
    dst_addr = ring_buf_base + target_tag * SLOT_SIZE
    MTE_copy(src=TILE.data, dst=dst_addr, size=SLOT_SIZE)
    WAIT mte_flag

    // Signal single AIV ready
    SET flag_ready[C2V, target_aiv]: target_tag
    target_tag = (target_tag + 1) % SLOT_NUM
```

**1:2 Mode** (`SPLIT == TILE_UP_DOWN` or `TILE_LEFT_RIGHT`):
```
function tpush_to_aiv(TILE, SPLIT):
    // Wait for BOTH AIV0 and AIV1 free flags
    WAIT flag_free[C2V, AIV0]: target_tag
    WAIT flag_free[C2V, AIV1]: target_tag

    // Split tile and DMA halves to both AIVs
    if SPLIT == TILE_UP_DOWN:
        // Upper half to AIV0, lower half to AIV1
        MTE_copy(src=TILE.upper_half, dst=aiv0_slot[target_tag], size=HALF_TILE_SIZE)
        MTE_copy(src=TILE.lower_half, dst=aiv1_slot[target_tag], size=HALF_TILE_SIZE)
    else:  // TILE_LEFT_RIGHT
        // Left half to AIV0, right half to AIV1
        MTE_copy(src=TILE.left_half,  dst=aiv0_slot[target_tag], size=HALF_TILE_SIZE)
        MTE_copy(src=TILE.right_half, dst=aiv1_slot[target_tag], size=HALF_TILE_SIZE)
    WAIT mte_flags

    // Signal BOTH AIV0 and AIV1 ready
    SET flag_ready[C2V, AIV0]: target_tag
    SET flag_ready[C2V, AIV1]: target_tag
    target_tag = (target_tag + 1) % SLOT_NUM
```

#### `tpush_to_aic(TILE, SPLIT)` — V2C Direction

**Executed on Vector (AIV).** Pushes a tile into the V2C ring buffer destined for the buddy Cube core.

**1:1 Mode** (`SPLIT == TILE_NO_SPLIT`):
```
function tpush_to_aic(TILE, TILE_NO_SPLIT):
    // Wait for my free flag
    WAIT flag_free[V2C, my_aiv_idx]: target_tag

    // DMA full tile to slot
    dst_addr = ring_buf_base + target_tag * SLOT_SIZE
    MTE_copy(src=TILE.data, dst=dst_addr, size=SLOT_SIZE)
    WAIT mte_flag

    // Signal my ready flag
    SET flag_ready[V2C, my_aiv_idx]: target_tag
    target_tag = (target_tag + 1) % SLOT_NUM
```

**1:2 Mode** (`SPLIT == TILE_UP_DOWN` or `TILE_LEFT_RIGHT`):
```
function tpush_to_aic(TILE, SPLIT):
    // Wait for my free flag
    WAIT flag_free[V2C, my_aiv_idx]: target_tag

    // Compute my write offset based on my AIV index and split axis
    if SPLIT == TILE_UP_DOWN:
        // AIV0 writes upper half (offset 0), AIV1 writes lower half
        dst_offset = my_aiv_idx * HALF_TILE_SIZE
    else:  // TILE_LEFT_RIGHT
        // AIV0 writes left half (offset 0), AIV1 writes right half
        dst_offset = my_aiv_idx * HALF_TILE_SIZE

    // DMA my half to shared L1 slot with correct offset
    dst_addr = ring_buf_base + target_tag * FULL_TILE_SIZE + dst_offset
    MTE_strided_copy(src=TILE.data, dst=dst_addr, ...)
    WAIT mte_flag

    // Signal my ready flag
    SET flag_ready[V2C, my_aiv_idx]: target_tag
    target_tag = (target_tag + 1) % SLOT_NUM
```

#### `tpop_from_aic(TILE, SPLIT)` — C2V Direction

**Executed on Vector (AIV).** Pops a tile from the C2V ring buffer (data that Cube pushed).

**Both 1:1 and 1:2 modes**: Each AIV receives its portion independently.
```
function tpop_from_aic(TILE, SPLIT):
    // Wait for my ready flag
    WAIT flag_ready[C2V, my_aiv_idx]: target_tag

    // Receive my tile (full in 1:1, half in 1:2)
    src_addr = my_slot[target_tag]
    if PLATFORM_A5:
        TILE.data = src_addr   // zero-copy, data already in UB
    else:  // PLATFORM_A2A3
        // Must load from GM (not a no-op sync)
        // Implementation: strided read or strided write, per GM access perf
        MTE_load(dst=TILE.data, src=src_addr, size=tile_size)
        WAIT mte_flag

    // Signal my free flag
    SET flag_free[C2V, my_aiv_idx]: target_tag
    target_tag = (target_tag + 1) % SLOT_NUM
```

**Note**: In 1:2 mode, each AIV's kernel knows which portion it received based on `SPLIT`:
- `TILE_UP_DOWN`: AIV0 has upper rows, AIV1 has lower rows
- `TILE_LEFT_RIGHT`: AIV0 has left cols, AIV1 has right cols

#### `tpop_from_aiv(TILE, SPLIT)` — V2C Direction

**Executed on Cube (AIC).** Pops a tile from the V2C ring buffer (data that Vector(s) pushed).

**1:1 Mode** (`SPLIT == TILE_NO_SPLIT`):
```
function tpop_from_aiv(TILE, TILE_NO_SPLIT):
    // Wait for single AIV's ready flag
    WAIT flag_ready[V2C, target_aiv]: target_tag

    // Receive full tile
    src_addr = slot[target_tag]
    if PLATFORM_A5:
        TILE.data = src_addr   // zero-copy, data already in L1
    else:  // PLATFORM_A2A3
        // Must load from GM (not a no-op sync)
        MTE_load(dst=TILE.data, src=src_addr, size=SLOT_SIZE)
        WAIT mte_flag

    // Signal single AIV free
    SET flag_free[V2C, target_aiv]: target_tag
    target_tag = (target_tag + 1) % SLOT_NUM
```

**1:2 Mode** (`SPLIT == TILE_UP_DOWN` or `TILE_LEFT_RIGHT`):
```
function tpop_from_aiv(TILE, SPLIT):
    // Wait for BOTH AIV0 and AIV1 ready flags
    WAIT flag_ready[V2C, AIV0]: target_tag
    WAIT flag_ready[V2C, AIV1]: target_tag

    // The combined tile is now complete in L1 slot (A5) or GM slot (A2A3)
    // (AIV0 wrote upper/left half, AIV1 wrote lower/right half)
    src_addr = slot[target_tag]
    if PLATFORM_A5:
        TILE.data = src_addr   // zero-copy, data already in L1
    else:  // PLATFORM_A2A3
        // Must load from GM with strided read to combine halves
        MTE_strided_load(dst=TILE.data, src=src_addr, ...)
        WAIT mte_flag

    // Signal BOTH AIV0 and AIV1 free
    SET flag_free[V2C, AIV0]: target_tag
    SET flag_free[V2C, AIV1]: target_tag
    target_tag = (target_tag + 1) % SLOT_NUM
```

### Flag Assignment

The 8 hardware flags per direction per peer are mapped as follows:

```
Unidirectional (DIR_C2V only, SLOT_NUM=8):

    flag_ready[C2V, aiv_idx] : flags 0..7   (Cube SETs, Vector WAITs)
    flag_free [C2V, aiv_idx] : flags 0..7   (Vector SETs, Cube WAITs)

Bidirectional (DIR_C2V | DIR_V2C, SLOT_NUM=4):

    flag_ready[C2V, aiv_idx] : flags 0..3   (Cube SETs, Vector WAITs)
    flag_free [C2V, aiv_idx] : flags 0..3   (Vector SETs, Cube WAITs)
    flag_ready[V2C, aiv_idx] : flags 4..7   (Vector SETs, Cube WAITs)
    flag_free [V2C, aiv_idx] : flags 4..7   (Cube SETs, Vector WAITs)
```

### Timing Diagram: C2V 1:2 Mode with TILE_UP_DOWN

```
          iter 0              iter 1              iter 2
tag:        0                   1                   2

AIC (Cube, producer):
          ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
          │ tpush_to_aiv     │  │ tpush_to_aiv     │  │ tpush_to_aiv     │
          │ WAIT f[0]:0      │  │ WAIT f[0]:1      │  │ WAIT f[0]:2      │
          │ WAIT f[1]:0      │  │ WAIT f[1]:1      │  │ WAIT f[1]:2      │
          │ MTE upper→AIV0   │  │ MTE upper→AIV0   │  │ MTE upper→AIV0   │
          │ MTE lower→AIV1   │  │ MTE lower→AIV1   │  │ MTE lower→AIV1   │
          │ SET r[0]:0       │  │ SET r[0]:1       │  │ SET r[0]:2       │
          │ SET r[1]:0       │  │ SET r[1]:1       │  │ SET r[1]:2       │
          └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
                   │                     │                     │
                   ▼ ready               ▼ ready               ▼ ready

AIV0 (Vector, consumer, receives UPPER half):
               ┌──────────────────┐  ┌──────────────────┐
               │ tpop_from_aic    │  │ tpop_from_aic    │
               │ WAIT r[0]:0      │  │ WAIT r[0]:1      │
               │ use upper half   │  │ use upper half   │
               │ SET f[0]:0       │  │ SET f[0]:1       │
               └────────┬─────────┘  └────────┬─────────┘

AIV1 (Vector, consumer, receives LOWER half):
               ┌──────────────────┐  ┌──────────────────┐
               │ tpop_from_aic    │  │ tpop_from_aic    │
               │ WAIT r[1]:0      │  │ WAIT r[1]:1      │
               │ use lower half   │  │ use lower half   │
               │ SET f[1]:0       │  │ SET f[1]:1       │
               └────────┬─────────┘  └────────┬─────────┘

Legend: r[x] = flag_ready[aiv_x], f[x] = flag_free[aiv_x]
```

### Timing Diagram: V2C 1:2 Mode with TILE_UP_DOWN

```
          iter 0              iter 1              iter 2
tag:        0                   1                   2

AIV0 (Vector, producer, writes UPPER half):
          ┌──────────────────┐  ┌──────────────────┐
          │ tpush_to_aic     │  │ tpush_to_aic     │
          │ WAIT f[0]:0      │  │ WAIT f[0]:1      │
          │ MTE upper→L1+0   │  │ MTE upper→L1+0   │
          │ SET r[0]:0       │  │ SET r[0]:1       │
          └────────┬─────────┘  └────────┬─────────┘

AIV1 (Vector, producer, writes LOWER half):
          ┌──────────────────┐  ┌──────────────────┐
          │ tpush_to_aic     │  │ tpush_to_aic     │
          │ WAIT f[1]:0      │  │ WAIT f[1]:1      │
          │ MTE lower→L1+off │  │ MTE lower→L1+off │
          │ SET r[1]:0       │  │ SET r[1]:1       │
          └────────┬─────────┘  └────────┬─────────┘
                   │                     │
                   ▼ ready               ▼ ready

AIC (Cube, consumer):
               ┌──────────────────┐  ┌──────────────────┐
               │ tpop_from_aiv    │  │ tpop_from_aiv    │
               │ WAIT r[0]:0      │  │ WAIT r[0]:1      │
               │ WAIT r[1]:0      │  │ WAIT r[1]:1      │
               │ use combined tile│  │ use combined tile│
               │ SET f[0]:0       │  │ SET f[0]:1       │
               │ SET f[1]:0       │  │ SET f[1]:1       │
               └────────┬─────────┘  └────────┬─────────┘
```

### Key Properties

1. **No deadlock**: The consumer side (`aiv_initialize_pipe` or `aic_initialize_pipe`) pre-signals all SLOT_NUM slots as free before the main loop begins, so the producer can fill up to SLOT_NUM slots before blocking.
2. **Backpressure**: If the producer is faster than the consumer, `tpush_*` blocks at `WAIT flag_free` when all slots are occupied; if the consumer is faster, `tpop_*` blocks at `WAIT flag_ready` when no data is ready.
3. **In-order delivery**: Both sides advance `target_tag` in strict round-robin order `(tag + 1) % SLOT_NUM`, guaranteeing FIFO semantics.
4. **Decoupled DMA**: `tpush_*` uses MTE for async data transfer with an explicit `mte_flag` wait to ensure completion before signaling the consumer.
5. **1:2 Mode Flow Control**:
   - **C2V**: Cube waits for **both** AIV0 and AIV1 free, then signals **both** ready.
   - **V2C**: Each AIV waits for its **own** free, signals its **own** ready. Cube waits for **both** ready, signals **both** free.
6. **Split Axis Semantic**: The `TileSplitAxis` set during init tells each kernel which portion of data it handles:
   - `TILE_UP_DOWN`: AIV0 = upper rows, AIV1 = lower rows
   - `TILE_LEFT_RIGHT`: AIV0 = left cols, AIV1 = right cols

### API Summary

| API | Called On | Role | Direction | Description |
|---|---|---|---|---|
| `aic_initialize_pipe(DIR_MASK, SLOT_SIZE, GM_SLOT_BUFFER, C2V_CONSUMER_BUF, V2C_CONSUMER_BUF)` | Cube (AIC) | Setup | — | Bind ring buffer, init tags, pre-signal free slots |
| `aiv_initialize_pipe(DIR_MASK, SLOT_SIZE, GM_SLOT_BUFFER, C2V_CONSUMER_BUF, V2C_CONSUMER_BUF)` | Vector (AIV) | Setup | — | Bind ring buffer, init tags, pre-signal free slots |
| `tpush_to_aiv(TILE, SPLIT)` | Cube (AIC) | Producer | C2V | 1:1: push to single AIV; 1:2: split and push to both AIVs |
| `tpush_to_aic(TILE, SPLIT)` | Vector (AIV) | Producer | V2C | 1:1: push full tile; 1:2: push my half with strided write |
| `tpop_from_aic(TILE, SPLIT)` | Vector (AIV) | Consumer | C2V | Receive my portion. A5: zero-copy; A2A3: load from GM |
| `tpop_from_aiv(TILE, SPLIT)` | Cube (AIC) | Consumer | V2C | 1:1: receive from single AIV; 1:2: wait both, receive combined. A5: zero-copy; A2A3: load from GM |

### DSL Grammar: `pl.reserve_buffer` — Reserved Address Space Declaration

*(This section remains unchanged from the original document.)*

The compiler must provide a **DSL-level mechanism** for InCore kernel programs to declare reserved address space for the SLOT_BUFFER. This is necessary because:

1. The address allocator must know which SRAM regions are off-limits **before** it runs.
2. For manually written InCore kernels, the programmer needs an explicit way to express "this region of my local SRAM is reserved for TPUSH/TPOP ring buffer."
3. For compiler-generated kernels (`auto_incore` / `ExpandMixedKernel`), the pass emits the same declaration into the generated IR, so the rest of the pipeline treats it uniformly.

---

## Version History

### v2.1 — 1:1 and 1:2 Modes with Tile Split/Combine Axis

**Date**: 2026-03-16

**Key Design Decisions**:

1. **Added `TileSplitAxis` enum** with three values:
   - `TILE_NO_SPLIT` — 1:1 mode, full tile to/from single AIV (AIV0 by default)
   - `TILE_UP_DOWN` — 1:2 mode, split/combine along rows (upper/lower halves)
   - `TILE_LEFT_RIGHT` — 1:2 mode, split/combine along cols (left/right halves)

2. **Split axis specified per-instruction** (not during init):
   - `initialize_pipe` does not take split params
   - Each `tpush_*` / `tpop_*` instruction takes `SPLIT` as second parameter
   - Rationale: Split axis is per-operation, not per-pipe property

3. **Platform-specific split/combine implementation**:
   - **A5**: Split/combine logic in `tpush` — producer directly writes to consumer's on-chip SRAM with correct layout; `tpop` is sync-only (zero-copy)
   - **A2A3**: Split/combine logic in `tpop` — consumer must load data from GM (not a no-op sync). Implementation options:
     - Strided read from GM and combine on consumer side, OR
     - Strided write to consumer's local buffer during read
     - Choice depends on GM access pattern performance on A3

4. **1:1 mode uses AIV0 by default** for forward compatibility:
   - When upgrading to 1:2 mode, AIV0's role remains the same
   - AIV1 is simply added for the other half

5. **Updated flow control semantics**:
   - **C2V 1:2**: Cube `tpush_to_aiv` waits for **both** AIV free flags, signals **both** ready
   - **V2C 1:2**: Each AIV `tpush_to_aic` waits its own free, signals its own ready; Cube `tpop_from_aiv` waits **both** ready, signals **both** free

6. **User-friendly naming**: Used `UP_DOWN` / `LEFT_RIGHT` instead of `M` / `N` axis — users immediately understand the tile partition without needing to know the M/N axis mapping.

**Rationale**: The split/combine axis **must be specified** so Vector kernel code knows which portion of the tile it receives or produces. Without this information, compute code cannot be correct.
