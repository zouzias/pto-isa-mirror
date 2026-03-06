# PTO TPUSH/TPOP C++ Interface v2

## Design Goals

1. **Kill subtile complexity** — No `pro_set_allocate`, no subtile tiling in API
2. **Use existing PTO Tile syntax** — Match FA kernel's TileDef conventions
3. **FIFO object encapsulates state** — Ring buffer counter, memory address, position
4. **User-controlled event IDs** — Cross-core events setup during init
5. **TPUSH handles sync internally** — Or provide explicit API for reserve/dependency
6. **Infer dual-dst split from tile shapes** — ROWS/COLS mismatch determines cutM/cutN

---

## Existing PTO Tile Syntax (from FA kernel)

The FA kernel uses this tile definition pattern:

```cpp
// Mat tiles (for Cube core L1)
using TileMatQData = Tile<TileType::Mat, half, ROWS, COLS, BLayout::ColMajor, 
                          ROWS, COLS, SLayout::RowMajor, 512>;
using TileMatKData = Tile<TileType::Mat, half, ROWS, COLS, BLayout::RowMajor,
                          ROWS, COLS, SLayout::ColMajor, 512>;

// Accumulator tiles (for Cube core output)
using TileQKData = TileAcc<float, Cube_S0, Cube_S1, Cube_S0, Cube_S1>;
using TilePVData = TileAcc<float, Cube_S0, HEAD_SIZE, Cube_S0, HEAD_SIZE>;

// Vector tiles (for Vec core UB)
using TileDataF_T = Tile<TileType::Vec, float, Vec_S0, Tile_S1, BLayout::RowMajor, Vec_S0, Tile_S1>;
using TileDataH_T = Tile<TileType::Vec, half, Vec_S0, Tile_S1, BLayout::RowMajor, Vec_S0, Tile_S1>;

// Reduce tiles (column vectors)
using ReduceTileF_T = Tile<TileType::Vec, float, SubblockRows, 1, BLayout::ColMajor, SubblockRows, 1>;
```

**Key observations:**
- `TileType::Mat` = L1 buffer for Cube matmul inputs
- `TileType::Vec` = UB buffer for Vector operations  
- `TileAcc<dtype, ROWS, COLS, ...>` = Accumulator output from Cube
- Template params: `<TileType, dtype, ROWS, COLS, BLayout, BRows, BCols, [SLayout, Align]>`
- ROWS/COLS are compile-time constants

---

## Core API

### Tile Type Definition

Use existing PTO Tile template syntax. Tile types carry shape info (ROWS/COLS) used to infer dual-dst split axis:

```cpp
// TileAcc definition (from pto_tile.hpp):
// template <typename Element_, const int Rows_, const int Cols_,
//           const int RowValid_ = Rows_, const int ColValid_ = Cols_>
// using TileAcc = Tile<TileType::Acc, Element_, Rows_, Cols_, BLayout::ColMajor,
//                      RowValid_, ColValid_, SLayout::RowMajor, TileConfig::fractalCSize>;

// Accumulator tile from Cube core (source for TPUSH)
// TileAcc<Element, Rows, Cols, RowValid=Rows, ColValid=Cols>
using AccTile_32x16_fp32 = TileAcc<float, 32, 16>;              // 2KB, RowValid=32, ColValid=16
using AccTile_16x16_fp32 = TileAcc<float, 16, 16>;              // 1KB

// Vector tile for Vec core (destination for TPOP)
// Tile<TileType::Vec, dtype, ROWS, COLS, BLayout, BRows, BCols>
using VecTile_16x16_half = Tile<TileType::Vec, half, 16, 16, BLayout::RowMajor, 16, 16>;  // 512B
using VecTile_32x16_half = Tile<TileType::Vec, half, 32, 16, BLayout::RowMajor, 32, 16>;  // 1KB

// Extract ROWS/COLS from tile type (Tile struct has Rows/Cols members)
template <typename TileT>
struct TileTraits {
    static constexpr int Rows = TileT::Rows;
    static constexpr int Cols = TileT::Cols;
    using DType = typename TileT::DType;
};
```

### FIFO Object

```cpp
template <typename AccTileT, typename VecTileT, int Depth = 2>
struct CrossCoreFIFO {
    // Ring buffer state
    uint32_t head;           // Producer position
    uint32_t tail;           // Consumer position
    
    // Memory layout
    void*    bufferBase;     // Base address of FIFO buffer
    
    // Cross-core sync
    uint16_t pushEventId;    // Event ID for producer → consumer signal
    uint16_t popEventId;     // Event ID for consumer → producer signal (backpressure)
    
    // Compile-time shape info extracted from tile types
    static constexpr int accRows = TileTraits<AccTileT>::Rows;
    static constexpr int accCols = TileTraits<AccTileT>::Cols;
    static constexpr int vecRows = TileTraits<VecTileT>::Rows;
    static constexpr int vecCols = TileTraits<VecTileT>::Cols;
    
    // Infer dual-dst split axis from shape mismatch
    // If accRows == 2 * vecRows → cut M (row split to AIV0/AIV1)
    // If accCols == 2 * vecCols → cut N (col split to AIV0/AIV1)
    static constexpr bool dualDst = (accRows == 2 * vecRows) || (accCols == 2 * vecCols);
    static constexpr bool cutM = (accRows == 2 * vecRows);  // Row split
    static constexpr bool cutN = (accCols == 2 * vecCols);  // Col split
    
    // Slot size based on accumulator tile
    static constexpr int slotSize = accRows * accCols * sizeof(typename TileTraits<AccTileT>::DType);
};
```

### Initialization

```cpp
// Called once per kernel launch, before any TPUSH/TPOP
template <typename AccTileT, typename VecTileT, int Depth>
void FIFO_INIT(
    CrossCoreFIFO<AccTileT, VecTileT, Depth>& fifo,
    void*    bufferBase,      // Pre-allocated buffer in L1/shared memory
    uint16_t pushEventId,     // User-assigned event ID for push signal
    uint16_t popEventId       // User-assigned event ID for pop signal
);
// Note: slotSize is inferred from AccTileT shape and dtype
```

### Push API (Producer Side)

```cpp
// AIC → AIV: Push accumulator tile to vector core(s)
// aivId: 0 = AIV0 only, 1 = AIV1 only, -1 = dual-dst (both AIV0 and AIV1)
template <typename AccTileT, typename VecTileT>
void PTO_PUSH_TO_AIV(
    const AccTileT& accTile,             // Source tile (accumulator)
    CrossCoreFIFO<AccTileT, VecTileT>& fifo,
    int aivId = -1                       // -1 = dual-dst, 0 = AIV0, 1 = AIV1
);
// When aivId = -1 (dual-dst), split axis inferred from tile shapes:
//   AccTile[128,128] → VecTile[64,128]: cutM, rows split to AIV0/AIV1
//   AccTile[64,256] → VecTile[64,128]: cutN, cols split to AIV0/AIV1

// AIV → AIC: Push vector tile back to cube core (rare, for gradients)
template <typename VecTileT, typename AccTileT>
void PTO_PUSH_TO_AIC(
    const VecTileT& vecTile,             // Source tile (vector)
    CrossCoreFIFO<VecTileT, AccTileT>& fifo
);
```

### Pop API (Consumer Side)

```cpp
// AIV: Pop tile from AIC
// Each AIV core receives its portion based on dual-dst split
template <typename AccTileT, typename VecTileT>
void PTO_POP_FROM_AIC(
    VecTileT& vecTile,                   // Destination tile (vector format)
    CrossCoreFIFO<AccTileT, VecTileT>& fifo
);
// If dualDst && cutM: AIV0 gets rows[0:R/2], AIV1 gets rows[R/2:R]
// If dualDst && cutN: AIV0 gets cols[0:C/2], AIV1 gets cols[C/2:C]

// AIC: Pop tile from AIV (rare)
template <typename VecTileT, typename AccTileT>
void PTO_POP_FROM_AIV(
    AccTileT& accTile,                   // Destination tile (accumulator format)
    CrossCoreFIFO<VecTileT, AccTileT>& fifo
);
```

---

## Implementation Details

### What TPUSH Does Internally

```cpp
template <typename AccTileDtype, typename VecTileDtype, int CoreId>
void PTO_PUSH_TO_AIV(
    const Tile<AccTileDtype>& accTile,
    CrossCoreFIFO<AccTileDtype, VecTileDtype>& fifo
) {
    // 1. Wait for slot available (backpressure from consumer)
    if (fifo.head - fifo.tail >= Depth) {
        WAIT_EVENT(fifo.popEventId);  // Consumer freed a slot
    }
    
    // 2. Calculate slot address
    uint32_t slotIdx = fifo.head % Depth;
    void* slotAddr = (char*)fifo.bufferBase + slotIdx * fifo.slotSize;
    
    // 3. Copy tile to FIFO slot (with optional dtype conversion)
    COPY_TILE_TO_BUFFER<AccTileDtype, VecTileDtype>(accTile, slotAddr);
    
    // 4. Increment head
    fifo.head++;
    
    // 5. Signal consumer
    SET_CROSS_CORE_EVENT(fifo.pushEventId, CoreId);
}
```

### What TPOP Does Internally

```cpp
template <typename AccTileDtype, typename VecTileDtype, int CoreId>
void PTO_POP_FROM_AIC(
    Tile<VecTileDtype>& vecTile,
    CrossCoreFIFO<AccTileDtype, VecTileDtype>& fifo
) {
    // 1. Wait for data available
    if (fifo.tail >= fifo.head) {
        WAIT_EVENT(fifo.pushEventId);  // Producer pushed data
    }
    
    // 2. Calculate slot address
    uint32_t slotIdx = fifo.tail % Depth;
    void* slotAddr = (char*)fifo.bufferBase + slotIdx * fifo.slotSize;
    
    // 3. Copy from FIFO slot to tile
    COPY_BUFFER_TO_TILE<AccTileDtype, VecTileDtype>(slotAddr, vecTile);
    
    // 4. Increment tail
    fifo.tail++;
    
    // 5. Signal producer (slot freed)
    SET_CROSS_CORE_EVENT(fifo.popEventId, CoreId);
}
```

---

## Usage Example: Flash Attention QK→Softmax

```cpp
// ============ TILE DEFINITIONS (using existing PTO syntax) ============
// From FA kernel: TileAcc<Element, Rows, Cols, RowValid, ColValid>
constexpr uint32_t Cube_M = 128;
constexpr uint32_t Cube_N = 128;
constexpr uint32_t Vec_M = Cube_M / 2;  // Each AIV gets half the rows (64)

// Accumulator: 128x128 fp32 from matmul
using TileQKData = TileAcc<float, Cube_M, Cube_N>;

// Vector: 64x128 half per AIV core (dual-dst cutM)
// Shape mismatch: 128 rows → 64 rows = cutM (row split)
using TileDataH_T = Tile<TileType::Vec, half, Vec_M, Cube_N, BLayout::RowMajor, Vec_M, Cube_N>;

// FIFO automatically infers: dualDst=true, cutM=true
CrossCoreFIFO<TileQKData, TileDataH_T, 2> qkFIFO;

// ============ INIT (called once) ============
FIFO_INIT(qkFIFO,
    L1_BUFFER_ADDR,           // Pre-allocated L1 buffer
    EVENT_QK_PUSH,            // User-defined event ID
    EVENT_QK_POP              // User-defined event ID
);

// ============ AIC KERNEL (Cube Core) ============
void aic_matmul_kernel() {
    TileQKData qkAccTile;     // 128x128 accumulator
    TASSIGN(qkAccTile, 0x0);  // Assign to accumulator buffer
    
    for (int i = 0; i < num_tiles; i++) {
        // Compute Q @ K^T → 128x128 result
        pto_macro_matmul<Cube_M, HEAD, Cube_N>(qMatTile, kMatTile, qkAccTile, AccMode::InitFinalSum);
        
        // Push to AIV with dual-dst (-1)
        // Automatically splits rows[0:64]→AIV0, rows[64:128]→AIV1
        PTO_PUSH_TO_AIV(qkAccTile, qkFIFO, -1);  // -1 = dual-dst
    }
}

// ============ AIV KERNEL (Vector Core) ============
void aiv_softmax_kernel() {
    TileDataH_T qkVecTile;    // 64x128 half (this core's portion)
    TASSIGN(qkVecTile, UB_OFFSET);
    
    for (int i = 0; i < num_tiles; i++) {
        // Pop from AIC — AIV0 gets rows[0:64], AIV1 gets rows[64:128]
        PTO_POP_FROM_AIC(qkVecTile, qkFIFO);
        
        // Compute softmax on our portion
        pto_macro_fa_softmax<...>(x_expT, qkVecTile, ...);
        
        // Store or continue pipeline...
    }
}
```

---

## Cross-Core Event Management

### Option A: Implicit (Handled by TPUSH/TPOP)

Events are set automatically inside `PTO_PUSH_*` / `PTO_POP_*` calls.

```cpp
// Inside PTO_PUSH_TO_AIV:
SET_CROSS_CORE_EVENT(fifo.pushEventId, CoreId);

// Inside PTO_POP_FROM_AIC:
SET_CROSS_CORE_EVENT(fifo.popEventId, CoreId);
```

### Option B: Explicit (User Controls Sync)

Separate APIs for fine-grained control:

```cpp
// Reserve dependency before push
void FIFO_WAIT_SLOT(CrossCoreFIFO& fifo);

// Copy without signaling
void FIFO_COPY_TO_SLOT(const Tile& src, CrossCoreFIFO& fifo);

// Explicit signal
void FIFO_SIGNAL_PUSH(CrossCoreFIFO& fifo, int coreId);

// Usage:
FIFO_WAIT_SLOT(qkFIFO);
FIFO_COPY_TO_SLOT(qkTile, qkFIFO);
// ... other work ...
FIFO_SIGNAL_PUSH(qkFIFO, 0);  // Deferred signal
```

---

## Questions for Lok

1. **Event ID allocation**: Should we provide a helper like `ALLOC_EVENT_PAIR()` or leave it fully manual?

2. **Dtype conversion**: Where does fp32→half conversion happen?
   - Option A: Inside TPUSH (on AIC side, before write to FIFO)
   - Option B: Inside TPOP (on AIV side, after read from FIFO)
   - Option C: User handles explicitly before/after

3. **Buffer ownership**: Who allocates the FIFO buffer?
   - Option A: User passes pre-allocated buffer to `FIFO_INIT`
   - Option B: FIFO allocates from a pool (`L1_ALLOC(size)`)

4. **Depth template param**: Fixed at compile time (current), or runtime configurable?

5. **Same-shape tiles**: When AccTile and VecTile have same ROWS/COLS (no dual-dst), should we require explicit `CoreId` template param, or use some other mechanism?

---

## Comparison: Old vs New

| Aspect | Old (Subtile-based) | New (FIFO-based) |
|--------|---------------------|------------------|
| Buffer management | `pro_set_allocate` per subtile | Single FIFO init |
| Sync | Manual `set_cross_core_event` | Implicit in push/pop |
| Dtype | Implicit | Explicit template params |
| Ring buffer | Manual index tracking | FIFO object encapsulates |
| Complexity | High (subtile loops) | Low (tile-level ops) |

---

## Next Steps

1. Finalize API based on feedback
2. Implement FIFO struct + init
3. Implement PTO_PUSH_TO_AIV / PTO_POP_FROM_AIC
4. Add CPU sim support
5. Add NPU codegen in compiler

---

## TMATMUL + TADD Fusion Example with TPUSH/TPOP

This example shows a simple fused kernel where:
- **AIC (Cube)**: Computes MATMUL → pushes result to AIV
- **AIV (Vector)**: Pops result → adds bias → stores to GM

### Sync Pattern

```
AIC (Cube)                          AIV (Vector)
-----------                         ------------
                                    
[TMATMUL] → accTile                 
    |                               
[TSTORE to FIFO]                    
    |                               
sync.record() ─────────────────────→ sync.wait()
                                        |
                                    [TLOAD from FIFO]
                                        |
                                    [TADD bias]
                                        |
                                    [TSTORE to GM]
                                        |
sync.allocate() ←─────────────────── sync.free()
    |
[next iteration...]
```

### Cross-Core Sync Methods (TSync_Custom)

| Method | Called By | Purpose | Underlying Op |
|--------|-----------|---------|---------------|
| `record()` | Producer (AIC) | Signal data ready | `set_intra_block(PIPE_FIX, flag_id)` |
| `wait()` | Consumer (AIV) | Wait for data | `wait_intra_block(PIPE_MTE2, flag_id)` |
| `allocate()` | Producer (AIC) | Wait for buffer free | `wait_intra_block(PIPE_FIX, flag_id+1)` |
| `free()` | Consumer (AIV) | Signal buffer consumed | `set_intra_block(PIPE_MTE2, flag_id+1)` |

### Complete Code Example

```cpp
#include <pto/pto-inst.hpp>
#include <pto/npu/a5/custom/TSync_Custom.hpp>

using namespace pto;

// ============ CONSTANTS ============
constexpr uint32_t M = 128;         // Output rows (Cube)
constexpr uint32_t K = 128;         // Inner dimension
constexpr uint32_t N = 128;         // Output cols
constexpr uint32_t Vec_M = M / 2;   // Each AIV gets half rows = 64 (dual-dst cutM)
constexpr int FIFO_DEPTH = 2;       // Double buffer

// ============ TILE DEFINITIONS ============
// Cube input tiles (L1)
using TileMatA = Tile<TileType::Mat, half, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
using TileMatB = Tile<TileType::Mat, half, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>;

// Cube accumulator output (on-chip CO buffer) - 128x128 fp32
using TileAccOut = TileAcc<float, M, N>;  // 128x128 fp32 = 64KB

// Vector tiles (UB) - each AIV gets 64x128 (half the rows)
using TileVecIn = Tile<TileType::Vec, float, Vec_M, N, BLayout::RowMajor, Vec_M, N>;   // 64x128 fp32 = 32KB
using TileBias  = Tile<TileType::Vec, float, Vec_M, N, BLayout::RowMajor, Vec_M, N>;   // 64x128 fp32
using TileVecOut = Tile<TileType::Vec, float, Vec_M, N, BLayout::RowMajor, Vec_M, N>;  // 64x128 fp32

// ============ SYNC FLAG IDs ============
enum SyncFlagID : uint16_t {
    MATMUL_TO_ADD_FLAG = 0,  // AIC→AIV data ready
    // Flag+1 is automatically used for backpressure (allocate/free)
};

// ============ SYNC OBJECT ============
// TSync_Custom: producer op = TSTORE_C2GM (Cube), consumer op = TLOAD
constexpr TSync_Custom<SyncOpType::TSTORE_C2GM, SyncOpType::TLOAD> matmul2addSync = {MATMUL_TO_ADD_FLAG};

// ============ AIC KERNEL (Cube Core) ============
__attribute__((aic))
void aic_matmul_kernel(__gm__ half* A, __gm__ half* B, __gm__ float* fifo_buffer, int num_tiles) {
    // Tile declarations
    TileMatA aTile;
    TileMatB bTile;
    TileAccOut accTile;
    
    // Assign buffers (L1 for mat tiles, CO for acc)
    uint32_t l1_offset = 0;
    TASSIGN(aTile, l1_offset); l1_offset += M * K * sizeof(half);
    TASSIGN(bTile, l1_offset);
    TASSIGN(accTile, 0x0);  // Accumulator at CO base
    
    // Init pipe flags
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    
    for (int tile_id = 0; tile_id < num_tiles; tile_id++) {
        // === BACKPRESSURE: Wait for buffer slot ===
        // Skip for first FIFO_DEPTH tiles (buffer not full yet)
        if (tile_id >= FIFO_DEPTH) {
            matmul2addSync.allocate();  // wait_intra_block(PIPE_FIX, flag_id+1)
        }
        
        // === LOAD A, B tiles ===
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        
        using GlobalA = GlobalTensor<half, Shape<1,1,1,M,K>, Stride<1,1,1,K,1>>;
        using GlobalB = GlobalTensor<half, Shape<1,1,1,K,N>, Stride<1,1,1,1,K>, Layout::DN>;
        GlobalA aGlobal(A + tile_id * M * K);
        GlobalB bGlobal(B);  // B is shared
        
        TLOAD(aTile, aGlobal);
        TLOAD(bTile, bGlobal);
        
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        
        // === MATMUL ===
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        TMATMUL(accTile, aTile, bTile, AccMode::InitFinalSum);
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        
        // === STORE to FIFO ===
        using GlobalFIFO = GlobalTensor<float, Shape<1,1,1,M,N>, Stride<1,1,1,N,1>>;
        uint32_t slot_idx = tile_id % FIFO_DEPTH;
        __gm__ float* slot_addr = fifo_buffer + slot_idx * M * N;
        GlobalFIFO fifoSlot(slot_addr);
        
        TSTORE(fifoSlot, accTile);
        
        // === SIGNAL: Data ready for AIV ===
        // aivId = -1 means dual-dst: signal both AIV0 and AIV1
        matmul2addSync.record();  // set_intra_block(PIPE_FIX, flag_id) to both AIV0/AIV1
        
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    }
    
    // Drain pending allocate waits
    for (int i = 0; i < (num_tiles < FIFO_DEPTH ? num_tiles : FIFO_DEPTH); i++) {
        matmul2addSync.allocate();
    }
    
    pipe_barrier(PIPE_ALL);
}

// ============ AIV KERNEL (Vector Core) ============
__attribute__((aiv))
void aiv_add_kernel(__gm__ float* fifo_buffer, __gm__ float* bias, __gm__ float* output, int num_tiles) {
    // Tile declarations - each AIV processes 64x128
    TileVecIn inTile;
    TileBias biasTile;
    TileVecOut outTile;
    
    // Assign UB buffers
    uint32_t ub_offset = 0;
    TASSIGN(inTile, ub_offset);  ub_offset += Vec_M * N * sizeof(float);
    TASSIGN(biasTile, ub_offset); ub_offset += Vec_M * N * sizeof(float);
    TASSIGN(outTile, ub_offset);
    
    // Init pipe flags
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    
    // Load bias once (shared across tiles)
    // Each AIV core loads its portion based on get_subblockid()
    uint32_t subblock_id = get_subblockid();  // 0 = AIV0, 1 = AIV1
    using GlobalBias = GlobalTensor<float, Shape<1,1,1,Vec_M,N>, Stride<1,1,1,N,1>>;
    GlobalBias biasGlobal(bias + subblock_id * Vec_M * N);
    TLOAD(biasTile, biasGlobal);
    
    for (int tile_id = 0; tile_id < num_tiles; tile_id++) {
        // === WAIT: Data ready from AIC ===
        matmul2addSync.wait();  // wait_intra_block(PIPE_MTE2, flag_id)
        
        // === LOAD from FIFO (each AIV gets its half) ===
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        
        uint32_t slot_idx = tile_id % FIFO_DEPTH;  // Consumer tracks its own tail
        // Dual-dst cutM: AIV0 gets rows[0:64], AIV1 gets rows[64:128]
        __gm__ float* slot_addr = fifo_buffer + slot_idx * M * N + subblock_id * Vec_M * N;
        
        using GlobalFIFOSlice = GlobalTensor<float, Shape<1,1,1,Vec_M,N>, Stride<1,1,1,N,1>>;
        GlobalFIFOSlice fifoSlice(slot_addr);
        TLOAD(inTile, fifoSlice);
        
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        
        // === TADD: inTile + biasTile → outTile ===
        TADD(outTile, inTile, biasTile);
        
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        
        // === SIGNAL: Buffer consumed, AIC can reuse ===
        matmul2addSync.free();  // set_intra_block(PIPE_MTE2, flag_id+1)
        
        // === STORE to GM ===
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        
        using GlobalOut = GlobalTensor<float, Shape<1,1,1,Vec_M,N>, Stride<1,1,1,N,1>>;
        GlobalOut outGlobal(output + tile_id * M * N + subblock_id * Vec_M * N);
        TSTORE(outGlobal, outTile);
    }
    
    pipe_barrier(PIPE_ALL);
}
```

### Key Sync Points Summary

| Location | Call | Direction | Purpose |
|----------|------|-----------|---------|
| AIC: before TMATMUL (tile_id >= FIFO_DEPTH) | `allocate()` | AIC ← AIV | Wait for buffer space |
| AIC: after TSTORE to FIFO | `record()` | AIC → AIV | Signal data ready |
| AIV: before TLOAD from FIFO | `wait()` | AIC → AIV | Wait for data |
| AIV: after using data | `free()` | AIV → AIC | Signal buffer consumed |

### Dual-Dst Row Split (cutM)

With `TileAccOut[128,128]` → `TileVecIn[64,128]`:
- FIFO::dualDst = true, cutM = true
- AIC writes full 128×128 tile to FIFO slot (64KB)
- `record()` signals BOTH AIV0 and AIV1 (via `flag_id` and `flag_id+16`)
- AIV0 reads rows[0:64], AIV1 reads rows[64:128] using `get_subblockid()` offset
- Each AIV processes 64×128 = 32KB per tile
