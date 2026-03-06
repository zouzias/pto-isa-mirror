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
// Accumulator tile from Cube core (source for TPUSH)
// TileAcc<dtype, ROWS, COLS, BlockRows, BlockCols>
using AccTile_32x16_fp32 = TileAcc<float, 32, 16, 32, 16>;  // 2KB
using AccTile_16x16_fp32 = TileAcc<float, 16, 16, 16, 16>;  // 1KB

// Vector tile for Vec core (destination for TPOP)
// Tile<TileType::Vec, dtype, ROWS, COLS, BLayout, BRows, BCols>
using VecTile_16x16_half = Tile<TileType::Vec, half, 16, 16, BLayout::RowMajor, 16, 16>;  // 512B
using VecTile_32x16_half = Tile<TileType::Vec, half, 32, 16, BLayout::RowMajor, 32, 16>;  // 1KB

// Extract ROWS/COLS from tile type
template <typename TileT>
struct TileTraits {
    static constexpr int Rows = TileT::Rows;  // or use template param position
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
// If FIFO::dualDst is true, automatically splits to both AIV cores
template <typename AccTileT, typename VecTileT>
void PTO_PUSH_TO_AIV(
    const AccTileT& accTile,             // Source tile (accumulator)
    CrossCoreFIFO<AccTileT, VecTileT>& fifo
);
// Dual-dst split axis inferred from tile shapes:
//   AccTile[32,16] → VecTile[16,16]: cutM, rows split to AIV0/AIV1
//   AccTile[16,32] → VecTile[16,16]: cutN, cols split to AIV0/AIV1

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
// Accumulator: 32x16 fp32 (2KB) from matmul
using AccTile = TileAcc<float, 32, 16, 32, 16>;

// Vector: 16x16 half (512B) per AIV core
// Shape mismatch: 32 rows → 16 rows = cutM (row split)
using VecTile = Tile<TileType::Vec, half, 16, 16, BLayout::RowMajor, 16, 16>;

// FIFO automatically infers: dualDst=true, cutM=true
CrossCoreFIFO<AccTile, VecTile, 2> qkFIFO;

// ============ INIT (called once) ============
FIFO_INIT(qkFIFO,
    L1_BUFFER_ADDR,           // Pre-allocated L1 buffer
    EVENT_QK_PUSH,            // User-defined event ID
    EVENT_QK_POP              // User-defined event ID
);

// ============ AIC KERNEL (Cube Core) ============
void aic_matmul_kernel() {
    AccTile qkAccTile;        // 32x16 accumulator
    TASSIGN(qkAccTile, 0x0);  // Assign to accumulator buffer
    
    for (int i = 0; i < num_tiles; i++) {
        // Compute Q @ K^T → 32x16 result
        pto_macro_matmul<32, HEAD, 16>(qMatTile, kMatTile, qkAccTile, AccMode::InitFinalSum);
        
        // Push to AIV — automatically splits rows[0:16]→AIV0, rows[16:32]→AIV1
        PTO_PUSH_TO_AIV(qkAccTile, qkFIFO);
    }
}

// ============ AIV KERNEL (Vector Core) ============
void aiv_softmax_kernel() {
    VecTile qkVecTile;        // 16x16 half (this core's portion)
    TASSIGN(qkVecTile, UB_OFFSET);
    
    for (int i = 0; i < num_tiles; i++) {
        // Pop from AIC — AIV0 gets rows[0:16], AIV1 gets rows[16:32]
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
