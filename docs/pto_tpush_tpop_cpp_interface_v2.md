# PTO TPUSH/TPOP C++ Interface v2

## Design Goals

1. **Kill subtile complexity** — No `pro_set_allocate`, no subtile tiling in API
2. **Explicit dtype conversion** — Template params for source and destination types
3. **FIFO object encapsulates state** — Ring buffer counter, memory address, position
4. **User-controlled event IDs** — Cross-core events setup during init
5. **TPUSH handles sync internally** — Or provide explicit API for reserve/dependency

---

## Core API

### Tile Type Definition

Tile types carry shape info used to infer dual-dst split axis:

```cpp
// Tile type with compile-time shape info
template <typename Dtype, int ROWS, int COLS>
struct TileDef {
    using dtype = Dtype;
    static constexpr int rows = ROWS;
    static constexpr int cols = COLS;
    static constexpr int size = ROWS * COLS * sizeof(Dtype);
};

// Common tile definitions
using AccTile_16x16_fp32 = TileDef<fp32, 16, 16>;   // 1KB
using AccTile_32x16_fp32 = TileDef<fp32, 32, 16>;   // 2KB
using VecTile_16x16_half = TileDef<half, 16, 16>;   // 512B
using VecTile_32x16_half = TileDef<half, 32, 16>;   // 1KB
using VecTile_16x32_half = TileDef<half, 16, 32>;   // 1KB
```

### FIFO Object

```cpp
template <typename AccTileDef, typename VecTileDef, int Depth = 2>
struct CrossCoreFIFO {
    // Ring buffer state
    uint32_t head;           // Producer position
    uint32_t tail;           // Consumer position
    
    // Memory layout
    void*    bufferBase;     // Base address of FIFO buffer
    
    // Cross-core sync
    uint16_t pushEventId;    // Event ID for producer → consumer signal
    uint16_t popEventId;     // Event ID for consumer → producer signal (backpressure)
    
    // Compile-time shape info from tile types
    static constexpr int accRows = AccTileDef::rows;
    static constexpr int accCols = AccTileDef::cols;
    static constexpr int vecRows = VecTileDef::rows;
    static constexpr int vecCols = VecTileDef::cols;
    
    // Infer dual-dst split axis from shape mismatch
    // If accRows == 2 * vecRows → cut M (row split)
    // If accCols == 2 * vecCols → cut N (col split)
    static constexpr bool dualDst = (accRows == 2 * vecRows) || (accCols == 2 * vecCols);
    static constexpr bool cutM = (accRows == 2 * vecRows);  // Row split to AIV0/AIV1
    static constexpr bool cutN = (accCols == 2 * vecCols);  // Col split to AIV0/AIV1
    
    // Slot size is the larger of acc/vec tile
    static constexpr int slotSize = AccTileDef::size;
};
```

### Initialization

```cpp
// Called once per kernel launch, before any TPUSH/TPOP
template <typename AccTileDef, typename VecTileDef, int Depth>
void FIFO_INIT(
    CrossCoreFIFO<AccTileDef, VecTileDef, Depth>& fifo,
    void*    bufferBase,      // Pre-allocated buffer in L1/shared memory
    uint16_t pushEventId,     // User-assigned event ID for push signal
    uint16_t popEventId       // User-assigned event ID for pop signal
);
// Note: slotSize is inferred from AccTileDef::size
```

### Push API (Producer Side)

```cpp
// AIC → AIV: Push accumulator tile to vector core(s)
// If FIFO::dualDst is true, automatically splits to both AIV cores
template <typename AccTileDef, typename VecTileDef>
void PTO_PUSH_TO_AIV(
    const Tile<AccTileDef>& accTile,     // Source tile (accumulator)
    CrossCoreFIFO<AccTileDef, VecTileDef>& fifo
);
// Dual-dst split axis inferred from tile shapes:
//   AccTile[32,16] → VecTile[16,16]: cutM, rows split to AIV0/AIV1
//   AccTile[16,32] → VecTile[16,16]: cutN, cols split to AIV0/AIV1

// AIV → AIC: Push vector tile back to cube core (rare, for gradients)
template <typename VecTileDef, typename AccTileDef>
void PTO_PUSH_TO_AIC(
    const Tile<VecTileDef>& vecTile,     // Source tile (vector)
    CrossCoreFIFO<VecTileDef, AccTileDef>& fifo
);
```

### Pop API (Consumer Side)

```cpp
// AIV: Pop tile from AIC
// Each AIV core receives its portion based on dual-dst split
template <typename AccTileDef, typename VecTileDef>
void PTO_POP_FROM_AIC(
    Tile<VecTileDef>& vecTile,           // Destination tile (vector format)
    CrossCoreFIFO<AccTileDef, VecTileDef>& fifo
);
// If dualDst && cutM: AIV0 gets rows[0:R/2], AIV1 gets rows[R/2:R]
// If dualDst && cutN: AIV0 gets cols[0:C/2], AIV1 gets cols[C/2:C]

// AIC: Pop tile from AIV (rare)
template <typename VecTileDef, typename AccTileDef>
void PTO_POP_FROM_AIV(
    Tile<AccTileDef>& accTile,           // Destination tile (accumulator format)
    CrossCoreFIFO<VecTileDef, AccTileDef>& fifo
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
// ============ TILE DEFINITIONS ============
// Accumulator: 32x16 fp32 (2KB) from matmul
using AccTile = TileDef<fp32, 32, 16>;

// Vector: 16x16 half (512B) per AIV core
// Shape mismatch: 32 rows → 16 rows = cutM (row split)
using VecTile = TileDef<half, 16, 16>;

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
    Tile<AccTile> qkTile;     // 32x16 accumulator
    
    for (int i = 0; i < num_tiles; i++) {
        // Compute Q @ K^T → 32x16 result
        MATMUL(Q_tile, K_tile, qkTile);
        
        // Push to AIV — automatically splits rows[0:16]→AIV0, rows[16:32]→AIV1
        PTO_PUSH_TO_AIV(qkTile, qkFIFO);
    }
}

// ============ AIV KERNEL (Vector Core) ============
void aiv_softmax_kernel() {
    Tile<VecTile> qkTile;     // 16x16 half (this core's portion)
    
    for (int i = 0; i < num_tiles; i++) {
        // Pop from AIC — AIV0 gets rows[0:16], AIV1 gets rows[16:32]
        PTO_POP_FROM_AIC(qkTile, qkFIFO);
        
        // Compute softmax on our portion
        SOFTMAX(qkTile);
        
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
