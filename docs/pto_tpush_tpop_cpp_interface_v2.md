# PTO TPUSH/TPOP C++ Interface v2

## Design Goals

1. **Kill subtile complexity** — No `pro_set_allocate`, no subtile tiling in API
2. **Explicit dtype conversion** — Template params for source and destination types
3. **FIFO object encapsulates state** — Ring buffer counter, memory address, position
4. **User-controlled event IDs** — Cross-core events setup during init
5. **TPUSH handles sync internally** — Or provide explicit API for reserve/dependency

---

## Core API

### FIFO Object

```cpp
template <typename SrcDtype, typename DstDtype, int Depth = 2>
struct CrossCoreFIFO {
    // Ring buffer state
    uint32_t head;           // Producer position
    uint32_t tail;           // Consumer position
    
    // Memory layout
    void*    bufferBase;     // Base address of FIFO buffer
    uint32_t slotSize;       // Size of each slot in bytes
    
    // Cross-core sync
    uint16_t pushEventId;    // Event ID for producer → consumer signal
    uint16_t popEventId;     // Event ID for consumer → producer signal (backpressure)
    
    // Direction info (compile-time)
    static constexpr bool isAicToAiv = std::is_same_v<SrcDtype, AccDtype>;
};
```

### Initialization

```cpp
// Called once per kernel launch, before any TPUSH/TPOP
template <typename SrcDtype, typename DstDtype, int Depth>
void FIFO_INIT(
    CrossCoreFIFO<SrcDtype, DstDtype, Depth>& fifo,
    void*    bufferBase,      // Pre-allocated buffer in L1/shared memory
    uint32_t slotSize,        // Tile size in bytes
    uint16_t pushEventId,     // User-assigned event ID for push signal
    uint16_t popEventId       // User-assigned event ID for pop signal
);
```

### Push API (Producer Side)

```cpp
// AIC → AIV: Push accumulator tile to vector core
template <typename AccTileDtype, typename VecTileDtype, int CoreId = 0>
void PTO_PUSH_TO_AIV(
    const Tile<AccTileDtype>& accTile,   // Source tile (accumulator)
    CrossCoreFIFO<AccTileDtype, VecTileDtype>& fifo
);

// AIV → AIC: Push vector tile back to cube core (rare, for gradients)
template <typename VecTileDtype, typename AccTileDtype, int CoreId = 0>
void PTO_PUSH_TO_AIC(
    const Tile<VecTileDtype>& vecTile,   // Source tile (vector)
    CrossCoreFIFO<VecTileDtype, AccTileDtype>& fifo
);
```

### Pop API (Consumer Side)

```cpp
// AIV: Pop tile from AIC
template <typename AccTileDtype, typename VecTileDtype, int CoreId = 0>
void PTO_POP_FROM_AIC(
    Tile<VecTileDtype>& vecTile,         // Destination tile (vector format)
    CrossCoreFIFO<AccTileDtype, VecTileDtype>& fifo
);

// AIC: Pop tile from AIV (rare)
template <typename VecTileDtype, typename AccTileDtype, int CoreId = 0>
void PTO_POP_FROM_AIV(
    Tile<AccTileDtype>& accTile,         // Destination tile (accumulator format)
    CrossCoreFIFO<VecTileDtype, AccTileDtype>& fifo
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
// ============ INIT (called once) ============
CrossCoreFIFO<fp32, half, 2> qkFIFO;   // 2-slot double buffer

// User allocates buffer and assigns event IDs
FIFO_INIT(qkFIFO,
    L1_BUFFER_ADDR,           // Pre-allocated L1 buffer
    TILE_SIZE_BYTES,          // e.g., 256 * sizeof(half)
    EVENT_QK_PUSH,            // User-defined event ID
    EVENT_QK_POP              // User-defined event ID
);

// ============ AIC KERNEL (Cube Core) ============
void aic_matmul_kernel() {
    Tile<fp32> qkTile;        // Accumulator tile
    
    for (int i = 0; i < num_tiles; i++) {
        // Compute Q @ K^T
        MATMUL(Q_tile, K_tile, qkTile);
        
        // Push to AIV for softmax
        PTO_PUSH_TO_AIV<fp32, half, 0>(qkTile, qkFIFO);
    }
}

// ============ AIV KERNEL (Vector Core) ============
void aiv_softmax_kernel() {
    Tile<half> qkTile;        // Vector tile
    
    for (int i = 0; i < num_tiles; i++) {
        // Pop from AIC
        PTO_POP_FROM_AIC<fp32, half, 0>(qkTile, qkFIFO);
        
        // Compute softmax
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
   - Option A: Inside TPUSH (on AIC side)
   - Option B: Inside TPOP (on AIV side)
   - Option C: User handles explicitly before/after

3. **A5 dual-AIV**: For 1:2 split to both vector cores, do we need:
   ```cpp
   PTO_PUSH_TO_AIV_DUAL<fp32, half>(accTile, fifo0, fifo1);  // Splits rows
   ```
   Or just two separate pushes with different FIFOs?

4. **Buffer ownership**: Who allocates the FIFO buffer?
   - Option A: User passes pre-allocated buffer to `FIFO_INIT`
   - Option B: FIFO allocates from a pool (`L1_ALLOC(size)`)

5. **Depth template param**: Fixed at compile time, or runtime configurable?

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
