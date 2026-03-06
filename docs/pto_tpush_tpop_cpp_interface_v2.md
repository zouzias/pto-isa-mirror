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
// ============================================================
// C2V FIFO: AIC → AIV (Cube to Vector)
// ============================================================

// === DUAL-DST MODE: Both AIV0 and AIV1 receive data (cutM/cutN) ===
template <typename AccTileT, typename VecTileT, int Depth = 2, int NumVecBuffers = Depth>
struct CrossCoreFIFO {
    // Ring buffer state
    uint32_t head;           // Producer position
    uint32_t tail;           // Consumer position
    
    // Memory layout - L1 buffer for cross-core transfer
    void*    l1BufferBase;   // Base address of FIFO buffer in L1
    
    // UB addresses for each AIV (dual-dst: different addrs per AIV)
    uint32_t ubAddrsAIV0[NumVecBuffers];  // VecTile addresses in AIV0's UB
    uint32_t ubAddrsAIV1[NumVecBuffers];  // VecTile addresses in AIV1's UB
    
    // Cross-core sync
    uint16_t pushEventId;
    uint16_t popEventId;
    
    // Compile-time shape info
    static constexpr int accRows = TileTraits<AccTileT>::Rows;
    static constexpr int accCols = TileTraits<AccTileT>::Cols;
    static constexpr int vecRows = TileTraits<VecTileT>::Rows;
    static constexpr int vecCols = TileTraits<VecTileT>::Cols;
    
    static constexpr bool dualDst = (accRows == 2 * vecRows) || (accCols == 2 * vecCols);
    static constexpr bool cutM = (accRows == 2 * vecRows);
    static constexpr bool cutN = (accCols == 2 * vecCols);
    
    static constexpr int slotSize = accRows * accCols * sizeof(typename TileTraits<AccTileT>::DType);
    static constexpr int vecTileSize = vecRows * vecCols * sizeof(typename TileTraits<VecTileT>::DType);
    
    uint32_t getCurrentUBAddr(int aivId) const {
        uint32_t bufIdx = head % NumVecBuffers;
        return (aivId == 0) ? ubAddrsAIV0[bufIdx] : ubAddrsAIV1[bufIdx];
    }
};

// === SIMD MODE: Same data to both AIVs (broadcast, no split) ===
// Use when AccTile and VecTile have same shape, or explicit broadcast
template <typename AccTileT, typename VecTileT, int Depth = 2, int NumVecBuffers = Depth>
struct CrossCoreFIFO_SIMD {
    uint32_t head;
    uint32_t tail;
    
    void*    l1BufferBase;
    
    // Single UB address array (same for both AIV0 and AIV1)
    uint32_t ubAddrsAIV[NumVecBuffers];   // Shared: both AIVs use same UB offset
    
    uint16_t pushEventId;
    uint16_t popEventId;
    
    static constexpr int accRows = TileTraits<AccTileT>::Rows;
    static constexpr int accCols = TileTraits<AccTileT>::Cols;
    static constexpr int vecRows = TileTraits<VecTileT>::Rows;
    static constexpr int vecCols = TileTraits<VecTileT>::Cols;
    
    // SIMD mode: same shape, broadcast to both
    static constexpr bool sameShape = (accRows == vecRows) && (accCols == vecCols);
    
    static constexpr int slotSize = accRows * accCols * sizeof(typename TileTraits<AccTileT>::DType);
    static constexpr int vecTileSize = vecRows * vecCols * sizeof(typename TileTraits<VecTileT>::DType);
    
    uint32_t getCurrentUBAddr() const {
        return ubAddrsAIV[head % NumVecBuffers];
    }
};
```

### Initialization

```cpp
// === DUAL-DST MODE: Separate UB addresses for AIV0/AIV1 ===
template <typename AccTileT, typename VecTileT, int Depth, int NumVecBuffers>
void FIFO_INIT(
    CrossCoreFIFO<AccTileT, VecTileT, Depth, NumVecBuffers>& fifo,
    void*    l1BufferBase,
    const uint32_t (&ubAddrsAIV0)[NumVecBuffers],
    const uint32_t (&ubAddrsAIV1)[NumVecBuffers],
    uint16_t pushEventId,
    uint16_t popEventId
) {
    fifo.head = 0;
    fifo.tail = 0;
    fifo.l1BufferBase = l1BufferBase;
    for (int i = 0; i < NumVecBuffers; i++) {
        fifo.ubAddrsAIV0[i] = ubAddrsAIV0[i];
        fifo.ubAddrsAIV1[i] = ubAddrsAIV1[i];
    }
    fifo.pushEventId = pushEventId;
    fifo.popEventId = popEventId;
}

// === SIMD MODE: Single UB address array (same for both AIVs) ===
template <typename AccTileT, typename VecTileT, int Depth, int NumVecBuffers>
void FIFO_INIT(
    CrossCoreFIFO_SIMD<AccTileT, VecTileT, Depth, NumVecBuffers>& fifo,
    void*    l1BufferBase,
    const uint32_t (&ubAddrsAIV)[NumVecBuffers],  // Same UB offset for both AIVs
    uint16_t pushEventId,
    uint16_t popEventId
) {
    fifo.head = 0;
    fifo.tail = 0;
    fifo.l1BufferBase = l1BufferBase;
    for (int i = 0; i < NumVecBuffers; i++) {
        fifo.ubAddrsAIV[i] = ubAddrsAIV[i];
    }
    fifo.pushEventId = pushEventId;
    fifo.popEventId = popEventId;
}

// === Simplified single-buffer init ===
template <typename AccTileT, typename VecTileT, int Depth>
void FIFO_INIT(
    CrossCoreFIFO<AccTileT, VecTileT, Depth, 1>& fifo,
    void*    l1BufferBase,
    uint32_t ubAddrAIV0,
    uint32_t ubAddrAIV1,
    uint16_t pushEventId,
    uint16_t popEventId
) {
    fifo.head = 0;
    fifo.tail = 0;
    fifo.l1BufferBase = l1BufferBase;
    fifo.ubAddrsAIV0[0] = ubAddrAIV0;
    fifo.ubAddrsAIV1[0] = ubAddrAIV1;
    fifo.pushEventId = pushEventId;
    fifo.popEventId = popEventId;
}
```

### Push API (Producer Side)

```cpp
// === STATIC SCHEDULE: FIFO manages buffer rotation automatically ===
template <typename AccTileT, typename VecTileT, int Depth, int NumVecBuffers>
void PTO_PUSH_TO_AIV(
    const AccTileT& accTile,
    CrossCoreFIFO<AccTileT, VecTileT, Depth, NumVecBuffers>& fifo,
    int aivId = -1
) {
    // Uses fifo.getCurrentUBAddr(aivId) to get destination based on head % NumVecBuffers
    // Automatic ping-pong buffer rotation
    // ...
}

// === DYNAMIC SCHEDULE: User provides current VecTile (for complex scheduling) ===
template <typename AccTileT, typename VecTileT>
void PTO_PUSH_TO_AIV(
    const AccTileT& accTile,
    const VecTileT& dstVecTileAIV0,    // Current VecTile for AIV0 (has UB addr)
    const VecTileT& dstVecTileAIV1,    // Current VecTile for AIV1 (has UB addr)
    uint16_t pushEventId,
    int aivId = -1
) {
    // User controls which VecTile buffer to use
    uint32_t ubAddrAIV0 = dstVecTileAIV0.getAddress();
    uint32_t ubAddrAIV1 = dstVecTileAIV1.getAddress();
    // ...
}

// === DYNAMIC SCHEDULE (single AIV): User provides single VecTile ===
template <typename AccTileT, typename VecTileT>
void PTO_PUSH_TO_AIV(
    const AccTileT& accTile,
    const VecTileT& dstVecTile,        // Target VecTile (has UB addr)
    uint16_t pushEventId,
    int aivId                          // Must be 0 or 1 (not -1)
) {
    assert(aivId == 0 || aivId == 1);
    uint32_t ubAddr = dstVecTile.getAddress();
    // ...
}
```

### Push to AIC (AIV → Cube) API

```cpp
// ============================================================
// AIV → AIC: Push vector tile back to cube core (for gradients, P*V results)
// ============================================================

// === FIFO for V2C direction (Vector → Cube) ===
template <typename VecTileT, typename MatTileT, int Depth = 2, int NumMatBuffers = Depth>
struct CrossCoreFIFO_V2C {
    uint32_t head;
    uint32_t tail;
    
    // L1 addresses for MatTile buffers (ping-pong/N-buffer)
    uint32_t l1AddrsMatTile[NumMatBuffers];
    
    // UB addresses of source VecTiles (dual-dst: separate per AIV)
    uint32_t ubAddrsAIV0[NumMatBuffers];
    uint32_t ubAddrsAIV1[NumMatBuffers];
    
    uint16_t pushEventId;
    uint16_t popEventId;
    
    // Shape info
    static constexpr int vecRows = TileTraits<VecTileT>::Rows;
    static constexpr int vecCols = TileTraits<VecTileT>::Cols;
    static constexpr int matRows = TileTraits<MatTileT>::Rows;
    static constexpr int matCols = TileTraits<MatTileT>::Cols;
    
    static constexpr int vecTileSize = vecRows * vecCols * sizeof(typename TileTraits<VecTileT>::DType);
    
    // Helper: get current L1 address for MatTile
    uint32_t getCurrentL1Addr() const {
        return l1AddrsMatTile[head % NumMatBuffers];
    }
};

// === SIMD V2C: Same UB offset for both AIVs ===
template <typename VecTileT, typename MatTileT, int Depth = 2, int NumMatBuffers = Depth>
struct CrossCoreFIFO_V2C_SIMD {
    uint32_t head;
    uint32_t tail;
    
    uint32_t l1AddrsMatTile[NumMatBuffers];
    uint32_t ubAddrsAIV[NumMatBuffers];    // Same UB offset for both AIVs
    
    uint16_t pushEventId;
    uint16_t popEventId;
    
    static constexpr int vecRows = TileTraits<VecTileT>::Rows;
    static constexpr int vecCols = TileTraits<VecTileT>::Cols;
    static constexpr int matRows = TileTraits<MatTileT>::Rows;
    static constexpr int matCols = TileTraits<MatTileT>::Cols;
    
    static constexpr int vecTileSize = vecRows * vecCols * sizeof(typename TileTraits<VecTileT>::DType);
    
    uint32_t getCurrentL1Addr() const {
        return l1AddrsMatTile[head % NumMatBuffers];
    }
    
    uint32_t getCurrentUBAddr() const {
        return ubAddrsAIV[head % NumMatBuffers];
    }
};

// === STATIC SCHEDULE: Array of MatTile L1 addresses (ping-pong) ===
template <typename VecTileT, typename MatTileT, int Depth, int NumMatBuffers>
void FIFO_INIT_V2C(
    CrossCoreFIFO_V2C<VecTileT, MatTileT, Depth, NumMatBuffers>& fifo,
    const uint32_t (&l1AddrsMatTile)[NumMatBuffers],  // Array of MatTile L1 addrs
    const uint32_t (&ubAddrsAIV0)[NumMatBuffers],     // Array of VecTile UB addrs for AIV0
    const uint32_t (&ubAddrsAIV1)[NumMatBuffers],     // Array of VecTile UB addrs for AIV1
    uint16_t pushEventId,
    uint16_t popEventId
) {
    fifo.head = 0;
    fifo.tail = 0;
    for (int i = 0; i < NumMatBuffers; i++) {
        fifo.l1AddrsMatTile[i] = l1AddrsMatTile[i];
        fifo.ubAddrsAIV0[i] = ubAddrsAIV0[i];
        fifo.ubAddrsAIV1[i] = ubAddrsAIV1[i];
    }
    fifo.pushEventId = pushEventId;
    fifo.popEventId = popEventId;
}

// === STATIC SCHEDULE: FIFO manages buffer rotation ===
template <typename VecTileT, typename MatTileT, int Depth, int NumMatBuffers>
void PTO_PUSH_TO_AIC(
    const VecTileT& vecTile,
    CrossCoreFIFO_V2C<VecTileT, MatTileT, Depth, NumMatBuffers>& fifo,
    int aivId                          // Which AIV is pushing (0 or 1)
) {
    // Get source UB addr based on current buffer index
    uint32_t bufIdx = fifo.head % NumMatBuffers;
    uint32_t srcUBAddr = (aivId == 0) ? fifo.ubAddrsAIV0[bufIdx] : fifo.ubAddrsAIV1[bufIdx];
    uint32_t dstL1Addr = fifo.l1AddrsMatTile[bufIdx];
    
    // DMA from UB[srcUBAddr] → L1[dstL1Addr]
    // ... implementation ...
}

// === DYNAMIC SCHEDULE: User provides MatTile with L1 address ===
template <typename VecTileT, typename MatTileT>
void PTO_PUSH_TO_AIC(
    const VecTileT& srcVecTile,        // Source VecTile (has UB addr via getAddress())
    const MatTileT& dstMatTile,        // Target MatTile (has L1 addr via getAddress())
    uint16_t pushEventId
) {
    uint32_t srcUBAddr = srcVecTile.getAddress();
    uint32_t dstL1Addr = dstMatTile.getAddress();
    // DMA from UB[srcUBAddr] → L1[dstL1Addr]
    // ... implementation ...
}
```

### Shape Inference for PTO_PUSH_TO_AIV

```cpp
// AIC → AIV: Push accumulator tile to vector core(s)
// aivId: 0 = AIV0 only, 1 = AIV1 only, -1 = dual-dst (both AIV0 and AIV1)
template <typename AccTileT, typename VecTileT>
void PTO_PUSH_TO_AIV(
    const AccTileT& accTile,             // Source tile (accumulator)
    CrossCoreFIFO<AccTileT, VecTileT>& fifo,
    int aivId = -1                       // -1 = dual-dst, 0 = AIV0, 1 = AIV1
) {
    // ============ COMPILE-TIME SHAPE INFERENCE ============
    constexpr int accM = TileTraits<AccTileT>::Rows;
    constexpr int accN = TileTraits<AccTileT>::Cols;
    constexpr int vecM = TileTraits<VecTileT>::Rows;
    constexpr int vecN = TileTraits<VecTileT>::Cols;
    
    // Infer split mode from shape relationship
    constexpr bool cutM = (accM == 2 * vecM) && (accN == vecN);  // Row split
    constexpr bool cutN = (accN == 2 * vecN) && (accM == vecM);  // Col split
    constexpr bool sameShape = (accM == vecM) && (accN == vecN); // 1:1 mapping
    
    // ============ ASSERTIONS ============
    // Must be exactly one of: cutM, cutN, or sameShape
    static_assert(cutM || cutN || sameShape,
        "Invalid tile shape relationship: AccTile must be either "
        "(2*vecM, vecN) for cutM, (vecM, 2*vecN) for cutN, or (vecM, vecN) for 1:1");
    
    static_assert(!(cutM && cutN),
        "Ambiguous tile shape: cannot have both cutM and cutN");
    
    // Dual-dst requires cutM or cutN (shape mismatch)
    static_assert(!(aivId == -1 && sameShape),
        "Dual-dst (aivId=-1) requires shape mismatch: "
        "AccTile[M,N] -> VecTile[M/2,N] (cutM) or AccTile[M,N] -> VecTile[M,N/2] (cutN). "
        "For 1:1 shape, use aivId=0 or aivId=1 explicitly.");
    
    // Single-dst with shape mismatch is allowed (send to one core only)
    // aivId=0 or aivId=1 with cutM/cutN: only that core receives its half
    
    // ============ IMPLEMENTATION ============
    if constexpr (sameShape) {
        // 1:1 mapping: send entire tile to specified AIV
        assert(aivId == 0 || aivId == 1);  // Must specify which AIV
        uint32_t dstUBAddr = (aivId == 0) ? fifo.ubAddrAIV0 : fifo.ubAddrAIV1;
        // Copy accTile → L1 slot → UB[dstUBAddr]
        // ... TSTORE accTile to L1, then DMA L1 → UB[dstUBAddr] ...
    } else if constexpr (cutM) {
        // Row split: AccTile[M,N] -> VecTile[M/2,N]
        if (aivId == -1) {
            // Dual-dst: AIV0 gets rows[0:M/2], AIV1 gets rows[M/2:M]
            // Copy accTile[0:M/2, :] → UB[ubAddrAIV0]
            // Copy accTile[M/2:M, :] → UB[ubAddrAIV1]
        } else {
            // Single-dst: only send to specified AIV
            uint32_t dstUBAddr = (aivId == 0) ? fifo.ubAddrAIV0 : fifo.ubAddrAIV1;
            uint32_t rowOffset = (aivId == 0) ? 0 : (accM / 2);
            // Copy accTile[rowOffset:rowOffset+M/2, :] → UB[dstUBAddr]
        }
    } else if constexpr (cutN) {
        // Col split: AccTile[M,N] -> VecTile[M,N/2]
        if (aivId == -1) {
            // Dual-dst: AIV0 gets cols[0:N/2], AIV1 gets cols[N/2:N]
            // Copy accTile[:, 0:N/2] → UB[ubAddrAIV0]
            // Copy accTile[:, N/2:N] → UB[ubAddrAIV1]
        } else {
            // Single-dst: only send to specified AIV
            uint32_t dstUBAddr = (aivId == 0) ? fifo.ubAddrAIV0 : fifo.ubAddrAIV1;
            uint32_t colOffset = (aivId == 0) ? 0 : (accN / 2);
            // Copy accTile[:, colOffset:colOffset+N/2] → UB[dstUBAddr]
        }
    }
}
```

**Shape Relationship Rules:**

| AccTile Shape | VecTile Shape | Mode | aivId Options |
|---------------|---------------|------|---------------|
| `[M, N]` | `[M/2, N]` | cutM (row split) | -1, 0, 1 |
| `[M, N]` | `[M, N/2]` | cutN (col split) | -1, 0, 1 |
| `[M, N]` | `[M, N]` | sameShape (1:1) | 0, 1 only |

**Examples:**
```cpp
// cutM: 128x128 -> 64x128 (row split)
using AccTile = TileAcc<float, 128, 128>;
using VecTile = Tile<TileType::Vec, half, 64, 128, ...>;
PTO_PUSH_TO_AIV(accTile, fifo, -1);  // OK: dual-dst cutM
PTO_PUSH_TO_AIV(accTile, fifo, 0);   // OK: AIV0 gets rows[0:64]
PTO_PUSH_TO_AIV(accTile, fifo, 1);   // OK: AIV1 gets rows[64:128]

// cutN: 64x256 -> 64x128 (col split)
using AccTile = TileAcc<float, 64, 256>;
using VecTile = Tile<TileType::Vec, half, 64, 128, ...>;
PTO_PUSH_TO_AIV(accTile, fifo, -1);  // OK: dual-dst cutN

// sameShape: 64x128 -> 64x128 (1:1)
using AccTile = TileAcc<float, 64, 128>;
using VecTile = Tile<TileType::Vec, float, 64, 128, ...>;
PTO_PUSH_TO_AIV(accTile, fifo, -1);  // ERROR: static_assert fails
PTO_PUSH_TO_AIV(accTile, fifo, 0);   // OK: send to AIV0
PTO_PUSH_TO_AIV(accTile, fifo, 1);   // OK: send to AIV1
```

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
template <typename AccTileT, typename VecTileT>
void PTO_PUSH_TO_AIV(
    const AccTileT& accTile,
    CrossCoreFIFO<AccTileT, VecTileT>& fifo,
    int aivId = -1
) {
    constexpr int accM = TileTraits<AccTileT>::Rows;
    constexpr int accN = TileTraits<AccTileT>::Cols;
    constexpr bool cutM = (accM == 2 * TileTraits<VecTileT>::Rows);
    constexpr bool cutN = (accN == 2 * TileTraits<VecTileT>::Cols);
    
    // 1. Wait for slot available (backpressure from consumer)
    if (fifo.head - fifo.tail >= Depth) {
        WAIT_EVENT(fifo.popEventId);  // Consumer freed a slot
    }
    
    // 2. Calculate slot address in L1
    uint32_t slotIdx = fifo.head % Depth;
    void* l1SlotAddr = (char*)fifo.l1BufferBase + slotIdx * fifo.slotSize;
    
    // 3. Copy tile to L1 slot
    TSTORE_TO_L1(accTile, l1SlotAddr);
    
    // 4. DMA from L1 to UB (based on split mode)
    if constexpr (cutM) {
        if (aivId == -1 || aivId == 0) {
            // Copy rows[0:M/2] to AIV0's UB
            DMA_L1_TO_UB(l1SlotAddr, fifo.ubAddrAIV0, fifo.vecTileSize, /*rowOffset=*/0);
        }
        if (aivId == -1 || aivId == 1) {
            // Copy rows[M/2:M] to AIV1's UB
            uint32_t srcOffset = (accM / 2) * accN * sizeof(AccTileT::DType);
            DMA_L1_TO_UB((char*)l1SlotAddr + srcOffset, fifo.ubAddrAIV1, fifo.vecTileSize, /*rowOffset=*/0);
        }
    } else if constexpr (cutN) {
        if (aivId == -1 || aivId == 0) {
            // Copy cols[0:N/2] to AIV0's UB (strided copy)
            DMA_L1_TO_UB_STRIDED(l1SlotAddr, fifo.ubAddrAIV0, accM, accN/2, /*srcStride=*/accN, /*colOffset=*/0);
        }
        if (aivId == -1 || aivId == 1) {
            // Copy cols[N/2:N] to AIV1's UB
            DMA_L1_TO_UB_STRIDED(l1SlotAddr, fifo.ubAddrAIV1, accM, accN/2, /*srcStride=*/accN, /*colOffset=*/accN/2);
        }
    } else {
        // sameShape: copy entire tile to specified AIV
        uint32_t dstUBAddr = (aivId == 0) ? fifo.ubAddrAIV0 : fifo.ubAddrAIV1;
        DMA_L1_TO_UB(l1SlotAddr, dstUBAddr, fifo.vecTileSize, 0);
    }
    
    // 5. Increment head
    fifo.head++;
    
    // 6. Signal consumer(s)
    if (aivId == -1) {
        // Dual-dst: signal both AIV0 and AIV1
        SET_CROSS_CORE_EVENT(fifo.pushEventId, 0);
        SET_CROSS_CORE_EVENT(fifo.pushEventId, 1);
    } else {
        SET_CROSS_CORE_EVENT(fifo.pushEventId, aivId);
    }
}
```

### What TPOP Does Internally

```cpp
template <typename AccTileT, typename VecTileT>
void PTO_POP_FROM_AIC(
    VecTileT& vecTile,
    CrossCoreFIFO<AccTileT, VecTileT>& fifo
) {
    // Get this core's ID (0 = AIV0, 1 = AIV1)
    int coreId = get_subblockid();
    
    // 1. Wait for data available
    if (fifo.tail >= fifo.head) {
        WAIT_EVENT(fifo.pushEventId);  // Producer pushed data
    }
    
    // 2. Data is already in our UB (TPUSH did the DMA)
    // The vecTile was assigned to ubAddrAIV0 or ubAddrAIV1 during TASSIGN
    // TPUSH copied data directly to that address
    
    // 3. Increment tail (local to this core)
    fifo.tail++;
    
    // 4. Signal producer (slot freed)
    SET_CROSS_CORE_EVENT(fifo.popEventId, coreId);
    
    // Note: vecTile is now ready to use - data was written by TPUSH
    // No additional copy needed since TPUSH targeted our UB address
}
```

**Key insight:** TPUSH writes directly to the AIV's UB address stored in the FIFO. When AIV calls TPOP, the data is already in its UB - TPOP just handles synchronization.
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
// === STATIC SCHEDULE: Ping-pong buffers (2 VecTiles per AIV) ===
constexpr uint32_t UB_ADDRS_AIV0[2] = {0x0000, 0x8000};  // Ping-pong for AIV0
constexpr uint32_t UB_ADDRS_AIV1[2] = {0x0000, 0x8000};  // Ping-pong for AIV1

// FIFO with Depth=2, NumVecBuffers=2 (ping-pong)
CrossCoreFIFO<TileQKData, TileDataH_T, 2, 2> qkFIFO;

FIFO_INIT(qkFIFO,
    L1_BUFFER_ADDR,           // Pre-allocated L1 buffer
    UB_ADDRS_AIV0,            // Array of AIV0's VecTile UB addresses
    UB_ADDRS_AIV1,            // Array of AIV1's VecTile UB addresses
    EVENT_QK_PUSH,            // User-defined event ID
    EVENT_QK_POP              // User-defined event ID
);

// === DYNAMIC SCHEDULE (alternative): User passes VecTile directly ===
// TileDataH_T vecTilePing, vecTilePong;
// TASSIGN(vecTilePing, 0x0000);
// TASSIGN(vecTilePong, 0x8000);
// PTO_PUSH_TO_AIV(accTile, vecTilePing, vecTilePong, EVENT_QK_PUSH, -1);

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
