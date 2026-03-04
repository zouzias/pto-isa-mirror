# CPU Simulation: TASSIGN Tile Aliasing

## Problem

On NPU hardware, `TASSIGN(tile, ubOffset)` makes a tile point to a physical Unified Buffer (UB) address. Multiple tiles can share the same UB address, enabling operations like:

```cpp
TASSIGN(tileA, 0x1000);  // tileA points to UB offset 0x1000
TASSIGN(tileB, 0x1000);  // tileB points to same location
// Now tileA and tileB share memory — writes to one are visible in the other
```

This is used in `batch_paged_attention` (PR #399) for `TFILLPAD_INPLACE`:

```cpp
TASSIGN(scores, SCORES_OFF);
ScoresDyn scoresDynTile(valid_len);
TASSIGN(scoresDynTile, SCORES_OFF);
ScoresPad scoresPadTile;
TASSIGN(scoresPadTile, SCORES_OFF);
TFILLPAD_INPLACE(scoresPadTile, scoresDynTile);
// scores now sees the padded result
```

**Prior CPU sim behavior:** `TASSIGN` was a no-op for tiles. Tiles used fixed arrays (`DType[N]`) embedded in the struct, making pointer redirection impossible. This broke any code relying on tile aliasing.

## Solution

### 1. Pointer-Based Tile Storage

Changed `TileDType` from fixed array to pointer:

```cpp
// Before (pto_tile.hpp line 1348)
using TileDType = typename Tile::DType[Rows * Cols];

// After
using TileDType = typename Tile::DType*;
```

### 2. TileBufferManager Singleton

New header `include/pto/cpu/TileBufferManager.hpp`:

```cpp
class TileBufferManager {
public:
    static TileBufferManager& Instance();

    // Allocate private buffer for tile constructor
    template<typename DType>
    DType* Allocate(std::size_t numElements);

    // Get shared buffer at UB offset (creates if needed)
    template<typename DType>
    DType* GetSharedBuffer(std::size_t ubOffset, std::size_t numElements);

    // Track first-assign per phase (loop iteration)
    bool IsFirstAssign(std::size_t ubOffset);
    void ResetPhase();

    void Clear();
};
```

### 3. Tile Constructor Allocation

All tile constructors now allocate via the manager:

```cpp
AICORE Tile() {
#ifdef __CPU_SIM
    data_ = TileBufferManager::Instance().Allocate<DType>(Rows * Cols);
#endif
}
```

### 4. TASSIGN Implementation

Updated `include/pto/cpu/TAssign.hpp`:

```cpp
template <typename T, typename AddrType>
void TASSIGN_IMPL(T &obj, AddrType addr) {
    if constexpr (is_tile_data_v<T>) {
        using DType = typename T::DType;
        constexpr std::size_t numElements = T::Rows * T::Cols;
        std::size_t ubOffset = static_cast<std::size_t>(addr);

        // First assign to this offset copies data to shared buffer
        bool isFirst = TileBufferManager::Instance().IsFirstAssign(ubOffset);
        DType* shared = TileBufferManager::Instance().GetSharedBuffer<DType>(
            ubOffset, numElements);

        if (isFirst && obj.data() != nullptr && obj.data() != shared) {
            std::memcpy(shared, obj.data(), numElements * sizeof(DType));
        }

        // Redirect tile to shared buffer
        obj.data() = shared;
    }
    // ... GlobalTensor handling unchanged
}
```

### 5. Phase Reset Helper

For loops that create new tiles each iteration:

```cpp
inline void TASSIGN_RESET_PHASE() {
    TileBufferManager::Instance().ResetPhase();
}
```

Usage:
```cpp
for (int bn = 0; bn < num_blocks; ++bn) {
    TASSIGN_RESET_PHASE();  // Reset first-assign tracking
    // ... tile operations with TASSIGN
}
```

## Test Case

`batch_paged_attention_demo` validates the implementation:

| Parameter | Value |
|-----------|-------|
| Batch | 2 |
| Heads | 16 |
| KV Heads | 1 |
| Head Dim | 16 |
| Block Size | 16 |
| Max Blocks | 4 |
| Context Lens | [33, 17] |

**Result:**
```
verify_allclose: abs_tol=0.01 rel_tol=0.01
  bad=0
  max_abs=2.98023e-08
  max_rel=1.18321e-07
[PASS] batch_paged_attention_demo
```

## Files Changed

| File | Change |
|------|--------|
| `include/pto/common/pto_tile.hpp` | TileDType → pointer; constructors allocate via manager |
| `include/pto/cpu/TileBufferManager.hpp` | New: buffer management singleton |
| `include/pto/cpu/TAssign.hpp` | Pointer aliasing logic + TASSIGN_RESET_PHASE |
| `demos/cpu/batch_paged_attention_demo/` | Uses TASSIGN API (same as NPU) |

## Design Rationale

1. **First-assign copies, subsequent assigns redirect** — Matches NPU semantics where the first tile written to a UB address establishes the data, and other tiles alias it.

2. **Phase reset per loop iteration** — Loop-created tiles need fresh tracking each iteration. Without reset, subsequent iterations would skip the data copy.

3. **Minimal API change** — `TASSIGN(tile, offset)` signature unchanged. Only CPU sim internals differ.

## Limitations

- `TASSIGN_RESET_PHASE()` must be called manually at loop boundaries in CPU sim demos
- Memory is not automatically freed (singleton lifetime)
- Not thread-safe (single-threaded CPU sim assumed)
