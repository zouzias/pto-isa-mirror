# Memory Optimization

This document provides memory optimization techniques for PTO operator development, helping developers fully utilize on-chip memory and reduce memory access overhead.

## Contents

- [1. Memory Hierarchy](#1-memory-hierarchy)
- [2. L1 Memory Optimization](#2-l1-memory-optimization)
- [3. L0 Memory Optimization](#3-l0-memory-optimization)
- [4. Memory Alignment](#4-memory-alignment)
- [5. Memory Reuse Strategies](#5-memory-reuse-strategies)
- [6. Memory Bandwidth Optimization](#6-memory-bandwidth-optimization)

---

## 1. Memory Hierarchy

### 1.1 Ascend Memory Architecture

```
Global Memory (GM)
    ↓ MTE2 (DMA)
L1 Cache (~512KB - 1MB/core)
    ↓ MTE1
L0 Cache (register level)
    ↓
Compute Units (CUBE/VECTOR)
```

**Access Latency Comparison**:

| Memory Level | Latency | Bandwidth |
|--------------|---------|-----------|
| L0 | ~1 cycle | Highest |
| L1 | ~10 cycles | High |
| GM | ~100+ cycles | Limited |

**Optimization Principles**:
- Complete computation in L0 as much as possible
- Cache frequently accessed data in L1
- Reduce GM access count

---

## 2. L1 Memory Optimization

### 2.1 L1 Capacity Planning

**Platform Capacity**:
- A2/A3: ~512 KB/core
- A5: ~1 MB/core

**Capacity Allocation Example (GEMM)**:
```cpp
// A2/A3 platform
constexpr int L1_CAPACITY = 512 * 1024;  // 512KB

// Double buffer A Tile: 2 × 128×64×2 = 32KB
constexpr int A_BUFFER_SIZE = 2 * 128 * 64 * sizeof(half);

// Double buffer B Tile: 2 × 64×256×2 = 64KB
constexpr int B_BUFFER_SIZE = 2 * 64 * 256 * sizeof(half);

// Accumulator: 128×256×4 = 128KB
constexpr int ACC_BUFFER_SIZE = 128 * 256 * sizeof(float);

// Total: 32 + 64 + 128 = 224KB < 512KB ✓
```

### 2.2 Avoid L1 Overflow

**Check Method**:
```cpp
// Compile-time check
static_assert(TOTAL_L1_USAGE < L1_CAPACITY, 
              "L1 memory overflow!");

// Runtime check (debug mode)
#ifdef DEBUG
  size_t total_usage = calculate_l1_usage();
  assert(total_usage < L1_CAPACITY);
#endif
```

**Overflow Consequences**:
- Performance drops dramatically (10-100×)
- Frequent L1 ↔ GM paging
- Pipeline stalls

**Solutions**:
```cpp
// Solution 1: Reduce Tile size
// Before: 
using TileT = Tile<TileType::Vec, float, 32, 512>;  // 64KB

// After:
using TileT = Tile<TileType::Vec, float, 16, 256>;  // 16KB

// Solution 2: Reduce buffer count
// Change from triple buffering to double buffering
```

### 2.3 L1 Data Reuse

**Strategy 1: K-dimension Blocking**
```cpp
// GEMM example: Reuse A and B
for (int k = 0; k < K; k += TILE_K) {
  // Load to L1 once
  TLOAD(tileA_L1, A[m:m+M, k:k+TILE_K]);
  TLOAD(tileB_L1, B[k:k+TILE_K, n:n+N]);
  
  // Reuse multiple times in L1
  for (int sub_k = 0; sub_k < TILE_K; sub_k += SUB_K) {
    TEXTRACT(tileA_L0, tileA_L1[sub_k:sub_k+SUB_K]);
    TEXTRACT(tileB_L0, tileB_L1[sub_k:sub_k+SUB_K]);
    TMATMUL(acc, tileA_L0, tileB_L0);
  }
}
```

**Strategy 2: Cache Constant Data**
```cpp
// Softmax example: Cache max and sum
TLOAD(input_L1, input);
TROWMAX(max_val, input_L1);  // Compute once, cache in L1
TROWEXPANDSUB(shifted, input_L1, max_val);  // Reuse max_val
TEXP(exp_vals, shifted);
TROWSUM(sum_val, exp_vals);  // Compute once, cache in L1
TROWEXPANDDIV(output, exp_vals, sum_val);  // Reuse sum_val
```

---

## 3. L0 Memory Optimization

### 3.1 L0 Capacity Management

**L0 Types**:
- Vector L0 (UB): Vector registers
- Matrix L0A/L0B: Matrix operand registers
- Matrix L0C: Accumulator registers

**Capacity Limits**:
- Typically tens of KB
- Requires precise management

### 3.2 Register Allocation

**Manual Allocation (Manual Mode)**:
```cpp
// Use TASSIGN to bind addresses
constexpr int ADDR_A = 0;
constexpr int ADDR_B = 32 * 1024;  // 32KB offset
constexpr int ADDR_C = 64 * 1024;  // 64KB offset

TileLeft tileA;
TileRight tileB;
TileAcc tileC;

TASSIGN(tileA, ADDR_A);
TASSIGN(tileB, ADDR_B);
TASSIGN(tileC, ADDR_C);
```

**Automatic Allocation (Auto Mode)**:
```cpp
// Compiler manages automatically
TileLeft tileA;
TileRight tileB;
TileAcc tileC;
// No TASSIGN needed
```

### 3.3 Register Reuse

**Strategy 1: Temporal Reuse**
```cpp
// Reuse same register in different stages
TileVec temp;

// Stage 1: Use for intermediate result
TADD(temp, a, b);
TSTORE(output1, temp);

// Stage 2: Reuse for other data
TMUL(temp, c, d);
TSTORE(output2, temp);
```

---

## 4. Memory Alignment

### 4.1 Alignment Requirements

**Basic Alignment**:
- Row-major: `Cols × sizeof(T)` aligned to 32 bytes
- Column-major: `Rows × sizeof(T)` aligned to 32 bytes

**Example**:
```cpp
// Correct: Aligned to 32 bytes
using TileT1 = Tile<TileType::Vec, float, 16, 8>;
// 16 × 8 × 4 = 512 bytes (multiple of 32) ✓

// Wrong: Not aligned
using TileT2 = Tile<TileType::Vec, float, 16, 7>;
// 16 × 7 × 4 = 448 bytes (not multiple of 32) ✗
```

### 4.2 Padding Techniques

**Method 1: Compile-time Padding**
```cpp
// Original size: Not aligned
constexpr int ORIG_COLS = 127;

// Pad to alignment
constexpr int PADDED_COLS = ((ORIG_COLS + 7) / 8) * 8;  // 128

using TileT = Tile<TileType::Vec, float, 16, PADDED_COLS>;
```

**Method 2: Runtime Padding**
```cpp
// Use valid region to handle non-aligned data
using TileT = Tile<TileType::Vec, float, 16, 128,
                   RowMajor, 16, DYNAMIC>;  // Dynamic column count

TileT tile(actual_cols);  // actual_cols can be 127
```

---

## 5. Memory Reuse Strategies

### 5.1 Data Reuse Patterns

**Pattern 1: Temporal Reuse**
```cpp
// Same data used multiple times at different times
TLOAD(tile, data);
for (int i = 0; i < N; i++) {
  TCOMPUTE(result[i], tile, other[i]);  // Reuse tile
}
```

**Pattern 2: Spatial Reuse**
```cpp
// Adjacent data loaded together, used separately
TLOAD(tile_block, data[0:BLOCK_SIZE]);
for (int i = 0; i < BLOCK_SIZE; i++) {
  TEXTRACT(tile_i, tile_block, i);
  TCOMPUTE(result[i], tile_i);
}
```

### 5.2 GEMM Reuse Strategy

**Three-level Reuse**:
```cpp
// M×K×N GEMM
for (int m = 0; m < M; m += TILE_M) {
  for (int n = 0; n < N; n += TILE_N) {
    // Accumulator reuse: Entire K loop
    TileAcc acc;
    TFILL(acc, 0);
    
    for (int k = 0; k < K; k += TILE_K) {
      // A reuse: N direction
      TLOAD(tileA, A[m:m+TILE_M, k:k+TILE_K]);
      
      // B reuse: M direction
      TLOAD(tileB, B[k:k+TILE_K, n:n+TILE_N]);
      
      TMATMUL_ACC(acc, tileA, tileB);
    }
    
    TSTORE(C[m:m+TILE_M, n:n+TILE_N], acc);
  }
}
```

**Reuse Analysis**:
- Each element of A is reused N/TILE_N times
- Each element of B is reused M/TILE_M times
- Accumulator is reused K/TILE_K times

---

## 6. Memory Bandwidth Optimization

### 6.1 Reduce Memory Access

**Strategy 1: Operator Fusion**
```cpp
// Before fusion: 3 GM accesses
TLOAD(a, input1);
TADD(b, a, scalar);
TSTORE(output1, b);

TLOAD(c, output1);  // Redundant access
TMUL(d, c, scalar2);
TSTORE(output2, d);

// After fusion: 2 GM accesses
TLOAD(a, input1);
TADD(b, a, scalar);
TMUL(d, b, scalar2);  // Use b directly, no store and reload
TSTORE(output2, d);
```

**Strategy 2: Compute-intensive Design**
```cpp
// Increase arithmetic intensity = FLOPs / Bytes
// Goal: More computation per byte of data

// Bad: Low arithmetic intensity
for (int i = 0; i < N; i++) {
  TLOAD(tile, data[i]);     // 1 load
  TADD(result, tile, 1);    // 1 compute
  TSTORE(output[i], result); // 1 store
}
// Arithmetic intensity = 1 FLOP / (2 × tile_size bytes)

// Good: High arithmetic intensity
TLOAD(tile, data);           // 1 load
for (int i = 0; i < 100; i++) {
  TADD(tile, tile, 1);       // 100 computes
}
TSTORE(output, tile);        // 1 store
// Arithmetic intensity = 100 FLOPs / (2 × tile_size bytes)
```

### 6.2 Contiguous Access Pattern

**Row-major Optimization**:
```cpp
// Good: Contiguous access
for (int i = 0; i < M; i++) {
  TLOAD(tile, A[i, :]);  // Row contiguous
  TCOMPUTE(result, tile);
}

// Bad: Strided access
for (int j = 0; j < N; j++) {
  TLOAD(tile, A[:, j]);  // Column access, may not be contiguous
  TCOMPUTE(result, tile);
}
```

**Solution**:
```cpp
// Solution 1: Transpose data
TTRANS(A_T, A);
for (int j = 0; j < N; j++) {
  TLOAD(tile, A_T[j, :]);  // Now contiguous
  TCOMPUTE(result, tile);
}

// Solution 2: Use column-major layout
using TileT = Tile<TileType::Vec, float, M, N, ColMajor>;
```

### 6.3 Prefetch Optimization

**Software Prefetch**:
```cpp
// Prefetch next batch of data
for (int i = 0; i < N; i++) {
  // Process current data
  TCOMPUTE(result[i], tile[i]);
  
  // Prefetch next batch
  if (i + 1 < N) {
    TPREFETCH(tile[i+1], data[i+1]);
  }
}
```

**Double Buffering Prefetch**:
```cpp
// Preload first batch
TLOAD(tile[0], data[0]);

for (int i = 0; i < N; i++) {
  int curr = i % 2;
  int next = (i + 1) % 2;
  
  // Process current
  TCOMPUTE(result[i], tile[curr]);
  
  // Load next batch simultaneously
  if (i + 1 < N) {
    TLOAD(tile[next], data[i+1]);
  }
}
```

---

## References

- [Performance Optimization Guide](opt.md)
- [Pipeline and Parallel Execution](pipeline-parallel.md)
- [Performance Best Practices](performance-best-practices.md)
- [GEMM Optimization Case](../../kernels/manual/a2a3/gemm_performance/README.md)

