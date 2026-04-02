# Pipeline and Parallel Execution

This document introduces PTO's pipeline model and parallel execution mechanisms, helping developers fully utilize hardware resources for high-performance operators.

## Contents

- [1. Pipeline Overview](#1-pipeline-overview)
- [2. Hardware Pipeline](#2-hardware-pipeline)
- [3. Software Pipeline](#3-software-pipeline)
- [4. Parallel Execution Model](#4-parallel-execution-model)
- [5. Performance Optimization Tips](#5-performance-optimization-tips)

---

## 1. Pipeline Overview

### 1.1 What is Pipeline

Pipeline is a parallel technique that decomposes tasks into multiple stages, allowing different stages to process different data simultaneously.

**Analogy**: Car assembly line
- Stage 1: Install chassis
- Stage 2: Install engine
- Stage 3: Install body
- Stage 4: Paint

When Stage 2 processes car B, Stage 1 can simultaneously process car C.

### 1.2 Pipeline in PTO

PTO operators typically include the following stages:

```
TLOAD → Transform → Compute → TSTORE
  ↓         ↓          ↓         ↓
 MTE2      MTE1      CUBE/VEC   MTE1
```

**Key Idea**: Overlap execution of different stages to improve hardware utilization.

---

## 2. Hardware Pipeline

### 2.1 Ascend Hardware Pipeline

Ascend AI processors contain multiple independent execution units:

| Pipeline | Function | Typical Instructions |
|----------|----------|---------------------|
| **MTE2** | GM → L1 data transfer | `TLOAD` |
| **MTE1** | L1 → L0 data transfer | `TEXTRACT`, `TMOV` |
| **CUBE** | Matrix multiplication | `TMATMUL` |
| **VECTOR** | Element-wise operations | `TADD`, `TEXP`, `TMAX` |
| **SCALAR** | Scalar operations and control flow | Address calculation, loop control |

### 2.2 Pipeline Parallelism

Different pipelines can execute **simultaneously**:

```cpp
// Time T0: TLOAD executes on MTE2
TLOAD(tileA[0], ...);

// Time T1: TLOAD continues, TEXTRACT executes on MTE1 simultaneously
TLOAD(tileA[1], ...);
TEXTRACT(tileLeft[0], tileA[0]);

// Time T2: Three pipelines work simultaneously
TLOAD(tileA[2], ...);
TEXTRACT(tileLeft[1], tileA[1]);
TMATMUL(acc, tileLeft[0], tileRight[0]);
```

**Performance Gain**: Ideally can achieve 3-4× throughput improvement.

---

## 3. Software Pipeline

### 3.1 Double Buffering

Double buffering is the most commonly used software pipeline technique:

```cpp
// Basic version (no pipeline)
for (int i = 0; i < N; i++) {
  TLOAD(tile, ...);       // Wait for load
  TCOMPUTE(result, tile); // Wait for compute
  TSTORE(..., result);    // Wait for store
}
// Total time = N × (T_load + T_compute + T_store)

// Double buffering version (with pipeline)
TLOAD(tile[0], ...);  // Preload first
for (int i = 0; i < N; i++) {
  int curr = i % 2;
  int next = (i + 1) % 2;
  
  // Compute current iteration
  TCOMPUTE(result[curr], tile[curr]);
  
  // Load next iteration simultaneously
  if (i + 1 < N) {
    TLOAD(tile[next], ...);
  }
  
  // Store previous result
  if (i > 0) {
    TSTORE(..., result[1 - curr]);
  }
}
TSTORE(..., result[N % 2]);  // Store last result

// Total time ≈ max(T_load, T_compute, T_store) × N
// Speedup ≈ (T_load + T_compute + T_store) / max(...)
```

### 3.2 Triple Buffering

For more complex scenarios, use triple buffering:

```cpp
TileT tile[3];
Event load_event[3], compute_event[3];

// Preload first two
TLOAD(tile[0], ..., load_event[0]);
TLOAD(tile[1], ..., load_event[1]);

for (int i = 0; i < N; i++) {
  int curr = i % 3;
  int next = (i + 1) % 3;
  int prev = (i + 2) % 3;
  
  // Load next
  if (i + 2 < N) {
    TLOAD(tile[next], ..., load_event[next]);
  }
  
  // Compute current
  WAIT(load_event[curr]);
  TCOMPUTE(result, tile[curr], compute_event[curr]);
  
  // Store previous
  if (i > 0) {
    WAIT(compute_event[prev]);
    TSTORE(..., result);
  }
}
```

### 3.3 Event-based Synchronization

Use events for fine-grained synchronization:

```cpp
// Define event types
Event<Op::TLOAD, Op::TADD> load_event;
Event<Op::TADD, Op::TSTORE> compute_event;

// Load with event
load_event = TLOAD(tile, ...);

// Compute depends on load
compute_event = TADD(result, tile, ..., load_event);

// Store depends on compute
TSTORE(..., result, compute_event);
```

---

## 4. Parallel Execution Model

### 4.1 Multi-core Parallelism

**SPMD (Single Program, Multiple Data)**:
```cpp
__global__ __aicore__ void ParallelKernel(...) {
  int block_idx = get_block_idx();
  int block_num = get_block_num();
  
  // Each core processes different data
  int start = block_idx * elements_per_core;
  int end = min(start + elements_per_core, total_elements);
  
  for (int i = start; i < end; i++) {
    process(i);
  }
}
```

### 4.2 Instruction-level Parallelism

Different instruction types can execute in parallel:

```cpp
// These can execute simultaneously
TLOAD(tile_a, ...);      // MTE2
TEXTRACT(tile_b, ...);   // MTE1
TMATMUL(acc, ...);       // CUBE
TADD(vec_result, ...);   // VECTOR
```

### 4.3 Data-level Parallelism

Process multiple data elements simultaneously:

```cpp
// Tile operations process all elements in parallel
using TileT = Tile<TileType::Vec, float, 16, 256>;
TileT a, b, c;

TLOAD(a, ...);
TLOAD(b, ...);
TADD(c, a, b);  // All 16×256 elements computed in parallel
```

---

## 5. Performance Optimization Tips

### 5.1 Maximize Pipeline Overlap

**Strategy 1: Preload Data**
```cpp
// Preload first batch
TLOAD(tile[0], data[0]);

for (int i = 0; i < N; i++) {
  // Load next while processing current
  if (i + 1 < N) {
    TLOAD(tile[(i+1)%2], data[i+1]);
  }
  
  TCOMPUTE(result, tile[i%2]);
  TSTORE(output[i], result);
}
```

**Strategy 2: Use Events Instead of Global Sync**
```cpp
// Bad: Global synchronization
TLOAD(tile, ...);
TSYNC<Op::TLOAD>();  // Wait for all TLOAD
TADD(result, tile, ...);

// Good: Event-based synchronization
Event e = TLOAD(tile, ...);
TADD(result, tile, ..., e);  // Only wait for this TLOAD
```

### 5.2 Balance Pipeline Stages

**Identify Bottleneck**:
```
TLOAD:    40%  ← Bottleneck
TCOMPUTE: 20%
TSTORE:   10%
Idle:     30%
```

**Solution**: Increase compute intensity
```cpp
// Increase computation per load
for (int k = 0; k < K; k += TILE_K) {
  TLOAD(tileA, ...);  // Load once
  TLOAD(tileB, ...);
  
  // Reuse multiple times
  for (int sub_k = 0; sub_k < TILE_K; sub_k++) {
    TMATMUL(acc, tileA[sub_k], tileB[sub_k]);
  }
}
```

### 5.3 Reduce Synchronization Overhead

**Minimize WAIT Calls**:
```cpp
// Bad: Frequent synchronization
for (int i = 0; i < N; i++) {
  Event e = TLOAD(tile, ...);
  WAIT(e);  // Sync every iteration
  TCOMPUTE(result, tile);
}

// Good: Batch synchronization
Event events[BATCH_SIZE];
for (int i = 0; i < N; i += BATCH_SIZE) {
  // Load batch
  for (int j = 0; j < BATCH_SIZE; j++) {
    events[j] = TLOAD(tiles[j], ...);
  }
  
  // Process batch
  for (int j = 0; j < BATCH_SIZE; j++) {
    WAIT(events[j]);
    TCOMPUTE(results[j], tiles[j]);
  }
}
```

### 5.4 Optimize Memory Access Pattern

**Contiguous Access**:
```cpp
// Good: Sequential access
for (int i = 0; i < M; i++) {
  TLOAD(tile, A[i, :]);  // Row-major, contiguous
}

// Bad: Strided access
for (int j = 0; j < N; j++) {
  TLOAD(tile, A[:, j]);  // Column access, may be strided
}
```

**Prefetch**:
```cpp
// Prefetch next data
TPREFETCH(next_data, ...);
TCOMPUTE(current_data);
```

---

## References

- [Multi-core Programming](multi-core-programming.md)
- [Performance Best Practices](performance-best-practices.md)
- [Memory Optimization](memory-optimization.md)
- [GEMM Optimization Case](../../kernels/manual/a2a3/gemm_performance/README.md)

