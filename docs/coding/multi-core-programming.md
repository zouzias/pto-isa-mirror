# Multi-core Programming

This document introduces PTO multi-core parallel programming techniques, helping developers fully utilize Ascend's multi-core architecture for high-performance operators.

## Contents

- [1. Multi-core Architecture Overview](#1-multi-core-architecture-overview)
- [2. SPMD Programming Pattern](#2-spmd-programming-pattern)
- [3. MPMD Programming Pattern](#3-mpmd-programming-pattern)
- [4. Load Balancing](#4-load-balancing)
- [5. Inter-core Communication](#5-inter-core-communication)
- [6. Performance Optimization](#6-performance-optimization)

---

## 1. Multi-core Architecture Overview

### 1.1 Ascend Multi-core Architecture

**Hardware Configuration**:
- A2/A3: 24 AI Cores
- A5: More cores (varies by model)

**Architecture Features**:
```
┌─────────────────────────────────┐
│         Host CPU                │
└────────────┬────────────────────┘
             │
    ┌────────┴────────┐
    │   NPU Device    │
    │  ┌───┬───┬───┐  │
    │  │C0 │C1 │...│  │  AI Cores
    │  └───┴───┴───┘  │
    │  ┌───────────┐  │
    │  │    GM     │  │  Global Memory
    │  └───────────┘  │
    └─────────────────┘
```

**Core Features**:
- Each core executes independently
- Shared global memory (GM)
- Independent L1 cache
- Inter-core communication via GM

### 1.2 Parallel Programming Models

**Two Main Patterns**:

| Pattern | Features | Use Case |
|---------|----------|----------|
| **SPMD** | All cores run same code | Regular data parallelism |
| **MPMD** | Different cores run different code | Pipeline, producer-consumer |

---

## 2. SPMD Programming Pattern

### 2.1 Basic Concept

**SPMD (Single Program, Multiple Data)**:
- All cores execute the same program
- Distinguish different data blocks via `block_idx`
- Most commonly used parallel pattern

### 2.2 Basic Example

**Vector Addition**:
```cpp
__global__ __aicore__ void VecAddKernel(__gm__ float* out,
                                        __gm__ const float* in0,
                                        __gm__ const float* in1,
                                        uint32_t totalLength) {
  // Get current core ID
  int block_idx = get_block_idx();
  int block_num = get_block_num();
  
  // Calculate data range for current core
  int elements_per_block = (totalLength + block_num - 1) / block_num;
  int start = block_idx * elements_per_block;
  int end = min(start + elements_per_block, totalLength);
  
  // Process current block
  for (int i = start; i < end; i += TILE_SIZE) {
    int size = min(TILE_SIZE, end - i);
    
    using TileT = Tile<TileType::Vec, float, 8, 256>;
    TileT a, b, c;
    
    TLOAD(a, GlobalTensor(in0 + i));
    TLOAD(b, GlobalTensor(in1 + i));
    TADD(c, a, b);
    TSTORE(GlobalTensor(out + i), c);
  }
}
```

### 2.3 2D Data Partitioning

**Matrix Multiplication Example**:
```cpp
__global__ __aicore__ void MatMulKernel(__gm__ float* C,
                                        __gm__ const float* A,
                                        __gm__ const float* B,
                                        int M, int K, int N) {
  // Get core ID
  int block_idx = get_block_idx();
  
  // 2D partitioning: M and N dimensions
  int blocks_m = (M + TILE_M - 1) / TILE_M;
  int blocks_n = (N + TILE_N - 1) / TILE_N;
  
  int block_m = block_idx / blocks_n;
  int block_n = block_idx % blocks_n;
  
  // Calculate matrix block for current core
  int m_start = block_m * TILE_M;
  int n_start = block_n * TILE_N;
  
  // Ensure no out-of-bounds
  if (m_start >= M || n_start >= N) return;
  
  int m_size = min(TILE_M, M - m_start);
  int n_size = min(TILE_N, N - n_start);
  
  // Execute matrix multiplication
  TileAcc acc;
  TFILL(acc, 0);
  
  for (int k = 0; k < K; k += TILE_K) {
    int k_size = min(TILE_K, K - k);
    
    TLOAD(tileA, A[m_start:m_start+m_size, k:k+k_size]);
    TLOAD(tileB, B[k:k+k_size, n_start:n_start+n_size]);
    TMATMUL_ACC(acc, tileA, tileB);
  }
  
  TSTORE(C[m_start:m_start+m_size, n_start:n_start+n_size], acc);
}
```

---

## 3. MPMD Programming Pattern

### 3.1 Basic Concept

**MPMD (Multiple Program, Multiple Data)**:
- Different cores execute different programs
- Suitable for pipeline and producer-consumer patterns
- Requires inter-core synchronization

### 3.2 Task Dispatch Pattern

**Method 1: Single Entry + Switch**
```cpp
__global__ __aicore__ void MPMDKernel(__gm__ float* out,
                                      __gm__ const float* in,
                                      uint32_t task_id) {
  switch (task_id) {
    case 0:
      ProducerTask(out, in);
      break;
    case 1:
      ConsumerTask(out, in);
      break;
    case 2:
      ProcessorTask(out, in);
      break;
    default:
      break;
  }
}
```

### 3.3 Pipeline Pattern

**Three-stage Pipeline**:
```cpp
__global__ __aicore__ void PipelineKernel(__gm__ float* out,
                                          __gm__ const float* in,
                                          uint32_t stage_id) {
  switch (stage_id) {
    case 0:  // Stage 1: Load
      for (int i = 0; i < N; i++) {
        TLOAD(buffer1[i], in[i]);
        signal_stage2();
      }
      break;
      
    case 1:  // Stage 2: Compute
      for (int i = 0; i < N; i++) {
        wait_stage1();
        TCOMPUTE(buffer2[i], buffer1[i]);
        signal_stage3();
      }
      break;
      
    case 2:  // Stage 3: Store
      for (int i = 0; i < N; i++) {
        wait_stage2();
        TSTORE(out[i], buffer2[i]);
      }
      break;
  }
}
```

---

## 4. Load Balancing

### 4.1 Static Load Balancing

**Uniform Partitioning**:
```cpp
// Method 1: Simple division
int elements_per_block = totalLength / block_num;
int start = block_idx * elements_per_block;
int end = (block_idx == block_num - 1) ? 
          totalLength : start + elements_per_block;

// Method 2: Ceiling division
int elements_per_block = (totalLength + block_num - 1) / block_num;
int start = block_idx * elements_per_block;
int end = min(start + elements_per_block, totalLength);
```

### 4.2 Load Imbalance Detection

**Detection Method**:
```cpp
// Record execution time for each core
#ifdef PROFILE
  auto start = GetTime();
  
  // Execute task
  process_block(block_idx);
  
  auto end = GetTime();
  execution_times[block_idx] = end - start;
#endif

// Analyze load balance
float max_time = *max_element(execution_times.begin(), 
                              execution_times.end());
float min_time = *min_element(execution_times.begin(), 
                              execution_times.end());
float imbalance = (max_time - min_time) / max_time;

if (imbalance > 0.2) {
  printf("Warning: Load imbalance detected: %.2f%%\n", 
         imbalance * 100);
}
```

---

## 5. Inter-core Communication

### 5.1 Communication via Global Memory

**Basic Pattern**:
```cpp
// Core 0: Write data
__global__ __aicore__ void Writer(__gm__ float* shared_buffer) {
  if (get_block_idx() == 0) {
    TLOAD(tile, local_data);
    TSTORE(shared_buffer, tile);
    // Set flag indicating data is ready
    shared_buffer[FLAG_OFFSET] = 1;
  }
}

// Core 1: Read data
__global__ __aicore__ void Reader(__gm__ float* shared_buffer) {
  if (get_block_idx() == 1) {
    // Wait for data ready
    while (shared_buffer[FLAG_OFFSET] != 1) {
      // Spin wait
    }
    TLOAD(tile, shared_buffer);
    process(tile);
  }
}
```

### 5.2 Synchronization with Atomic Operations

**Counter Synchronization**:
```cpp
__gm__ atomic<int> counter = 0;

__global__ __aicore__ void SyncKernel(...) {
  // Each core increments counter after completing work
  process_local_work();
  
  counter.fetch_add(1);
  
  // Wait for all cores to complete
  while (counter.load() < block_num) {
    // Spin wait
  }
  
  // Continue to next stage
  next_stage_work();
}
```

---

## 6. Performance Optimization

### 6.1 Reduce Inter-core Communication

**Strategy 1: Increase Data Block Size**
```cpp
// Bad: Frequent communication
for (int i = 0; i < N; i++) {
  process_small_block(i);
  sync_with_other_cores();  // Sync every time
}

// Good: Batch processing
for (int i = 0; i < N; i += BATCH_SIZE) {
  process_large_block(i, BATCH_SIZE);
  sync_with_other_cores();  // Batch sync
}
```

**Strategy 2: Localize Computation**
```cpp
// Make each core complete work independently
__global__ __aicore__ void LocalizedKernel(...) {
  int block_idx = get_block_idx();
  
  // Each core processes complete subproblem
  // No need to communicate with other cores
  process_independent_subproblem(block_idx);
}
```

### 6.2 Optimize Data Partitioning

**Consider Data Locality**:
```cpp
// 2D matrix: Partition by blocks rather than rows/columns
// Good: Each core accesses contiguous memory blocks
for (int bm = 0; bm < blocks_m; bm++) {
  for (int bn = 0; bn < blocks_n; bn++) {
    int block_id = bm * blocks_n + bn;
    if (block_id == get_block_idx()) {
      process_block(bm, bn);
    }
  }
}
```

### 6.3 Avoid False Sharing

**Problem**:
```cpp
// Bad: Multiple cores write to adjacent locations
__gm__ float results[NUM_CORES];

__global__ __aicore__ void BadKernel(...) {
  int idx = get_block_idx();
  results[idx] = compute();  // May cause cache line conflicts
}
```

**Solution**:
```cpp
// Good: Use padding to avoid false sharing
constexpr int CACHE_LINE_SIZE = 64;
constexpr int PADDING = CACHE_LINE_SIZE / sizeof(float);

__gm__ float results[NUM_CORES * PADDING];

__global__ __aicore__ void GoodKernel(...) {
  int idx = get_block_idx();
  results[idx * PADDING] = compute();  // Avoid cache line conflicts
}
```

---

## References

- [Programming Model](ProgrammingModel.md)
- [Pipeline and Parallel Execution](pipeline-parallel.md)
- [Performance Best Practices](performance-best-practices.md)
- [GEMM Optimization Case](../../kernels/manual/a2a3/gemm_performance/README.md)

