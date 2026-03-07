# Operator Fusion

This document introduces PTO operator fusion techniques, helping developers reduce memory access and improve overall performance by fusing multiple operators.

## Contents

- [1. Fusion Overview](#1-fusion-overview)
- [2. Fusion Pattern Classification](#2-fusion-pattern-classification)
- [3. Fusion Implementation](#3-fusion-implementation)
- [4. Fusion Benefits Analysis](#4-fusion-benefits-analysis)
- [5. Best Practices](#5-best-practices)

---

## 1. Fusion Overview

### 1.1 What is Operator Fusion

**Definition**: Combine multiple independent operators into a single operator, completing all computations in on-chip memory to reduce intermediate result storage and loading from GM.

**Core Idea**:
```
Traditional Approach:
  Kernel1: GM → L1 → Compute → L1 → GM
  Kernel2: GM → L1 → Compute → L1 → GM
  Kernel3: GM → L1 → Compute → L1 → GM

Fused Approach:
  FusedKernel: GM → L1 → Compute1 → Compute2 → Compute3 → L1 → GM
```

### 1.2 Fusion Advantages

#### Advantage 1: Reduce Memory Access

**Example: Add + ReLU + Mul**
```cpp
// Before fusion: 3 independent kernels
y = Add(x, bias);      // Load x, Store y
z = ReLU(y);           // Load y, Store z
out = Mul(z, scale);   // Load z, Store out

// Memory access statistics:
// - Load: 3 times (x, y, z)
// - Store: 3 times (y, z, out)
// - Total: 6 GM accesses

// After fusion: 1 fused kernel
out = FusedAddReLUMul(x, bias, scale);

// Memory access statistics:
// - Load: 1 time (x)
// - Store: 1 time (out)
// - Total: 2 GM accesses

// Memory access reduction: (6 - 2) / 6 = 67%
```

#### Advantage 2: Reduce Kernel Launch Overhead

**Kernel Launch Overhead**:
- Each kernel launch: ~10-50 μs
- 3 independent kernels: 30-150 μs
- 1 fused kernel: 10-50 μs
- Savings: 20-100 μs

#### Advantage 3: Improve Data Locality

**Cache Hit Rate Improvement**:
```
Before fusion:
  - Intermediate results written back to GM
  - May be evicted by other cores' data
  - Next operator reloads (cache miss)

After fusion:
  - Intermediate results stay in L1
  - No cache eviction
  - 100% cache hit
```

---

## 2. Fusion Pattern Classification

### 2.1 Element-wise Fusion

**Characteristics**:
- All operators are element-wise operations
- No data dependencies
- Simplest fusion, highest benefits

**Common Patterns**:
```cpp
// Pattern 1: Add + ReLU
out = ReLU(Add(x, bias));

// Pattern 2: Add + ReLU + Mul
out = Mul(ReLU(Add(x, bias)), scale);

// Pattern 3: Add + BatchNorm + ReLU
out = ReLU(BatchNorm(Add(x, bias)));

// Pattern 4: Mul + Add + Sigmoid
out = Sigmoid(Add(Mul(x, scale), bias));
```

**Implementation Example**:
```cpp
__global__ __aicore__ void FusedAddReLUMul(
    __gm__ float* out,
    __gm__ const float* in,
    float bias,
    float scale,
    uint32_t length) {
  
  int block_idx = get_block_idx();
  int block_num = get_block_num();
  
  int elements_per_block = (length + block_num - 1) / block_num;
  int start = block_idx * elements_per_block;
  int end = min(start + elements_per_block, length);
  
  using TileT = Tile<TileType::Vec, float, 16, 256>;
  
  for (int i = start; i < end; i += 16 * 256) {
    TileT tile;
    
    // Load data
    TLOAD(tile, GlobalTensor(in + i));
    
    // Fused computation: Add + ReLU + Mul
    TADDS(tile, tile, bias);    // Add
    TRELU(tile, tile);          // ReLU
    TMULS(tile, tile, scale);   // Mul
    
    // Store result
    TSTORE(GlobalTensor(out + i), tile);
  }
}
```

**Performance Analysis**:
```
Data size: 1M elements (4 MB)
Platform: A3 (24 cores)

Before fusion:
- Add: 0.05 ms
- ReLU: 0.05 ms
- Mul: 0.05 ms
- Total: 0.15 ms

After fusion:
- FusedAddReLUMul: 0.05 ms

Speedup: 3×
```

### 2.2 Reduction Fusion

**Characteristics**:
- Includes reduction operations (sum, max, min)
- Need to preserve reduction results
- Medium complexity fusion

**Common Patterns**:
```cpp
// Pattern 1: Softmax
// max → sub → exp → sum → div
out = exp(x - max(x)) / sum(exp(x - max(x)))

// Pattern 2: LayerNorm
// mean → sub → square → mean → sqrt → div
out = (x - mean(x)) / sqrt(mean((x - mean(x))^2) + eps)

// Pattern 3: RMSNorm
// square → mean → sqrt → div
out = x / sqrt(mean(x^2) + eps)
```

**Softmax Fusion Implementation**:
```cpp
__global__ __aicore__ void FusedSoftmax(
    __gm__ float* out,
    __gm__ const float* in,
    int rows,
    int cols) {
  
  int block_idx = get_block_idx();
  
  // Each core processes one row
  if (block_idx >= rows) return;
  
  using TileVec = Tile<TileType::Vec, float, 1, 256>;
  using TileScalar = Tile<TileType::Vec, float, 1, 1>;
  
  TileVec input, shifted, exp_vals, output;
  TileScalar max_val, sum_val;
  
  for (int col = 0; col < cols; col += 256) {
    int size = min(256, cols - col);
    
    // Load input
    TLOAD(input, in[block_idx * cols + col : size]);
    
    // Step 1: Compute max
    TROWMAX(max_val, input);
    
    // Step 2: Subtract max (numerical stability)
    TROWEXPANDSUB(shifted, input, max_val);
    
    // Step 3: Compute exponential
    TEXP(exp_vals, shifted);
    
    // Step 4: Compute sum
    TROWSUM(sum_val, exp_vals);
    
    // Step 5: Normalize
    TROWEXPANDDIV(output, exp_vals, sum_val);
    
    // Store result
    TSTORE(out[block_idx * cols + col : size], output);
  }
}
```

### 2.3 Matrix Fusion

**Characteristics**:
- Includes matrix multiplication
- Fuse post-processing (Bias, Activation)
- High performance benefits

**Common Patterns**:
```cpp
// Pattern 1: GEMM + Bias
out = MatMul(A, B) + bias

// Pattern 2: GEMM + Bias + ReLU
out = ReLU(MatMul(A, B) + bias)

// Pattern 3: GEMM + Bias + GELU
out = GELU(MatMul(A, B) + bias)

// Pattern 4: GEMM + Residual + LayerNorm
out = LayerNorm(MatMul(A, B) + residual)
```

**GEMM + Bias + ReLU Fusion Implementation**:
```cpp
__global__ __aicore__ void FusedGEMMBiasReLU(
    __gm__ float* C,
    __gm__ const float* A,
    __gm__ const float* B,
    __gm__ const float* bias,
    int M, int K, int N) {
  
  int block_idx = get_block_idx();
  
  // 2D partitioning
  int blocks_n = (N + TILE_N - 1) / TILE_N;
  int block_m = block_idx / blocks_n;
  int block_n = block_idx % blocks_n;
  
  int m_start = block_m * TILE_M;
  int n_start = block_n * TILE_N;
  
  if (m_start >= M || n_start >= N) return;
  
  using TileLeft = TileLeft<half, 128, 64>;
  using TileRight = TileRight<half, 64, 256>;
  using TileAcc = TileAcc<float, 128, 256>;
  using TileBias = Tile<TileType::Vec, float, 1, 256>;
  
  TileAcc acc;
  TFILL(acc, 0);
  
  // Matrix multiplication
  for (int k = 0; k < K; k += 64) {
    TileLeft tileA;
    TileRight tileB;
    
    TLOAD(tileA, A[m_start:128, k:64]);
    TLOAD(tileB, B[k:64, n_start:256]);
    TMATMUL_ACC(acc, tileA, tileB);
  }
  
  // Fuse Bias
  TileBias bias_tile;
  TLOAD(bias_tile, bias[n_start:256]);
  TROWEXPANDADD(acc, acc, bias_tile);
  
  // Fuse ReLU
  TRELU(acc, acc);
  
  // Store result
  TSTORE(C[m_start:128, n_start:256], acc);
}
```

---

## 3. Fusion Implementation

### 3.1 Manual Fusion Steps

**Step 1: Identify Fusion Opportunities**
```python
# Analyze computation graph
def analyze_fusion_opportunities(graph):
    candidates = []
    
    for node in graph.nodes:
        # Find consecutive element-wise operations
        if is_elementwise(node):
            chain = find_elementwise_chain(node)
            if len(chain) >= 2:
                candidates.append(chain)
    
    return candidates
```

**Step 2: Verify Fusion Feasibility**
```cpp
// Checklist
bool can_fuse(Op op1, Op op2) {
  // 1. Check data dependencies
  if (op2.input != op1.output) return false;
  
  // 2. Check if intermediate result is used by other operators
  if (op1.output.num_users > 1) return false;
  
  // 3. Check on-chip memory capacity
  size_t required_memory = op1.memory + op2.memory;
  if (required_memory > L1_CAPACITY) return false;
  
  // 4. Check data type compatibility
  if (op1.output_type != op2.input_type) return false;
  
  return true;
}
```

**Step 3: Implement Fused Kernel**
```cpp
// Templated fused kernel
template<typename Op1, typename Op2, typename Op3>
__global__ __aicore__ void FusedKernel(
    __gm__ float* out,
    __gm__ const float* in,
    Op1 op1, Op2 op2, Op3 op3) {
  
  using TileT = Tile<TileType::Vec, float, 16, 256>;
  TileT tile;
  
  TLOAD(tile, in);
  
  // Execute fused operations sequentially
  op1(tile, tile);
  op2(tile, tile);
  op3(tile, tile);
  
  TSTORE(out, tile);
}
```

---

## 4. Fusion Benefits Analysis

### 4.1 Theoretical Benefits Calculation

**Formula**:
```
Speedup = T_unfused / T_fused

Where:
T_unfused = Σ(T_compute_i + T_memory_i + T_launch_i)
T_fused = T_compute_fused + T_memory_fused + T_launch_fused

Typically:
T_compute_fused ≈ Σ T_compute_i (compute time unchanged)
T_memory_fused << Σ T_memory_i (memory access greatly reduced)
T_launch_fused << Σ T_launch_i (launch overhead reduced)
```

**Example Calculation**:
```
Add + ReLU + Mul fusion:

Before fusion:
- Add: 0.01 ms (compute) + 0.04 ms (memory) + 0.02 ms (launch) = 0.07 ms
- ReLU: 0.01 ms + 0.04 ms + 0.02 ms = 0.07 ms
- Mul: 0.01 ms + 0.04 ms + 0.02 ms = 0.07 ms
- Total: 0.21 ms

After fusion:
- Compute: 0.03 ms (3 operations)
- Memory: 0.04 ms (load and store once only)
- Launch: 0.02 ms (launch once only)
- Total: 0.09 ms

Speedup: 0.21 / 0.09 = 2.3×
```

---

## 5. Best Practices

### 5.1 Design Principles

✅ **DO**:
- Prioritize fusing element-wise operations
- Fuse Softmax and other reduction operations
- Fuse Bias and Activation after GEMM
- Keep fused kernels simple and understandable
- Measure actual performance benefits

❌ **DON'T**:
- Don't fuse operators whose intermediate results are used multiple times
- Don't fuse operations that cause L1 overflow
- Don't over-fuse (maintain maintainability)
- Don't assume fusion is always faster (need to measure)

### 5.2 Fusion Checklist

**Before Fusion**:
- [ ] Intermediate result used only once
- [ ] Fusion doesn't exceed L1 capacity
- [ ] Data types compatible
- [ ] No complex control flow

**After Fusion**:
- [ ] Numerical correctness verified
- [ ] Performance improvement > 20%
- [ ] Code maintainability good
- [ ] Performance regression tests established

---

## References

- [Performance Optimization Guide](opt.md)
- [Memory Optimization](memory-optimization.md)
- [Performance Best Practices](performance-best-practices.md)
- [Pipeline and Parallel Execution](pipeline-parallel.md)

