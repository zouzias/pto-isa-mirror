# 流水线与并行执行

本文档介绍 PTO 的流水线模型和并行执行机制，帮助开发者充分利用硬件资源实现高性能算子。

## 目录

- [1. 流水线概述](#1-流水线概述)
- [2. 硬件流水线](#2-硬件流水线)
- [3. 软件流水线](#3-软件流水线)
- [4. 并行执行模型](#4-并行执行模型)
- [5. 性能优化技巧](#5-性能优化技巧)

---

## 1. 流水线概述

### 1.1 什么是流水线

流水线是一种并行技术，将任务分解为多个阶段，不同阶段可以同时处理不同的数据。

**类比**：汽车装配线
- 阶段1：安装底盘
- 阶段2：安装发动机
- 阶段3：安装车身
- 阶段4：喷漆

当阶段2在处理车辆B时，阶段1可以同时处理车辆C。

### 1.2 PTO 中的流水线

PTO 算子通常包含以下阶段：

```
TLOAD → Transform → Compute → TSTORE
  ↓         ↓          ↓         ↓
 MTE2      MTE1      CUBE/VEC   MTE1
```

**关键思想**：让不同阶段重叠执行，提高硬件利用率。

---

## 2. 硬件流水线

### 2.1 Ascend 硬件流水线

Ascend AI 处理器包含多个独立的执行单元：

| 流水线 | 功能 | 典型指令 |
|--------|------|----------|
| **MTE2** | GM → L1 数据搬运 | `TLOAD` |
| **MTE1** | L1 → L0 数据搬运 | `TEXTRACT`, `TMOV` |
| **CUBE** | 矩阵乘法 | `TMATMUL` |
| **VECTOR** | 逐元素运算 | `TADD`, `TEXP`, `TMAX` |
| **SCALAR** | 标量运算和控制流 | 地址计算、循环控制 |

### 2.2 流水线并行

不同流水线可以**同时执行**：

```cpp
// 时间 T0: TLOAD 在 MTE2 上执行
TLOAD(tileA[0], ...);

// 时间 T1: TLOAD 继续，同时 TEXTRACT 在 MTE1 上执行
TLOAD(tileA[1], ...);
TEXTRACT(tileLeft[0], tileA[0]);

// 时间 T2: 三个流水线同时工作
TLOAD(tileA[2], ...);
TEXTRACT(tileLeft[1], tileA[1]);
TMATMUL(acc, tileLeft[0], tileRight[0]);
```

**性能提升**：理想情况下可以达到 3-4× 的吞吐量提升。

---

## 3. 软件流水线

### 3.1 双缓冲技术

双缓冲是最常用的软件流水线技术：

```cpp
// 基础版本（无流水线）
for (int i = 0; i < N; i++) {
  TLOAD(tile, ...);      // 等待加载
  TCOMPUTE(result, tile); // 等待计算
  TSTORE(..., result);    // 等待存储
}
// 总时间 = N × (T_load + T_compute + T_store)

// 双缓冲版本（有流水线）
TLOAD(tile[0], ...);  // 预加载第一个
for (int i = 0; i < N; i++) {
  int curr = i % 2;
  int next = (i + 1) % 2;
  
  // 当前迭代计算
  TCOMPUTE(result[curr], tile[curr]);
  
  // 同时加载下一个
  if (i + 1 < N) {
    TLOAD(tile[next], ...);
  }
  
  // 存储结果
  TSTORE(..., result[curr]);
}
// 总时间 ≈ N × max(T_load, T_compute, T_store)
```

**性能提升**：当三个阶段时间相近时，可以达到接近 3× 的加速。

### 3.2 多级流水线

对于复杂算子，可以使用多级流水线：

```cpp
// GEMM 的三级流水线
for (int k = 0; k < K; k += tileK) {
  int curr = k % 2;
  int next = (k + tileK) % 2;
  
  // 阶段1: MTE2 加载下一批数据
  if (k + tileK < K) {
    TLOAD(tileA_L1[next], ...);
    TLOAD(tileB_L1[next], ...);
  }
  
  // 阶段2: MTE1 提取当前数据到 L0
  TEXTRACT(tileA_L0[curr], tileA_L1[curr]);
  TEXTRACT(tileB_L0[curr], tileB_L1[curr]);
  
  // 阶段3: CUBE 计算
  TMATMUL(acc, tileA_L0[curr], tileB_L0[curr]);
}
```

### 3.3 事件同步

使用 Event 确保流水线的正确性：

```cpp
Event<Op::TLOAD, Op::TEXTRACT> e_load;
Event<Op::TEXTRACT, Op::TMATMUL> e_extract;
Event<Op::TMATMUL, Op::TMOV> e_compute;

for (int k = 0; k < K; k += tileK) {
  // 加载并记录事件
  e_load = TLOAD(tileA, ...);
  
  // 等待加载完成，然后提取
  e_extract = TEXTRACT(tileLeft, tileA, e_load);
  
  // 等待提取完成，然后计算
  e_compute = TMATMUL(acc, tileLeft, tileRight, e_extract);
}
```

**关键原则**：
- 只等待真实的数据依赖
- 避免不必要的全局同步
- 使用细粒度的 producer-consumer 事件

---

## 4. 并行执行模型

### 4.1 多核并行（Block 级）

PTO 支持多核并行执行，每个核处理不同的数据块：

```cpp
__global__ __aicore__ void MatMulKernel(...) {
  // 获取当前核的 ID
  int block_idx = get_block_idx();
  int block_m = block_idx / N_blocks;
  int block_n = block_idx % N_blocks;
  
  // 计算当前核负责的数据范围
  int m_start = block_m * TILE_M;
  int n_start = block_n * TILE_N;
  
  // 处理当前块
  for (int k = 0; k < K; k += TILE_K) {
    TLOAD(tileA, A[m_start:m_start+TILE_M, k:k+TILE_K]);
    TLOAD(tileB, B[k:k+TILE_K, n_start:n_start+TILE_N]);
    TMATMUL(acc, tileA, tileB);
  }
  
  TSTORE(C[m_start:m_start+TILE_M, n_start:n_start+TILE_N], acc);
}
```

**并行策略**：
- 2D 划分：同时切分 M 和 N 维度
- 负载均衡：确保每个核的工作量相近
- 数据局部性：减少核间通信

### 4.2 核内并行（Tile 级）

单个核内，Tile 操作本身是并行的：

```cpp
// TADD 会并行处理 Tile 中的所有元素
TADD(c, a, b);  // 16×16 = 256 个元素并行相加
```

**硬件实现**：
- Vector 单元：SIMD 并行处理多个元素
- Cube 单元：矩阵乘法的并行计算

### 4.3 流水线并行（阶段级）

如前所述，不同流水线阶段可以并行执行。

**三级并行**：
```
Block 级并行（多核）
  ↓
Tile 级并行（SIMD）
  ↓
Pipeline 级并行（阶段重叠）
```

---

## 5. 性能优化技巧

### 5.1 识别瓶颈

使用 profiler 分析各阶段的时间占比：

```bash
msprof --application="your_app" --output=./profiling_data
```

**瓶颈类型**：
- **TLOAD 占主导**：内存带宽受限 → 提升数据复用
- **TMATMUL 占主导**：计算受限 → 已接近理论峰值
- **TEXTRACT 占主导**：布局转换开销大 → 优化数据布局

### 5.2 优化流水线重叠

**目标**：让最慢的阶段决定总时间。

```cpp
// 不好的例子：串行执行
for (int i = 0; i < N; i++) {
  TLOAD(tile, ...);       // 10ms
  TCOMPUTE(result, tile); // 5ms
  TSTORE(..., result);    // 3ms
}
// 总时间 = N × 18ms

// 好的例子：流水线重叠
// 预加载 + 双缓冲
// 总时间 ≈ N × 10ms（由最慢的 TLOAD 决定）
```

### 5.3 调整 Tile 大小

**权衡**：
- **大 Tile**：更好的数据复用，但可能超出片上容量
- **小 Tile**：更灵活，但开销占比增大

```cpp
// 示例：GEMM 的 Tile 大小选择
// A2/A3: baseM=128, baseK=64, baseN=256
// A5: baseM=256, baseK=128, baseN=512（更大的片上容量）
```

### 5.4 减少同步开销

**原则**：
- 只在必要时同步
- 使用细粒度事件而非全局屏障
- 在稳态循环中避免 drain

```cpp
// 不好：每次迭代都全局同步
for (int i = 0; i < N; i++) {
  TLOAD(tile, ...);
  TSYNC<Op::TLOAD>();  // 全局同步，开销大
  TCOMPUTE(result, tile);
}

// 好：使用事件表达依赖
Event<Op::TLOAD, Op::TADD> e;
for (int i = 0; i < N; i++) {
  e = TLOAD(tile, ...);
  TCOMPUTE(result, tile, e);  // 只等待 TLOAD 完成
}
```

### 5.5 数据复用

**策略**：
- 在 L1 中缓存频繁访问的数据
- 在 K 维度分块以复用 A 和 B

```cpp
// GEMM 示例：K 维度分块
for (int k = 0; k < K; k += TILE_K) {
  TLOAD(tileA, A[m:m+M, k:k+TILE_K]);  // 加载一次
  TLOAD(tileB, B[k:k+TILE_K, n:n+N]);  // 加载一次
  TMATMUL(acc, tileA, tileB);          // 复用计算
}
// 每个元素被加载一次，但参与多次计算
```

---

## 6. 实战案例

### 6.1 GEMM 流水线优化

**优化前**：
```cpp
for (int k = 0; k < K; k += tileK) {
  TLOAD(tileA, ...);
  TLOAD(tileB, ...);
  TEXTRACT(tileLeft, tileA);
  TEXTRACT(tileRight, tileB);
  TMATMUL(acc, tileLeft, tileRight);
}
// TLOAD 占比 80%，TMATMUL 占比 15%
```

**优化后（双缓冲 + 流水线）**：
```cpp
// 预加载
TLOAD(tileA[0], ...);
TLOAD(tileB[0], ...);

for (int k = 0; k < K; k += tileK) {
  int curr = k % 2;
  int next = (k + tileK) % 2;
  
  // 提取当前数据
  TEXTRACT(tileLeft[curr], tileA[curr]);
  TEXTRACT(tileRight[curr], tileB[curr]);
  
  // 同时加载下一批
  if (k + tileK < K) {
    TLOAD(tileA[next], ...);
    TLOAD(tileB[next], ...);
  }
  
  // 计算
  TMATMUL(acc, tileLeft[curr], tileRight[curr]);
}
// TLOAD 占比 45%，TMATMUL 占比 55%
// 性能提升 3.2×
```

详细分析：[GEMM 性能优化案例](../../kernels/manual/a2a3/gemm_performance/README_zh.md)

### 6.2 Flash Attention 多阶段流水线

Flash Attention 包含多个计算阶段，需要精心设计流水线：

```cpp
// 阶段1: 计算 QK^T
// 阶段2: Softmax
// 阶段3: 计算 PV
// 阶段4: 更新输出

// 使用多级流水线重叠这些阶段
```

详细实现：[Flash Attention 优化](../../kernels/manual/common/flash_atten/README_zh.md)

---

## 7. 调试技巧

### 7.1 验证流水线正确性

**步骤**：
1. 先实现串行版本，验证正确性
2. 逐步添加流水线优化
3. 每次优化后验证数值结果

```cpp
#ifdef DEBUG
  // 在关键点检查中间结果
  float max_diff = CheckNumericalError(result, expected);
  assert(max_diff < 1e-5);
#endif
```

### 7.2 性能分析

**工具**：
- `msprof`：硬件性能分析
- 手动计时：关键阶段的时间测量

```cpp
// 手动计时示例
auto start = GetTime();
for (int i = 0; i < N; i++) {
  TLOAD(tile, ...);
}
auto end = GetTime();
printf("TLOAD time: %f ms\n", (end - start) / N);
```

---

## 8. 最佳实践总结

✅ **DO**：
- 使用双缓冲实现流水线重叠
- 用事件表达细粒度依赖
- 分析 profiler 数据识别瓶颈
- 调整 Tile 大小平衡复用和容量
- 在稳态循环中最大化重叠

❌ **DON'T**：
- 过度同步（避免不必要的全局屏障）
- 忽略数据复用机会
- Tile 过大导致溢出片上容量
- 在优化前不验证正确性

---

## 参考资源

- [编程模型](ProgrammingModel_zh.md)
- [事件与同步](Event_zh.md)
- [性能优化指南](opt_zh.md)
- [GEMM 优化案例](../../kernels/manual/a2a3/gemm_performance/README_zh.md)
- [Flash Attention 案例](../../kernels/manual/common/flash_atten/README_zh.md)





